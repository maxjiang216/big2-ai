"""Training loop for the imperfect-information PI-distillation net (Big2NetII).

ONE model, joint-trained per-GAME (same data path as train_az_seq): each batch
is a set of games; the belief transformer runs once per game over its token
sequence, hidden states are gathered at every sample's hist_idx, and one readout
serves all heads.

Heads / losses:
  * value    — BCE vs series outcome, PLAYER rows only (owner-to-move). There is
               no qa head: the az_ii search backs opponent nodes up from child
               values, so only player-to-move positions need a net value.
  * policy   — masked CE of composed (head @ C.T) logits vs the chosen move
               (distill data records {played:1}, so the visit dist IS one-hot).
  * behavior — masked NLL vs the chosen opponent move slot (opp rows).
  * outcome  — CE over the 32-way signed terminal-margin bucket (all rows, aux).
  * bomb     — BCE, opponent holds a 4-of-a-kind (all rows, aux, off belief e).
  * ar       — autoregressive opp-hand NLL (all rows, aux, off belief e). Logged
               as nats/row and per-rank perplexity.

Aux losses are weighted by has_aux (0 for legacy parquets w/o the columns), so
mixing old data stays safe.

Usage:
    uv run python nn/train_az_ii.py --data data/az_ii_gen1 \\
        --series-v data/series_v_pi.csv --out models/az_ii_gen1.pt \\
        --epochs 10 --batch-games 256

--data PREFIX expands to PREFIX_{games,player,opp}.parquet (the az_pi_selfplay
--distill-out triple). Pass it N times for generation mixing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from nn.dataset_seq import make_seq_split
from nn.model_az import load_compose_matrix
from nn.model_az_ii import AR_ORDER, Big2NetII, margin_to_bucket
from nn.model_az_seq import load_token_feats
from nn.train_az import _cur_lr, _device, _make_optim, _make_sched, masked_log_softmax

EPS = 1e-7


def _gather_h(H, local, hist):
    return H[local, hist]


def _step(model, C, batch, lam_outcome=1.0, lam_bomb=1.0, lam_ar=1.0, lam_play=1.0):
    H = model.forward_full(batch["tokens"])
    hp = _gather_h(H, batch["p_local"], batch["p_hist"])
    ho = _gather_h(H, batch["o_local"], batch["o_hist"])
    n_p, n_o = hp.shape[0], ho.shape[0]
    h = torch.cat([hp, ho], dim=0)
    hand = torch.cat([batch["p_hand"], batch["o_hand"]])
    oppm = torch.cat([batch["p_opp"], batch["o_opp"]])
    osz = torch.cat([batch["p_osz"], batch["o_osz"]])
    usz = torch.cat([batch["p_usz"], batch["o_usz"]])
    otm = torch.cat(
        [torch.ones(n_p, device=h.device), torch.zeros(n_o, device=h.device)]
    )
    mpts = torch.cat([batch["p_mpts"], batch["o_mpts"]])
    opts = torch.cat([batch["p_opts"], batch["o_opts"]])
    thermo_cnt = torch.cat([batch["p_oppmax_cnt"], batch["o_oppmax_cnt"]]).long()
    opp_cnt = torch.cat([batch["p_opp_hand_cnt"], batch["o_opp_hand_cnt"]]).long()

    value, policy, behavior, qa, outcome, bomb, ar_nll = model.readout_train(
        h, hand, oppm, osz, usz, otm, mpts, opts, thermo_cnt, opp_cnt
    )

    wp, wo = batch["p_w"], batch["o_w"]

    def _wmean(loss_per_row, w):
        return (loss_per_row * w).sum() / w.sum().clamp_min(EPS)

    # player rows: value BCE + composed-policy CE (one-hot chosen move)
    loss_v = _wmean(
        F.binary_cross_entropy(
            value[:n_p].clamp(EPS, 1 - EPS), batch["p_value"], reduction="none"
        ),
        wp,
    )
    logp = masked_log_softmax(policy[:n_p] @ C.t(), batch["p_mask"])
    loss_p = _wmean(-(batch["p_policy"] * logp).sum(-1), wp)
    # opp rows: behavior NLL vs chosen opp move
    tgt = batch["o_target"]
    logb = masked_log_softmax(behavior[n_p:], batch["o_mask"])
    loss_b = _wmean(F.nll_loss(logb, tgt, reduction="none"), wo)
    # qa BCE at the played opp slot (opp rows) — grounds opp-to-move node values
    # for the MCTS, vs the series outcome from the observer's POV.
    q = qa[n_p:].gather(1, tgt.unsqueeze(1)).squeeze(1)
    loss_q = _wmean(
        F.binary_cross_entropy(q.clamp(EPS, 1 - EPS), batch["o_value"], reduction="none"),
        wo,
    )

    # aux over ALL rows (gated by has_aux)
    waux = torch.cat([wp * batch["p_haux"], wo * batch["o_haux"]])
    margin_raw = torch.cat([batch["p_margin_raw"], batch["o_margin_raw"]])
    bucket = margin_to_bucket(margin_raw)
    loss_o = _wmean(F.cross_entropy(outcome, bucket, reduction="none"), waux)
    bomb_tgt = (opp_cnt == 4).any(1).float()
    loss_bomb = _wmean(
        F.binary_cross_entropy_with_logits(bomb, bomb_tgt, reduction="none"), waux
    )
    loss_ar = _wmean(ar_nll, waux)

    loss = (
        lam_play * (loss_v + loss_p + loss_b + loss_q)
        + lam_outcome * loss_o + lam_bomb * loss_bomb + lam_ar * loss_ar
    )
    return (
        loss,
        (loss_v, loss_p, loss_b, loss_q, loss_o, loss_bomb, loss_ar),
        (n_p, n_o),
    )


def _save(model, out_path, cfg, device):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "arch": {
                "d_model": cfg.d_model,
                "n_layers": cfg.layers,
                "n_heads": cfg.heads,
                "d_ff": cfg.d_ff,
            },
        },
        out_path,
    )
    model.to(device)


def train(cfg) -> None:
    device = _device(cfg.device)
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)

    triples = [(f"{p}_games.parquet", f"{p}_player.parquet", f"{p}_opp.parquet")
               for p in cfg.data]
    tl, vl = make_seq_split(
        triples, cfg.batch_games, device, val_frac=cfg.val_frac, seed=cfg.seed,
        mix_decay=cfg.mix_decay, validate=not cfg.no_validate,
        series_v=cfg.series_v or None,
    )
    print(
        f"[ii] games train={tl.rows.numel():,} val={vl.rows.numel():,}  "
        f"player rows={tl.n_p:,}/{vl.n_p:,}  opp rows={tl.n_o:,}/{vl.n_o:,}"
    )

    model = Big2NetII(
        load_token_feats(), d_model=cfg.d_model, n_layers=cfg.layers,
        n_heads=cfg.heads, d_ff=cfg.d_ff,
    ).to(device)
    if cfg.init:
        # Warm-start from a prior checkpoint so training REFINES it. Accepts a
        # plain az_ii checkpoint OR a TorchScript net (e.g. the II champion
        # models/az_seq.pt — Big2NetII shares its trunk + value/policy/behavior/qa
        # tensor names, so those copy; the aux heads start fresh).
        try:
            sd = dict(torch.jit.load(cfg.init, map_location=device).state_dict())
        except RuntimeError:
            ck = torch.load(cfg.init, map_location=device, weights_only=False)
            sd = ck["state_dict"] if "state_dict" in ck else ck
        msd = model.state_dict()
        copied = 0
        for k in msd:
            if k in sd and sd[k].shape == msd[k].shape:
                msd[k] = sd[k].to(device)
                copied += 1
        model.load_state_dict(msd)
        print(f"[ii] warm-start from {cfg.init}: {copied}/{len(msd)} tensors")
    C = load_compose_matrix().to(device)
    opt = _make_optim(model, cfg.lr, cfg.weight_decay)
    sched, per_batch = _make_sched(opt, cfg, cfg.epochs * len(tl))
    best = math.inf
    n_ranks = len(AR_ORDER)
    # Belief-only: zero the play heads + outcome aux so the trunk specialises on
    # opp-hand prediction (the only thing the determinization sampler uses).
    lam_play = 0.0 if cfg.belief_only else 1.0
    lam_outcome = 0.0 if cfg.belief_only else cfg.lam_outcome

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in tl:
            loss, _, _ = _step(model, C, batch, lam_outcome, cfg.lam_bomb,
                               cfg.lam_ar, lam_play)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if per_batch:
                sched.step()

        model.eval()
        sums = [0.0] * 7
        nps = nos = 0
        with torch.no_grad():
            for batch in vl:
                _, parts, (n_p, n_o) = _step(
                    model, C, batch, lam_outcome, cfg.lam_bomb, cfg.lam_ar, lam_play
                )
                lv, lp, lb, lq, lo, lbomb, lar = (float(x) for x in parts)
                n_all = n_p + n_o
                sums[0] += lv * n_p
                sums[1] += lp * n_p
                sums[2] += lb * n_o
                sums[3] += lq * n_o
                sums[4] += lo * n_all
                sums[5] += lbomb * n_all
                sums[6] += lar * n_all
                nps += n_p
                nos += n_o
        vv, vp = sums[0] / max(nps, 1), sums[1] / max(nps, 1)
        vb, vq = sums[2] / max(nos, 1), sums[3] / max(nos, 1)
        n_all = max(nps + nos, 1)
        vo, vbomb, var = sums[4] / n_all, sums[5] / n_all, sums[6] / n_all
        ar_ppx = math.exp(var / n_ranks)  # geo-mean per-rank perplexity
        blended = vv + vp + vb + vq
        if cfg.belief_only:
            mon = var  # checkpoint on AR NLL — the belief is all that matters
        else:
            mon = vp + vb if cfg.ckpt_metric == "head" else blended
        if not per_batch:
            sched.step(mon)
        print(
            f"[ii] epoch {epoch}/{cfg.epochs}  val={blended:.4f} "
            f"[v={vv:.3f} p={vp:.3f} b={vb:.3f} q={vq:.3f}]  "
            f"aux[outcome={vo:.3f} bomb={vbomb:.3f} ar={var:.3f} ppx/rank={ar_ppx:.3f}]  "
            f"lr={_cur_lr(opt):.2e}"
        )
        if mon < best:
            best = mon
            _save(model, cfg.out, cfg, device)
            print(f"  ✓ saved → {cfg.out}  (mon={mon:.4f})")
        if _cur_lr(opt) < cfg.lr_floor:
            print(f"[ii] lr below floor {cfg.lr_floor:.1e} — early stop.")
            break
    print(f"[ii] done. best mon={best:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the az_ii PI-distillation net")
    p.add_argument("--data", nargs="+", required=True,
                   help="triple PREFIX(es): PREFIX_{games,player,opp}.parquet")
    p.add_argument("--out", default="models/az_ii.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-games", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup", type=float, default=0.05)
    p.add_argument("--final-div-factor", type=float, default=100.0)
    p.add_argument("--sched", choices=["onecycle", "plateau"], default="onecycle")
    p.add_argument("--ckpt-metric", choices=["blended", "head"], default="blended")
    p.add_argument("--plateau-factor", type=float, default=0.3)
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--lr-floor", type=float, default=1e-6)
    p.add_argument("--mix-decay", type=float, default=1.0)
    p.add_argument("--series-v", default="",
                   help="series V/natural-freq CSV; enables natural-freq weights")
    p.add_argument("--init", default="",
                   help="warm-start from this az_ii checkpoint (previous gen)")
    p.add_argument("--lam-outcome", type=float, default=1.0)
    p.add_argument("--lam-bomb", type=float, default=1.0)
    p.add_argument("--lam-ar", type=float, default=1.0)
    p.add_argument("--belief-only", action="store_true",
                   help="train ONLY the belief trunk + AR/bomb heads (zero the "
                   "value/policy/behavior/qa + outcome play losses); checkpoint "
                   "by AR NLL. For the determinization belief model.")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-validate", action="store_true")
    cfg = p.parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
