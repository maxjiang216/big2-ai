"""Training loop for the unified history-transformer net (Big2NetSeqAZ).

ONE model, four heads, trained per-GAME: each batch is a set of games; the
causal trunk runs once per game over its token sequence, hidden states are
gathered at every sample's hist_idx, and one readout pass serves all heads.

Usage:
    uv run python nn/train_az_seq.py \\
        --games-data  data/az_games_gen0.parquet \\
        --player-data data/az_player_gen0.parquet \\
        --opp-data    data/az_opp_gen0.parquet \\
        --out         models/az_seq_gen0.pt \\
        --epochs 10 --batch-games 256

The three --*-data flags accept N paths each (generation mixing); path i of
each flag must come from the same generation run.

Losses (per-head means over that head's own row count):
  * value    — BCE vs outcome, PLAYER rows only (owner-to-move; opp rows are
               grounded through q_a instead)
  * policy   — masked CE of composed (head @ C.T) logits vs visit distribution
               (zero-policy tablebase rows contribute no policy gradient)
  * behavior — masked NLL vs the played move slot (opp rows)
  * q_a      — BCE at the played slot vs outcome (opp rows)

The dataset is device-resident; --device cuda is the (default-on-auto) GPU
path, --device cpu the same code on host (debug/parity).
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from nn.dataset_seq import make_seq_split
from nn.model_az_seq import Big2NetSeqAZ, load_token_feats
from nn.model_az import load_compose_matrix
from nn.train_az import (
    _cur_lr,
    _device,
    _make_optim,
    _make_sched,
    _save_scripted,
    masked_log_softmax,
)

EPS = 1e-7


def _gather_h(H: torch.Tensor, local: torch.Tensor, hist: torch.Tensor):
    """H [b, T+1, d] trunk hiddens; rows (local game idx, hist_idx) -> [n, d]."""
    return H[local, hist]


def _step(model, C, batch, no_memory=False):
    """Forward + the four losses for one game-batch. Returns (loss, parts, ns).

    no_memory: zero the trunk hidden before the junction — an apples-to-apples
    memory-off control (identical side-input pathways, transformer contributes
    nothing and receives no gradient)."""
    H = model.forward_full(batch["tokens"])
    hp = _gather_h(H, batch["p_local"], batch["p_hist"])
    ho = _gather_h(H, batch["o_local"], batch["o_hist"])
    n_p, n_o = hp.shape[0], ho.shape[0]
    h = torch.cat([hp, ho], dim=0)
    if no_memory:
        h = torch.zeros_like(h)
    hand = torch.cat([batch["p_hand"], batch["o_hand"]])
    oppm = torch.cat([batch["p_opp"], batch["o_opp"]])
    osz = torch.cat([batch["p_osz"], batch["o_osz"]])
    usz = torch.cat([batch["p_usz"], batch["o_usz"]])
    otm = torch.cat(
        [
            torch.ones(n_p, device=h.device),
            torch.zeros(n_o, device=h.device),
        ]
    )
    value, policy, behavior, qa = model.readout(h, hand, oppm, osz, usz, otm)

    # player rows: value BCE + composed-policy CE
    loss_v = F.binary_cross_entropy(value[:n_p].clamp(EPS, 1 - EPS), batch["p_value"])
    logp = masked_log_softmax(policy[:n_p] @ C.t(), batch["p_mask"])
    loss_p = -(batch["p_policy"] * logp).sum(-1).mean()
    # opp rows: behavior NLL + q_a BCE at the played slot
    tgt = batch["o_target"]
    logb = masked_log_softmax(behavior[n_p:], batch["o_mask"])
    loss_b = F.nll_loss(logb, tgt)
    q = qa[n_p:].gather(1, tgt.unsqueeze(1)).squeeze(1)
    loss_q = F.binary_cross_entropy(q.clamp(EPS, 1 - EPS), batch["o_value"])

    loss = loss_v + loss_p + loss_b + loss_q
    return loss, (loss_v, loss_p, loss_b, loss_q), (n_p, n_o)


def train(cfg) -> None:
    device = _device(cfg.device)
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)

    triples = list(zip(cfg.games_data, cfg.player_data, cfg.opp_data))
    tl, vl = make_seq_split(
        triples,
        cfg.batch_games,
        device,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        mix_decay=cfg.mix_decay,
        validate=not cfg.no_validate,
    )
    print(
        f"[seq] games train={tl.rows.numel():,} val={vl.rows.numel():,}  "
        f"player rows={tl.n_p:,}/{vl.n_p:,}  opp rows={tl.n_o:,}/{vl.n_o:,}"
    )

    model = Big2NetSeqAZ(
        load_token_feats(),
        d_model=cfg.d_model,
        n_layers=cfg.layers,
        n_heads=cfg.heads,
        d_ff=cfg.d_ff,
    ).to(device)
    C = load_compose_matrix().to(device)
    opt = _make_optim(model, cfg.lr, cfg.weight_decay)
    sched, per_batch = _make_sched(opt, cfg, cfg.epochs * len(tl))
    best = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in tl:
            loss, _, _ = _step(model, C, batch, cfg.no_memory)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if per_batch:
                sched.step()

        model.eval()
        sums = [0.0] * 4
        nps = nos = 0
        with torch.no_grad():
            for batch in vl:
                _, parts, (n_p, n_o) = _step(model, C, batch, cfg.no_memory)
                lv, lp, lb, lq = (float(x) for x in parts)
                sums[0] += lv * n_p
                sums[1] += lp * n_p
                sums[2] += lb * n_o
                sums[3] += lq * n_o
                nps += n_p
                nos += n_o
        vv, vp = sums[0] / max(nps, 1), sums[1] / max(nps, 1)
        vb, vq = sums[2] / max(nos, 1), sums[3] / max(nos, 1)
        blended = vv + vp + vb + vq
        mon = vp + vb if cfg.ckpt_metric == "head" else blended
        if not per_batch:
            sched.step(mon)
        print(
            f"[seq] epoch {epoch}/{cfg.epochs}  val={blended:.4f} "
            f"[v={vv:.3f} p={vp:.3f} b={vb:.3f} q={vq:.3f}]  lr={_cur_lr(opt):.2e}"
        )
        if mon < best:
            best = mon
            _save_scripted(model, cfg.out, device)
            print(f"  ✓ saved → {cfg.out}  (mon={mon:.4f})")
        if _cur_lr(opt) < cfg.lr_floor:
            print(f"[seq] lr below floor {cfg.lr_floor:.1e} — early stop.")
            break
    print(f"[seq] done. best mon={best:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the az_search seq (memory) net")
    p.add_argument("--games-data", nargs="+", required=True)
    p.add_argument("--player-data", nargs="+", required=True)
    p.add_argument("--opp-data", nargs="+", required=True)
    p.add_argument("--out", default="models/az_seq.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch-games",
        type=int,
        default=256,
        help="games per batch (~20-25 sample rows per game)",
    )
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
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="ablation: zero the transformer hidden (memory-off control)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the load-time replay cross-checks (sizes vs move list)",
    )
    cfg = p.parse_args()
    if not (len(cfg.games_data) == len(cfg.player_data) == len(cfg.opp_data)):
        raise SystemExit("--games-data/--player-data/--opp-data counts must match")
    train(cfg)


if __name__ == "__main__":
    main()
