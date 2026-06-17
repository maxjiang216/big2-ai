"""Training loop for the perfect-information AlphaZero net (Big2NetAZPI).

Single net, single Parquet file (the player schema in nn/dataset.py): masked
cross-entropy of the composed policy logits vs the MCTS visit distribution plus
BCE of the value head vs the realised game outcome. Exports TorchScript for the
C++ LibTorch evaluator (src/players/az_pi/pi_nn_eval).

Reuses the az_search player data path verbatim (make_az_split / GpuLoader /
collate_az_player), since the PI Parquet matches the player schema (the only
difference — exact instead of thermometer opponent encoding in enc[48:96] — is
invisible to the loader).

Usage:
    # gen0 bootstrap: export a random-init scripted net, no training.
    uv run python nn/train_az_pi.py --export-init models/az_pi_gen0.pt

    # train a generation:
    uv run python nn/train_az_pi.py --data data/az_pi_gen1.parquet \\
        --out models/az_pi_gen1.pt --epochs 10 --batch 2048
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import ConcatDataset

from nn.dataset import GpuLoader, collate_az_player, make_az_split
from nn.model_az import NUM_MOVES, load_compose_matrix
from nn.model_az_pi import Big2NetAZPI
from nn.train_az import (
    _cur_lr,
    _device,
    _make_optim,
    _make_sched,
    _save_scripted,
    _use_gpu_resident,
    masked_log_softmax,
)


def load_series_v(csv_path: str) -> torch.Tensor:
    """50x50 V table (a,b,v,natural_freq CSV from analysis/series_markov.py)."""
    V = torch.zeros(50, 50)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("a,"):
                continue
            a, b, v = line.split(",")[:3]
            V[int(a), int(b)] = float(v)
    return V


def relabel_series(ds, V: torch.Tensor) -> None:
    """Replace the binary game outcome with the series target, in place:
    value = P(mover wins the SERIES) = series_value_after_win(margin), using
    the per-row scores and the per-game loser card count stored by selfplay."""
    parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
    for part in parts:
        if not getattr(part, "series", False):
            raise SystemExit("--series-table given but data has no my_pts column")
        win = part.value > 0.5  # mover won the game (C++ writes 1/0)
        a, b, lc = part.my_pts_raw, part.opp_pts_raw, part.loser_cards
        p = torch.where(lc <= 12, lc, 10 * (lc - 11))  # points_for_cards
        wpts = torch.where(win, a, b)
        lpts = torch.where(win, b, a)
        nxt = wpts + p
        tw = torch.where(nxt >= 50, torch.ones(1), V[nxt.clamp(max=49), lpts]).float()
        part.value = torch.where(win, tw, 1.0 - tw).float()


def train(paths, out_path, cfg, device):
    train_ds, val_ds = make_az_split(
        paths, "player", cfg.val_frac, cfg.seed, cfg.mix_decay
    )
    if cfg.series_table:
        V = load_series_v(cfg.series_table)
        relabel_series(train_ds, V)
        relabel_series(val_ds, V)
        print(f"[pi] series targets from {cfg.series_table}  V(0,0)={V[0,0]:.3f}")
    print(f"[pi] train={len(train_ds):,}  val={len(val_ds):,}")
    if _use_gpu_resident(cfg, device):
        tl = GpuLoader(train_ds, NUM_MOVES, cfg.batch, device, shuffle=True)
        vl = GpuLoader(val_ds, NUM_MOVES, cfg.batch * 2, device, shuffle=False)
    else:
        tl, vl = (
            DataLoader(
                train_ds,
                batch_size=cfg.batch,
                shuffle=True,
                num_workers=cfg.workers,
                pin_memory=True,
                persistent_workers=(cfg.workers > 0),
                collate_fn=collate_az_player,
            ),
            DataLoader(
                val_ds,
                batch_size=cfg.batch * 2,
                shuffle=False,
                num_workers=cfg.workers,
                pin_memory=True,
                persistent_workers=(cfg.workers > 0),
                collate_fn=collate_az_player,
            ),
        )

    model = Big2NetAZPI(cfg.width, cfg.blocks, cfg.embed).to(device)
    if cfg.init_from:
        src = torch.jit.load(cfg.init_from, map_location="cpu").state_dict()
        tgt = model.state_dict()
        # Name+shape matching warm start. The old net's size_embed never maps
        # onto pts_embed (different semantics); the value head transfers as a
        # same-shape init and adapts to the series target.
        hit = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
        model.load_state_dict(hit, strict=False)
        print(f"[pi] warm start {cfg.init_from}: {len(hit)}/{len(tgt)} tensors")
    C = load_compose_matrix().to(device)
    opt = _make_optim(model, cfg.lr, cfg.weight_decay)
    sched, per_batch = _make_sched(opt, cfg, cfg.epochs * len(tl))
    eps = 1e-7
    best = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # Slots 3/4 carry my_pts/opp_pts for series files (see dataset.py).
        for hand, opp, trick, mpts, opts, value, mask, policy in tl:
            hand, opp, trick = hand.to(device), opp.to(device), trick.to(device)
            mpts, opts, value = mpts.to(device), opts.to(device), value.to(device)
            mask, policy = mask.to(device), policy.to(device)

            v, head = model(hand, opp, trick, mpts, opts)
            loss_v = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
            logp = masked_log_softmax(head @ C.t(), mask)
            loss_p = -(policy * logp).sum(-1).mean()
            loss = loss_v + loss_p

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if per_batch:
                sched.step()

        model.eval()
        vl_loss = vv = vp = 0.0
        n = 0
        with torch.no_grad():
            for hand, opp, trick, mpts, opts, value, mask, policy in vl:
                hand, opp, trick = hand.to(device), opp.to(device), trick.to(device)
                mpts, opts, value = mpts.to(device), opts.to(device), value.to(device)
                mask, policy = mask.to(device), policy.to(device)
                v, head = model(hand, opp, trick, mpts, opts)
                lv = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
                lp = -(policy * masked_log_softmax(head @ C.t(), mask)).sum(-1).mean()
                bs = len(value)
                vl_loss += float(lv + lp) * bs
                vv += float(lv) * bs
                vp += float(lp) * bs
                n += bs
        vl_loss, vv, vp = vl_loss / n, vv / n, vp / n
        mon = vp if cfg.ckpt_metric == "head" else vl_loss
        if not per_batch:
            sched.step(mon)
        print(
            f"[pi] epoch {epoch}/{cfg.epochs}  val={vl_loss:.4f} "
            f"[v={vv:.3f} p={vp:.3f}]  lr={_cur_lr(opt):.2e}"
        )
        if mon < best:
            best = mon
            _save_scripted(model, out_path, device)
            print(f"  ✓ saved → {out_path}  (mon={mon:.4f})")
        if _cur_lr(opt) < cfg.lr_floor:
            print(f"[pi] lr below floor {cfg.lr_floor:.1e} — early stop.")
            break
    print(f"[pi] done. best mon={best:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the perfect-info AZ net")
    p.add_argument("--data", nargs="+", help="PI self-play Parquet (player schema)")
    p.add_argument("--out", default="models/az_pi.pt")
    p.add_argument(
        "--export-init",
        metavar="PATH",
        help="export a random-init scripted net to PATH and exit (gen0 bootstrap)",
    )
    # Model size.
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--embed", type=int, default=64)
    # Training.
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2048)
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
    p.add_argument(
        "--series-table",
        help="series V CSV (analysis/series_markov.py); relabels values as "
        "P(win series) at load time. Requires my_pts/opp_pts/loser_cards "
        "columns in the data (series selfplay).",
    )
    p.add_argument(
        "--init-from",
        help="warm-start weights from a scripted checkpoint (name+shape match)",
    )
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpu-resident", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)

    if cfg.export_init:
        model = Big2NetAZPI(cfg.width, cfg.blocks, cfg.embed).to(device)
        _save_scripted(model, cfg.export_init, device)
        print(
            f"✓ exported random-init net → {cfg.export_init} "
            f"(width={cfg.width} blocks={cfg.blocks} embed={cfg.embed})"
        )
        return

    if not cfg.data:
        p.error("--data is required unless --export-init is given")
    print(f"Device: {device}")
    train(cfg.data, cfg.out, cfg, device)


if __name__ == "__main__":
    main()
