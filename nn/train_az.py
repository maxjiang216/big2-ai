"""Training loop for the az_search networks (Big2NetAZ + Big2NetOpp).

Trains both nets from their respective Parquet files (see nn/dataset.py for the
schemas) and exports each as TorchScript for C++ LibTorch loading.

Usage:
    uv run python nn/train_az.py \\
        --player-data data/az_player_gen0.parquet \\
        --opp-data    data/az_opp_gen0.parquet \\
        --player-out  models/az_player_gen0.pt \\
        --opp-out     models/az_opp_gen0.pt \\
        --epochs 10 --batch 2048

Losses:
  * value heads        — BCE vs realized outcome.
  * player policy head — masked cross-entropy vs the MCTS visit distribution.
  * opp behavior head  — masked cross-entropy vs the one-hot played move.
The legal mask sets illegal logits to -inf before log_softmax (a correct
renormalisation over the legal subset).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nn.dataset import make_az_split
from nn.model_az import Big2NetAZ, Big2NetOpp

NEG = -1e30  # stand-in for -inf in masked logits (rows always have >=1 legal)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, NEG), dim=-1)


def _make_optim(model, lr, weight_decay):
    emb_ids = model.embedding_param_ids()
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in emb_ids],
         "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if id(p) in emb_ids],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def _scheduler(optimizer, lr, total_steps, warmup, final_div):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=warmup,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=final_div,
    )


def _loaders(train_ds, val_ds, batch, workers):
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=(workers > 0))
    vl = DataLoader(val_ds, batch_size=batch * 2, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=(workers > 0))
    return tl, vl


def _device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _save_scripted(model, out_path, device):
    scripted = torch.jit.script(model.cpu().eval())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    model.to(device)
    model.train()


# ---------------------------------------------------------------------------
# Player net
# ---------------------------------------------------------------------------


def train_player(paths, out_path, cfg, device):
    train_ds, val_ds = make_az_split(paths, "player", cfg.val_frac, cfg.seed, cfg.mix_decay)
    print(f"[player] train={len(train_ds):,}  val={len(val_ds):,}")
    tl, vl = _loaders(train_ds, val_ds, cfg.batch, cfg.workers)

    model = Big2NetAZ().to(device)
    opt = _make_optim(model, cfg.lr, cfg.weight_decay)
    sched = _scheduler(opt, cfg.lr, cfg.epochs * len(tl), cfg.lr_warmup, cfg.final_div_factor)
    eps = 1e-7
    best = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for hand, opp, trick, osz, usz, value, mask, policy in tl:
            hand, opp, trick = hand.to(device), opp.to(device), trick.to(device)
            osz, usz, value = osz.to(device), usz.to(device), value.to(device)
            mask, policy = mask.to(device), policy.to(device)

            v, logits = model(hand, opp, trick, osz, usz)
            loss_v = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
            logp = masked_log_softmax(logits, mask)
            loss_p = -(policy * logp).sum(-1).mean()
            loss = loss_v + loss_p

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

        # validate
        model.eval()
        vl_loss = vv = vp = 0.0
        n = 0
        with torch.no_grad():
            for hand, opp, trick, osz, usz, value, mask, policy in vl:
                hand, opp, trick = hand.to(device), opp.to(device), trick.to(device)
                osz, usz, value = osz.to(device), usz.to(device), value.to(device)
                mask, policy = mask.to(device), policy.to(device)
                v, logits = model(hand, opp, trick, osz, usz)
                lv = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
                lp = -(policy * masked_log_softmax(logits, mask)).sum(-1).mean()
                bs = len(value)
                vl_loss += float(lv + lp) * bs
                vv += float(lv) * bs
                vp += float(lp) * bs
                n += bs
        vl_loss, vv, vp = vl_loss / n, vv / n, vp / n
        print(f"[player] epoch {epoch}/{cfg.epochs}  val={vl_loss:.4f} "
              f"[v={vv:.3f} p={vp:.3f}]")
        if vl_loss < best:
            best = vl_loss
            _save_scripted(model, out_path, device)
            print(f"  ✓ saved → {out_path}  (val={vl_loss:.4f})")
    print(f"[player] done. best val={best:.4f}")


# ---------------------------------------------------------------------------
# Opponent net
# ---------------------------------------------------------------------------


def train_opp(paths, out_path, cfg, device):
    train_ds, val_ds = make_az_split(paths, "opp", cfg.val_frac, cfg.seed, cfg.mix_decay)
    print(f"[opp] train={len(train_ds):,}  val={len(val_ds):,}")
    tl, vl = _loaders(train_ds, val_ds, cfg.batch, cfg.workers)

    model = Big2NetOpp().to(device)
    opt = _make_optim(model, cfg.lr, cfg.weight_decay)
    sched = _scheduler(opt, cfg.lr, cfg.epochs * len(tl), cfg.lr_warmup, cfg.final_div_factor)
    eps = 1e-7
    best = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for opp, trick, osz, usz, value, mask, tgt in tl:
            opp, trick = opp.to(device), trick.to(device)
            osz, usz, value = osz.to(device), usz.to(device), value.to(device)
            mask, tgt = mask.to(device), tgt.to(device)

            v, logits = model(opp, trick, osz, usz)
            loss_v = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
            logp = masked_log_softmax(logits, mask)
            loss_b = F.nll_loss(logp, tgt)
            loss = loss_v + loss_b

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

        model.eval()
        vl_loss = vv = vb = 0.0
        n = 0
        with torch.no_grad():
            for opp, trick, osz, usz, value, mask, tgt in vl:
                opp, trick = opp.to(device), trick.to(device)
                osz, usz, value = osz.to(device), usz.to(device), value.to(device)
                mask, tgt = mask.to(device), tgt.to(device)
                v, logits = model(opp, trick, osz, usz)
                lv = F.binary_cross_entropy(v.clamp(eps, 1 - eps), value)
                lb = F.nll_loss(masked_log_softmax(logits, mask), tgt)
                bs = len(value)
                vl_loss += float(lv + lb) * bs
                vv += float(lv) * bs
                vb += float(lb) * bs
                n += bs
        vl_loss, vv, vb = vl_loss / n, vv / n, vb / n
        print(f"[opp] epoch {epoch}/{cfg.epochs}  val={vl_loss:.4f} "
              f"[v={vv:.3f} b={vb:.3f}]")
        if vl_loss < best:
            best = vl_loss
            _save_scripted(model, out_path, device)
            print(f"  ✓ saved → {out_path}  (val={vl_loss:.4f})")
    print(f"[opp] done. best val={best:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Train az_search nets")
    p.add_argument("--player-data", nargs="+", required=True)
    p.add_argument("--opp-data", nargs="+", required=True)
    p.add_argument("--player-out", default="models/az_player.pt")
    p.add_argument("--opp-out", default="models/az_opp.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup", type=float, default=0.05)
    p.add_argument("--final-div-factor", type=float, default=100.0)
    p.add_argument("--mix-decay", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-player", action="store_true")
    p.add_argument("--skip-opp", action="store_true")
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    print(f"Device: {device}")

    if not cfg.skip_player:
        train_player(cfg.player_data, cfg.player_out, cfg, device)
    if not cfg.skip_opp:
        train_opp(cfg.opp_data, cfg.opp_out, cfg, device)


if __name__ == "__main__":
    main()
