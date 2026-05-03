"""Training loop for Big2Net.

Usage:
    uv run python nn/train.py data/dnn_gen0_turn.parquet --out models/gen0.pt
    uv run python nn/train.py data/dnn_gen0_turn.parquet --out models/gen0.pt \\
        --epochs 30 --batch 2048 --val-frac 0.1

The trained model is saved as TorchScript (torch.jit.script) so it can be
loaded from C++ via LibTorch without the Python runtime.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nn.dataset import make_train_val_split
from nn.model import Big2Net

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    v_init: torch.Tensor,
    v_no_init: torch.Tensor,
    p_trick: torch.Tensor,
    y_trick: torch.Tensor,
    y_win: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Three-headed BCE loss with per-head masking.

    p_trick is trained on all normal moves (pass/flag rows contribute zero
    gradient via the post-processing in model.forward).
    v_init is trained only on rows where the player won the trick (y_trick=1).
    v_no_init is trained only on rows where the player lost the trick (y_trick=0).
    """
    eps = 1e-7
    p_trick_c = p_trick.clamp(eps, 1 - eps)
    v_init_c = v_init.clamp(eps, 1 - eps)
    v_no_init_c = v_no_init.clamp(eps, 1 - eps)

    loss_p = F.binary_cross_entropy(p_trick_c, y_trick)

    trick_mask = y_trick.bool()
    no_trick_mask = ~trick_mask

    if trick_mask.any():
        loss_vi = F.binary_cross_entropy(v_init_c[trick_mask], y_win[trick_mask])
    else:
        loss_vi = torch.tensor(0.0, device=p_trick.device)

    if no_trick_mask.any():
        loss_vn = F.binary_cross_entropy(
            v_no_init_c[no_trick_mask], y_win[no_trick_mask]
        )
    else:
        loss_vn = torch.tensor(0.0, device=p_trick.device)

    total = loss_p + loss_vi + loss_vn
    metrics = {
        "loss_p_trick": loss_p.item(),
        "loss_v_init": loss_vi.item(),
        "loss_v_no_init": loss_vn.item(),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    parquet_path: str,
    out_path: str,
    epochs: int = 20,
    batch_size: int = 2048,
    val_frac: float = 0.1,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    num_workers: int = 8,
    seed: int = 0,
    device_str: str = "auto",
) -> None:
    torch.manual_seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    print("Loading dataset...")
    t0 = time.time()
    train_ds, val_ds = make_train_val_split(parquet_path, val_frac=val_frac, seed=seed)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  ({time.time()-t0:.1f}s)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = Big2Net().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    emb_ids = model.embedding_param_ids()
    param_groups = [
        {
            "params": [p for p in model.parameters() if id(p) not in emb_ids],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in model.parameters() if id(p) in emb_ids],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    log_every = max(1, len(train_loader) // 5)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        tr_p = tr_vi = tr_vn = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            hand, opp, move, hint, flag, pass_, y_trick, y_win = (
                t.to(device) for t in batch
            )

            optimizer.zero_grad()
            v_init, v_no_init, p_trick = model(hand, opp, move, hint, flag, pass_)
            loss, metrics = compute_loss(v_init, v_no_init, p_trick, y_trick, y_win)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            tr_p += metrics["loss_p_trick"]
            tr_vi += metrics["loss_v_init"]
            tr_vn += metrics["loss_v_no_init"]
            n_batches += 1

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch}/{epochs}  step {batch_idx+1}/{len(train_loader)}"
                    f"  loss={avg:.4f}  lr={lr_now:.2e}"
                )

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_p = val_vi = val_vn = 0.0
        val_p_brier = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                hand, opp, move, hint, flag, pass_, y_trick, y_win = (
                    t.to(device) for t in batch
                )
                v_init, v_no_init, p_trick = model(hand, opp, move, hint, flag, pass_)
                loss, metrics = compute_loss(v_init, v_no_init, p_trick, y_trick, y_win)
                bs = len(y_trick)
                val_loss += loss.item() * bs
                val_p += metrics["loss_p_trick"] * bs
                val_vi += metrics["loss_v_init"] * bs
                val_vn += metrics["loss_v_no_init"] * bs
                val_p_brier += ((p_trick - y_trick) ** 2).sum().item()
                n_val += bs

        val_loss /= n_val
        val_p /= n_val
        val_vi /= n_val
        val_vn /= n_val
        val_p_brier /= n_val
        train_avg = train_loss / n_batches
        tr_p /= n_batches
        tr_vi /= n_batches
        tr_vn /= n_batches

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train={train_avg:.4f} [p={tr_p:.3f} vi={tr_vi:.3f} vn={tr_vn:.3f}]  "
            f"val={val_loss:.4f} [p={val_p:.3f} vi={val_vi:.3f} vn={val_vn:.3f}]  "
            f"brier={val_p_brier:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save as TorchScript for C++ LibTorch loading.
            scripted = torch.jit.script(model.cpu())
            scripted.save(str(out_path))
            model.to(device)
            print(f"  ✓ saved best model → {out_path}  (val_loss={val_loss:.4f})")

    print(f"\nTraining complete.  Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Big2Net")
    parser.add_argument("parquet", help="Path to *_turn.parquet from dnn datagen")
    parser.add_argument(
        "--out", default="models/model.pt", help="Output TorchScript model path"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        parquet_path=args.parquet,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        val_frac=args.val_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.workers,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
