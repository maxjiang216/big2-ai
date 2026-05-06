"""Break down p_trick BCE loss by move combination type across a parquet file."""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nn.dataset import encode_exact_np, encode_upper_bound_np

ENCODING_DIM = 48


def combo_label(move_counts: np.ndarray) -> str:
    """Classify a move from its rank-count array (shape 13)."""
    total = move_counts.sum()
    if total == 0:
        return "pass"
    if total == 1:
        return "single"
    if total == 2:
        return "pair"
    if total == 3:
        if move_counts.max() == 3:
            return "triple"
        return "straight3?"
    if total == 4:
        if move_counts.max() == 4:
            return "bomb"
        return "other4"
    if total == 5:
        nonzero = (move_counts > 0).sum()
        if nonzero == 2 and move_counts.max() == 3:
            return "full_house"
        if nonzero == 5:
            return "str5"
        return "other5"
    if total <= 13:
        # straight or double/triple straight
        nonzero = (move_counts > 0).sum()
        maxcount = move_counts.max()
        if maxcount == 1:
            return f"str{total}"
        if maxcount == 2:
            return f"dstr{total//2}"
        if maxcount == 3:
            return f"tstr{total//3}"
        return f"other{total}"
    return f"other{total}"


def run_analysis(parquet_path: str, model_path: str, device_str: str = "auto"):
    device = torch.device(
        "cuda"
        if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "cpu" else device_str)
    )
    print(f"Device: {device}")

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} rows")

    # Encode features
    hand_counts = df[[f"hand_after_{r}" for r in range(13)]].to_numpy(dtype=np.int32)
    opp_counts = df[[f"opp_cnt_{r}" for r in range(13)]].to_numpy(dtype=np.int32)
    move_counts = df[[f"move_at{r}" for r in range(13)]].to_numpy(dtype=np.int32)

    hand_enc = torch.from_numpy(encode_exact_np(hand_counts)).float()
    opp_enc = torch.from_numpy(encode_upper_bound_np(opp_counts)).float()
    move_enc = torch.from_numpy(encode_exact_np(move_counts)).float()
    hint = torch.tensor(df["hint"].to_numpy(dtype=np.float32))
    flag = torch.tensor(df["flag"].to_numpy(dtype=bool))
    pass_ = torch.tensor(df["pass_"].to_numpy(dtype=bool))
    y_trick = torch.tensor(df["trick_winner"].to_numpy(dtype=np.float32))

    print(f"Loading model {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)

    # Run inference in batches
    BATCH = 8192
    all_pt = []
    n = len(df)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            sl = slice(i, i + BATCH)
            out = model.forward(
                hand_enc[sl].to(device),
                opp_enc[sl].to(device),
                move_enc[sl].to(device),
                hint[sl].to(device),
                flag[sl].to(device),
                pass_[sl].to(device),
            )
            pt = out[2].cpu()  # p_trick output
            all_pt.append(pt)
    pt_all = torch.cat(all_pt).numpy()

    # Classify each move
    labels = np.array([combo_label(move_counts[i]) for i in range(n)])

    # Exclude pass and flag rows from analysis (they contribute no gradient)
    pass_mask = df["pass_"].to_numpy(dtype=bool)
    flag_mask = df["flag"].to_numpy(dtype=bool)
    active = ~pass_mask  # include flag (it's a real move) but exclude pass

    y = y_trick.numpy()

    print(
        f"\n{'Combo':<12} {'N':>8}  {'y=1%':>6}  {'BCE':>6}  {'pred_pt':>8}  {'brier':>7}"
    )
    print("-" * 60)

    combo_order = [
        "single",
        "pair",
        "triple",
        "full_house",
        "bomb",
        "str5",
        "str6",
        "str7",
        "str8",
        "str9",
        "str10",
        "str11",
        "str12",
        "str13",
        "dstr2",
        "dstr3",
        "dstr4",
        "dstr5",
        "dstr6",
        "dstr7",
        "dstr8",
        "tstr2",
        "tstr3",
        "tstr4",
        "tstr5",
    ]

    seen = set()
    rows = []
    for combo in combo_order + sorted(set(labels) - set(combo_order)):
        mask = (labels == combo) & active
        if mask.sum() == 0:
            continue
        seen.add(combo)
        pt_c = np.clip(pt_all[mask], 1e-7, 1 - 1e-7)
        y_c = y[mask]
        bce = -(y_c * np.log(pt_c) + (1 - y_c) * np.log(1 - pt_c)).mean()
        brier = ((pt_c - y_c) ** 2).mean()
        y1_pct = y_c.mean() * 100
        pt_mean = pt_c.mean()
        rows.append((combo, mask.sum(), y1_pct, bce, pt_mean, brier))
        print(
            f"{combo:<12} {mask.sum():>8,}  {y1_pct:>5.1f}%  {bce:>6.4f}  {pt_mean:>8.4f}  {brier:>7.4f}"
        )

    # Overall (excluding pass)
    mask = active
    pt_c = np.clip(pt_all[mask], 1e-7, 1 - 1e-7)
    y_c = y[mask]
    bce = -(y_c * np.log(pt_c) + (1 - y_c) * np.log(1 - pt_c)).mean()
    brier = ((pt_c - y_c) ** 2).mean()
    print("-" * 60)
    print(
        f"{'ALL':<12} {mask.sum():>8,}  {y_c.mean()*100:>5.1f}%  {bce:>6.4f}  {pt_c.mean():>8.4f}  {brier:>7.4f}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.model is None:
        # infer model from parquet name: data/genN.parquet -> models/genN.pt
        stem = Path(args.parquet).stem  # e.g. "gen6"
        args.model = f"models/{stem}.pt"

    run_analysis(args.parquet, args.model, args.device)
