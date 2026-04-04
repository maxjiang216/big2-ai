#!/usr/bin/env python3
"""
Train a Ridge linear rollout model on turn Parquet (e.g. from pimc(20) self-play).

Mirrors ``get_training_rows`` / ``FEATURE_COLS`` from ``train_tree_greedy.py`` so
C++ ``LinearEvaluator`` matches training.

Steps:
  1) Load Parquet, ``ensure_derived_features``, ``get_training_rows`` (last-player
     rows only; drops TB territory and any turn with ``tb_case != -1``).
  2) Optional RF feature importances → top-``--top-k`` columns (fixed order).
  3) Train/val split by ``game_index`` (no turn leakage).
  4) ``StandardScaler`` + ``Ridge``; export ``data/linear_rollout_w.txt`` for C++.

Example:
    python3 analysis/train_linear_rollout.py data/greedy_all_features_100k_turn.parquet \\
        --out data/linear_rollout_w.txt --top-k 35 --alpha 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_tree_greedy import (
    FEATURE_COLS,
    TARGET_COL,
    ensure_derived_features,
    get_training_rows,
    load_data,
)

# Canonical tree feature order (must match C++ ``extract_tree_features`` / ``FEATURE_COLS``).
assert len(FEATURE_COLS) == 60

# After training with feature selection, this list is the subset used (order = matrix columns).
# The export file stores indices into FEATURE_COLS; keep this in sync when changing selection.
LINEAR_FEATURE_COLS: list[str] | None = None


def _split_games(
    df: pd.DataFrame, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(df["game_index"].values)
    if len(games) < 2:
        raise ValueError(
            "Need at least 2 distinct game_index values for train/val split"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    n_val = min(n_val, len(games) - 1)
    val_games = set(games[:n_val].tolist())
    train_games = set(games[n_val:].tolist())
    train_mask = df["game_index"].isin(train_games).values
    val_mask = df["game_index"].isin(val_games).values
    return train_mask, val_mask


def _select_columns(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
    seed: int,
) -> list[int]:
    """Return indices into FEATURE_COLS for the selected subset (fixed order)."""
    d = len(FEATURE_COLS)
    if top_k <= 0 or top_k >= d:
        return list(range(d))

    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=50,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_
    order = np.argsort(-imp)
    chosen = sorted(order[:top_k].tolist())
    return chosen


def export_weights(
    path: Path,
    indices: list[int],
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> None:
    k = len(indices)
    lines = [
        str(k),
        repr(intercept),
        " ".join(str(indices[j]) for j in range(k)),
        " ".join(repr(float(mean[j])) for j in range(k)),
        " ".join(repr(float(scale[j])) for j in range(k)),
        " ".join(repr(float(coef[j])) for j in range(k)),
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path} ({k} features)")


def main() -> None:
    global LINEAR_FEATURE_COLS

    parser = argparse.ArgumentParser(
        description="Train Ridge linear rollout + export weights"
    )
    parser.add_argument("parquet", help="Path to *_turn.parquet")
    parser.add_argument(
        "--out",
        default="data/linear_rollout_w.txt",
        help="Output path for C++ LinearEvaluator (default: data/linear_rollout_w.txt)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    parser.add_argument(
        "--top-k",
        type=int,
        default=35,
        help="Number of features after RF screening (0 = use all 60)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.2, help="Validation fraction by game"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--meta-json",
        default="",
        help="Optional path to write JSON with LINEAR_FEATURE_COLS and metrics",
    )
    args = parser.parse_args()

    df = load_data(args.parquet)
    df = ensure_derived_features(df)
    n_raw = len(df)
    df = get_training_rows(df)
    print(
        f"Rows: {n_raw} in Parquet → {len(df)} last-player rows "
        "(TB segment + per-turn tb_case != -1 excluded)"
    )

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print("Missing columns:", missing, file=sys.stderr)
        sys.exit(1)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    try:
        train_mask, val_mask = _split_games(df, args.val_frac, args.seed)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
        print("Empty train or validation set after split.", file=sys.stderr)
        sys.exit(1)

    col_idx = _select_columns(X_tr, y_tr, args.top_k, args.seed)
    LINEAR_FEATURE_COLS = [FEATURE_COLS[j] for j in col_idx]
    print("LINEAR_FEATURE_COLS (order):", json.dumps(LINEAR_FEATURE_COLS))

    X_tr_s = X_tr[:, col_idx]
    X_va_s = X_va[:, col_idx]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha, random_state=args.seed)),
        ]
    )
    pipe.fit(X_tr_s, y_tr)
    pred_va = pipe.predict(X_va_s)

    mse = mean_squared_error(y_va, pred_va)
    brier = brier_score_loss(y_va, np.clip(pred_va, 0.0, 1.0))
    print(f"Val MSE: {mse:.6f}  Brier (clipped pred): {brier:.6f}")

    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]
    coef = ridge.coef_.ravel()
    intercept = float(ridge.intercept_)
    mean = scaler.mean_
    scale = scaler.scale_.copy()
    scale = np.where(scale < 1e-12, 1.0, scale)

    out_path = Path(args.out)
    export_weights(out_path, col_idx, mean, scale, coef, intercept)

    if args.meta_json:
        meta = {
            "LINEAR_FEATURE_COLS": LINEAR_FEATURE_COLS,
            "indices_into_FEATURE_COLS": col_idx,
            "val_mse": mse,
            "val_brier": brier,
            "alpha": args.alpha,
            "top_k_requested": args.top_k,
        }
        Path(args.meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {args.meta_json}")


if __name__ == "__main__":
    main()
