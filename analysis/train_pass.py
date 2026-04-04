#!/usr/bin/env python3
"""
Train a Ridge linear model to predict Δ = p_pass − p_greedy from pass_greedy_datagen CSV.

Features f0..f59 match C++ extract_tree_features / FEATURE_COLS in train_tree_greedy.py.
Export format: same as LinearEvaluator / train_linear_rollout.py (not logistic_pass).

At play time: pass when predict(features) > 0 (GreedyPassPlayer, PimcPassRolloutPlayer).

For non-linearity, consider KernelRidge / sklearn.svm.SVC with RBF later; this script
stays a fast linear separator.

Example:
  bin/pass_greedy_datagen --games 2000 --dets 20 --seed 42 --output data/pass_train.csv
  # optional: --player pimc --player-param 20
  uv run python analysis/train_pass.py data/pass_train.csv --out data/pass_ridge_w.txt

Progress: tqdm for CSV chunks + training; use --no-progress to disable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from train_tree_greedy import FEATURE_COLS  # noqa: E402

assert len(FEATURE_COLS) == 60


def _feat_columns() -> list[str]:
    return [f"f{i}" for i in range(60)]


def load_xy(
    path: str, *, progress: bool = True
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = _feat_columns()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    if progress and tqdm is not None:
        chunks: list[pd.DataFrame] = []
        iterator = pd.read_csv(path, chunksize=100_000)
        for chunk in tqdm(
            iterator,
            desc="Loading CSV",
            unit="chunk",
            dynamic_ncols=True,
        ):
            chunks.append(chunk)
        if not chunks:
            df = pd.read_csv(path)
        else:
            df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path)

    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    if "p_pass" not in df.columns or "p_greedy" not in df.columns:
        raise ValueError(f"Need p_pass and p_greedy columns in {path}")
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["p_pass"] - df["p_greedy"]).to_numpy(dtype=np.float64)
    return df, X, y


def split_by_game(
    df: pd.DataFrame, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(df["game_index"].values)
    if len(games) < 2:
        train_mask = np.ones(len(df), dtype=bool)
        val_mask = np.zeros(len(df), dtype=bool)
        return train_mask, val_mask
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    n_val = min(n_val, len(games) - 1)
    val_set = set(games[:n_val].tolist())
    val_mask = df["game_index"].isin(val_set).values
    train_mask = ~val_mask
    return train_mask, val_mask


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
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from pass_greedy_datagen")
    ap.add_argument(
        "--out",
        default="data/pass_ridge_w.txt",
        help="Output weights (LinearEvaluator)",
    )
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength"
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress (CSV chunks + training step)",
    )
    args = ap.parse_args()

    show_progress = not args.no_progress
    df, X, y = load_xy(args.csv, progress=show_progress)
    if len(y) < 10:
        print("Warning: very few rows; metrics may be meaningless.")

    train_mask, val_mask = split_by_game(df, args.val_frac, args.seed)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha, random_state=args.seed)),
        ]
    )
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    if show_progress and tqdm is not None:
        with tqdm(
            total=1, desc="Training Ridge", unit="fit", dynamic_ncols=True
        ) as pbar:
            pipe.fit(X_train, y_train)
            pbar.update(1)
    else:
        pipe.fit(X_train, y_train)

    ridge = pipe.named_steps["ridge"]
    scaler = pipe.named_steps["scaler"]
    coef_full = ridge.coef_.ravel()
    intercept = float(ridge.intercept_)
    mean = scaler.mean_
    scale = scaler.scale_.copy()
    scale = np.where(scale < 1e-12, 1.0, scale)

    if X_val.shape[0] > 0:
        pred_va = pipe.predict(X_val)
        print("Val MSE:", mean_squared_error(y_val, pred_va))
        print("Val MAE:", mean_absolute_error(y_val, pred_va))
    else:
        print("Val MSE/MAE: n/a (no validation rows)")

    feat_idx = list(range(60))
    export_weights(
        Path(args.out),
        feat_idx,
        mean,
        scale,
        coef_full,
        intercept,
    )


if __name__ == "__main__":
    main()
