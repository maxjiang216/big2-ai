#!/usr/bin/env python3
"""
Explore turn features vs ``turn_outcome``: correlations, mutual information,
binned nonlinearities, and candidate feature interactions.

Reads the same filtered rows as ``train_linear_rollout.py`` (via
``get_training_rows``). Writes figures and CSV/text under ``--out-dir``.

Example:
    uv run python analysis/feature_interaction_explore.py data/pimc20_selfplay_turn.parquet \\
        --out-dir analysis/feature_explore_out --sample-n 80000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from train_tree_greedy import (
    FEATURE_COLS,
    TARGET_COL,
    ensure_derived_features,
    get_training_rows,
    load_data,
)


def _subsample(
    X: np.ndarray, y: np.ndarray, df: pd.DataFrame, n: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if n is None or len(y) <= n:
        return X, y, df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx], df.iloc[idx].reset_index(drop=True)


def plot_correlation_heatmap(
    df_feat: pd.DataFrame, names: list[str], out_path: Path, title: str
) -> None:
    c = df_feat[names].corr()
    fig, ax = plt.subplots(
        figsize=(max(8, 0.25 * len(names)), max(7, 0.25 * len(names)))
    )
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar(
    labels: list[str],
    values: np.ndarray,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    order = np.argsort(-values)
    labels_s = [labels[i] for i in order]
    vals_s = values[order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(labels))))
    ax.barh(np.arange(len(labels)), vals_s, color="steelblue")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels_s, fontsize=8)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_binned_outcome(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    bins: int,
    out_path: Path,
) -> None:
    """Mean turn_outcome vs quantile bins of ``x`` (nonlinearity)."""
    sx = pd.Series(x)
    sy = pd.Series(y)
    try:
        q = pd.qcut(sx, q=bins, duplicates="drop")
    except ValueError:
        return
    g = sy.groupby(q, observed=True)
    means = g.mean()
    counts = g.size()
    mid = getattr(means.index, "mid", None)
    centers = np.asarray(mid, dtype=float) if mid is not None else np.arange(len(means))
    if len(centers) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, means.values, "o-", color="darkblue", label="mean(turn_outcome)")
    ax.set_xlabel(name)
    ax.set_ylabel("mean turn_outcome")
    ax.set_title(f"Binned outcome vs {name}")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    w = (np.nanmax(x) - np.nanmin(x) + 1e-9) / max(len(centers), 1) * 0.8
    ax2.bar(centers, counts.values, width=w, alpha=0.25, color="gray")
    ax2.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_squared_correlations(
    Xs: np.ndarray, y: np.ndarray, names: list[str]
) -> list[tuple[str, float]]:
    """|corr(y, x_i^2)| after standardizing x_i (simple quadratic probe)."""
    out = []
    for j in range(Xs.shape[1]):
        z = Xs[:, j]
        z2 = z * z
        if np.std(z2) < 1e-12:
            continue
        r = np.corrcoef(y, z2)[0, 1]
        out.append((names[j], float(abs(r))))
    out.sort(key=lambda t: -t[1])
    return out


def top_interaction_correlations(
    Xs: np.ndarray, y: np.ndarray, names: list[str], top_idx: list[int], max_pairs: int
) -> list[tuple[str, str, float]]:
    """|corr(y, z_i * z_j)| for standardized features, i<j."""
    Z = Xs[:, top_idx]
    subnames = [names[i] for i in top_idx]
    pairs: list[tuple[str, str, float]] = []
    d = Z.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            p = Z[:, i] * Z[:, j]
            if np.std(p) < 1e-12:
                continue
            r = np.corrcoef(y, p)[0, 1]
            pairs.append((subnames[i], subnames[j], float(abs(r))))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:max_pairs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot features, MI, binned nonlinearities, interaction hints"
    )
    parser.add_argument("parquet", help="Path to *_turn.parquet")
    parser.add_argument(
        "--out-dir",
        default="analysis/feature_explore_out",
        help="Output directory for PNG/CSV/JSON",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Subsample rows for speed (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--heatmap-features",
        type=int,
        default=28,
        help="Submatrix size for readable correlation heatmap (top N by MI)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Quantile bins for binned outcome plots",
    )
    parser.add_argument(
        "--interaction-top",
        type=int,
        default=18,
        help="Scan pairwise products among top-N features by MI",
    )
    parser.add_argument(
        "--binned-top",
        type=int,
        default=12,
        help="How many top-MI features get binned outcome plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.parquet)
    df = ensure_derived_features(df)
    df = get_training_rows(df)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print("Missing columns:", missing, file=sys.stderr)
        sys.exit(1)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    X, y, df = _subsample(X, y, df, args.sample_n, args.seed)
    n = len(y)
    print(f"Using {n} rows after filter" + (f" (subsampled)" if args.sample_n else ""))

    # --- Mutual information (nonlinear relevance) ---
    mi = mutual_info_regression(X, y, random_state=args.seed)
    mi_order = np.argsort(-mi)
    plot_bar(
        FEATURE_COLS,
        mi,
        out_dir / "mutual_info_target.png",
        "Mutual information vs turn_outcome",
        "MI",
    )
    pd.DataFrame({"feature": FEATURE_COLS, "mi": mi}).sort_values(
        "mi", ascending=False
    ).to_csv(out_dir / "mutual_info_target.csv", index=False)

    # --- Random Forest importances (interaction-capable model) ---
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=max(5, n // 5000),
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    imp = rf.feature_importances_
    plot_bar(
        FEATURE_COLS,
        imp,
        out_dir / "random_forest_importance.png",
        "RandomForestRegressor feature importances",
        "Importance",
    )
    pd.DataFrame({"feature": FEATURE_COLS, "importance": imp}).sort_values(
        "importance", ascending=False
    ).to_csv(out_dir / "random_forest_importance.csv", index=False)

    # --- Full correlation matrix ---
    corr_df = df[FEATURE_COLS].corr()
    corr_df.to_csv(out_dir / "correlation_pearson_full.csv")

    # --- Top-N by MI: correlation submatrix (readable) ---
    hm_n = min(args.heatmap_features, len(FEATURE_COLS))
    top_names = [FEATURE_COLS[i] for i in mi_order[:hm_n]]
    plot_correlation_heatmap(
        df,
        top_names,
        out_dir / f"correlation_top{hm_n}_by_mi.png",
        f"Pearson correlation (top {hm_n} features by MI)",
    )

    # --- Standardized X for quadratic / interaction probes ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sq = top_squared_correlations(Xs, y, FEATURE_COLS)
    plot_bar(
        [s[0] for s in sq[:30]],
        np.array([s[1] for s in sq[:30]]),
        out_dir / "abs_corr_y_x_squared.png",
        "|corr(y, x_i^2)| after standardizing x_i (quadratic probe)",
        "|corr|",
    )
    with open(out_dir / "quadratic_probe.json", "w") as f:
        json.dump(
            {"top_k": [{"feature": a, "abs_corr_y_x2": b} for a, b in sq[:25]]},
            f,
            indent=2,
        )

    top_idx = mi_order[: min(args.interaction_top, len(FEATURE_COLS))].tolist()
    pairs = top_interaction_correlations(Xs, y, FEATURE_COLS, top_idx, max_pairs=120)
    pd.DataFrame(
        pairs, columns=["feature_i", "feature_j", "abs_corr_y_product"]
    ).to_csv(out_dir / "interaction_product_correlations.csv", index=False)

    # --- Binned outcome plots (nonlinearity) ---
    top_b = mi_order[: args.binned_top]
    bdir = out_dir / "binned_outcome"
    bdir.mkdir(exist_ok=True)
    for j in top_b:
        name = FEATURE_COLS[j]
        safe = name.replace("/", "_")
        plot_binned_outcome(X[:, j], y, name, args.bins, bdir / f"{safe}.png")

    # --- Summary text ---
    summary_lines = [
        f"Rows: {n}",
        "",
        "Top 5 by mutual information with turn_outcome:",
        *[f"  {FEATURE_COLS[i]}: {mi[i]:.5f}" for i in mi_order[:5]],
        "",
        "Top 5 by |corr(y, x_i^2)| (quadratic hint on standardized x):",
        *[f"  {a}: {b:.4f}" for a, b in sq[:5]],
        "",
        "Top 10 pairwise product interactions |corr(y, z_i*z_j)| (standardized):",
        *[f"  {a} * {b}: {c:.4f}" for a, b, c in pairs[:10]],
        "",
        "Suggestions:",
        "- If MI and RF disagree, trust RF for tree-like interactions; MI for smooth nonlinearity.",
        "- Large |corr(y, x_i^2)| suggests adding x_i^2 (after standardization) to Ridge.",
        "- Large pairwise product score suggests adding x_i * x_j (or tree / MLP).",
        "- C++ linear rollout uses only linear features; extending it needs new features or a second model.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nWrote plots and tables under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
