#!/usr/bin/env python3
"""
Train DecisionTreeRegressor models to predict win probability for
"last-player" positions (after the perspective player just moved).

Tablebase-territory turns are excluded: once a player plays a TB move in a
game, all subsequent "last-player" rows from that game/perspective are dropped.

The trained models are saved to data/tree_model_d{depth}.joblib.
Writes plots: feature importances (horizontal bar) and CV Brier vs max_depth.

Outputs (under --output-dir / tree_greedy_plots):
    tree_greedy_cv_brier_vs_depth.png   — CV Brier vs max_depth (optionally two curves)
    tree_greedy_feature_importance_best.png — horizontal bar chart (best model)
    tree_greedy_depth_sweep.csv         — all sweep rows

If CV Brier flatlines while max_depth increases, n_nodes usually stops growing:
min_samples_leaf caps tree size before max_depth is reached. Lower
--min-samples-leaf or pass --compare-min-leaf 100 to overlay a second curve.

Example (faster dev: subsample + 3-fold CV + compare):
    conda run -n big2 python analysis/train_tree_greedy.py \\
        --sample-n 500000 --cv-folds 3 --compare-min-leaf 100

Full data (slow):
    conda run -n big2 python analysis/train_tree_greedy.py --compare-min-leaf 100
"""

from __future__ import annotations

import argparse
import csv
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Feature columns — fixed order.  Must match C++ TreeEvaluator::extract_features().
#
# All numeric turn_level features from feature_registry.h except:
#   turn_outcome (label), next_player, tb_case (filters only).
#
# Includes n_ge_* / n_le_* cumulative rank counts, last_move_* indicators,
# highest_*_not_bomb, trick_rank, etc.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "player_hand_size",
    "opponent_hand_size",
    "only_single",
    "n_3",
    "n_4",
    "n_5",
    "n_6",
    "n_7",
    "n_8",
    "n_9",
    "n_10",
    "n_j",
    "n_q",
    "n_k",
    "n_a",
    "n_2",
    "n_ge_4",
    "n_ge_5",
    "n_ge_6",
    "n_ge_7",
    "n_ge_8",
    "n_ge_9",
    "n_ge_10",
    "n_ge_j",
    "n_ge_q",
    "n_ge_k",
    "n_ge_a",
    "n_le_3",
    "n_le_4",
    "n_le_5",
    "n_le_6",
    "n_le_7",
    "n_le_8",
    "n_le_9",
    "n_le_10",
    "n_le_j",
    "n_le_q",
    "n_le_k",
    "n_le_a",
    "highest_single",
    "highest_double",
    "highest_triple",
    "highest_bomb",
    "highest_single_not_bomb",
    "highest_double_not_bomb",
    "highest_triple_not_bomb",
    "last_move_is_pass",
    "last_move_is_single",
    "last_move_is_double",
    "last_move_is_triple",
    "last_move_is_full_house",
    "last_move_is_bomb",
    "last_move_is_single_straight",
    "last_move_is_double_straight",
    "last_move_is_triple_straight",
    "last_move_card_count",
    "n_bombs",
    "possible_moves",
    "possible_moves_not_bomb",
    "trick_rank",
]

assert len(FEATURE_COLS) == 60

# n_3 .. n_2 in FEATURE_COLS order (matches C++ hand rank layout).
RANK_COUNT_COLS = FEATURE_COLS[3:16]


def ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill columns that older Parquet exports may omit (must match C++ TreeEvaluator)."""
    if "only_single" not in df.columns:
        missing_ranks = [c for c in RANK_COUNT_COLS if c not in df.columns]
        if missing_ranks:
            raise ValueError(
                "Cannot derive only_single: missing rank columns: "
                + ", ".join(missing_ranks)
            )
        df = df.copy()
        df["only_single"] = (df[RANK_COUNT_COLS].max(axis=1) <= 1).astype(np.float32)
        print(
            "  Derived only_single from rank counts (column absent in Parquet — "
            "regenerate data with only_single for consistency)"
        )
    return df


# For ``bin/generate_data``: comma-separated turn features (X + labels/filters).
TURN_FEATURES_FOR_DATAGEN = ",".join(
    FEATURE_COLS + ["turn_outcome", "next_player", "tb_case"]
)
# Full pipeline (data → train → export → report): scripts/pipeline_tree_greedy.sh

TARGET_COL = "turn_outcome"
NEXT_PLAYER_COL = "next_player"
TB_CASE_COL = "tb_case"

DEFAULT_DEPTHS = [
    3,
    5,
    7,
    10,
    15,
    18,
    20,
    22,
    25,
    30,
    35,
    40,
    45,
    50,
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Last-player rows with tablebase-heavy data removed.

    1) Keep only last-player rows (``next_player == 0``), excluding the synthetic
       pre-deal row.
    2) Drop all turns on or after the first tablebase move in each
       (``game_index``, ``perspective``) segment. Uses every row with
       ``tb_case != -1`` (not only ``next_player == 1`` rows) so TB turns are not
       missed when tagging differs by row type.
    3) Drop any remaining row where ``tb_case != -1`` on that turn (the moving
       player used a tablebase move), so training never includes turns that the
       engine resolves via TB (matches rollout after ``peek_tablebase``).
    """
    mask = (df[NEXT_PLAYER_COL] == 0) & ~(
        (df["player_hand_size"] == 16) & (df["opponent_hand_size"] == 16)
    )
    lp = df[mask].copy()

    if TB_CASE_COL not in df.columns:
        raise ValueError(
            f"get_training_rows requires column {TB_CASE_COL!r} (turn Parquet from "
            "datagen with tb_case in turn_features)."
        )

    tb_hits = df[df[TB_CASE_COL] != -1]
    first_tb = (
        tb_hits.groupby(["game_index", "perspective"])["turn_idx"]
        .min()
        .rename("first_tb_turn")
        .reset_index()
    )

    lp = lp.merge(first_tb, on=["game_index", "perspective"], how="left")
    lp["first_tb_turn"] = lp["first_tb_turn"].fillna(np.inf)
    lp = lp[lp["turn_idx"] < lp["first_tb_turn"]]
    lp = lp.drop(columns=["first_tb_turn"])

    lp = lp[lp[TB_CASE_COL] == -1]

    return lp


def brier_neg(estimator, X, y):
    return -brier_score_loss(y, estimator.predict(X))


def maybe_subsample(
    X: np.ndarray, y: np.ndarray, n: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n is None or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def plot_feature_importances(
    importances: np.ndarray,
    names: list[str],
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(importances)
    names_sorted = [names[i] for i in order]
    imp_sorted = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, imp_sorted, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("Importance (Gini decrease, normalized)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote feature importance plot: {out_path}")


def plot_cv_brier_vs_depth(
    series: list[tuple[str, list[int], list[float], str]],
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot one or more CV-Brier vs depth curves.

    `series`: list of (label, depths, briers, color).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    global_best = None
    for lab, depths, briers, color in series:
        ax.plot(
            depths, briers, "o-", color=color, linewidth=1.5, markersize=5, label=lab
        )
        bi = int(np.argmin(briers))
        cand = (briers[bi], depths[bi], lab)
        if global_best is None or cand[0] < global_best[0]:
            global_best = cand

    if global_best is not None:
        ax.scatter(
            [global_best[1]],
            [global_best[0]],
            color="crimson",
            s=130,
            zorder=5,
            label=f"global best: {global_best[2]} depth={global_best[1]}, "
            f"Brier={global_best[0]:.5f}",
        )
    ax.set_xlabel("max_depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote CV Brier vs depth plot: {out_path}")


def run_depth_sweep(
    X: np.ndarray,
    y: np.ndarray,
    depths: list[int],
    min_samples_leaf: int,
    kf,
) -> tuple[list[dict], list[tuple[float, int, int, DecisionTreeRegressor]],]:
    """Returns sweep_rows and (cv_brier, depth, min_samples_leaf, fitted_model)."""
    sweep_rows: list[dict] = []
    results: list[tuple[float, int, int, DecisionTreeRegressor]] = []

    print(f"\n--- min_samples_leaf={min_samples_leaf} ---")
    print(f"{'depth':>6}  {'CV Brier':>12}  {'std':>10}  {'n_nodes':>10}")
    print("=" * 72)

    for depth in depths:
        dt = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            criterion="squared_error",
            random_state=42,
        )
        cv_scores = cross_val_score(dt, X, y, cv=kf, scoring=brier_neg, n_jobs=-1)
        cv_brier = float(-cv_scores.mean())
        cv_std = float(cv_scores.std())

        dt.fit(X, y)
        n_nodes = dt.tree_.node_count

        print(f"{depth:>6}  {cv_brier:>12.5f}  {cv_std:>10.5f}  {n_nodes:>10}")
        sweep_rows.append(
            {
                "min_samples_leaf": min_samples_leaf,
                "max_depth": depth,
                "cv_brier_mean": cv_brier,
                "cv_brier_std": cv_std,
                "n_nodes": n_nodes,
            }
        )
        results.append((cv_brier, depth, min_samples_leaf, dt))

    print("=" * 72)
    return sweep_rows, results


def print_plateau_hint(depths: list[int], sweep_rows: list[dict]) -> None:
    """Warn when n_nodes stops growing while max_depth increases."""
    nodes = [r["n_nodes"] for r in sweep_rows]
    stagnant = 0
    for i in range(1, len(depths)):
        if nodes[i] == nodes[i - 1]:
            stagnant += 1
    if stagnant >= 2:
        print(
            "\nNote: n_nodes is unchanged across several increasing depths — the "
            "tree likely hit the min_samples_leaf limit before max_depth. "
            "Deeper max_depth will not help until you lower --min-samples-leaf "
            "or use --compare-min-leaf."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/greedy_200k_turn.parquet",
        help="Path to the turn-level parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write .joblib model files and CSV/plots",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for PNG plots (default: <output-dir>/tree_greedy_plots)",
    )
    parser.add_argument(
        "--depths",
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="Comma-separated max_depth values to sweep",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=500,
        help="min_samples_leaf for all trees in the depth sweep",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional: subsample this many rows (stratified by y) for faster CV",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --sample-n",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG plots",
    )
    parser.add_argument(
        "--no-save-per-depth",
        action="store_true",
        help="Only save the overall best model as tree_model_best.joblib",
    )
    parser.add_argument(
        "--compare-min-leaf",
        type=int,
        default=None,
        metavar="N",
        help="Optional second depth sweep with this min_samples_leaf; overlays "
        "on the CV plot (use when primary sweep plateaus due to leaf limit)",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help="Comma-separated feature names to omit from X (ablation). "
        "Cannot be used for C++ export unless columns still match TreeEvaluator.",
    )
    args = parser.parse_args()

    exclude_set = {x.strip() for x in args.exclude_features.split(",") if x.strip()}
    feature_cols = [c for c in FEATURE_COLS if c not in exclude_set]
    if not feature_cols:
        raise SystemExit("No features left after --exclude-features")
    if exclude_set:
        print(
            f"Ablation: omitting {sorted(exclude_set)} → using {len(feature_cols)} "
            f"of {len(FEATURE_COLS)} features\n"
        )

    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    if not depths:
        raise SystemExit("No depths parsed from --depths")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = args.plot_dir or os.path.join(args.output_dir, "tree_greedy_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading {args.input} …")
    df = load_data(args.input)
    print(f"  Total rows: {len(df):,}")

    df = ensure_derived_features(df)

    df = get_training_rows(df)
    print(f"  After last-player + TB filter: {len(df):,} rows")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    if args.sample_n is not None:
        n = min(args.sample_n, len(y))
        rng = np.random.default_rng(args.sample_seed)
        # Stratified sample: keep class balance roughly
        idx0 = np.flatnonzero(y < 0.5)
        idx1 = np.flatnonzero(y >= 0.5)
        half = n // 2
        take0 = rng.choice(idx0, size=min(half, len(idx0)), replace=False)
        take1 = rng.choice(idx1, size=min(n - len(take0), len(idx1)), replace=False)
        idx = np.concatenate([take0, take1])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]
        print(f"  Subsampled to {len(y):,} rows (--sample-n {args.sample_n})")

    print(f"\nTarget mean (win rate): {y.mean():.4f}")
    print(
        f"Depth sweep: depths={depths}, min_samples_leaf={args.min_samples_leaf}, "
        f"cv_folds={args.cv_folds}"
    )

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    sweep_rows, results = run_depth_sweep(X, y, depths, args.min_samples_leaf, kf)
    print_plateau_hint(depths, sweep_rows)

    compare_rows: list[dict] = []
    compare_results: list[tuple[float, int, int, DecisionTreeRegressor]] = []
    if args.compare_min_leaf is not None:
        compare_rows, compare_results = run_depth_sweep(
            X, y, depths, args.compare_min_leaf, kf
        )
        print_plateau_hint(depths, compare_rows)

    all_rows = sweep_rows + compare_rows
    csv_path = os.path.join(plot_dir, "tree_greedy_depth_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote depth sweep CSV: {csv_path}")

    all_results = results + compare_results
    best_brier, best_depth, best_leaf, best_model = min(all_results, key=lambda r: r[0])
    print(
        f"\nBest by CV Brier: depth={best_depth}  min_samples_leaf={best_leaf}  "
        f"Brier={best_brier:.5f}  nodes={best_model.tree_.node_count}"
    )

    # Δ Brier vs previous depth (primary sweep only)
    briers = [r[0] for r in results]
    dlist = [r[1] for r in results]
    print("\nΔ Brier vs previous depth — primary sweep (negative = improved):")
    for i, d in enumerate(dlist):
        if i == 0:
            print(f"  depth {d:3d}:  —")
        else:
            delta = briers[i] - briers[i - 1]
            print(f"  depth {d:3d}:  {delta:+.6f}  (vs depth {dlist[i - 1]})")

    ylabel = f"{args.cv_folds}-fold CV Brier (lower is better)"
    if not args.no_plots:
        series = [
            (
                f"min_leaf={args.min_samples_leaf}",
                dlist,
                briers,
                "steelblue",
            ),
        ]
        if compare_results:
            cb = [r[0] for r in compare_results]
            cd = [r[1] for r in compare_results]
            series.append(
                (
                    f"min_leaf={args.compare_min_leaf}",
                    cd,
                    cb,
                    "darkorange",
                )
            )
        plot_cv_brier_vs_depth(
            series,
            os.path.join(plot_dir, "tree_greedy_cv_brier_vs_depth.png"),
            title=f"CV Brier vs max_depth (n={len(y):,} rows)",
            ylabel=ylabel,
        )
        imp = best_model.feature_importances_
        plot_feature_importances(
            imp,
            feature_cols,
            os.path.join(plot_dir, "tree_greedy_feature_importance_best.png"),
            title=f"Feature importances (best: depth={best_depth}, "
            f"min_leaf={best_leaf}, Brier={best_brier:.4f})",
        )

    if not args.no_save_per_depth:
        print("\nSaving one model per depth (primary min_samples_leaf):")
        for brier, depth, _ml, model in results:
            path = os.path.join(args.output_dir, f"tree_model_d{depth}.joblib")
            joblib.dump({"model": model, "feature_cols": feature_cols}, path)
            print(
                f"  depth={depth:3d}  CV Brier={brier:.5f}  "
                f"nodes={model.tree_.node_count:5d}  → {path}"
            )
        if compare_results:
            cb, cd, _cml, cm = min(compare_results, key=lambda r: r[0])
            cp = os.path.join(
                args.output_dir,
                f"tree_model_best_leaf{args.compare_min_leaf}.joblib",
            )
            joblib.dump({"model": cm, "feature_cols": feature_cols}, cp)
            print(
                f"\nBest from compare sweep (min_leaf={args.compare_min_leaf}): "
                f"depth={cd} Brier={cb:.5f} → {cp}"
            )

    best_bundle = {"model": best_model, "feature_cols": feature_cols}
    best_path = os.path.join(args.output_dir, "tree_model_best.joblib")
    joblib.dump(best_bundle, best_path)
    print(f"\nGlobal best model saved: {best_path}")

    print("\nFeature importances (best model):")
    order = np.argsort(best_model.feature_importances_)[::-1]
    for rank, idx in enumerate(order):
        print(
            f"  {rank+1:2d}. {feature_cols[idx]:<30}  "
            f"{best_model.feature_importances_[idx]:.4f}"
        )


if __name__ == "__main__":
    main()
