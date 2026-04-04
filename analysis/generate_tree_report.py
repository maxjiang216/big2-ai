#!/usr/bin/env python3
"""
Generate an HTML report with figures for the tree-greedy model at max_depth=15.

Reads ``data/tree_model_d15.joblib`` (trains it if missing using the same
pipeline as ``train_tree_greedy.py``).  The model uses the numeric turn-level
features from ``FEATURE_COLS`` in ``train_tree_greedy.py`` (count varies), including
``n_ge_*``, ``n_le_*``, last-move indicators, ``trick_rank``, etc.

Writes to ``analysis/reports/tree_depth10/`` (default; see ``--out-dir``):
  - report.html          — summary + embedded figures
  - feature_importance_bar.png
  - feature_importance_cumulative.png
  - feature_importances.csv

Usage (from repo root, conda env with sklearn):
    conda run -n big2 python analysis/generate_tree_report.py
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

# Reuse training helpers from the same package directory
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "analysis"))

import train_tree_greedy as tg  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402


def ensure_model(
    joblib_path: str,
    parquet_path: str,
    max_depth: int,
    min_samples_leaf: int,
) -> dict:
    if os.path.isfile(joblib_path):
        print(f"Loading model: {joblib_path}")
        return joblib.load(joblib_path)

    print(f"No model at {joblib_path} — training depth={max_depth} …")
    df = tg.load_data(parquet_path)
    df = tg.get_training_rows(df)
    missing = [c for c in tg.FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in parquet: {missing}")

    X = df[tg.FEATURE_COLS].values.astype(np.float32)
    y = df[tg.TARGET_COL].values.astype(np.float32)

    dt = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion="squared_error",
        random_state=42,
    )
    dt.fit(X, y)
    bundle = {"model": dt, "feature_cols": tg.FEATURE_COLS}
    os.makedirs(os.path.dirname(joblib_path) or ".", exist_ok=True)
    joblib.dump(bundle, joblib_path)
    print(f"Saved {joblib_path}  (n_nodes={dt.tree_.node_count})")
    return bundle


def plot_bar(
    names: list[str],
    importances: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(importances)
    names_sorted = [names[i] for i in order]
    imp_sorted = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(names))))
    y_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(names)))
    ax.barh(y_pos, imp_sorted, align="center", color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xlabel("Importance (Gini decrease, normalized)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative(
    names: list[str],
    importances: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(importances)[::-1]
    names_desc = [names[i] for i in order]
    imp_desc = importances[order]
    cum = np.cumsum(imp_desc)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(cum) + 1), cum, "o-", color="steelblue", linewidth=2)
    ax.set_xticks(np.arange(1, len(cum) + 1))
    ax.set_xticklabels(names_desc, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative importance")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_csv(
    names: list[str],
    importances: np.ndarray,
    path: str,
) -> None:
    order = np.argsort(importances)[::-1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance"])
        for r, i in enumerate(order, start=1):
            w.writerow([r, names[i], f"{importances[i]:.8f}"])


def write_html(
    out_dir: str,
    depth: int,
    min_leaf: int,
    n_nodes: int,
    n_features: int,
    parquet_used: str,
) -> None:
    title = f"Tree greedy — max_depth={depth}"
    bar_rel = "feature_importance_bar.png"
    cum_rel = "feature_importance_cumulative.png"
    csv_rel = "feature_importances.csv"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; color: #1a1a1a; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
    .meta {{ color: #444; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 1rem 0; }}
    code {{ background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 3px; }}
    a {{ color: #0b5; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">
    Generated {html.escape(now)}<br/>
    Training data: <code>{html.escape(parquet_used)}</code><br/>
    <code>min_samples_leaf={min_leaf}</code> ·
    tree nodes: <strong>{n_nodes}</strong> ·
    features: <strong>{n_features}</strong>
  </p>
  <p>
    Importance values are the sklearn tree’s normalized Gini decrease
    (same as <code>model.feature_importances_</code>).
  </p>

  <h2>Bar chart (all features, low → high)</h2>
  <p><img src="{html.escape(bar_rel)}" alt="Feature importance bar chart"/></p>

  <h2>Cumulative importance (sorted high → low)</h2>
  <p><img src="{html.escape(cum_rel)}" alt="Cumulative feature importance"/></p>

  <h2>Data</h2>
  <p>CSV: <a href="{html.escape(csv_rel)}">{html.escape(csv_rel)}</a></p>
</body>
</html>
"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML report for tree greedy depth-15")
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="max_depth (default: 10)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=100,
        help="min_samples_leaf used when training a missing model",
    )
    parser.add_argument(
        "--parquet",
        default=os.path.join(_REPO, "data", "greedy_200k_turn.parquet"),
        help="Turn-level parquet for training if joblib missing",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(_REPO, "data"),
        help="Directory containing tree_model_d{{depth}}.joblib",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_REPO, "analysis", "reports", "tree_depth10"),
        help="Output directory for report.html and figures",
    )
    args = parser.parse_args()

    depth = args.depth
    joblib_path = os.path.join(args.model_dir, f"tree_model_d{depth}.joblib")
    bundle = ensure_model(
        joblib_path,
        args.parquet,
        max_depth=depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    model = bundle["model"]
    names: list[str] = list(bundle["feature_cols"])
    imp = np.asarray(model.feature_importances_, dtype=np.float64)

    os.makedirs(args.out_dir, exist_ok=True)

    bar_path = os.path.join(args.out_dir, "feature_importance_bar.png")
    cum_path = os.path.join(args.out_dir, "feature_importance_cumulative.png")
    csv_path = os.path.join(args.out_dir, "feature_importances.csv")

    plot_bar(
        names,
        imp,
        bar_path,
        title=f"Feature importances — max_depth={depth} (bar, low→high)",
    )
    plot_cumulative(
        names,
        imp,
        cum_path,
        title=f"Cumulative importance — max_depth={depth} (sorted high→low)",
    )
    write_csv(names, imp, csv_path)

    write_html(
        args.out_dir,
        depth=depth,
        min_leaf=args.min_samples_leaf,
        n_nodes=model.tree_.node_count,
        n_features=len(names),
        parquet_used=os.path.relpath(args.parquet, _REPO),
    )

    print(f"Figures: {bar_path}")
    print(f"         {cum_path}")
    print(f"CSV:      {csv_path}")
    print(
        f"Open:     file://{os.path.abspath(os.path.join(args.out_dir, 'report.html'))}"
    )


if __name__ == "__main__":
    main()
