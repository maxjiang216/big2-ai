#!/usr/bin/env python3
"""
Two separate L1 logistic models on PIMC self-play Parquet slices:
  initiative  — fresh trick, your turn (next_player==1, trick_rank==-1)
  post_play   — right after your move while opponent chooses (derived via shift)

Applies TB prefix truncation: drop all turns >= first ply with tb_case != -1.
Also use --compare-selection for grouped-CV comparison of feature pruning (L1
SelectFromModel, RF importance, mutual information, L1 top-|coef|) vs full
logistic; writes figures under <out-dir>/feature_selection/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]


RANK_COUNT_COLS = [
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
]

META_COLS = frozenset(
    {
        "game_index",
        "turn_idx",
        "perspective",
        "turn_outcome",
        "next_player",
        "trick_rank",
        "tb_case",
    }
)


def apply_tb_prefix_truncation(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows strictly before the first mover ply where tb_case != -1."""
    tb_first = df[df["tb_case"] != -1].groupby("game_index")["turn_idx"].min()
    thresh = df["game_index"].map(tb_first)
    kept = thresh.isna() | (df["turn_idx"] < thresh)
    return df.loc[kept].copy()


def build_slice_initiative(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["next_player"] == 1) & (df["trick_rank"] == -1)]
    return out.copy()


def build_slice_post_play(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values(["game_index", "perspective", "turn_idx"])
    prev = g.groupby(["game_index", "perspective"], sort=False)["next_player"].shift(1)
    g = g.assign(_prev_np=prev)
    out = g[(g["next_player"] == 0) & (g["_prev_np"] == 1)]
    return out.drop(columns=["_prev_np"]).copy()


def rank_and_numeric_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    rk = [c for c in RANK_COUNT_COLS if c in df.columns]
    drop = META_COLS | set(rk)
    num = sorted(c for c in df.columns if c not in drop)
    return rk, num


def slice_diagnostics(name: str, sub: pd.DataFrame) -> None:
    nw = int((sub["turn_outcome"] == 1).sum())
    nl = int((sub["turn_outcome"] == 0).sum())
    ng = sub["game_index"].nunique()
    print(f"\n--- slice: {name} ---")
    print(f"rows={len(sub):,} wins={nw:,} losses={nl:,} distinct_games={ng:,}")
    if nw > 0 and nl > 0:
        print(f"balance min(win,loss)={min(nw, nl):,}")


def build_grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int):
    """List of train/val integer index arrays for LogisticRegressionCV(cv=...,)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(y)), y, groups))


def column_transform(rank_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """Fit on dataframe columns rank_cols | numeric_cols; returns sparse-wide matrix."""
    feats: list[tuple[str, object, list[str]]] = []
    if rank_cols:
        feats.append(
            (
                "rank_oh",
                OneHotEncoder(
                    sparse_output=True,
                    categories=[[0, 1, 2, 3, 4]] * len(rank_cols),
                    handle_unknown="ignore",
                ),
                rank_cols,
            )
        )
    feats.append(("num", "passthrough", numeric_cols))
    return ColumnTransformer(feats, remainder="drop", sparse_threshold=1.0)


def make_pipeline(
    rank_cols: list[str],
    numeric_cols: list[str],
    cv_splits: list,
    cs_grid: np.ndarray,
) -> Pipeline:
    preprocess = column_transform(rank_cols, numeric_cols)
    clf = LogisticRegressionCV(
        Cs=cs_grid,
        cv=cv_splits,
        penalty="elasticnet",
        l1_ratios=(1.0,),
        solver="saga",
        scoring="neg_log_loss",
        max_iter=6000,
        n_jobs=-1,
        tol=1e-3,
        refit=True,
        random_state=0,
        use_legacy_attributes=True,
    )
    return Pipeline([("prep", preprocess), ("logit", clf)])


def coef_table(pipe: Pipeline) -> pd.DataFrame:
    lr: LogisticRegressionCV = pipe.named_steps["logit"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = lr.coef_.ravel()
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    return df.iloc[np.argsort(-np.abs(df["coef"]))].reset_index(drop=True)


def train_one_slice(name: str, sub: pd.DataFrame, out_dir: Path, n_splits: int) -> None:
    slice_diagnostics(name, sub)
    rank_cols, num_cols = rank_and_numeric_cols(sub)

    missing_r = set(RANK_COUNT_COLS) - set(rank_cols)
    if missing_r:
        print(f"warning {name}: missing rank cols {sorted(missing_r)}")

    if not rank_cols and not num_cols:
        print(f"skip {name}: no feature columns.")
        return

    y = sub["turn_outcome"].astype(int).to_numpy()
    if np.unique(y).size < 2:
        print(f"skip {name}: constant label.")
        return

    groups = sub["game_index"].to_numpy()
    n_group = int(pd.Series(groups).nunique())
    n_folds = max(2, min(n_splits, n_group))
    if n_folds < n_splits:
        print(
            f"warning {name}: capped CV folds from {n_splits} to {n_folds} "
            "(distinct games in slice)"
        )

    splits = build_grouped_splits(y, groups, n_splits=n_folds)
    if len(splits) < 2:
        print(f"skip {name}: not enough splits for grouped CV.")
        return

    X = sub[rank_cols + num_cols]
    cs = np.logspace(-3, 1, 12)
    pipe = make_pipeline(rank_cols, num_cols, splits, cs_grid=cs)

    pipe.fit(X, y)
    lr = pipe.named_steps["logit"]

    tab = coef_table(pipe)
    out_csv = out_dir / f"hand_strength_{name}_coef.csv"
    tab.to_csv(out_csv, index=False)

    cv_scores = getattr(lr, "scores_", None)
    print(f"{name}: C selected = {lr.C_[0]:.6f}")
    if cv_scores is not None and hasattr(cv_scores, "mean"):
        mean_cv = cv_scores.mean(axis=(0,))
        print(
            "  mean CV neg_log_loss (best-C axis): "
            f"{mean_cv[np.argmax(mean_cv)].item():.6f}"
            if getattr(mean_cv, "ndim", 0) == 1
            else "  (CV scores ndarray shape unsupported for summary)"
        )
    print(f"wrote {out_csv}\n(top 18 by |coef|)\n{tab.head(18).to_string(index=False)}\n")


def _safe_probs(p: np.ndarray) -> np.ndarray:
    """Clip probabilities for numerical stability."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return p


def _metrics_cls(y_va: np.ndarray, probs_pos: np.ndarray) -> tuple[float, float, float]:
    pp = _safe_probs(probs_pos)
    ll = float(log_loss(y_va, pp))
    roc = (
        float(roc_auc_score(y_va, pp))
        if len(np.unique(y_va)) >= 2
        else float("nan")
    )
    br = float(brier_score_loss(y_va, pp))
    return ll, roc, br


def _sparse_col_subset(
    xt: sp.spmatrix, support: np.ndarray
) -> sp.spmatrix:
    cols = np.flatnonzero(support)
    if cols.size == 0:
        raise ValueError("empty feature mask")
    return xt.tocsc()[:, cols].tocsr()


def _dense_subsample_for_mi(
    x_d: np.ndarray, y_sub: np.ndarray, rng: np.random.Generator, max_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    if x_d.shape[0] <= max_rows:
        return x_d, y_sub
    idx = rng.choice(x_d.shape[0], max_rows, replace=False)
    return x_d[idx], y_sub[idx]



def _register_methods(
    *,
    mutual_info_k: int,
    rf_trees: int,
    mi_rows: int,
) -> dict[str, object]:
    """Factory dict: method_name -> callable (Xt_tr, y_tr, Xt_va, rnd) -> result dict."""

    def full_l2(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=7000,
            tol=3e-4,
            random_state=int(rng.integers(2**31 - 1)),
        )
        clf.fit(xt_tr, y_tr)
        pr = clf.predict_proba(xt_va)[:, 1]
        return {
            "probs_pos": pr,
            "n_selected": xt_tr.shape[1],
        }

    def sfm_lr_median(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        rf_seed = int(rng.integers(2**31 - 1))
        gate = LogisticRegression(
            C=0.25,
            penalty="l1",
            solver="saga",
            max_iter=8000,
            tol=5e-4,
            random_state=rf_seed,
        )
        sfm = SelectFromModel(gate, threshold="median")
        sfm.fit(xt_tr, y_tr)
        xts = sfm.transform(xt_tr)
        xvs = sfm.transform(xt_va)
        if xts.shape[1] == 0:
            return full_l2(xt_tr, y_tr, xt_va, rng)
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=5000,
            random_state=rf_seed ^ 31337,
        )
        clf.fit(xts, y_tr)
        return {"probs_pos": clf.predict_proba(xvs)[:, 1], "n_selected": xts.shape[1]}

    def sfm_rf_median(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        rf_seed = int(rng.integers(2**31 - 1))
        rf = RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=16,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=rf_seed,
            n_jobs=-1,
        )
        rf.fit(xt_tr, y_tr)
        sfm = SelectFromModel(rf, prefit=True, threshold="median")
        xts = sfm.transform(xt_tr)
        xvs = sfm.transform(xt_va)
        if xts.shape[1] == 0:
            return full_l2(xt_tr, y_tr, xt_va, rng)
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=4000,
            random_state=rf_seed ^ 99991,
        )
        clf.fit(xts, y_tr)
        return {"probs_pos": clf.predict_proba(xvs)[:, 1], "n_selected": xts.shape[1]}

    def mi_select_refit(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        k_feat = min(mutual_info_k, xt_tr.shape[1])
        if k_feat <= 1:
            return full_l2(xt_tr, y_tr, xt_va, rng)
        xd = xt_tr.toarray()
        xd_s, ys = _dense_subsample_for_mi(xd, y_tr, rng, mi_rows)
        mi_seed = int(rng.integers(2**31 - 1))
        sk = SelectKBest(
            mutual_info_classif,
            k=k_feat,
            # MI is stochastic; stabilize for replication
            # (passed through partial not directly — sklearn uses RNG from random_state via np)
        )
        # sklearn SelectKBest doesn't pass rng to scorer; tolerate noise
        _ = mi_seed  # noqa: F841
        sk.fit(xd_s, ys)
        sup = sk.get_support()
        xts = _sparse_col_subset(xt_tr, sup)
        xvs = _sparse_col_subset(xt_va, sup)
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=5000,
            random_state=int(rng.integers(2**31 - 1)),
        )
        clf.fit(xts, y_tr)
        return {"probs_pos": clf.predict_proba(xvs)[:, 1], "n_selected": xts.shape[1]}

    def sparse_l1_prune_then_l2(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        seed = int(rng.integers(2**31 - 1))
        gate = LogisticRegression(
            C=0.2,
            penalty="l1",
            solver="saga",
            max_iter=12000,
            tol=5e-4,
            random_state=seed,
        )
        gate.fit(xt_tr, y_tr)
        coef = np.abs(gate.coef_.ravel())
        top_k = min(52, coef.size)
        order = np.argsort(-coef)[:top_k]
        mask = np.zeros(coef.size, dtype=bool)
        mask[order] = True
        nzero = mask.sum()
        if nzero < 5:
            return full_l2(xt_tr, y_tr, xt_va, rng)
        xts = _sparse_col_subset(xt_tr, mask)
        xvs = _sparse_col_subset(xt_va, mask)
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=4500,
            random_state=seed ^ 424242,
        )
        clf.fit(xts, y_tr)
        return {"probs_pos": clf.predict_proba(xvs)[:, 1], "n_selected": int(nzero)}

    def ridge_strong(
        xt_tr: sp.spmatrix, y_tr: np.ndarray, xt_va: sp.spmatrix, rng: np.random.Generator
    ):
        """Stronger ridge (fewer dof) baseline."""
        clf = LogisticRegression(
            C=0.15,
            penalty="l2",
            solver="saga",
            max_iter=8000,
            random_state=int(rng.integers(2**31 - 1)),
        )
        clf.fit(xt_tr, y_tr)
        return {
            "probs_pos": clf.predict_proba(xt_va)[:, 1],
            "n_selected": xt_tr.shape[1],
        }

    out = {
        "full_l2": full_l2,
        "ridge_c015": ridge_strong,
        "select_l1median_sfm_then_l2": sfm_lr_median,
        "select_rf_median_sfm_then_l2": sfm_rf_median,
        f"mutual_inf_top_{mutual_info_k}_then_l2": mi_select_refit,
        "l1_top52cols_then_l2": sparse_l1_prune_then_l2,
    }
    return out


def run_feature_selection_comparison(
    initiative: pd.DataFrame,
    post_play: pd.DataFrame,
    out_dir: Path,
    n_splits: int,
    mutual_info_k: int,
    rf_trees: int,
    mi_rows: int,
    random_state: int,
) -> None:
    """Compare feature-selection strategies via grouped CV on both slices."""
    fig_dir = out_dir / "feature_selection"
    fig_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for name, sub in (("initiative", initiative), ("post_play", post_play)):
        rk, nc = rank_and_numeric_cols(sub)
        if not rk and not nc:
            print(f"comparison: skip {name} (no features).")
            continue
        print(f"\n=== feature-selection CV: {name} ===")
        slice_diagnostics(name, sub)
        dfm = grouped_selection_comparison(
            name,
            sub,
            rk,
            nc,
            fig_dir,
            n_splits=n_splits,
            random_state=random_state,
            mutual_info_k=mutual_info_k,
            rf_trees=rf_trees,
            mi_rows=mi_rows,
        )
        parts.append(dfm)

    if not parts:
        print("No comparison results.")
        return

    combo = pd.concat(parts, axis=0, ignore_index=True)
    combo.to_csv(fig_dir / "feature_selection_fold_metrics.csv", index=False)
    save_selection_visualizations(combo, fig_dir)

    summ = combo.groupby(["slice", "method"], dropna=False).agg(
        log_loss=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"),
        roc_auc=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        n_feat=("n_selected", "mean"),
        brier=("brier_score", "mean"),
    )
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print("\n=== Pruned vs full summary (mean ± std across folds, grouped val) ===\n")
    try:
        print(summ.to_string())
    finally:
        pd.reset_option("display.max_rows")
        pd.reset_option("display.width")
    with open(fig_dir / "feature_selection_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summ.to_string())
        fh.write("\n")
    names = [
        "feature_selection_performance.png",
        "feature_selection_counts.png",
        "feature_selection_fold_boxplots.png",
        "feature_selection_logloss_heatmap.png",
    ]
    print(f"\nWrote comparison artefacts under {fig_dir}:")
    print("  CSV: feature_selection_fold_metrics.csv, feature_selection_aggregate.csv")
    for n in names:
        print(f"  PNG: {fig_dir / n}")


def grouped_selection_comparison(
    slice_name: str,
    df_slice: pd.DataFrame,
    rank_cols: list[str],
    num_cols: list[str],
    _fig_dir: Path,
    n_splits: int,
    random_state: int,
    mutual_info_k: int,
    rf_trees: int,
    mi_rows: int,
) -> pd.DataFrame:
    y = df_slice["turn_outcome"].astype(int).to_numpy()
    groups = df_slice["game_index"].to_numpy()
    n_grp = int(pd.Series(groups).nunique())
    n_folds = max(3, min(n_splits, n_grp))
    splits = list(GroupKFold(n_splits=n_folds).split(df_slice, y, groups))

    methods = _register_methods(
        mutual_info_k=mutual_info_k,
        rf_trees=rf_trees,
        mi_rows=mi_rows,
    )
    method_names = [
        "full_l2",
        "ridge_c015",
        "select_l1median_sfm_then_l2",
        "select_rf_median_sfm_then_l2",
        f"mutual_inf_top_{mutual_info_k}_then_l2",
        "l1_top52cols_then_l2",
    ]

    rows: list[dict] = []
    n_feat_total: int | None = None

    for fold_idx, (tr_ix, va_ix) in enumerate(splits):
        prep = column_transform(rank_cols, num_cols)
        x_tr_df = df_slice.iloc[tr_ix][rank_cols + num_cols]
        x_va_df = df_slice.iloc[va_ix][rank_cols + num_cols]
        y_tr, y_va = y[tr_ix], y[va_ix]
        prep.fit(x_tr_df)
        xt_tr = prep.transform(x_tr_df).tocsr()
        xt_va = prep.transform(x_va_df).tocsr()
        if n_feat_total is None:
            n_feat_total = xt_tr.shape[1]

        for mi, mname in enumerate(method_names):
            rng = np.random.default_rng(random_state + 997 * fold_idx + 53 * mi)
            fn = methods[mname]
            err_note: str | None = None
            try:
                res = fn(xt_tr, y_tr, xt_va, rng)
                pr = res["probs_pos"]
                n_sel = int(res["n_selected"])
                ll, auc, bri = _metrics_cls(y_va, pr)
            except Exception as exc:  # noqa: BLE001
                ll, auc, bri = float("nan"), float("nan"), float("nan")
                n_sel = -1
                err_note = repr(exc)
            rows.append(
                {
                    "slice": slice_name,
                    "fold": fold_idx,
                    "method": mname,
                    "log_loss": ll,
                    "roc_auc": auc,
                    "brier_score": bri,
                    "n_selected": n_sel if n_sel >= 0 else np.nan,
                    "n_features_total": n_feat_total,
                    **({"exception": err_note} if err_note else {}),
                }
            )

    return pd.DataFrame(rows)


def save_selection_visualizations(
    all_metrics: pd.DataFrame, fig_dir: Path
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg = (
        all_metrics.groupby(["slice", "method"], dropna=False)
        .agg(
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            brier_mean=("brier_score", "mean"),
            n_feat_mean=("n_selected", "mean"),
            n_fold=("fold", lambda s: len(np.unique(s))),
        )
        .reset_index()
    )

    agg.to_csv(fig_dir / "feature_selection_aggregate.csv", index=False)

    slices = list(agg["slice"].unique())
    methods_display = agg["method"].unique().tolist()
    palette = plt.cm.tab10(np.linspace(0, 1, len(methods_display)))

    fig, axes = plt.subplots(len(slices), 2, figsize=(12, max(6, len(slices) * 4)))
    if len(slices) == 1:
        axes = np.asarray([axes])
    metric_specs = (
        ("log_loss_mean", "log_loss_std", "mean log-loss (↓ better)"),
        ("roc_auc_mean", "roc_auc_std", "mean ROC-AUC (↑ better)"),
    )
    width = 0.65

    for i, sl in enumerate(slices):
        row = agg[agg["slice"] == sl]
        for ax_idx, (mcol, scol, ylabel) in enumerate(metric_specs):
            ax = axes[i, ax_idx]
            for j, m in enumerate(methods_display):
                r = row[row["method"] == m]
                if len(r) != 1:
                    continue
                v = float(r[mcol].iloc[0])
                std = (
                    float(r[scol].iloc[0])
                    if pd.notna(r[scol].iloc[0])
                    else 0.0
                )
                ax.bar(
                    j,
                    v,
                    yerr=std,
                    width=width,
                    color=palette[j % len(palette)],
                )
            ax.set_xticks(range(len(methods_display)))
            ax.set_xticklabels(methods_display, rotation=28, ha="right", fontsize=8)
            ax.set_ylabel("value")
            ax.set_title(f"{sl} · {ylabel}")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    png = fig_dir / "feature_selection_performance.png"
    fig.savefig(png, dpi=144)
    plt.close(fig)

    # N features horizontal bar chart (means)
    agg_nf = agg[["slice", "method", "n_feat_mean"]].drop_duplicates()
    fig2, axarr = plt.subplots(1, len(slices), figsize=(max(8, len(slices) * 5), 4))
    if len(slices) == 1:
        axarr = [axarr]
    for ax, sl in zip(axarr, slices, strict=True):
        r = agg_nf[agg_nf["slice"] == sl].sort_values("n_feat_mean")
        colors = palette[: len(r)]
        ax.barh(r["method"], r["n_feat_mean"], color=colors)
        ax.set_xlabel("mean # features retained (approx)")
        ax.set_title(f"{sl} · feature-count")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    png2 = fig_dir / "feature_selection_counts.png"
    fig2.savefig(png2, dpi=144)
    plt.close(fig2)

    # Fold-level distributions for the larger slice (post_play if present)
    target_sl = "post_play" if (all_metrics["slice"] == "post_play").any() else slices[0]
    subm = all_metrics[all_metrics["slice"] == target_sl].copy()
    fig3, (ax_ll, ax_auc) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    meth_order_fold = sorted(subm["method"].unique())
    for ax, metric, title in (
        (
            ax_ll,
            "log_loss",
            f"Log-loss by CV fold ({target_sl}) lower is better",
        ),
        (ax_auc, "roc_auc", f"ROC-AUC by CV fold ({target_sl}) higher is better"),
    ):
        data = [subm.loc[subm["method"] == m, metric].dropna().values for m in meth_order_fold]
        bp = ax.boxplot(
            data, tick_labels=meth_order_fold, patch_artist=True, showmeans=True
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.7)
        ax.set_xticklabels(meth_order_fold, rotation=32, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "feature_selection_fold_boxplots.png", dpi=144)
    plt.close(fig3)

    piv = agg.pivot(index="method", columns="slice", values="log_loss_mean")
    fig4, ax4 = plt.subplots(figsize=(5.8, max(4.2, piv.shape[0] * 0.38)))
    im = ax4.imshow(piv.values, aspect="auto", cmap="RdYlGn_r")
    ax4.set_yticks(np.arange(len(piv.index)))
    ax4.set_yticklabels(piv.index, fontsize=7)
    ax4.set_xticks(np.arange(len(piv.columns)))
    ax4.set_xticklabels(piv.columns)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax4.text(
                j,
                i,
                f"{piv.values[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )
    plt.colorbar(im, ax=ax4, shrink=0.55, label="mean log-loss")
    ax4.set_title("Mean log-loss (grouped CV val folds)")
    fig4.tight_layout()
    fig4.savefig(fig_dir / "feature_selection_logloss_heatmap.png", dpi=144)
    plt.close(fig4)


def snapshot_check_post_play(sample: pd.DataFrame, parquet_path: Path) -> None:
    if sample.empty:
        print("snapshot_check: empty post_play slice, skip.")
        return
    rk = [c for c in RANK_COUNT_COLS if c in sample.columns]
    if len(rk) < len(RANK_COUNT_COLS):
        print(f"snapshot_check: incomplete rank cols ({len(rk)}), skip.")
        return

    need = sorted((META_COLS | set(RANK_COUNT_COLS)))
    full = pd.read_parquet(parquet_path)
    cols = [c for c in need if c in full.columns]
    df = full[cols]

    row = sample.iloc[0]
    gi = int(row["game_index"])
    ti = int(row["turn_idx"])
    persp = int(row["perspective"])
    prev_mask = (
        (df["game_index"] == gi)
        & (df["perspective"] == persp)
        & (df["turn_idx"] == ti - 1)
    )
    prev = df.loc[prev_mask]
    if prev.empty:
        print("snapshot_check: no predecessor row — skip.")
        return
    a = prev.iloc[0][rk].astype(np.int64).to_numpy()
    b = row[rk].astype(np.int64).to_numpy()
    if np.array_equal(a, b):
        print(
            "snapshot_check WARNING: predecessor hand rank counts identical to "
            "post-play row (unexpected)."
        )
    else:
        print(
            "snapshot_check ok: post-play differs from predecessor own-turn ranks "
            f"(Δ sum |diff|={(np.abs(a - b)).sum()})"
        )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*penalty.*was deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*l1_ratios.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*use_legacy_attributes.*",
        category=FutureWarning,
    )
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "turn_parquet",
        nargs="?",
        default=str(REPO_ROOT / "data/pimc20_selfplay_turn.parquet"),
        help="Turn-level Parquet",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "research" / "hand_strength_models"),
        help="Coefficient CSV export directory",
    )
    ap.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only TB truncate + slices + sanity print; skip fitting",
    )
    ap.add_argument(
        "--compare-selection",
        action="store_true",
        help=(
            "After default L1CV export, run grouped-CV comparison of pruned "
            "models (plots + CSV in out-dir/feature_selection/)"
        ),
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run feature-selection comparison (skip LogisticRegressionCV coef export)",
    )
    ap.add_argument(
        "--mi-top-k",
        type=int,
        default=45,
        help="Mutual-information SelectKBest K (per train fold)",
    )
    ap.add_argument(
        "--rf-trees",
        type=int,
        default=120,
        help="RandomForest trees for SelectFromModel gate",
    )
    ap.add_argument(
        "--mi-max-rows",
        type=int,
        default=70_000,
        help="Row cap when fitting MI (subsample train fold)",
    )
    ap.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="RNG base for selection-study methods",
    )
    ap.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="GroupKFold n_splits (bounded by distinct games)",
    )
    args = ap.parse_args()

    tp = Path(args.turn_parquet)
    if not tp.is_absolute():
        tp = REPO_ROOT / tp
    if not tp.is_file():
        raise SystemExit(f"Missing parquet: {tp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(tp)
    need_cols = META_COLS - frozenset(df.columns)
    if need_cols:
        raise SystemExit(f"Parquet missing columns: {sorted(need_cols)}")

    frac_before = len(df)
    tb_touch = df.loc[df["tb_case"] != -1, "game_index"].nunique()

    df_t = apply_tb_prefix_truncation(df)
    print(
        json.dumps(
            {
                "turn_rows_before_tb_trunc": frac_before,
                "turn_rows_after_tb_trunc": len(df_t),
                "games_touching_tb": int(tb_touch),
            },
            indent=2,
        )
    )

    initiative = build_slice_initiative(df_t)

    sorted_df = df_t.sort_values(["game_index", "turn_idx"]).copy()
    post_play = build_slice_post_play(sorted_df)

    snapshot_check_post_play(post_play.head(50), tp)

    if args.diagnostics_only:
        slice_diagnostics("initiative", initiative)
        slice_diagnostics("post_play", post_play)
        dpath = out_dir / "slice_diagnostics.json"
        dpath.write_text(
            json.dumps(
                {
                    "initiative": {
                        "n_rows": len(initiative),
                        "n_distinct_games": int(initiative["game_index"].nunique()),
                        "turn_outcome_1": int((initiative["turn_outcome"] == 1).sum()),
                        "turn_outcome_0": int((initiative["turn_outcome"] == 0).sum()),
                    },
                    "post_play": {
                        "n_rows": len(post_play),
                        "n_distinct_games": int(post_play["game_index"].nunique()),
                        "turn_outcome_1": int((post_play["turn_outcome"] == 1).sum()),
                        "turn_outcome_0": int((post_play["turn_outcome"] == 0).sum()),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {dpath}")
        return

    if args.compare_only:
        run_feature_selection_comparison(
            initiative,
            post_play,
            out_dir,
            n_splits=min(args.cv_splits, 10),
            mutual_info_k=args.mi_top_k,
            rf_trees=args.rf_trees,
            mi_rows=args.mi_max_rows,
            random_state=args.selection_seed,
        )
        return

    train_one_slice(
        "initiative", initiative, out_dir, n_splits=min(args.cv_splits, 10)
    )
    train_one_slice("post_play", post_play, out_dir, n_splits=min(args.cv_splits, 10))

    if args.compare_selection:
        run_feature_selection_comparison(
            initiative,
            post_play,
            out_dir,
            n_splits=min(args.cv_splits, 10),
            mutual_info_k=args.mi_top_k,
            rf_trees=args.rf_trees,
            mi_rows=args.mi_max_rows,
            random_state=args.selection_seed,
        )


if __name__ == "__main__":
    main()
