#!/usr/bin/env python3
"""
Hyper-parameter search for XGBoost regression on "last-player" positions.
Reports Brier score; prints progress with ETA.
"""

import os, time, joblib, itertools, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import xgboost as xgb

TURN_FILE      = "turn_features.parquet"
OUTPUT_DIR     = "xgb_tuned_results"
TARGET_COL     = "turn_outcome"
NEXT_PLAYER_COL= "next_player"

#─────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def mask_last_player(df: pd.DataFrame) -> pd.DataFrame:
    m = (df[NEXT_PLAYER_COL] == 0) & ~(
        (df["player_hand_size"] == 16) & (df["opponent_hand_size"] == 16)
    )
    return df.loc[m]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
#─────────────────────────────────────────────────────────────────────────
def main():
    ensure_dir(OUTPUT_DIR)

    df = load_data(TURN_FILE)
    print(f"Loaded {len(df):,} turn rows.")
    df = mask_last_player(df)
    print(f"After filtering last-player positions: {len(df):,} rows.")

    #────────────── feature matrix ────────────────────────────────────
    skip = {TARGET_COL, NEXT_PLAYER_COL}
    num_like = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
    feature_cols = [c for c in df.columns if c not in skip and df[c].dtype in num_like]
    X = df[feature_cols].values
    y = df[TARGET_COL].values.astype(np.float32)

    #────────────── split: train / valid / test ───────────────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=42
    )  # 0.8*0.125 ≈ 0.10 overall

    #────────────── hyper-parameter grid ──────────────────────────────
    grid = {
        "max_depth":       [6, 8, 10],
        "learning_rate":   [0.05, 0.10],
        "n_estimators":    [400, 600],
        "min_child_weight":[100, 300],
    }
    combos = list(itertools.product(*grid.values()))
    total   = len(combos)
    print(f"Searching {total} hyper-parameter combos …\n")
    best = {"val_brier": math.inf}

    start_all = time.time()
    for idx, (md, lr, n_est, mcw) in enumerate(combos, 1):
        t0 = time.time()
        print(f"[{idx:2d}/{total}] depth={md:2d}, lr={lr:.2f}, "
              f"n={n_est}, mcw={mcw} … ", end="", flush=True)

        model = xgb.XGBRegressor(
            objective          ="reg:squarederror",
            tree_method        ="hist",
            n_jobs             =-1,
            max_depth          =md,
            learning_rate      =lr,
            n_estimators       =n_est,
            min_child_weight   =mcw,
            subsample          =0.8,
            colsample_bytree   =0.8,
            random_state       =42
        )

        # Try different early stopping approaches based on XGBoost version
        try:
            # Try newer callback-based approach first
            early_stopping = xgb.callback.EarlyStopping(
                rounds=40,
                metric_name='rmse',
                data_name='validation_0',
                save_best=True
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                callbacks=[early_stopping]
            )
        except (TypeError, AttributeError):
            # Fall back to older parameter-based approach
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False,
                    early_stopping_rounds=40
                )
            except TypeError:
                # If both fail, just fit without early stopping
                model.fit(
                    X_train, y_train,
                    verbose=False
                )

        y_val_pred = np.clip(model.predict(X_valid), 0, 1)
        val_brier  = brier_score_loss(y_valid, y_val_pred)
        dur        = time.time() - t0
        eta        = (time.time()-start_all)/idx * (total-idx)
        print(f"val Brier={val_brier:.4f}  (took {dur:.1f}s, ETA {eta/60:.1f} min)")

        if val_brier < best["val_brier"]:
            best.update(dict(
                val_brier=val_brier,
                params   = dict(max_depth=md, learning_rate=lr,
                                n_estimators=n_est, min_child_weight=mcw),
                model    = model
            ))

    #────────────── final test evaluation ─────────────────────────────
    print("\nBest params:", best["params"])
    y_test_pred = np.clip(best["model"].predict(X_test), 0, 1)
    test_brier  = brier_score_loss(y_test, y_test_pred)
    print(f"\n✅  Best validation Brier: {best['val_brier']:.4f}")
    print(f"🏁  Test Brier with best model: {test_brier:.4f}")

    #────────────── feature importances plot ──────────────────────────
    importances = best["model"].feature_importances_
    fi_sorted = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 15 feature importances:")
    for feat, val in fi_sorted[:15]:
        print(f"{feat:32s} {val:.4f}")

    plt.figure(figsize=(9,8))
    names, vals = zip(*fi_sorted[:20])
    plt.barh(names[::-1], vals[::-1])
    plt.title("Top-20 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_feature_importances.png"), dpi=120)
    plt.close()

    #────────────── save model ────────────────────────────────────────
    joblib.dump(best["model"],
                os.path.join(OUTPUT_DIR, "xgb_best_win_prob.joblib"))
    print("\nModel & plot saved to", OUTPUT_DIR)

#─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()