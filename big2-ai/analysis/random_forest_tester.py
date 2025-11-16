#!/usr/bin/env python3
"""
Train a Random-Forest regressor that predicts win-probability for
“last-player” positions and report the Brier score.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss

TURN_FILE      = "turn_features.parquet"
OUTPUT_DIR     = "turn_tree_results"           # reuse the same folder
TARGET_COL     = "turn_outcome"                # 0 / 1
NEXT_PLAYER_COL = "next_player"                # 1 if it is *this* player's turn

# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def get_last_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where it's the *opponent's* turn (a.k.a. last-player state)."""
    mask = (df[NEXT_PLAYER_COL] == 0) & ~(
        (df["player_hand_size"] == 16) & (df["opponent_hand_size"] == 16)
    )
    return df.loc[mask]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    # 1. Load + filter
    df = load_data(TURN_FILE)
    print(f"Loaded {len(df):,} turn rows.")
    df = get_last_player_rows(df)
    print(f"After filtering last-player positions: {len(df):,} rows.")

    # 2. Feature matrix
    skip_cols   = {TARGET_COL, NEXT_PLAYER_COL}
    numeric_dtypes = [np.int16, np.int32, np.int64, np.float32, np.float64]
    feat_cols = [c for c in df.columns
                 if c not in skip_cols and df[c].dtype in numeric_dtypes]

    X = df[feat_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    print("Begin fitting process...")

    # 4. Fit Random-Forest
    rf = RandomForestRegressor(
        n_estimators     = 100,      # plenty of trees for stability
        max_depth        = 5,       # deep enough to capture interactions
        min_samples_leaf = 1000,      # keep leaves “big” -> smoother probs
        bootstrap        = True,
        oob_score        = True,     # quick OOB sanity-check
        n_jobs           = -1,       # use all CPU cores
        random_state     = 42,
        verbose = 2,
    )
    rf.fit(X_train, y_train)
    print(f"OOB Brier (quick check): {brier_score_loss(y_train, rf.oob_prediction_):.4f}")

    # 5. Evaluate
    train_brier = brier_score_loss(y_train, rf.predict(X_train))
    test_brier  = brier_score_loss(y_test,  rf.predict(X_test))
    print(f"Brier score – train: {train_brier:.4f} | test: {test_brier:.4f}")

    # 6. Save for later port / inspection
    model_path = os.path.join(OUTPUT_DIR, "win_prob_random_forest.joblib")
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
