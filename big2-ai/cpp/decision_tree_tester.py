#!/usr/bin/env python3
"""
Train a regression tree that predicts win-probability for “last-player”
positions and print / plot the resulting tree, as well as feature importances.
"""
import os
import joblib  # for optional model persistence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import brier_score_loss

TURN_FILE = "turn_features.parquet"
OUTPUT_DIR = "turn_tree_results"
TARGET_COL = "turn_outcome"     # 0/1 win indicator
NEXT_PLAYER_COL = "next_player" # 1 if it is *this* player's turn
TOP_N_IMPORTANCES = 15          # Number of top features to show

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def get_last_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df[NEXT_PLAYER_COL] == 0) & ~(
        (df["player_hand_size"] == 16) & (df["opponent_hand_size"] == 16)
    )
    return df.loc[mask]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main():
    ensure_dir(OUTPUT_DIR)

    df = load_data(TURN_FILE)
    print(f"Loaded {len(df):,} turn rows.")
    df = get_last_player_rows(df)
    print(f"After filtering last-player positions: {len(df):,} rows.")

    # ----------------------------- feature matrix --------------------
    skip = {TARGET_COL, NEXT_PLAYER_COL}
    num_like = [np.int8, np.int16, np.int32, np.int64,
                np.float32, np.float64]
    feature_cols = [c for c in df.columns
                    if c not in skip and df[c].dtype in num_like]

    X = df[feature_cols].values
    y = df[TARGET_COL].values.astype(np.float32)

    # ----------------------------- train / test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    # ----------------------------- fit tree --------------------------
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=15,
        min_samples_leaf=500,
        random_state=42
    ).fit(X_train, y_train)

    # ----------------------------- tree structure print -------------
    txt = export_text(tree, feature_names=feature_cols)
    print("\nDecision tree structure:\n")
    print(txt)

    # ----------------------------- evaluation ------------------------
    train_brier = brier_score_loss(y_train, tree.predict(X_train))
    test_brier  = brier_score_loss(y_test,  tree.predict(X_test))
    print(f"Brier score – train: {train_brier:.4f},  test: {test_brier:.4f}")

    # ----------------------------- feature importances --------------
    importances = tree.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop Feature Importances:")
    for i in range(TOP_N_IMPORTANCES):
        idx = sorted_idx[i]
        print(f"{i+1:2d}. {feature_cols[idx]:30s} {importances[idx]:.4f}")

    # Plot feature importances (bar plot)
    plt.figure(figsize=(10, 7))
    top_idx = sorted_idx[:TOP_N_IMPORTANCES]
    plt.barh(
        [feature_cols[i] for i in top_idx][::-1],   # reverse for descending
        importances[top_idx][::-1]
    )
    plt.xlabel("Feature Importance")
    plt.title(f"Top {TOP_N_IMPORTANCES} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"), dpi=150)
    plt.close()
    print(f"Feature importances plot saved to {OUTPUT_DIR}/feature_importances.png")

    # ----------------------------- plot tree ------------------------
    plt.figure(figsize=(22, 10))
    plot_tree(
        tree,
        feature_names=feature_cols,
        filled=True,
        impurity=False,
        rounded=True,
        max_depth=3
    )
    plt.title("Top of decision tree predicting win-probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tree_top_levels.png"), dpi=150)
    plt.close()
    print(f"Tree plot saved to {OUTPUT_DIR}/tree_top_levels.png")

    # ----------------------------- optional save model ---------------
    joblib.dump(tree, os.path.join(OUTPUT_DIR, "win_prob_tree.joblib"))
    print("Model saved for later C++ port / inspection.")

if __name__ == "__main__":
    main()
