import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

TURN_FILE = "turn_features.parquet"
OUTCOME_COL = "turn_outcome"
NEXT_PLAYER_COL = "next_player"  # 1 if it is *this* player's turn
OUTPUT_DIR = "turn_analysis_results"  # <--- Change folder name as needed


def load_data(turn_file):
    return pd.read_parquet(turn_file)


def get_last_player_rows(df):
    mask = (df[NEXT_PLAYER_COL] == 0) & ~(
        (df["player_hand_size"] == 16) & (df["opponent_hand_size"] == 16)
    )
    return df[mask]


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output directory: {path}")


def main():
    df = load_data(TURN_FILE)
    print(f"Loaded {len(df):,} turn-level rows.")
    print(df.columns)

    outcome_col = None
    for col in df.columns:
        if "outcome" in col.lower():
            outcome_col = col
            break
    if not outcome_col:
        raise RuntimeError("Could not find outcome column.")

    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)

    # Filter to "last player" positions (after move, not first turn)
    last_df = get_last_player_rows(df)
    print(f"Filtered to {len(last_df):,} rows for last-player positions.")

    # 2D Heatmap: Player vs Opponent hand size → Average win rate
    if (
        "player_hand_size" in last_df.columns
        and "opponent_hand_size" in last_df.columns
    ):
        print("\n=== 2D Win Rate Heatmap: player_hand_size vs opponent_hand_size ===")
        pivot = (
            last_df.groupby(["player_hand_size", "opponent_hand_size"])[outcome_col]
            .mean()
            .unstack(fill_value=np.nan)
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Average Win Rate"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Mean Win Rate by Player & Opponent Hand Size")
        plt.xlabel("Opponent Hand Size")
        plt.ylabel("Player Hand Size")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "hand_size_2d_mean_outcome.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
        plt.close()

    # 2D Heatmap: Normalized Count of (Player Hand Size, Opponent Hand Size)
    if (
        "player_hand_size" in last_df.columns
        and "opponent_hand_size" in last_df.columns
    ):
        count_pivot = (
            last_df.groupby(["player_hand_size", "opponent_hand_size"])[outcome_col]
            .count()
            .unstack(fill_value=0)
        )
        norm_count = count_pivot / count_pivot.values.max()  # Normalize to [0,1]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            norm_count,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar_kws={"label": "Normalized Cell Count"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Normalized Count by Player & Opponent Hand Size")
        plt.xlabel("Opponent Hand Size")
        plt.ylabel("Player Hand Size")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "hand_size_2d_normalized_count.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
        plt.close()

    # Features to analyze
    skip_cols = {outcome_col, NEXT_PLAYER_COL}
    feature_cols = [
        col
        for col in df.columns
        if col not in skip_cols
        and df[col].dtype in [np.int64, np.int32, np.float64, np.float32]
    ]

    for feat in feature_cols:
        print(f"\n=== {feat} ===")
        means = last_df.groupby(feat)[outcome_col].mean()
        counts = last_df[feat].value_counts().sort_index()
        print("Counts per value:")
        print(counts)

        # Barplot: mean outcome vs. feature value
        plt.figure(figsize=(6, 4))
        sns.barplot(x=means.index, y=means.values, palette="viridis")
        plt.title(f"Mean Outcome by {feat}")
        plt.xlabel(feat)
        plt.ylabel(f"Mean {outcome_col} (Win Rate)")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{feat}_mean_outcome.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
        plt.close()

        # Correlation
        if last_df[feat].nunique() > 1:
            corr = last_df[[feat, outcome_col]].corr().iloc[0, 1]
            print(f"Correlation coefficient with outcome: {corr:.3f}")
        else:
            print("Feature is constant; correlation undefined.")


if __name__ == "__main__":
    main()
