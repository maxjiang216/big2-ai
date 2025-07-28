#!/usr/bin/env python3
"""
Parquet Data Analysis Script

Loads and analyzes the game-level and turn-level features exported from
the GameCoordinator simulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def load_data(game_file="game_features.parquet", turn_file="turn_features.parquet"):
    """Load the parquet files and return DataFrames."""
    try:
        game_df = pd.read_parquet(game_file)
        print(f"✓ Loaded game features: {game_file}")
        print(f"  Shape: {game_df.shape}")
    except FileNotFoundError:
        print(f"✗ Game features file not found: {game_file}")
        game_df = None
    
    try:
        turn_df = pd.read_parquet(turn_file)
        print(f"✓ Loaded turn features: {turn_file}")
        print(f"  Shape: {turn_df.shape}")
    except FileNotFoundError:
        print(f"✗ Turn features file not found: {turn_file}")
        turn_df = None
    
    return game_df, turn_df

def analyze_game_features(df):
    """Analyze game-level features."""
    if df is None:
        return
    
    print("\n" + "="*60)
    print("GAME-LEVEL FEATURE ANALYSIS")
    print("="*60)
    
    print(f"\nDataset Info:")
    print(f"  Number of games: {len(df)}")
    print(f"  Features: {list(df.columns)}")
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # If there's an outcome column, analyze win rates
    if 'outcome' in df.columns or any('outcome' in col.lower() for col in df.columns):
        outcome_col = next((col for col in df.columns if 'outcome' in col.lower()), None)
        if outcome_col:
            print(f"\nOutcome Distribution:")
            outcome_counts = df[outcome_col].value_counts().sort_index()
            print(outcome_counts)
            
            win_rates = df[outcome_col].value_counts(normalize=True).sort_index()
            print(f"\nWin Rates:")
            for outcome, rate in win_rates.items():
                print(f"  Player {outcome}: {rate:.3f} ({rate*100:.1f}%)")
    
    # Correlation matrix if multiple columns
    if len(df.columns) > 1:
        print(f"\nCorrelation Matrix:")
        corr_matrix = df.corr()
        print(corr_matrix)
    
    return df

def analyze_turn_features(df):
    """Analyze turn-level features."""
    if df is None:
        return
    
    print("\n" + "="*60)
    print("TURN-LEVEL FEATURE ANALYSIS")
    print("="*60)
    
    print(f"\nDataset Info:")
    print(f"  Number of turns: {len(df)}")
    print(f"  Features: {list(df.columns)}")
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # Hand size analysis if present
    if 'player_hand_size' in df.columns or any('hand' in col.lower() for col in df.columns):
        hand_col = next((col for col in df.columns if 'hand' in col.lower()), None)
        if hand_col:
            print(f"\nHand Size Distribution:")
            hand_dist = df[hand_col].value_counts().sort_index()
            print(hand_dist)
            
            print(f"\nHand Size Statistics:")
            print(f"  Mean: {df[hand_col].mean():.2f}")
            print(f"  Median: {df[hand_col].median():.2f}")
            print(f"  Min: {df[hand_col].min()}")
            print(f"  Max: {df[hand_col].max()}")
            print(f"  Std Dev: {df[hand_col].std():.2f}")
    
    return df

def create_visualizations(game_df, turn_df):
    """Create visualizations of the data."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Game Simulation Analysis', fontsize=16, fontweight='bold')
    
    # Game-level visualizations
    if game_df is not None:
        # Outcome distribution
        outcome_col = next((col for col in game_df.columns if 'outcome' in col.lower()), None)
        if outcome_col:
            ax = axes[0, 0]
            outcome_counts = game_df[outcome_col].value_counts().sort_index()
            bars = ax.bar(outcome_counts.index, outcome_counts.values, 
                         color=['skyblue', 'lightcoral'], alpha=0.7)
            ax.set_title('Game Outcomes Distribution')
            ax.set_xlabel('Winner (Player)')
            ax.set_ylabel('Number of Games')
            
            # Add percentage labels on bars
            total = len(game_df)
            for bar, count in zip(bars, outcome_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                       f'{count}\n({count/total*100:.1f}%)', 
                       ha='center', va='bottom')
        else:
            axes[0, 0].text(0.5, 0.5, 'No outcome data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Game Outcomes')
    
    # Turn-level visualizations
    if turn_df is not None:
        # Hand size distribution
        hand_col = next((col for col in turn_df.columns if 'hand' in col.lower()), None)
        if hand_col:
            ax = axes[0, 1]
            ax.hist(turn_df[hand_col], bins=range(int(turn_df[hand_col].min()), 
                                                 int(turn_df[hand_col].max()) + 2),
                   alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('Hand Size Distribution')
            ax.set_xlabel('Hand Size')
            ax.set_ylabel('Frequency')
            
            # Hand size over time (if we assume sequential turns)
            ax = axes[1, 0]
            sample_size = min(1000, len(turn_df))  # Sample for readability
            sample_indices = np.linspace(0, len(turn_df)-1, sample_size, dtype=int)
            ax.plot(sample_indices, turn_df[hand_col].iloc[sample_indices], 
                   alpha=0.6, linewidth=0.5)
            ax.set_title(f'Hand Size Over Time (Sample of {sample_size} turns)')
            ax.set_xlabel('Turn Index')
            ax.set_ylabel('Hand Size')
        else:
            axes[0, 1].text(0.5, 0.5, 'No hand size data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Hand Size Distribution')
            
            axes[1, 0].text(0.5, 0.5, 'No hand size data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Hand Size Over Time')
    
    # Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Summary Statistics\n\n"
    if game_df is not None:
        summary_text += f"Games analyzed: {len(game_df):,}\n"
        if outcome_col:
            win_rates = game_df[outcome_col].value_counts(normalize=True).sort_index()
            for player, rate in win_rates.items():
                summary_text += f"Player {player} win rate: {rate:.3f}\n"
        summary_text += "\n"
    
    if turn_df is not None:
        summary_text += f"Turns analyzed: {len(turn_df):,}\n"
        if hand_col:
            summary_text += f"Avg hand size: {turn_df[hand_col].mean():.2f}\n"
            summary_text += f"Hand size range: {turn_df[hand_col].min()}-{turn_df[hand_col].max()}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('game_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'game_analysis.png'")
    plt.show()

def main():
    """Main analysis function."""
    print("Big 2 Game Data Analysis")
    print("=" * 40)
    
    # Check if files exist
    game_file = "game_features.parquet"
    turn_file = "turn_features.parquet"
    
    if not Path(game_file).exists() and not Path(turn_file).exists():
        print(f"Error: Neither {game_file} nor {turn_file} found.")
        print("Make sure to run the C++ simulation first to generate the data files.")
        sys.exit(1)
    
    # Load data
    game_df, turn_df = load_data(game_file, turn_file)
    
    # Analyze features
    game_df = analyze_game_features(game_df)
    turn_df = analyze_turn_features(turn_df)
    
    # Create visualizations
    if game_df is not None or turn_df is not None:
        create_visualizations(game_df, turn_df)
    else:
        print("\nNo data available for visualization.")
    
    # Export summary to CSV for further analysis
    if game_df is not None:
        game_df.to_csv('game_features_summary.csv', index=False)
        print(f"✓ Game features exported to 'game_features_summary.csv'")
    
    if turn_df is not None:
        turn_df.to_csv('turn_features_summary.csv', index=False)
        print(f"✓ Turn features exported to 'turn_features_summary.csv'")

if __name__ == "__main__":
    main()