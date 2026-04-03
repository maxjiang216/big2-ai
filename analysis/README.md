# Analysis

Python notebooks-style scripts for exploring Parquet exports from `bin/generate_data`. Run from a directory that contains the input files (usually the repo root, after generation), unless you pass explicit paths inside each script.

**Typical inputs:** `game_features.parquet` and/or `turn_features.parquet` (names can be overridden in code or via defaults in each script).

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`; `random_forest_tester.py` also needs `scikit-learn` and `joblib`.

## `analysis.py`

Loads game- and turn-level Parquet files, prints summaries, and plots distributions (length, outcomes, rank histograms, tablebase-related columns when present).

**Defaults:** `game_features.parquet`, `turn_features.parquet` in the current working directory.

**Outputs:** Figures displayed interactively; no fixed output path (see script for any saved files).

## `last_player_analysis.py`

Filters **last-player** rows from `turn_features.parquet` (opponent’s turn, excluding the synthetic pre-deal state) and analyzes / plots `turn_outcome` and related columns.

**Defaults:** reads `turn_features.parquet`; writes under `turn_analysis_results/` (configurable near the top of the file).

## `random_forest_tester.py`

Trains a **RandomForestRegressor** to predict `turn_outcome` on last-player rows, reports Brier score, and saves the model under `turn_tree_results/` (configurable).

**Defaults:** `turn_features.parquet` for training data.

## Tablebase columns

Game-level aggregates such as `tb_case1`, `tb_case2`, `tb_hits` live in **`game_features.parquet`**. Turn-only scripts do not see those columns unless you merge game features into the turn frame in Python.
