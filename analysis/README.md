# Analysis

Python scripts for exploring **Parquet** exports from `bin/generate_data`, plus helpers that summarize **eval** logs. Run from the repo root (or pass explicit paths), unless noted.

**Typical Parquet inputs:** `<prefix>_game.parquet` and `<prefix>_turn.parquet` produced by `scripts/generate_data.py` / `bin/generate_data`.

**Eval logs:** `scripts/analyze_eval_match.py` parses `bin/eval_match` output (or a file saved by `scripts/run_eval_match.py`, e.g. `analysis/eval_match_last.log`) and prints wins, Wilson CI, and verdict lines. See [`scripts/README.md`](../scripts/README.md).

**Dependencies:** managed with **[uv](https://docs.astral.sh/uv/)** — run `uv sync` at the repo root (see root `README.md`). Core stack: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `joblib`, `matplotlib`; optional `seaborn` via `uv sync --extra analysis`.

## `feature_interaction_explore.py`

Uses the same **last-player / no-TB** table as `train_linear_rollout.py`. Writes MI and Random Forest bar charts, **Pearson** correlation heatmap (top features by MI), **binned mean** `turn_outcome` vs feature (nonlinearity), **quadratic** probes `|corr(y, x_i^2)|`, and pairwise **product** correlations `|corr(y, z_i z_j)|` with `summary.txt`. Example:

```bash
uv run python analysis/feature_interaction_explore.py data/pimc20_selfplay_turn.parquet --out-dir analysis/feature_explore_out
```

## End-to-end: `scripts/run.sh`

From the repo root:

```bash
./scripts/run.sh                    # default: scripts/configs/greedy.json
./scripts/run.sh scripts/configs/test.json
```

Runs `scripts/generate_data.py` with the JSON config, then `analysis/analysis.py` on the resulting Parquet files. Set `samples_md_path` in the config (e.g. `analysis/sample_games.md`) so `bin/generate_data` writes anomaly histories; the report can reference that file via `--samples-md-hint`.

## `analysis.py`

Loads game- and turn-level Parquet, prints column summaries and anomaly one-liners, and saves a **2×3** multi-panel figure plus an anomaly text row (longest/shortest game, most/fewest opening legal moves). Optionally points readers to `sample_games.md` for full hand histories.

**CLI:** `--game-parquet`, `--turn-parquet`, `--output-dir`, `--report-name`, `--samples-md-hint`, `--title` (see `python3 analysis/analysis.py --help`).

**Defaults:** `game_features.parquet`, `turn_features.parquet` in the current working directory (for backward compatibility).

**Outputs:** `report.png` (or chosen name) and CSV exports under `--output-dir`.

## `last_player_analysis.py`

Filters **last-player** rows from turn Parquet (opponent’s turn, excluding the synthetic pre-deal state) and analyzes / plots `turn_outcome` and related columns.

**Defaults:** reads `turn_features.parquet`; writes under `turn_analysis_results/` (configurable near the top of the file).

## `random_forest_tester.py`

Trains a **RandomForestRegressor** to predict `turn_outcome` on last-player rows, reports Brier score, and saves the model under `turn_tree_results/` (configurable).

**Defaults:** `turn_features.parquet` for training data.

## Tablebase columns

Game-level aggregates such as `tb_case1`, `tb_case2`, `tb_hits` live in **`<prefix>_game.parquet`**. Turn-only scripts do not see those columns unless you merge game features into the turn frame in Python.
