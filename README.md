<p align="center">
  <img src="assets/big2-ai-cards.png" alt="Big 2 AI Project Banner" width="400"/>
</p>

# Big 2 AI

A C++ implementation of the Big 2 card game (Shanghainese variant) with AI players and data collection for machine learning research.

## Overview

- Full rule implementation for the 2-player variant ([RULES.md](RULES.md); the C++ engine in `src/core/` is the source of truth for this repo)
- Built-in policies: **random**, **greedy**, and **greedy** variants (`greedy_random`, `greedy_random_pass`, `greedy_no_bomb`) with tunable parameters
- Multi-threaded self-play and Parquet export (`bin/generate_data`)
- Head-to-head **evaluation** with paired deals and Wilson confidence intervals (`bin/eval_match`), plus **round-robin** tooling for many players
- Pluggable feature extraction (game-level and turn-level)
- Tablebase support for opponent-has-one-card endgames
- Unit tests (`make test_core`)

## Rules

Human-readable rules for the Shanghainese-style 2-player variant are in **[RULES.md](RULES.md)** (combinations, bombs, series scoring, etc.). If anything disagrees with the implementation, treat **`src/core/`** as authoritative for this codebase.

## Project layout

```
.
├── src/              # Engine: core, simulation, players, features, datagen (see src/README.md)
├── test/             # Unit tests and perf benchmark source (see test/README.md)
├── scripts/          # Data generation, eval_match / round-robin drivers, JSON configs, tablebase precompute (see scripts/README.md)
├── analysis/         # Analyze self-play output, train models, experiments (see analysis/README.md)
├── research/         # Exploratory C++/Python; binaries under build/research/ via make research (see research/README.md)
├── build/            # Compiled objects and research binaries (gitignored)
├── bin/              # Main binaries (gitignored): generate_data, eval_match, test_core, …
├── assets/           # Images for docs
├── Makefile
├── README.md
└── RULES.md          # Full game rules (variant description)
```

## Architecture

| Component | Role |
|-----------|------|
| `Game` | Full game state (hands, discards, legal moves) |
| `PartialGame` | Imperfect-information view for a player |
| `Move` | Encoded moves (472 legal move ids) |
| `Player` | Abstract policy; `RandomPlayer`, `GreedyPlayer`, greedy variants (see `src/players/`) |
| `player_factory_registry.h` | String → factory for datagen / eval CLIs |
| `GameSimulator` | one full game |
| `GameCoordinator` | many games, threaded, writes Parquet |
| `eval_match` | Pairwise matchups with swapped seats per deal + Wilson CI (`src/datagen/eval_match.cpp`) |

## Prerequisites

- C++17 (GCC or Clang)
- OpenMP (`-fopenmp` in Makefile)
- Optional: Apache Arrow / Parquet for `make generate_data`

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential libarrow-dev libparquet-dev
```

## Build

From the **repository root** (directory containing `Makefile`):

```bash
make help              # list targets
make test_core         # unit tests (no Arrow)
make benchmark         # perf binary (no Arrow)
make eval_match        # head-to-head evaluation binary (no Arrow)
make generate_data     # self-play + Parquet (needs libarrow)
make tablebase_opp1_gen  # build tablebase precompute tool
make research          # standalone research/*.cpp -> build/research/
```

## Data generation

`bin/generate_data` is the single self-play binary: stats on stdout, optional Parquet export, optional `--samples-md` for anomaly game write-ups.

```bash
# Example: greedy self-play with full Parquet export
python3 scripts/generate_data.py scripts/configs/greedy.json --compile

# Or: generate Parquet then the analysis report (see scripts/configs/*.json)
./scripts/run.sh scripts/configs/test.json
```

Writes `<output_path>_game.parquet` and `<output_path>_turn.parquet` when `output_path` and feature lists are set in the config.

Self-play `--player` uses the same policy **names** as evaluation (`random`, `greedy`, `greedy_random`, `greedy_random_pass`, `greedy_no_bomb`, …). The `bin/generate_data` CLI currently instantiates variants with **parameter 0**; use `eval_match` or the round-robin scripts to compare non-zero parameters (see `scripts/README.md`).

## Evaluation (two policies)

`bin/eval_match` runs **paired** games: for each deal, both seats are played so card luck averages out. It prints win counts, a **95% Wilson** interval, and a significance line.

```bash
make eval_match
./bin/eval_match --p0 greedy --p1 random --deals 10000 --seed 42
```

**Convenience:** `scripts/run_eval_match.sh` loads JSON, saves `analysis/eval_match_last.log`, and runs `scripts/analyze_eval_match.py` on the output.

**Round robin:** `scripts/run_round_robin.sh` (or `scripts/round_robin_eval.py`) runs `eval_match` for every unordered pair of players from a JSON list—useful for comparing baselines (`random`, `greedy`) against many variant settings. See `scripts/configs/round_robin_example.json` and `scripts/README.md`.

## Tablebase

Precompute the opponent-1-card tablebase binary (used at runtime if loaded):

```bash
make tablebase_opp1_gen
./bin/tablebase_opp1_gen [out.bin] [samples.txt]
```

Point the engine at the file via your player/load path (see `src/core/tablebase_opp1.*`).

## Tests

```bash
make test_core
./bin/test_core
```

## Python analysis

Dependencies are declared in **`pyproject.toml`** and locked with **[uv](https://docs.astral.sh/uv/)**. From the repo root:

```bash
uv sync                    # create .venv and install numpy, pandas, pyarrow, scikit-learn, …
uv run python analysis/train_linear_rollout.py data/pimc20_selfplay_turn.parquet --out data/linear_rollout_w.txt --top-k 35
# or: source .venv/bin/activate && python analysis/train_linear_rollout.py …
```

Optional extras (e.g. Seaborn for some notebooks/scripts): `uv sync --extra analysis`.

**Parquet reports** (after `generate_data`): with `game_features.parquet` / `turn_features.parquet` in the current directory, or pass paths explicitly:

```bash
uv run python analysis/analysis.py
```

**Eval logs:** summarize a saved `eval_match` transcript:

```bash
python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
```

## See also

- **[RULES.md](RULES.md)** — full rules for the card variant
- Per-directory docs: [`scripts/README.md`](scripts/README.md), [`src/README.md`](src/README.md), [`analysis/README.md`](analysis/README.md), [`test/README.md`](test/README.md), [`research/README.md`](research/README.md)
