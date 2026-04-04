<p align="center">
  <img src="assets/big2-ai-cards.png" alt="Big 2 AI Project Banner" width="400"/>
</p>

# Big 2 AI

A C++ implementation of the Big 2 card game (Shanghainese variant) with AI players and data collection for machine learning research.

## Overview

- Full rule implementation (2-player, 49-card deck)
- Multi-threaded self-play and Parquet export
- Pluggable feature extraction (game-level and turn-level)
- Tablebase support for opponent-has-one-card endgames
- Unit tests (`make test_core`)

## Game Rules

Big 2 is a shedding-type card game. This repo follows the **Shanghainese variant**:

- **Modified deck**: 49 cards (standard 52 minus three 2s and one ace)
- **2-player format**: 16 cards each, 17 unused
- **Card ranking**: 3 (lowest) through K, A, 2 (highest)
- **Combinations**: Singles, doubles, triples, full houses, straights, sister straights, triple straights, bombs
- **Special rules**: Triple aces count as a bomb; bombs can burn other combinations

## Project layout

```
.
├── src/              # Engine: core, simulation, players, features, datagen (see src/README.md)
├── test/             # Unit tests and perf benchmark source (see test/README.md)
├── scripts/          # Drivers (e.g. data generation), JSON configs, tablebase precompute source (see scripts/README.md)
├── analysis/         # Analyze self-play output, train models, experiments (see analysis/README.md)
├── research/         # Exploratory C++/Python; binaries under build/research/ via make research (see research/README.md)
├── build/            # Compiled objects and research binaries (gitignored)
├── bin/              # Main binaries (gitignored)
├── assets/           # Images for docs
├── Makefile
└── README.md
```

## Architecture

| Component | Role |
|-----------|------|
| `Game` | Full game state (hands, discards, legal moves) |
| `PartialGame` | Imperfect-information view for a player |
| `Move` | Encoded moves (472 legal move ids) |
| `Player` | Abstract policy; `RandomPlayer`, `GreedyPlayer` |
| `GameSimulator` | one full game |
| `GameCoordinator` | many games, threaded, writes Parquet |

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

With `game_features.parquet` / `turn_features.parquet` in the current directory:

```bash
python3 analysis/analysis.py
```

## See also

- `DEVLOG.md`, `ROADMAP.md`
- Per-directory docs: `research/README.md`, `scripts/README.md`, `src/README.md`, `test/README.md`, `analysis/README.md`
