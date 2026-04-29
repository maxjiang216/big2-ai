<p align="center">
  <img src="assets/big2-ai-cards.png" alt="Big 2 AI Project Banner" width="400"/>
</p>

# Big 2 AI

A **Rust** implementation of the Big 2 card game (Shanghainese variant) with AI players and data collection for machine learning research.

## Overview

- Full rule implementation for the 2-player variant ([RULES.md](RULES.md); **`crates/core`** is the authoritative implementation)
- Built-in policies: **random**, **greedy**, greedy variants (`greedy_random`, `greedy_random_pass`, `greedy_no_bomb`, …), **PIMC** rollouts, and learned evaluators (`greedy_linear`, `tree_greedy`, …) — registered in `crates/players/src/registry.rs`
- Multi-threaded self-play and Parquet export (`target/release/generate_data`)
- Head-to-head **evaluation** with paired deals and Wilson confidence intervals (`target/release/eval_match`), plus **round-robin** scripts for many players
- Pluggable feature extraction (game-level and turn-level) in `crates/features`
- Tablebase for opponent-has-one-card endgames (loader in `crates/core`; data file `data/tablebase_opp1.bin`)
- Tests: `make test` runs `cargo test --all` (see [test/README.md](test/README.md))

## Rules

Human-readable rules for the Shanghainese-style 2-player variant are in **[RULES.md](RULES.md)** (combinations, bombs, series scoring, etc.). If anything disagrees with the implementation, treat **`crates/core`** as authoritative.

## Project layout

```
.
├── crates/           # Rust workspace: core, players, simulation, features, datagen
├── test/             # Legacy C++ unit-test sources (not built by Makefile; see test/README.md)
├── scripts/          # Python drivers, JSON configs, tablebase helper source (see scripts/README.md)
├── analysis/         # Training and analysis scripts (see analysis/README.md)
├── research/         # Exploratory code (see research/README.md)
├── data/             # Datasets, model weights, tablebase binary (local; large files may be gitignored)
├── projects/         # Vendored standard-linter (git submodule after make standards_init)
├── bin/              # make tablebase_opp1_gen → bin/tablebase_opp1_gen (gitignored)
├── target/           # Rust build output (gitignored)
├── assets/           # Images for docs
├── Cargo.toml
├── Makefile
├── README.md
└── RULES.md
```

Day-to-day tooling notes for agents and contributors: **[CLAUDE.md](CLAUDE.md)**.

## Architecture

| Component | Role |
|-----------|------|
| `Game` | Full game state (hands, discards, legal moves, turn order) |
| `PartialGame` | Imperfect-information view for a player |
| `Move` | Encoded moves: id `0` = pass, ids `1..=468` non-pass (`Move::encode` / `decode`) |
| `Strategy` / `Player` | Policies in `crates/players` (`make_player` in `registry.rs`) |
| `simulator::play_game` | One full game between two players |
| `coordinator` | Parallel self-play and paired-deal evaluation |
| `generate_data` / `eval_match` | CLI binaries in `crates/datagen` |

## Prerequisites

- **Rust** (stable), **Cargo**
- **Python 3** + **[uv](https://docs.astral.sh/uv/)** for analysis scripts and pinned tool versions
- **C++17 compiler** only if you run `make tablebase_opp1_gen` (standalone precompute tool)
- Parquet export uses the **Rust `parquet` crate** (bundled); system `libarrow` / `libparquet` are not required for the Rust build

## Build

From the **repository root**:

```bash
make help              # list Makefile targets
make build             # cargo build (debug)
make release           # cargo build --release → target/release/eval_match, generate_data
make test              # cargo test --all
make tablebase_opp1_gen  # bin/tablebase_opp1_gen (C++ helper for data/tablebase_opp1.bin)
```

## Data generation

`target/release/generate_data` runs self-play: stats on stdout, optional Parquet export, optional `--samples-md` for anomaly write-ups.

```bash
make release
python3 scripts/generate_data.py scripts/configs/greedy.json --compile

# Or: generate Parquet then the analysis report (see scripts/configs/*.json)
./scripts/run.sh scripts/configs/test.json
```

Writes `<output_path>_game.parquet` and `<output_path>_turn.parquet` when `output_path` and feature lists are set in the config.

Policy **names** match evaluation (`random`, `greedy`, `pimc`, …). For parameter sweeps on variants, use `eval_match` or the round-robin scripts (see `scripts/README.md`).

## Evaluation (two policies)

`target/release/eval_match` runs **paired** games: for each deal, both seats are played so card luck averages out. It prints win counts, a **95% Wilson** interval, and a significance line.

```bash
make release
./target/release/eval_match --p0 greedy --p1 random --deals 10000 --seed 42
```

**Convenience:** `scripts/run_eval_match.py` loads JSON, saves `analysis/eval_match_last.log`, and runs `scripts/analyze_eval_match.py` on the output.

**Round robin:** `scripts/round_robin_eval.py` runs `eval_match` for every unordered pair of players from a JSON list. See `scripts/configs/round_robin_example.json` and `scripts/README.md`.

## Tablebase

Precompute the opponent-1-card tablebase binary (read at runtime from `data/tablebase_opp1.bin`):

```bash
make tablebase_opp1_gen
./bin/tablebase_opp1_gen [out.bin] [samples.txt]
```

## Tests

```bash
make test
```

Same command runs in CI (`.github/workflows/ci.yml`). Details: [test/README.md](test/README.md).

## Python analysis

Dependencies are declared in **`pyproject.toml`** and installed with **[uv](https://docs.astral.sh/uv/)**:

```bash
uv sync
uv run python analysis/train_linear_rollout.py data/pimc20_selfplay_turn.parquet --out data/linear_rollout_w.txt --top-k 35
```

Optional extras (e.g. Seaborn): `uv sync --extra analysis`.

**Parquet reports** (after `generate_data`): with Parquet outputs in the current directory, or pass paths explicitly:

```bash
uv run python analysis/analysis.py
```

**Eval logs:** summarize a saved `eval_match` transcript:

```bash
python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
```

## See also

- **[RULES.md](RULES.md)** — full rules for the card variant
- **[CLAUDE.md](CLAUDE.md)** — build, evaluation, and crate-level reference
- Per-directory docs: [`scripts/README.md`](scripts/README.md), [`analysis/README.md`](analysis/README.md), [`test/README.md`](test/README.md), [`research/README.md`](research/README.md)
