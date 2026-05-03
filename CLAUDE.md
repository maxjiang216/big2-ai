# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & test

All commands run from the **repository root** (where `Makefile` lives).

```bash
make test_core                    # compile unit tests (no Arrow dependency)
./bin/test_core                   # run all suites → "All tests passed."

make benchmark && ./bin/benchmark # perf timing (not part of CI)

make eval_match
./bin/eval_match --p0 greedy --p1 random --deals 10000 --seed 42

make generate_data        # requires libarrow-dev
python3 scripts/generate_data.py scripts/configs/greedy.json --compile
```

Prerequisites: `sudo apt install build-essential libarrow-dev libparquet-dev` (Arrow only needed for `generate_data`).

Python analysis uses **uv**:
```bash
uv sync          # creates .venv from pyproject.toml
uv run python analysis/analysis.py
```

CI (`.github/workflows/ci.yml`) runs `make test_core && ./bin/test_core`.

## Architecture

Big 2 card game engine (2-player Shanghainese variant) with AI players and ML data pipelines. Human-readable rules are in RULES.md; `src/core/` is the authoritative implementation.

**Data flow:**
```
Game + PartialGame  (src/core/)
  └── GameSimulator (src/simulation/)  ← one full game, two Player instances
        └── GameCoordinator            ← many games, threaded, Parquet export
              └── generate_data CLI    (src/datagen/generate_data.cpp)
```

**Key types:**
- `Game` — full game state (both hands, discard pile, legal moves)
- `PartialGame` — imperfect-information view for a single player; each `Player` instance keeps its own and updates it each turn
- `Move` — 472 encoded legal move IDs (singles, pairs, triples, straights, full houses, four-of-a-kind, bombs, pass)
- `Player` (abstract in `src/players/player.h`) — `select_move(PartialGame&)` → `Move`; tablebase injected by simulator
- `player_factory_registry.h` — `make_player_factory(name, param, seed)` → factory; used by all CLIs and scripts

**Policy names** (used in `--p0`/`--p1` flags and JSON configs):
`random`, `greedy`, `greedy_random`, `greedy_random_pass`, `greedy_no_bomb`

Greedy heuristic: among non-pass moves, select the state with the best lexicographic score — (bombs held, cards remaining, rank counts 2-down-to-4).

**eval_match** pairs each deal: same shuffle played both ways (seats swapped) so card luck cancels. Reports win counts, a Wilson 95% CI, and a significance line.

**Features** (`src/features/`) are header-only extractors registered by name and wired into the Parquet schema via `src/datagen/feature_registry.h`. Two levels: `game_level/` (per-game scalars) and `turn_level/` (per-turn hand/move features).

**Tablebase** (`src/core/tablebase_opp1.*`) — optional runtime lookup for opponent-has-one-card endgames; precomputed via `make tablebase_opp1_gen && ./bin/tablebase_opp1_gen`.
