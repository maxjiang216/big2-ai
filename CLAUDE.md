# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build commands

All commands run from the repository root:

```bash
make test_core         # build + run unit tests (no Arrow required)
./bin/test_core        # re-run tests without rebuilding

make eval_match        # head-to-head evaluation binary (no Arrow)
make generate_data     # self-play + Parquet export (requires libarrow-dev, libparquet-dev)
make benchmark         # perf binary (no Arrow)
make tablebase_opp1_gen
make research          # standalone research/*.cpp → build/research/
make clean
```

**Git submodule:** after a fresh clone, run `git submodule update --init projects/standard-linter` (or `make standards_init`). The vendored [standard-linter](https://github.com/maxjiang216/standard-linter) repo holds shared formatter/linter configs. Run `make standards_configs` to copy them into `.code-standards/` for local tooling or `pre-commit`. Optional GitHub reuse: `uses: maxjiang216/standard-linter/.github/workflows/standards.yml@<ref>` (pin a tag when publishing).

Ubuntu/Debian dependencies: `sudo apt install build-essential libarrow-dev libparquet-dev`

## Running evaluations

```bash
# Quick head-to-head
./bin/eval_match --p0 greedy --p1 random --deals 10000 --seed 42

# Via JSON config (saves log + runs analysis)
./scripts/run_eval_match.sh scripts/configs/eval_match.json

# Round-robin across many players
python3 scripts/round_robin_eval.py scripts/configs/round_robin_example.json --compile
```

## Python analysis

```bash
uv sync                                   # install deps (numpy, pandas, pyarrow, scikit-learn, …)
uv sync --extra analysis                  # + seaborn

uv run python analysis/train_linear_rollout.py data/pimc20_selfplay_turn.parquet \
    --out data/linear_rollout_w.txt --top-k 35
uv run python analysis/analysis.py        # Parquet report after generate_data
python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
```

## Architecture

### Core engine (`src/core/`)

- **`Game`** — full game state: both hands, discard pile, legal moves, turn order. Source of truth for rules.
- **`PartialGame`** — imperfect-information view for one player (built from `Game` + seat number; opponent hand hidden).
- **`Move`** — encoded move with 472 legal integer IDs. Fields: `combination` (enum), `rank`, `auxiliary`.
- **`GameRecord`** — turn-by-turn replay / training record.
- **`tablebase_opp1.*`** — runtime lookup for precomputed opponent-has-one-card endgame table.

### Simulation (`src/simulation/`)

- **`GameSimulator`** — runs one full game between two `Player` instances, returns a `GameRecord`.
- **`GameCoordinator`** — many games, OpenMP threading, hooks into Parquet export.

### Player abstraction (`src/players/`)

All players inherit from `Player` (non-virtual `select_move()` checks the tablebase first, then calls `select_move_impl()`). Players never access the full `Game`; they operate only on their `PartialGame game_` member which is kept in sync by the base class.

**Player families:**
- `random/` — uniform random legal move.
- `greedy/` — lexicographic post-move hand eval (`GreedyEval`): win-now > bomb count > fewest cards > high ranks. Variants: `greedy_random`, `greedy_random_pass`, `greedy_no_bomb` (each takes a float `param`), `greedy_linear` (Ridge evaluator), `greedy_pass` (Ridge pass/greedy decision), `tree_greedy` (decision-tree evaluator, loads `data/tree_model_d{depth}.txt`).
- `pimc/` — Perfect Information Monte Carlo: sample N opponent hands, run full rollouts for each candidate move, pick the move with most wins. Template `pimc_select_move_impl` is the shared core; rollout policy (greedy/tree/linear/pass-aware) and resampling mode (`_redet`) are injected as template params.

**Player name registry:** `src/players/player_factory_registry.h` — `make_player_factory(name, param, seed)` maps string names to factories. Add new player types here to make them available to all CLI tools.

### Features (`src/features/`)

Header-only extractors registered by name. Two scopes:
- `game_level/` — per-game scalars (outcome, length, tablebase hits, …).
- `turn_level/` — per-turn features (hand shape, last move type, bomb count, possible moves, …).

Feature lists in JSON configs control which columns are written to Parquet.

### Data generation

`bin/generate_data` — threaded self-play; outputs `<output_path>_game.parquet` and `<output_path>_turn.parquet`. Configured via JSON (see `scripts/configs/greedy.json` for schema). `param` is fixed at 0; use `eval_match` for parameter sweeps.

`bin/eval_match` — paired deals (both seats per deal to cancel luck). Reports win counts and 95% Wilson CI. Compiled with `-DBIG2_PIMC_STATS=1` so PIMC determinization stats are printed at the end.

## Key invariants

- Ranks are 0-indexed: 0=3, 1=4, …, 10=K, 11=A, 12=2. The deck has 4 of each 0–10, 3 of rank 11 (A), 1 of rank 12 (2).
- Suits are ignored entirely; only rank matters.
- 48-card deck (52 − 3×2s − 1A), dealt 16 each; third pile unseen.
- AAA is a bomb (not a triple); it cannot appear in a triple straight. See `RULES.md` for the rationale.
- `bin/generate_data` and `bin/eval_match` look for model files relative to the **binary's working directory** (`data/tree_model_d{n}.txt`, `data/linear_rollout_w.txt`, `data/pass_ridge_w.txt`).

## Adding a new player

1. Create `src/players/<family>/<name>_player.h` implementing `select_move_impl()`.
2. Create `src/players/<family>/<name>_player_factory.h` with a `PlayerFactory` subclass.
3. Register the name in `src/players/player_factory_registry.h`.
4. No Makefile changes needed — players are header-only; the factory registry is `#include`d by each binary's `.cpp` entry point.
