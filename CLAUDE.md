# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build commands

All commands run from the repository root:

```bash
make release       # cargo build --release → target/release/eval_match + generate_data
make test          # cargo test --all
make build         # cargo build (debug)
make tablebase_opp1_gen  # standalone C++ tool to regenerate data/tablebase_opp1.bin
```

**Git submodule:** after a fresh clone, run `git submodule update --init projects/standard-linter` (or `make standards_init`). The vendored [standard-linter](https://github.com/maxjiang216/standard-linter) repo holds shared formatter/linter configs. Run `make standards_configs` to copy them into `.code-standards/` for local tooling or `pre-commit`.

Ubuntu/Debian dependencies for Arrow/Parquet (only needed if building the Rust `parquet` feature): `sudo apt install libarrow-dev libparquet-dev` (the Rust crate bundles its own; system libs not required).

## Running evaluations

```bash
# Quick head-to-head
./target/release/eval_match --p0 greedy --p1 random --deals 10000 --seed 42

# Via JSON config (saves log + runs analysis)
python3 scripts/run_eval_match.py scripts/configs/eval_match.json --compile

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

The game engine is a **Rust workspace** under `crates/`. All production binaries are Rust.

### `crates/core`

- **`Game`** — full game state: both hands, discard pile, legal moves, turn order.
- **`PartialGame`** — imperfect-information view for one player (opponent hand hidden).
- **`Move`** — encoded move with 469 legal integer IDs (0 = PASS). Fields: `combination` (enum), `rank`, `auxiliary`. `Move::encode()` / `Move::decode(id)` for serialization.
- **`GameRecord`** / **`TurnRecord`** — turn-by-turn replay and training record.
- **`tablebase`** — runtime lookup for precomputed opponent-has-one-card endgame table (loads `data/tablebase_opp1.bin`).

### `crates/players`

All players implement the `Strategy` trait wrapped by `Player<S>`. The wrapper calls `peek_tablebase_move` before delegating to the strategy, so strategies never handle the tablebase directly. `AnyPlayer` is the type-erased trait for `Box<dyn AnyPlayer>`.

**Player families:**
- `random` — uniform random legal move (including pass when forced).
- `greedy` — lexicographic post-move hand eval (`GreedyEval`): win-now > bomb count > fewest cards > high ranks. Variants: `greedy_random`, `greedy_random_pass`, `greedy_no_bomb` (float param), `greedy_linear` (Ridge evaluator), `greedy_pass` (Ridge pass/greedy), `tree_greedy` (decision-tree evaluator).
- `pimc` — Perfect Information Monte Carlo: sample N opponent hands, run full rollouts for each candidate move, pick the move with most wins. Hoeffding early-stop, optional re-determinization. Rollout policies: greedy, tree, linear, pass-aware.

**Player registry:** `crates/players/src/registry.rs` — `make_player(name, param, seed)` maps string names to `Box<dyn AnyPlayer>`. Add new players here to make them available to all CLI tools.

Model files are loaded relative to the **working directory**: `data/tree_model_d{n}.txt`, `data/linear_rollout_w.txt`, `data/pass_ridge_w.txt`.

### `crates/simulation`

- **`simulator::play_game`** — plays one game between two `AnyPlayer`s, returns `GameRecord`.
- **`coordinator::run_games_parallel`** — rayon-parallel self-play, returns `Vec<(u32, GameRecord)>`.
- **`coordinator::run_paired_deals`** — paired deals (seat-swap) for eval; returns `(p0_wins, total)`.

### `crates/features`

Feature extractors registered by name via `create_feature(name)`. Two scopes:
- `game_level` — per-game scalars (outcome, length, tablebase hits, …).
- `turn_level` — per-turn features (hand shape, last move type, bomb count, possible moves, …).

Feature names match the C++ column names exactly (Python analysis is unchanged).

### `crates/datagen`

- **`target/release/generate_data`** — threaded self-play → `<prefix>_game.parquet` + `<prefix>_turn.parquet`.
- **`target/release/eval_match`** — paired deals, Wilson 95% CI. Flags: `--p0/--p1`, `--deals`, `--seed`, `--threads`.
- **`parquet.rs`** — Arrow `RecordBatch` + `ArrowWriter` export; schema matches C++ column names.

## Key invariants

- Ranks 0-indexed: 0=3, 1=4, …, 10=K, 11=A, 12=2. Deck: 4 copies of 0–10, 3 of rank 11 (A), 1 of rank 12 (2). Total 48 cards, dealt 16 each; 16 unseen.
- Suits are ignored entirely; only rank matters.
- AAA is a bomb (not a triple); it cannot appear in a triple straight. See `RULES.md`.
- Move ID 0 = PASS. IDs 1–468 are legal non-pass moves. `MOVE_TO_CARDS[id]` gives the rank costs.

## Adding a new player

1. Create `crates/players/src/<family>/` with a struct implementing `Strategy`.
2. Register the name in `crates/players/src/registry.rs` — `make_player(name, ...)`.
3. No other changes needed; the registry is imported by all binaries.

## Exact-match verification

`crates/simulation/tests/greedy_exact_match.rs` runs 200 greedy-vs-greedy games from C++-dealt hands (golden file `greedy_golden.txt`) and asserts Rust produces identical move sequences. To regenerate the golden file: `make export_greedy_games` (requires the old C++ source) or check git history for `src/datagen/export_greedy_games.cpp`.
