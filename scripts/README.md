# Scripts

Drivers and tooling that live outside the main `src/` library. Paths below are relative to the **repository root**.

## `generate_data.py`

Python wrapper around `bin/generate_data`: compiles (optional), loads JSON, runs the Parquet self-play pipeline.

**Prerequisites:** C++17, OpenMP, Apache Arrow / Parquet (`libarrow-dev`, `libparquet-dev` on Debian/Ubuntu).

**Build the binary (once):**

```bash
make generate_data
```

**Run:**

```bash
python3 scripts/generate_data.py scripts/configs/greedy.json
python3 scripts/generate_data.py scripts/configs/greedy.json --compile   # build then run
```

**Input:** A JSON config with at least `player`, `num_games`, and `output_path`. Optional keys include `game_features`, `turn_features`, `threads`, `seed` (see `scripts/configs/greedy.json` and `test.json`).

**Output:** Two Parquet files next to the configured base path:

- `<output_path>_game.parquet` — one row per game (game-level features)
- `<output_path>_turn.parquet` — one row per turn (turn-level features)

Paths are resolved from the repo root unless `output_path` is absolute.

## `tablebase_opp1_gen.cpp`

Source for the opponent-has-one-card tablebase **precompute** tool (not built by `generate_data`).

**Build:**

```bash
make tablebase_opp1_gen
```

**Run:**

```bash
./bin/tablebase_opp1_gen [out.bin] [samples.txt]
```

**Output:** A binary tablebase file (loaded at runtime via `src/core/tablebase_opp1.*`) and an optional text sample file for debugging. See the root `README.md` tablebase section.

## `configs/`

Example JSON configs for `generate_data.py`. Copy and edit `output_path`, `num_games`, feature lists, and `threads` for your machine.
