# Source layout (`src/`)

C++17 code for the Big 2 engine, simulation, players, features, and data export. Include paths used by the `Makefile` are `-Isrc/core`, `-Isrc/simulation`, `-Isrc/players`, `-Isrc/datagen`, `-Isrc/features`.

## `core/`

Game rules and shared state:

- **`game.*`, `move.*`, `partial_game.*`** — full game state, encoded moves (472 legal ids), and imperfect-information view for a player.
- **`game_record.*`** — turn-by-turn replay / training record.
- **`util.*`** — deck limits, helpers shared with tests.
- **`tablebase_opp1.*`** — runtime lookup for the precomputed opponent-1-card endgame table (load path configured where players are built).

## `simulation/`

- **`game_simulator.*`** — runs one full game between two `Player` instances and returns a `GameRecord`.
- **`game_coordinator.*`** — many games, threading, hooks into Parquet export (`datagen`).

## `players/`

Abstract **`player.h`** and factories:

- **`random/`** — uniform random legal move.
- **`greedy/`** — heuristic policy (used in self-play and tests).

Policies can use the tablebase when the engine provides it.

## `features/`

Header-only **feature extractors** registered by name for datagen:

- **`game_level/`** — per-game scalars (e.g. length, outcome, tablebase hit counts, start legal moves).
- **`turn_level/`** — per-turn features (hand shape, last move, bombs, possible moves, etc.).
- **`feature_extractor.h`** — registry / dispatch used by the coordinator and Parquet writer.

## `datagen/`

Binaries are linked from the repo `Makefile`, not as a separate library:

- **`generate_data.cpp`** — CLI entry for threaded self-play, stats, optional Parquet + sample Markdown (`bin/generate_data`).
- **`samples_md.*`** — writes anomaly game histories for `--samples-md`.
- **`parquet_export.*`, `feature_registry.h`** — Arrow/Parquet schema and feature column wiring.
