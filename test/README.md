# Tests

Unit and integration tests for the C++ engine. Built and run from the **repository root**.

## How to run

```bash
make test_core
./bin/test_core
```

CI (`.github/workflows/ci.yml`) runs the same commands on pushes and pull requests to `main`.

**Not part of `test_core`:** `test/test_perf.cpp` is compiled separately as the performance benchmark (its own `main` when `BENCHMARK_MAIN` is defined):

```bash
make benchmark
./bin/benchmark
```

## What each file tests

| File | Suite label | What it covers |
|------|----------------|----------------|
| `test_move.cpp` | `move` | Move id encode/decode roundtrip; `MOVE_TO_CARDS` consistency and deck limits; legal move enumeration invariants. |
| `test_game.cpp` | `game` | Deal invariants, turn order, playing moves, game completion, discard and hand conservation. |
| `test_partial_game.cpp` | `partial` | `PartialGame` vs `Game` perspective, legal move subset on your turn, consistency across turns. |
| `test_greedy_player.cpp` | `greedy` | Greedy vs random full games (played move ∈ legal set); greedy wins more often than random over many seeds. |
| `test_random_games.cpp` | `random_games` | Long random simulations: card conservation, `GameRecord` integrity, `PartialGame` vs `Game` after each turn. |
| `test_tablebase.cpp` | `tablebase` | When the opponent-has-one-card tablebase applies, the played move matches the expected highest single (and related integration cases). |
| `test_main.cpp` | — | Invokes all of the above suites in order; prints `All tests passed.` on success. |
| `test_perf.cpp` | *(benchmark only)* | Single-threaded throughput and tablebase-related counters over many games (not asserted in CI). |

The order of suites in `test_main.cpp` matches the table above.
