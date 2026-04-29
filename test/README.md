# Tests

The **active** test suite is the Rust workspace. Run it from the **repository root**:

```bash
make test
```

This is equivalent to `cargo test --all`. CI (`.github/workflows/ci.yml`) runs `make test` on pushes and pull requests to `main`.

## Where tests live

| Area | Location | What it covers |
|------|-----------|----------------|
| Core engine | `crates/core/src/*` (`#[cfg(test)]` modules) | Deals, legal moves, pass rules, `PartialGame` vs `Game`, move encode/decode, tablebase hooks, move enumeration helpers |
| Players | `crates/players/src/greedy/evaluators.rs` | Greedy evaluator ordering and tree feature vector shape |
| Simulation | `crates/simulation/src/simulator.rs` | Random self-play completes; card conservation |
| Golden parity | `crates/simulation/tests/greedy_exact_match.rs` | Rust greedy-vs-greedy move sequences match the C++ golden file |

Add new integration tests under `crates/<crate>/tests/` or unit tests next to the code under `src/`, following existing patterns.

## Legacy C++ sources

The `test/*.cpp` files are **not** built by the root `Makefile`. They are the old C++ unit-test sources kept for reference (and for parity with `greedy_exact_match`, which compares against a C++-derived golden transcript). Rust tests above supersede them for day-to-day development.

The former layout was: `test_main.cpp` driving suites for moves, game, partial game, greedy vs random, long random runs, tablebase, plus `test_perf.cpp` as a separate benchmark entry point. There is no `make test_core` or `bin/test_core` in the current tree.
