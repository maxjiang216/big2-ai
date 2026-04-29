# Scripts

Python drivers and JSON configs. Paths are relative to the **repository root**. The native CLIs they invoke live under **`target/release/`** (build with `make release` or `--compile`).

## Policy names (`crates/players/src/registry.rs`)

These strings are accepted by **`make_player`** and surfaced on the **`target/release/generate_data`** (`--player`) and **`target/release/eval_match`** (`--p0` / `--p1`) CLIs:

| Name | Meaning |
|------|---------|
| `random` | Uniform random legal move |
| `greedy` | Greedy heuristic (lexicographic score after each candidate move) |
| `greedy_random` | Usually greedy; with probability `param`, play a random legal move |
| `greedy_random_pass` | Usually greedy; with probability `param`, pass when legal |
| `greedy_no_bomb` | Usually greedy; when greedy would play a bomb, only do so with probability `param` |
| `pimc`, `pimc_linear`, `pimc_tree`, … | See **`crates/players/src/registry.rs`** (e.g. `pimc_redet` = PIMC with opponent hand re-sampled each opponent turn inside rollouts) |

Parameterized variants use `--p0-param` / `--p1-param` on `eval_match`, or `p0_param` / `p1_param` in JSON for `eval_match` / round robin.

For **self-play**, `generate_data` defaults **`--player-param` to `0.0`**; set **`player_param`** in the JSON config when you need a non-zero variant parameter. Use **`eval_match`** or the round-robin scripts for systematic parameter sweeps across matchups.

## `compare_pimc_rollout_strategies.py`

Runs a **fixed matrix** of `eval_match` jobs (greedy vs random; `pimc(N)` vs random/greedy; `pimc_redet(N)` vs random/greedy; `pimc` vs `pimc_redet`) with the same `--deals`, `--seed`, and `--threads`, and prints P0 win rate, Wilson CI, `eval_match` elapsed line, and process wall time. Optional `--csv` writes a summary table.

```bash
uv run python scripts/compare_pimc_rollout_strategies.py --deals 2000 --seed 42
```

Individual matchups are also available as JSON under `scripts/configs/eval_pimc20_vs_greedy.json`, `eval_pimc_redet20_vs_greedy.json`, `eval_pimc20_vs_random.json`, `eval_pimc_redet20_vs_random.json`, `eval_pimc20_vs_pimc_redet20.json`.

---

## `generate_data.py`

Python wrapper around **`target/release/generate_data`**: optional **`cargo build --release -p big2-datagen`** (`--compile`), JSON config, Parquet self-play.

**Prerequisites:** **Rust / Cargo** (same as the workspace). Parquet is handled by the Rust `parquet` crate; no system Arrow install is required.

```bash
make release
python3 scripts/generate_data.py scripts/configs/greedy.json
python3 scripts/generate_data.py scripts/configs/greedy.json --compile
```

**Config:** `player`, `num_games`, optional `player_param`, and (for Parquet) `output_path` plus `game_features` / `turn_features`. Optional: `threads`, `seed`, `samples_md_path`. See `scripts/configs/greedy.json`, `test.json`.

**Output:** `<output_path>_game.parquet` and `<output_path>_turn.parquet`.

---

## `run.sh` (self-play + Parquet analysis)

Runs `generate_data.py` then `analysis/analysis.py` on the produced Parquet files.

```bash
./scripts/run.sh                    # default: scripts/configs/greedy.json
./scripts/run.sh scripts/configs/test.json
```

Set `samples_md_path` in the config if you want anomaly game histories from `generate_data`.

---

## `run_eval_match.sh` / `run_eval_match.py`

Head-to-head evaluation via **`target/release/eval_match`**: **paired deals** (both seats per deal), **Wilson 95% CI**, full log + optional `scripts/analyze_eval_match.py` summary.

**Prerequisites:** **`make release`** (or pass **`--compile`** so the script runs `cargo build --release -p big2-datagen`).

```bash
./scripts/run_eval_match.sh
./scripts/run_eval_match.sh scripts/configs/eval_match.json
python3 scripts/run_eval_match.py scripts/configs/eval_match.json --compile
```

**Config** (`scripts/configs/eval_match.json`): `p0`, `p1`, `deals`; optional `p0_param`, `p1_param`, `seed`, `threads`, `output_log` (default `analysis/eval_match_last.log`).

**Analyze an existing log:**

```bash
python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
./target/release/eval_match ... 2>&1 | python3 scripts/analyze_eval_match.py
```

---

## `analyze_eval_match.py`

Parses `eval_match` stdout and prints a short markdown-style summary (wins, CI, result line). Used by `run_eval_match.py`; can be run standalone on a saved log.

---

## `round_robin_eval.py` / `run_round_robin.sh`

**Round robin:** for each unordered pair of players, runs **`target/release/eval_match`** once (same fairness as a single head-to-head). Prints a **win-rate matrix** and **aggregate standings**.

**Prerequisites:** **`make release`** (or **`--compile`**).

```bash
./scripts/run_round_robin.sh
./scripts/run_round_robin.sh scripts/configs/round_robin_example.json
python3 scripts/round_robin_eval.py scripts/configs/round_robin_example.json --compile
python3 scripts/round_robin_eval.py scripts/configs/round_robin_example.json --quiet --deals 500
```

**Config** (`scripts/configs/round_robin_example.json`):

- `players`: list of strings (`"greedy"`, `"random"`) or objects `{ "type": "...", "param": 0.2, "label": "..." }`.
- The example includes **`random`** and **`greedy`**, plus each greedy variant at **five** `param` values **0.1–0.5** (17 players → **136** pairwise matchups—use `--deals` to shorten test runs).
- `deals_per_matchup` or `k`: deals per pair (total games per pair = `2 × deals`). Override with `--deals N`.

---

## `tablebase_opp1_gen.cpp`

Source for the opponent-has-one-card tablebase **precompute** (standalone C++; not part of the Rust workspace). Built from the root `Makefile`:

```bash
make tablebase_opp1_gen
./bin/tablebase_opp1_gen [out.bin] [samples.txt]
```

Runtime loader: **`crates/core`** (reads `data/tablebase_opp1.bin` when present).

---

## `configs/`

| File | Used by |
|------|---------|
| `greedy.json`, `test.json` | `generate_data.py` |
| `eval_match.json` | `run_eval_match.py` |
| `round_robin_example.json` | `round_robin_eval.py` |

Copy and edit paths, `num_games` / `deals`, feature lists, and `threads` for your machine.
