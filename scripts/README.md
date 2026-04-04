# Scripts

Drivers and tooling outside `src/`. Paths are relative to the **repository root**.

## Policy names (`player_factory_registry.h`)

These strings work anywhere the code calls `make_player_factory` (e.g. `bin/eval_match`, `bin/generate_data` for `--player`):

| Name | Meaning |
|------|---------|
| `random` | Uniform random legal move |
| `greedy` | Greedy heuristic (lexicographic score after each candidate move) |
| `greedy_random` | Usually greedy; with probability `param`, play a random legal move |
| `greedy_random_pass` | Usually greedy; with probability `param`, pass when legal |
| `greedy_no_bomb` | Usually greedy; when greedy would play a bomb, only do so with probability `param` |

Parameterized variants take `--p0-param` / `--p1-param` (or `param` in JSON configs for `eval_match` / round robin). **`bin/generate_data` fixes `param` at 0** for self-play; use `eval_match` or round robin for parameter sweeps.

---

## `generate_data.py`

Python wrapper around `bin/generate_data`: optional compile, JSON config, Parquet self-play.

**Prerequisites:** C++17, OpenMP, Apache Arrow / Parquet (`libarrow-dev`, `libparquet-dev` on Debian/Ubuntu).

```bash
make generate_data
python3 scripts/generate_data.py scripts/configs/greedy.json
python3 scripts/generate_data.py scripts/configs/greedy.json --compile
```

**Config:** `player`, `num_games`, and (for Parquet) `output_path` plus `game_features` / `turn_features`. Optional: `threads`, `seed`, `samples_md_path`. See `scripts/configs/greedy.json`, `test.json`.

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

Head-to-head evaluation via `bin/eval_match`: **paired deals** (both seats per deal), **Wilson 95% CI**, full log + optional `scripts/analyze_eval_match.py` summary.

**Prerequisites:** `make eval_match` (no Arrow).

```bash
./scripts/run_eval_match.sh
./scripts/run_eval_match.sh scripts/configs/eval_match.json
python3 scripts/run_eval_match.py scripts/configs/eval_match.json --compile
```

**Config** (`scripts/configs/eval_match.json`): `p0`, `p1`, `deals`; optional `p0_param`, `p1_param`, `seed`, `threads`, `output_log` (default `analysis/eval_match_last.log`).

**Analyze an existing log:**

```bash
python3 scripts/analyze_eval_match.py --file analysis/eval_match_last.log
./bin/eval_match ... 2>&1 | python3 scripts/analyze_eval_match.py
```

---

## `analyze_eval_match.py`

Parses `eval_match` stdout and prints a short markdown-style summary (wins, CI, result line). Used by `run_eval_match.py`; can be run standalone on a saved log.

---

## `round_robin_eval.py` / `run_round_robin.sh`

**Round robin:** for each unordered pair of players, runs `bin/eval_match` once (same fairness as a single head-to-head). Prints a **win-rate matrix** and **aggregate standings**.

**Prerequisites:** `make eval_match`

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

Source for the opponent-has-one-card tablebase **precompute** (not built by `generate_data`).

```bash
make tablebase_opp1_gen
./bin/tablebase_opp1_gen [out.bin] [samples.txt]
```

See root `README.md` and `src/core/tablebase_opp1.*`.

---

## `configs/`

| File | Used by |
|------|---------|
| `greedy.json`, `test.json` | `generate_data.py` |
| `eval_match.json` | `run_eval_match.py` |
| `round_robin_example.json` | `round_robin_eval.py` |

Copy and edit paths, `num_games` / `deals`, feature lists, and `threads` for your machine.
