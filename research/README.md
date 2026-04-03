# Research

Exploratory scripts and small standalone programs. They are **not** required for the main training pipeline (`scripts/generate_data.py`). Python tools assume you run commands from the **repository root** unless noted.

## Build C++ research binaries

From the repo root:

```bash
make research
```

Produces `build/research/best_hand`, `build/research/multi_comb`, and `build/research/play_probs` (no Arrow dependency). Run a binary:

```bash
./build/research/best_hand
```

## Python: engine-backed game statistics (`game_feature_stats.py`)

Aggregates game-level fields that align with **`src/features/game_level/`** extractors, via the real engine self-play binary:

- `length` — turns in the game (`GameLength` / length feature)
- `outcome` — winner index (`OutcomeFeature`)
- `tb_hits` — tablebase episodes (`TablebaseHitsFeature`)
- `start_legal_moves` — legal move count for the opening player on turn 0 (`StartLegalMovesFeature`)

**Prerequisite:**

```bash
make selfplay    # produces bin/selfplay
```

**Run:**

```bash
python3 research/game_feature_stats.py --games 10000 --player random
```

**Inputs:** CLI flags (`--games`, `--player`, `--threads`, `--seed`, `--game-features`, etc.; see `--help`).

**Outputs:** Printed mean / stdev / min / max (and winner counts for `outcome`). Optional `--jsonl-out path.jsonl` / `--keep-jsonl` to retain per-game JSON lines. The C++ self-play process also prints aggregate lines on stdout.

## Python: hand enumeration (`count_num_hands.py`)

Exact DP count of **distinct rank-composition hands** (multisets of rank counts summing to 16). No simulation.

**Output:** one number printed to stdout.

## C++: `best_hand.cpp`

Enumerates rank-count compositions for a 16-card Big 2 hand and finds the composition that maximizes the **total count of legal moves** (per this program’s move generator).

**Output:** printed composition and move count.

## C++: `multi_comb.cpp`

Monte Carlo (10 000 000 trials): estimates the probability that a random 16-card hand has **at least one triple** (ranks 3–K) **and** at least **two doubles** (ranks 3–A).

**Output:** printed probability.

## C++: `play_probs.cpp`

Exact enumeration over weighted hand compositions: for each legal move type, the **exact probability** that a random starting hand can play that move.

**Output:** printed probability table.

## Generated data files (optional / local)

If you regenerate large artifacts locally, you may see files such as `endgame_straight_dp*.jsonl` from older experiments; they are gitignored where applicable. The **runtime** tablebase file used by the engine is built with `make tablebase_opp1_gen` and `bin/tablebase_opp1_gen`, not from these research binaries.

| Artifact | Producer | Notes |
|----------|----------|--------|
| `tablebase_opp1.bin` | `bin/tablebase_opp1_gen` | Loaded by `src/core/tablebase_opp1.*` |
| `tablebase_opp1_samples.txt` | `bin/tablebase_opp1_gen` | Optional diagnostic sample |
