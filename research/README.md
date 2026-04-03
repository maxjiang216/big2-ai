# Research Scripts

Standalone exploratory scripts for analysing Big-2 hand structure and ad-hoc
experiments. They are not part of the main `Makefile` targets except where they
call `bin/selfplay` (see below).

---

## Engine game-level statistics (`game_feature_stats.py`)

Use this for aggregate statistics that match the **core game** and **feature
extractors** in `src/features/game_level/`:

- `length` — number of turns in the game (same as `GameLengthExtractor`)
- `outcome` — winner index (same as `OutcomeFeature`)
- `tb_hits` — turns where a tablebase case fired (same as `TablebaseHitsFeature`)
- `start_legal_moves` — count of legal moves for the **opening** player on turn
  0 (replaces the old `avg_moves.cpp` idea using real engine legality)

**Prerequisite:** build the self-play binary from the repo root:

```bash
make selfplay
```

**Run:**

```bash
python3 research/game_feature_stats.py --games 10000 --player random
```

Optional: `--game-features length,outcome,...` (comma-separated), `--jsonl-out
path.jsonl` to keep per-game JSON, `--keep-jsonl` when using a temp file.

Under the hood this invokes `bin/selfplay` with `--game-features` and
`--game-jsonl`, then prints mean / stdev / min / max (and winner counts for
`outcome`). The C++ tool also prints aggregate means on stdout.

This supersedes the removed scripts `avg_game_length.cpp` and `avg_moves.cpp`,
which used a separate 2-player toy simulator or raw hand enumeration instead of
the shared `Game` / `GameRecord` pipeline.

---

## `count_num_hands.py`

Short DP script that counts the number of **distinct rank-composition hands**
(multisets of rank counts summing to 16) possible in Big-2. No simulation —
exact result in milliseconds.

**Produces:** a single printed number.

---

## `endgame.cpp`

DP solver over every hand of ≤ 15 cards. The heuristic score measures how
cleanly a hand can be played down using straights; the DP finds the best
straight to lead in order to maximise that score. Only hands where the optimal
move is a straight ("non-trivial" positions) are written to disk.

**Produces:**

- `endgame_straight_dp.jsonl` — one JSON line per non-trivial hand:
  `{"h":[…], "s":<score>, "m":[<len>,<endRank>]}`
- `endgame_straight_dp_sample.txt` — first 1 000 of those lines in a compact
  plain-text format for quick sanity checks

Compiled binary: `build/research/endgame` (`make research` from repo root)

**Relationship to `tablebase_opp1`:** **No.** `endgame.cpp` is a separate
research tool (straight-first DP, JSONL dump). The opponent-1-card tablebase is
built by `scripts/tablebase_opp1_gen.cpp` → `bin/tablebase_opp1_gen`, loaded at
runtime via `src/core/tablebase_opp1.*`. The generator only mentions
`endgame.cpp` in a comment comparing sample-file purposes.

---

## `best_hand.cpp`

Exact enumeration over all rank-count compositions of a 16-card Big-2 hand.
Finds and prints the composition that maximises the **total count of legal
moves** (under the same counting rules as that program’s move generator).

**Produces:** printed composition and move count.

Compiled binary: `build/research/best_hand` (`make research`)

---

## `multi_comb.cpp`

Monte Carlo simulation (10 000 000 trials) estimating the probability that a
random 16-card hand satisfies a structural condition: it must contain **at
least one triple** (three-of-a-kind among ranks 3–K) **and at least two
doubles** (pairs among ranks 3–A).

**Produces:** printed probability.

Compiled binary: `build/research/multi_comb` (`make research`)

---

## `play_probs.cpp`

Exact enumeration over all weighted hand compositions. For every specific legal
move (single 3, single 4, … double 3, … full house, bomb, every straight
variant, etc.) it computes the **exact probability** that a random starting hand
can play that move.

**Produces:** printed probability table.

Compiled binary: `build/research/play_probs` (`make research`)

---

## Output / data files

| File | Generator | Notes |
|------|-----------|-------|
| `endgame_straight_dp.jsonl` | `endgame.cpp` | Large; regenerable |
| `endgame_straight_dp_sample.txt` | `endgame.cpp` | Small sanity sample |
| `tablebase_opp1_samples.txt` | `bin/tablebase_opp1_gen` | Optional diagnostic |
| `tablebase_opp1.bin` | `bin/tablebase_opp1_gen` | **Used at runtime** — see `src/core/tablebase_opp1.h` |

---

## Compiled binaries

Prefer `make selfplay` / `make tablebase_opp1_gen` for supported tools. For the
standalone C++ programs in `research/`, use:

```bash
make research   # builds endgame, best_hand, multi_comb, play_probs -> build/research/
```

Ad-hoc copies in `research/` can be removed; rebuild from source via `make research`.
`a.out` / `big2_moves` without source should be deleted.
