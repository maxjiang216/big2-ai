# az_pi series line — 2026-06-12

Series-aware version of the perfect-information AlphaZero teacher: the value
head predicts **P(win the series to 50 points)** conditioned on the series
score, instead of P(win this game). Motivation: (1) the eventual
imperfect-information student must be series-aware; (2) points-shaped values
reward shedding cards and winning big — more "intuitive" play that should
transfer to the II game better than edgy binary-PI lines.

## Design

- **Net** (`nn/model_az_pi.py`): `size_embed` → `pts_embed(Linear(2,16))` fed
  `(my_pts/50, opp_pts/50)`. Hand sizes dropped as inputs (derivable from the
  exact encodings). Value = sigmoid P(win series).
- **Solver** (`pi_solver.cpp`): exact **margin minimax**. No first-win break —
  a node is scored only after all (pruned) moves; WIN takes the max margin
  (loser's final cards), LOSS the min. Only shortcut: a win at the opponent's
  *current* card count is the ceiling at that state. Series points are
  monotone in the margin, so margin-optimal = series-optimal. Memo layout
  unchanged ([sig:57][margin:5][proof:2]); a WIN proven past unresolved
  siblings is sound but its margin may improve, so it is not memoized.
- **Search** (`pi_search`): `PiSearchConfig{const SeriesTable*, int pts[2]}`.
  Every proven/terminal backup goes through `exact_value()` =
  `series_value_after_win(table, pts, margin)` (classic 1/0 when no table, so
  the binary line is untouched). PUCT, mean backup, margin tie-breaks, greedy
  tie-breaks all unchanged.
- **V table** (`analysis/series_markov.py`, pre-existing): per-state outcome
  distributions shrunk toward the pooled global (k=100), backward induction
  over the score DAG (`V(a,b) = Σ_p P(win,p)·[a+p≥50 ? 1 : V(a+p,b)] +
  Σ_p P(lose,p)·[b+p≥50 ? 0 : 1−V(b+p,a)]`; loser follows = initiative rule).
  Refreshed **every generation** from the last 4 gens' outcome CSVs.
- **Labeling at Python load time**: the parquet stores the 1/0 winner flag
  plus `my_pts/opp_pts/loser_cards`; `train_az_pi --series-table` computes
  `value = mover won ? tw : 1−tw` with `tw = series_value_after_win` at load.
  No relabel pass; every gen trains against the freshest table. Dataset slots
  3/4 carry the pts scalars when the `my_pts` column exists, so the GpuLoader
  and collate plumbing is unchanged.
- **Self-play** (`az_pi_selfplay`): uniform series-state sampling over 50×50
  (coverage matters, realism doesn't — the target is table-derived);
  `--outcomes` sidecar CSV in the series_markov format; `--fixed-pts` pins a
  state for the sample-game HTMLs (now rendered at 0-0 / 40-0 / 0-40).
- **Eval** (`eval_az_pi_match --series N`): N **paired full series**. Game k of
  both orientations replays the identical shuffle (deal RNG seeded on
  (pair, game_idx) only); game 1's lead follows the 3♠ holder
  (`sample_first_player_3s` — hands stay with seats, the player assignment
  flips); the winner leads every later game. Promotion gate = paired-series
  sweep-share Wilson CI (same sign-test logic as decisive deals). Also reports
  the per-GAME win rate, to show how a small per-game edge amplifies over a
  first-to-50 series.
- **Bootstrap (gen 1)**: old gen-15 binary-line champion plays 20k fresh games
  score-blind (`--series-states`); the run's own sidecar builds the initial
  table with `--bootstrap-global` (per-state splits of score-blind play are
  noise); training warm-starts with `--init-from models/az_pi.pt` (30/32
  tensors; pts_embed reinit). Old gen-15/16 parquets could NOT be relabeled —
  forced moves write no rows, so loser's final card count is unrecoverable.
- **Evaluator** (`PiNNEvaluator(path, device, pts_inputs)`): pts inputs ⟺ a
  series table is loaded; legacy score-blind nets keep their size inputs.

## Results (gens 1–3, sims 1600, 200 paired series)

| gen | result | detail |
|---|---|---|
| 1 | champion seeded | bootstrap table V(0,0)=0.550, ~8.6 games/series expected |
| 2 | **PROMOTED** | raw series 56.8% (227/400); decisive pairs 36–9 → sweep share **0.80** [0.66, 0.89] |
| 3 | tie | 200/400 series, decisive pairs 13–13 |

- Score conditioning is learned almost immediately: after 2 smoke epochs the
  mean value over fixed positions was 0.80 at 49-0 ahead and 0.06 at 0-49
  behind (0.47 at 0-0).
- Series amplification is real and matches the motivation for series-level
  eval: gen 2's modest per-game edge became an 80% paired-series sweep share.
- Value loss descends monotonically across gens (.598 → .595 → .593) even
  through the gen-3 strength tie.
- Observed series length ≈ 11.4 games (both players score; the DP's 8.6 is
  winner-leads expectation from (0,0) under the pooled kernel).

## Self-escalating run (gens 4–11, killed)

`run_az_pi_series_overnight.sh`: 3-gen stall → sims ×2, eval pairs ×2,
slots ÷2; caps 12800/800/128; 9G cgroup cap. Per-gen CSV:
`logs/az_pi_series_overnight.csv` (includes the per-game rate column).

Result: **no promotion after gen 2.** Seven straight challengers tied or
lost to the gen-2 champion; the run was killed mid-gen-11 eval after ~8 h.

| gen | sims | eval pairs | game rate | raw series | sweep share | decisive |
|---|---|---|---|---|---|---|
| 4 | 1600 | 200 | 0.5024 | 0.5050 | 0.533 | 30 |
| 5 | 1600 | 200 | 0.4962 | 0.4875 | 0.432 | 37 |
| 6 | 3200 | 400 | 0.4960 | 0.4800 | 0.371 | 62 |
| 7 | 3200 | 400 | 0.5023 | 0.4775 | 0.375 | 72 |
| 8 | 3200 | 400 | 0.4999 | 0.4850 | 0.403 | 62 |
| 9 | 6400 | 800 | 0.5007 | 0.4900 | 0.437 | 126 |
| 10 | 6400 | 800 | 0.5014 | 0.4875 | 0.412 | 114 |

Observations:

- **Plateau hit at gen 2**, far earlier than the binary line's gen 15–16.
  The big gen-2 jump was score conditioning; sims-doubling found nothing
  past it.
- Gens 6–10 sweep shares sit **below** 0.5 (challengers slightly worse
  than champion in decisive pairs) while per-game rates are dead even —
  the series eval resolves a deficit the per-game rate can't see, which
  is the amplification working as designed, just in the wrong direction.
- Value loss kept creeping down every gen (v≈0.615 → 0.606 by gen 11)
  with zero strength gain — same loss/strength decoupling the binary
  line and the II memory line showed.
- Cost scaling: 1600-sim gens ≈ 15 min, 3200 ≈ 35 min, 6400 ≈ 2.5 h.

Follow-up: run killed; round-robin re-eval of gens 2/3/4 at 800 paired
series each (sims 1600, fresh seed 1000 — the promotion evals all reused
seed 42, i.e. identical deal streams) to pick the line's champion on
more data (`logs/az_pi_s_roundrobin_234.txt`).
