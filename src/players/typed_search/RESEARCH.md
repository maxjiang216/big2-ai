# Typed-Search Research Log

## Algorithm overview

Trick-bounded DAG search for the 2-player Big 2 engine. Key design choices:

- **Truncation at trick boundaries.** Search recurses through our turn → opp turn → our turn → ... until the first PASS, then evaluates the resulting just-passed leaf via a tabular evaluator.
- **Two tabular evaluators** (sparse hashmap, binary on-disk):
  - `eval_main` — ~442k state space encoded from hand-shape features (initiative, hand-size buckets, bomb / straight / double-triple-straight tiers, and per-rank counts of singles/doubles/triples bucketed with "guaranteed largest" promotion).
  - `eval_fallback` — ~9k state space, simpler features. Queried when main has < `min_visits` (default 5) at the queried state.
- **Two opponent-move-probability tables**:
  - `mp_main` — keyed on `(player_move_id, opp_count)`.
  - `mp_fallback` — keyed on `player_move_id` only.
- **DAG memoization** keyed on `(hand_packed, opp_count, last_move.combination, last_move.rank)` — discard composition intentionally ignored (transposition approximation; same approximation also justifies merging move-groups across opp ranks that produce identical sets of our responses).
- **Forced-move sub-search** at "we hold initiative" leaves: DFS over our unbeatable moves, evaluates the eval table at every visited node (not just terminals), takes max.
- **Generations training**: bootstrap gen-0 from random self-play populating `eval_fallback` only; each subsequent generation runs typed-search self-play, decays existing tables by α (default 0.7), then merges new counts.

## Files

| Path | Purpose |
|---|---|
| `eval_features.{h,cpp}` | State-ID encoding + impute-terminal + guaranteed-largest helper |
| `eval_table.{h,cpp}` | Eval table with main+fallback chain, decay, query-stats counters |
| `move_prob_table.{h,cpp}` | Opp-move distribution table, same shape |
| `move_grouping.{h,cpp}` | Group opp moves into response-equivalence classes |
| `forced_search.{h,cpp}` | "We have initiative" leaf forced-move sub-search |
| `typed_search.{h,cpp}` | DAG search with memoization |
| `typed_search_player.{h,cpp}` | Player subclass; per-call RNG mix |
| `typed_search_player_factory.{h,cpp}` | Loads 4 table files, shares across players |

Driver / orchestration:
- `src/datagen/typed_search_train.cpp` — multithreaded self-play + table updates with decay.

## Reference results

### Strength (after gen-0 random + 4 × 5k typed_search generations, α=0.7)

Paired-deal evaluation, 500 deals = 1000 games, seed 100:

| Generation | vs random | vs greedy |
|---|---|---|
| 0 (bootstrap only) | 87.0% | 53.0% (no signal; CI [49.9, 56.1]) |
| 1 | 89.8% | 59.4% |
| 2 | 90.7% | 62.0% |
| 3 | 90.0% | 63.3% |

vs greedy improves monotonically with training. vs random saturates around 90%.

### Throughput (typed_search self-play, current trained tables)

| Threads | Games/sec | Per-game wall | 5k | 50k | 500k |
|---|---|---|---|---|---|
| 1 | 75.8 | 13.2 ms | 66 s | 11 min | 110 min |
| 8 | 736 | 1.36 ms | 6.8 s | 68 s | 11 min |

Parallel efficiency at 8 threads: ~82%.

### Static table coverage (after the runs above)

| Table | Entries / total | Coverage | ≥1 visit | ≥5 visits | ≥20 visits | ≥100 visits |
|---|---|---|---|---|---|---|
| `eval_main` | 37,105 / 442,368 | 8.4% | 29,306 | 11,027 | 3,712 | 422 |
| `eval_fallback` | 3,112 / 9,216 | 33.8% | 2,882 | 2,110 | 1,540 | 727 |
| `mp_main` | 5,265 / 7,956 | 66.2% | — | — | — | — |
| `mp_fallback` | 434 / 468 | 92.7% | — | — | — | — |

### Empirical query hit rates (during 5k-game typed_search run)

`eval_table` — 25.3M leaf queries:
- main hit (≥5 visits): **73.2%**
- fallback hit: **24.7%**
- both miss → default 0.5: **2.1%**

`mp_table` — 9.3M opp-distribution queries:
- main hit: **98.0%**
- fallback hit: **2.0%**
- both empty → uniform: ~0%

The empirical hit rates are much higher than static coverage % because game-state distribution is Zipfian: a small core of hot states gets most queries.

## Profiling — baseline

Callgrind on 10 games, single-thread, with trained tables loaded. 1.17B instructions / ~6.1s wall.

```
12.24%  TypedSearch::visit_opp (self)
12.08%  opponent_can_respond(int, hand, discard, opp_count)   — array overload, used by find_forced_win
 5.52%  compute_legal_moves(HandBits, Move)
 5.14%  _int_free
 3.78%  malloc
 3.53%  Move::Move(int)
 3.42%  main_state_id
 2.84%  opponent_can_respond(int, HandBits, int)               — fast overload
 2.36%  free
 2.32%  all_moves()                                            — table accessor
 2.10%  _int_malloc
 1.65%  fallback_state_id
 1.65%  __memcpy_avx_unaligned_erms
 1.19%  group_opp_moves (self)
 1.08%  compute_r_star
 ~9%    libstdc++ I/O (one-shot table load + save)
```

`visit_opp` accounts for ~82.5% cumulative cost (self + recursion). Inside it:
- The `for (int mid : beating[our_move_id])` filter loop — `opp_could_play` body is the bulk of `visit_opp`'s self time.
- `group_opp_moves`: 1.82% per call site
- `mp_table_.query`: 0.06% (cheap)

`main_state_id` cost is dominated by `straight_tier` and `double_triple_straight_tier` calls, each of which calls `compute_legal_moves` on a bomb-rank-stripped hand copy.

## Optimization candidates (baseline: 75.8 g/s single-thread)

| # | Target | Estimated win | Complexity |
|---|---|---|---|
| 1 | Cache `main_state_id` / `fallback_state_id` per `OurNode` (don't recompute per leaf visit) | ~6% | low |
| 2 | Hoist `opp_max[r]` out of `opp_could_play` (precompute once per `visit_opp`) | ~5–8% | low |
| 3 | Use `HandBits`-based opp-can-hold check (compose with existing fast overloads) | extra on top of #2 | low–med |
| 4 | Thread-local scratch buffers (eliminate per-call vector allocations in hot path) | ~10% from allocation pressure | medium |
| 5 | `find_forced_win` uses array-version of `opponent_can_respond`; switch to HandBits-version | ~5% | low |
| 6 | `Move::Move(int)` accessor inlining / `all_moves()` direct member ref | ~2% | low |

Optimizations applied below are profiled one at a time so we can attribute the delta to each change.

---

## Optimization 1 — Bitset `opp_could_play` (HandBits-based)

**Change:** Replaced the per-move 13-rank loop in `opp_could_play` with an O(1) bitset check using a precomputed `move_needs_table()` (HandBits per move ID). Hoisted `opp_upper_bound_bits` to be computed once per `visit_opp` call instead of once per (move × rank).

**Files:** `src/players/typed_search/typed_search.cpp` only.

**Profile (callgrind, 10 games):**

| Metric | Baseline | After opt 1 | Δ |
|---|---|---|---|
| Total instructions | 1,168M | 1,090M | **−6.7%** |
| `visit_opp` (self) | 12.24% | 6.83% | −5.4 pp |
| `compute_legal_moves(HandBits, Move)` | 5.52% | 6.02% | +0.5 pp (relative grew) |
| `opponent_can_respond(arr, arr)` | 12.08% | 13.17% | +1.1 pp (still hot via `find_forced_win`; absolute Ir nearly unchanged at 141M→143M) |

**Wall clock (1 thread, 1000 games, trained tables):** 75.8 g/s → **88.4 g/s** (+16.6%).

The wall-clock gain (16.6%) is larger than the instruction-count drop (6.7%) because the new path is branch-free + cache-friendly: 4 ANDs + 4 equality checks vs. an unrolled 13-iter loop with conditionals.

**Strength check (paired-deal, 500 deals vs greedy, seed 100):** 65.5% (CI excludes 50%) — unchanged within noise vs. baseline 63.3%.

---

## Optimization 2 — Direct bitset straight enumeration in eval_features

**Change:** Replaced `compute_legal_moves(mod_hand, kPass)` calls inside `straight_tier`, `double_triple_straight_tier`, and the `has_straight` check in `fallback_state_id` with direct calls to the existing `straight_moves_for_pass_masks_into(at1, at2, at3, out)` bitmask helper. This skips the bomb/full-house enumeration that we never inspect, and reuses thread-local scratch buffers for the result vector.

**Files:** `src/players/typed_search/eval_features.cpp` only.

**Profile (callgrind, 10 games):**

| Metric | After opt 1 | After opt 2 | Δ vs opt 1 | Δ vs baseline |
|---|---|---|---|---|
| Total instructions | 1,090M | 1,038M | −4.8% | **−11.2%** |
| `Move::Move(int)` | 3.85% | 1.41% | −2.4 pp | −2.1 pp (12.5M of 14.6M Ir avoided in eval-features) |
| `compute_legal_moves(HandBits, Move)` | 6.02% | 3.13% | −2.9 pp | −2.4 pp |
| `main_state_id` (self only — internals shifted) | 3.73% | 4.14% | +0.4 pp | +0.7 pp |

**Wall clock (1 thread, 1000 games):** 88.4 g/s → **103.8 g/s** (+17.4% from opt 1; **+37.0% from baseline**).

**Strength check:** 65.8% vs greedy at seed 100 — unchanged within noise.

**New hot spot — `find_forced_win` (`opponent_can_respond(arr, arr)`):** Now 17.08% / 177M Ir, the single largest cost. Each call rebuilds opp's HandBits from scratch via the array overload. Switching `find_forced_win` (in `src/core/util.cpp`) and `forced_search_recursive` (in `forced_search.cpp`) to compute opp_bits once per call site and use `opponent_can_respond(int, HandBits, int)` would target this directly.

---

## Optimization 3 — Hoist opp_bits in find_forced_win and forced_search

**Insight:** During the forced-win recursion the opponent's possible-card upper bound is invariant. Each forced move moves cards from `hand` to `discard`, and `opp_max[r] = max_in_deck[r] − hand[r] − discard[r]` is algebraically conserved. `opp_count` is also invariant (opp passes each turn). So `opp_bits` can be built once at the top of the recursion and reused.

**Change:** Refactored `find_forced_win` in `src/core/util.cpp` to build `opp_bits` once and pass via a `ForcedWinCtx` to a private `find_forced_win_inner`. The inner function uses the HandBits overload of `opponent_can_respond` and no longer needs `discard`. Same refactor in `src/players/typed_search/forced_search.cpp::forced_search_recursive` (it still tracks `discard` for eval_table queries at intermediate nodes, but unbeatability checks use the precomputed `opp_bits`).

**Aborted intermediate variant:** First tried precomputing the full `unbeatable[mid]` boolean array (468 entries) at each `forced_search_value` entry. That was a 3.5× regression — the 468-call precompute was too eager when most invocations only check ~30 moves at ~5 recursion levels. Reverted to lazy per-need bitmask checks.

**Files:** `src/core/util.cpp`, `src/players/typed_search/forced_search.cpp`.

**Profile (callgrind, 10 games):**

| Metric | After opt 2 | After opt 3 | Δ vs opt 2 | Δ vs baseline |
|---|---|---|---|---|
| Total instructions | 1,038M | 796M | **−23.3%** | **−31.8%** |
| `opponent_can_respond(arr, arr)` | 17.08% / 177M | absent from top 20 | gone | gone |
| `opponent_can_respond(HandBits, int)` | 3.83% | 4.55% (relative grew; absolute ~36M, similar) | — | — |
| `visit_opp` (self) | 7.33% | 9.41% (relative grew) | — | — |

**Wall clock (1 thread, 1000 games):** 103.8 g/s → **141.1 g/s** (+36% from opt 2; **+86% cumulative from baseline 75.8 g/s**).

**Strength check:** 65.0% vs greedy at seed 100 — unchanged within noise.

**New top hot spot — allocator pressure:** combined malloc/free now ~18% of program total. Per-call vector allocations in `visit_opp` (legal-moves vector, opp_legal, opp_probs, group buckets) and in `compute_legal_moves` are the main contributors. Threadlocal scratch buffers + reservation tuning could halve this.

---

## Optimization 4 — Scratch pool + sort-walk grouping (allocator pressure)

**Two changes in this commit:**

1. Thread-local depth-indexed scratch pool (`SearchScratch` in `typed_search.cpp`) holding reusable `opp_legal`, `opp_probs`, and `groups` buffers per recursion level. RAII `ScratchGuard` pushes/pops. Vectors retain capacity across calls so steady-state pushes don't reallocate.

2. Rewrite `group_opp_moves_into` to use a sort-walk algorithm instead of two `std::unordered_map`s + per-call `vector<Bucket>`. Single pass: build `(combo, rank, orig_idx)` infos in a thread-local scratch vector, sort by `(combo, rank)`, walk consecutive runs to emit groups (collapse over auxiliary for bombs/full-houses, merge consecutive ranks for other combinations when our hand has no in-range response). Output goes into a caller-provided `std::vector<MoveGroup>` (the scratch buffer); recycled `MoveGroup` slots preserve their inner-vector capacity via `clear()` instead of being destroyed.

**Files:** `src/players/typed_search/typed_search.cpp`, `src/players/typed_search/move_grouping.{h,cpp}`.

**Calibration note** — measured both before/after across 5 runs to control for variance. The single-run readings vary by ~10% game-to-game due to RNG-driven exploration depth. Median of 5 runs is the reliable signal.

**Profile (callgrind, 10 games):**

| Metric | After opt 3 | After opt 4 | Δ vs opt 3 | Δ vs baseline |
|---|---|---|---|---|
| Total instructions | 796M | 662M | **−16.8%** | **−43.3%** |
| `_int_free` | 6.78% | 3.24% | −3.5 pp | gone halved |
| `malloc` | 4.93% | 2.29% | −2.6 pp | halved |
| `free` | 3.07% | 1.40% | −1.7 pp | halved |
| Total allocator | ~17.85% | ~9.91% | **−7.9 pp** | (combined malloc/free) |

**Wall clock (1 thread, 1000 games, median of 5 runs):** 141 g/s → **170.6 g/s** (+21% from opt 3; **+125% cumulative from baseline 75.8 g/s**).

**Strength check:** 67.1% vs greedy at seed 100 — unchanged within noise.

**Remaining hot spots** (in order):
1. `visit_opp` (self): 10.39% — recursion + bookkeeping; not much more to squeeze without algorithmic changes.
2. `opponent_can_respond(HandBits, int)`: 6.74% — already the fast overload; bound by table-driven walk over `BeatEntry`s.
3. `main_state_id`: 5.91% — eval-feature encoding; per-leaf computation. Could be cached per memo node (each `OurNode` would compute once, reuse on memo hits).
4. `compute_legal_moves(HandBits, Move)`: 4.39% — central. An `_into` overload that takes a caller buffer (instead of allocating + returning) would let `visit_our` skip its per-call alloc.

The remaining allocator pressure (~10%) is mostly inside `compute_legal_moves` and incidental `std::vector` allocations in `move_grouping_into`'s emit path. The next target with clear payoff is plan #3 (`compute_legal_moves_into` in `src/core/util.cpp`).

---

## Search-node distribution (gen-4 self-play, 200 games / 3711 decisions)

Tablebase fast-path: 4.2% of decisions; the rest go through search.

| Trick context | Decisions | Mean | p25 | p50 | p75 | p95 | Max |
|---|---|---|---|---|---|---|---|
| PASS (we lead) | 1186 | **272** | 22 | 121 | 430 | 911 | 1274 |
| Single response | 1471 | 42 | 1 | 5 | 32 | 264 | 558 |
| Double response | 417 | 6 | 1 | 1 | 5 | 24 | 106 |
| Bomb response | 53 | 1.0 | 1 | 1 | 1 | 1 | 1 |
| Other response | rest | 1-3 | 1 | 1 | 1-2 | 4-13 | 31 |

Heavy skew: initiative leads dominate the search budget; bomb/triple/straight responses are trivial because legal moves are sparse (pass + at most one same-type higher-rank). Singles are moderately expensive due to many candidate beats; pairs less so; everything else essentially constant-time.

By hand size (initiative-only): monotone — hand=1 → 17 nodes, hand=16 → 864.
By opp size (initiative-only): also monotone — opp=1 → 1 node, opp=16 → 720.

**Implication for budget tuning.** Most search cost is concentrated at full-hand opening + mid-game initiative leads. If we ever want to bound per-move cost, capping the initiative-lead branching (e.g., greedy pre-pruning of clearly-dominated lead options) would buy more than capping responses.

---

## Bayesian shrinkage + dominance prune (gen-4 inspection follow-up)

User-reported anomaly. At gen-4 game 0 turn 2, the search ranked candidates for an A-bomb by the auxiliary kicker:

```
hand = [5,5,5,8,9,0,0,K,A,A,A]   (3 fives, 8, 9, two 10s, K, three As)
opp = 11   last move = 77700 (full house)

  1. AAAK   v=0.663
  2. AAA0   v=0.445  (single 10 aux — not loose; breaks the pair of 10s)
  3. AAA00  v=0.445  (pair of 10s aux — loose)
  4. AAA5   v=0.443  (single 5 aux — not loose; breaks the triple of 5s)
  5. AAA8   v=0.434
  6. AAA9   v=0.434
  7. AAA55  v=0.405
```

Counterintuitive: keeping the K (`AAA8` or `AAA9`) should *dominate* discarding it (`AAAK`) — playing the higher loose single only loses optionality. The `eval_inspect` probe showed:

| Move | main visits | main wp | fb visits | fb wp |
|---|---|---|---|---|
| AAAK | 34 | 0.663 | 91 | 0.338 |
| AAA8/9 | 11 | 0.434 | 735 | 0.707 |
| AAA0/00 | 73 | 0.445 | 456 | 0.359 |

`AAAK`'s main entry has only 34 observations, `AAA8/9`'s only 11; both are noisy. The fallback table — coarser features but ~10× more data per entry — points the *opposite* way: `AAAK` is worse than `AAA8/9`. With the original hard threshold (`visit_count ≥ 5` → use main), the noisy main entry shadowed the well-supported fallback.

### Two fixes layered together

**(1) Bayesian shrinkage in `EvalTable::query`.** Replaced the hard threshold with a Beta-Binomial posterior mean:

```
value = (κ · fb_prior + main_total_wins) / (κ + main_visit_count)
```

- `fb_prior` = fallback's `total_wins / visit_count` (when `fb_visits ≥ fb_min_visits`, default 5; otherwise `default_value` 0.5).
- κ = effective prior weight in equivalent visits (default 20).
- Smooth crossover: at `main_visits ≈ κ`, blend is 50/50; at `main_visits = 5κ`, ~83% main.

User's example, after shrinkage with κ=20:

| Move | raw main | shrunken |
|---|---|---|
| AAAK | 0.663 | **0.543** |
| AAA8/9 | 0.434 | **0.609** |
| AAA0/00 | 0.445 | 0.427 |
| AAA5 | 0.443 | 0.426 |
| AAA55 | 0.405 | 0.400 |

Ranking flips: `AAA8/9` overtakes `AAAK`, matching the dominance argument.

**(2) Dominance prune in `visit_our`.** Strict-domination prune of the legal-move set before recursion. Two complementary rules:

- **Single-move dominance.** A `Single` move at rank X is "loose" iff `hand[X] == 1` *and* removing X doesn't change the set of straight / DS / TS moves available (checked by re-running `straight_moves_for_pass_masks_into` on the modified bitmasks). Among loose-single moves in the legal set, the smallest-rank wins; larger ones are dominated and dropped.
- **Aux dominance for bombs / full houses.**
  - For each bomb at rank R, an aux X (single-card with `cost[X] == 1`) is "loose" iff in the post-bomb-base hand `count[X] == 1` and X not in any straight. The smallest loose-single aux per bomb_rank wins; larger loose-single auxes dropped.
  - Pair auxes (only ace bomb has these, plus all full houses) are "loose" iff `count == 2` post-base and the rank is not in any straight / DS / TS. Smallest loose-pair aux wins per `(bomb_rank | triple_rank)`.
- **Not pruned**: bare bombs (no aux), non-loose auxes (where the dominance argument doesn't extend without more analysis).

Both rules are applied at *every* `visit_our` call, not just at the root, so the dominance propagates through the search tree.

Why both fixes together:

- Dominance prune removes the "obviously dominated" moves before the table even sees them — so `AAAK` no longer shows up at all when `AAA8` is loose-and-smaller-rank. The table doesn't need to learn this.
- Shrinkage handles the residual noise where dominance doesn't apply (the bulk of states), keeping the well-evidenced fallback in charge until main accumulates enough data.

### Effects (gen-4 tables, no retraining)

Strength (paired-deal eval, `--p0 typed_search --p0-param 4`):

| Match | gen-4 raw | gen-4 + shrinkage + dominance | Δ |
|---|---|---|---|
| vs random (1000 deals) | 92.3% | **93.3%** | +1.0 pp |
| vs greedy (1000 deals) | 66.8% | **71.3%** | +4.5 pp |
| vs pimc(20) (200 deals) | 55.0% | **58.5%** | +3.5 pp |

Throughput (1 thread, 1000 games, gen-4 tables):

| | g/s | per-game wall |
|---|---|---|
| Before | 141 | 7.10 ms |
| After | **205** | 4.88 ms |

The +45% throughput is the main surprise: I'd expected a small slowdown from the extra check + the shrinkage-side double-lookup, but the dominance prune drops a meaningful fraction of legal moves at most decision points, which removes recursive `visit_opp` calls that would otherwise inflate the search tree.

### Risks / caveats

- The dominance argument assumes "remove a loose card → keep a strictly more capable hand." The check uses straight-set equivalence as the loose-test, which is sufficient but not necessary — there are positions where breaking a straight you'd never want to play *would* still be safe. Conservative: we under-prune, never over-prune.
- Shrinkage with a poorly-calibrated prior (e.g., a coarse fallback whose buckets average over very different positions) can drag well-evidenced main entries toward a worse value. With κ=20 and `main_visits >> 20` the prior gets out-weighted quickly; this should self-correct as gens accrue.
- κ is hand-tuned; unclear whether 20 is optimal across all states. Could be made adaptive (e.g., scale with fallback's own visit count) in a follow-up.

The strength gains carry forward without retraining, but **a fresh multi-gen run is likely to compound the wins** — generations 1+ will accrue main-table data faster on the *correct* states (no longer wasting visits on dominated moves), and shrinkage means smaller per-state samples translate to usable signal sooner.
