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
