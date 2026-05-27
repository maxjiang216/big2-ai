# az_search — open questions / deferred simplifications

Self-play (`src/datagen/az_selfplay.cpp`) and the search core work and are
validated end-to-end, but four things were deliberately simplified. None are
correctness bugs — they're throughput / coverage / memory tradeoffs. Each below
is a decision to make; fill in **Decision:** when ready.

---

## 1. Single inference thread vs double-buffer CPU/GPU pipeline

**Now:** one thread, strictly alternating (a) CPU phase — pump every slot's search
to collect leaf requests, apply moves, record samples — and (b) NN phase — flush
the player/opp batches. They're serial: the GPU idles during the CPU phase and
vice-versa.

**Full version** (`src/datagen/generate_nn_selfplay.cpp`): two game pools on two
CPU threads interleaved against one GPU stream, so CPU work for pool B is hidden
inside the GPU inference for pool A (~50% throughput on GPU per its header).

**Impact:** mostly on GPU runs (keeps the GPU saturated). On CPU, torch intra-op
threads already parallelize each forward, so the loss is smaller.

**Cost:** moderate — port the barrier/double-buffer threading; resolve shared-
`NNEvaluator` concurrent-`forward` safety (dedicated inference thread, or one
module per worker).

**Recommendation:** defer until we generate on GPU at scale.

**Decision:**

---

## 2. No out-of-tree NN-eval cache in self-play

**Now:** the batched loop calls the evaluator directly; it does NOT use
`CachingEvaluator` (`eval_cache.h`, used by the step-5 player). The transposition
DAG inside each search already dedups within one search, so there is no
*intra-search* redundancy.

**Missed:** dedup *across* searches/slots/turns — e.g. two games' opponent nodes
with the same public-only opp-net input. Also, even within a single flushed batch
I don't collapse duplicate inputs before the forward pass.

**Impact:** extra forward passes (wasted compute, identical results). Modest.

**Cost:** low for in-batch dedup (hash inputs, eval uniques, scatter back);
moderate for a persistent shared cache (shared map + thread-safety if combined
with #1).

**Recommendation:** do the cheap in-batch dedup; consider a shared cache later.

**Decision:**

---

## 3. Start-state sampler: random-forward vs from-scratch sampling

**Now:** `setup_game` deals a full random hand then plays 1–12 uniform-random
legal moves forward and records from there. Consistent by construction; legal.

**Full version** (plan §5): sample a mid-game state directly — both hands size
k ≤ 16, a random current trick — subject to deck consistency
(`hand0[r]+hand1[r]+discard[r] ≤ max_per_rank(r)`) and last-trick consistency
(the last move's cards sit in the discard, jointly consistent with both hands).
Purpose: cover positions humans / other bots create that random play rarely
reaches.

**Impact:** random-forward states are a SUBSET of all legal mid-game states —
narrower off-policy coverage (won't produce hoarded-bomb hands, odd size
imbalances, etc.). Minor for pure self-play; matters for robustness vs outside
opponents.

**Cost:** moderate and fiddly — constrained sampling is easy to get subtly wrong
(illegal/inconsistent states).

**Recommendation:** keep random-forward unless we specifically target vs-human/
vs-bot robustness; then add the from-scratch sampler with a legality assert.

**Decision:**

---

## 4. No subtree pruning (memo grows over a game)

**Now:** `advance_root` re-roots but keeps the whole transposition memo; every
node created during a game stays in the arena until `finalize_game` frees both
seat trees at game end. The plan wanted transposition-aware mark-and-sweep to
free nodes unreachable from the new root.

**Impact:** memory only. Peak ≈ `slots × 2 × game_length × sims × node_size`
(hundreds of MB to low GB at slots=256, sims=200). Bounded per game-in-flight
(freed at game end) and tunable via fewer `--slots` / `--sims`.

**Cost:** highest of the four — DAG pruning must not free a node still reachable
via a different parent (refcount / mark-and-sweep over shared nodes).

**Recommendation:** rely on the `--slots` bound; implement pruning only if peak
RSS becomes a problem.

**Decision:**

---

## Priority if scaling up
1 (threading) and the cheap slice of 2 (in-batch dedup) first — they're
samples/sec. 3 is data coverage; 4 is memory.
