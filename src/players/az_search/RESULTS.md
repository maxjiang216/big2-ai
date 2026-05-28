# az_search — results, timing, and scores

AlphaZero-style best-first search + two neural nets for the 2-player Shanghainese
Big 2 engine. The thesis: the NN supplies positional judgement (including discard
reading) that the tabular typed-Shannon search lacked, and the search supplies the
concrete calculation that the search-less `Big2Net` lacked — so the combination
should break past where each alone plateaued (~62–63% vs greedy).

This doc covers the work done, correctness, timing/throughput, and win rates as of
2026-05-27 (branch `az-search`).

## What was built

Steps 1–6 (prior): move-set audit, the two PyTorch nets (`Big2NetAZ` player,
`Big2NetOpp` opponent), C++ LibTorch inference, the transposition-DAG search core
(hierarchical PUCT at our nodes, undersampling at opponent nodes, expectimax
backup, fused forced-passes, tablebase, subtree reuse), the player/factory/registry,
and batched self-play + paired-deal eval.

This session resolved the deferred design decisions and added the training loop:

| Commit | What |
|---|---|
| `ac79be2` | Opponent net takes our hand (inputs identical to the player net) + a **per-move value head**; opponent-node backup is now an unbiased **control variate** (`V = Σ_a π(a)·(expanded ? child : q_a)`, no renormalization) instead of the biased prior-mass-renormalized average. |
| `bd4e73f` | **In-edge-refcount GC** for the search DAG: re-rooting eager-cascades the freed subtree (worklist), so memory stays bounded across turns. |
| `69ec26f` | **Async self-play pipeline**: W CPU worker threads (disjoint slot blocks, single-threaded search) + one inference thread (sole `nn.eval_*` caller, batches across all slots, flushes at `batch_target` or tail-drain). |
| `0ab48e0` | **Generational training loop** `scripts/run_az_training_loop.sh`: gen0 bootstrap → self-play → mixed train → eval vs champion (promote on Wilson lower bound > 0.5) + vs greedy. |
| `2972c3e` | **NN-valued forced-move expansion** at the root: play our provably-unbeatable moves and raise the root value to the max over **every** forced-reachable node (not just leaves); override the move when a forced line beats the searched value. Self-play keeps win-proof only. |
| `1491387` | `--device cpu|cuda` for self-play (GPU inference). |
| `7f48b90` | Training loop runs self-play on GPU by default. |

Design decisions worth recording:
- **Opp net takes our hand** (dropped the public-only constraint) rather than
  hand-engineering response-class buckets — the net learns response-class and
  hand-quality effects itself; forfeits the opp-eval cache, judged low-value.
- **Bitsets, bigger batches, model-compile, encoding-offload were all tried and
  measured as no-ops** for self-play throughput (see Timing); reverted.

## Correctness

- `make test_core && ./bin/test_core` — all suites pass, including the rewritten
  control-variate backup test (+ a regression guard against renormalization), the
  GC re-root test (shared transposition survives, discarded subtree freed/recycled),
  and the forced-move-expansion isolation test (`sims=0` so only the extension acts).
- `az_nn_check`: C++ inference matches the Python forward pass to float32 tolerance
  (last-digit only) on both heads of both nets; argmax exact.
- `eval_az_match` and `az_selfplay` run end-to-end on CPU and CUDA.

## Timing / throughput

Self-play is **NN-forward-bound**. Profiling (single inference thread + 19 CPU
workers, 1000 games):
- inference thread ≈ **94% of wall**; CPU workers ≈ **2% busy** (search/enumeration
  is essentially free — the async design hides it behind the NN).
- within an inference batch: GPU forward + host↔device transfer + copy-back ≈
  **6.3s**, vs CPU tensor stacking (alloc + copy + encode) ≈ **0.1s**.
- the forward is **overhead-bound, not compute-bound** (tiny net; ~1 ms/batch is
  kernel launch + transfer + sync). Batch size sits ~350 at 2048 slots and grows
  with slot count, so larger generations get bigger batches (better amortization)
  for free.

Measured rates:

| Config | Rate / time |
|---|---|
| Self-play, GPU, `--slots 4096`, sims 100 | **~167 games/s** (5000 games in ~30s) |
| Self-play, GPU vs CPU (1000 games) | ~9.4s vs ~25.4s → **~2.7× GPU speedup** |
| Per generation (5k games, sims 100, 5 epochs, eval) | self-play ~30s + train ~80s + eval → **~3.5 min/gen** |
| Eval (`eval_az_match`, single-game, batch-1, OMP=1) | sims 100: 23s / 60 deals · sims 1000: 2m41s / 60 deals (~2.7 s/deal, ~45 games/s) |

CUDA inference needs the venv CUDA libs on `LD_LIBRARY_PATH` (for nvrtc, which the
JIT fuser compiles at startup); the training-loop script sets this.

## Scores (win rate vs greedy)

All paired-deal (seats swapped, same shuffle) so card luck cancels.

| Approach | vs greedy |
|---|---|
| greedy (self) | 50% |
| `Big2Net` — NN, **no search**, 120 gens | ~62% (range 58–68%, peak ~68%) — *plateau* |
| typed-Shannon search — tabular eval+search, gen3 | ~63% (53 → 59 → 62 → 63 over gen0–3, still climbing) |
| pimc(20) | ~mid-50s (beat `Big2Net` ~50–54%) |
| **az_search gen1 (toy), sims 100** | **57–58%** |
| **az_search gen1 (toy), sims 1000** | **68.3%** [0.596, 0.760] |

**Key signal — same gen1 net, same 60 deals, only the search budget changed:**

| Search | vs greedy |
|---|---|
| sims = 100 | 70/120 = **58.3%** [0.494, 0.668] |
| sims = 1000 | 82/120 = **68.3%** [0.596, 0.760] |

+10 points from search alone. 68% already **matches the `Big2Net` peak and clears
both ~62–63% plateaus** — with a gen1 net trained only on random self-play data.
This is direct evidence the search adds real strength on top of the net.

Quick-check generational run (5k games/gen, sims 100, slots 4096, 2 gens): gen1/gen2
each beat greedy ~57%, but came out ~50% vs the gen0 champion → no gen-over-gen gain
at this toy scale, so the promotion gate correctly held them back.

**Caveats:** the 68.3% is 60 deals (wide CI, overlaps the sims-100 CI at the edges),
and the nets are toy (gen1, 5k games, 5 epochs). These are encouraging directional
signals, not a converged verdict.

## Status & next step

All code is committed on `az-search`; tests pass. The remaining work is a **proper
multi-generation run** (e.g. 50–100k games/gen, sims 200, more slots, 8–10 gens) to
see whether a *trained* net + search climbs past the ~63–68% ceiling. For a fresh
run, clear `models/az_*` and `data/az_*` (the loop auto-resumes from the highest
existing generation), then:

```bash
scripts/run_az_training_loop.sh --start 1 --end 8 --games 50000 --sims 200 \
    --slots 8192 --epochs 5 --eval-deals 400 --eval-sims 200 --device cuda
```
