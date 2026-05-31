# az_search self-play limit test (v1) — infrastructure, calibration, and results

Status: **COMPLETE. gen1 blocker fixed; experiment ran. Verdict: self-play does NOT
improve this architecture past the teacher-bootstrapped gen0 at any search budget
(sims 1/10/100). The only lever that moves strength is search depth at PLAY time.**

## RESULTS (2026-05-31)

Ran the 3-arm limit test from the shared gen0. Completed clean: **sims=1 — 10/10
gens, sims=10 — 10/10 gens, sims=100 — 6/10 gens** (gen 7 self-play wedged when the
laptop was suspended — see "Self-play hang" below; cross-play never ran). That is more
than enough to settle the question.

**Zero promotions in any arm, ever.** Every generation tied gen0 (vs-champion win rate
≈ 0.47–0.48, Wilson lower bound never cleared 0.5), and every panel metric is flat
across all 10 (resp. 6) generations — no upward trajectory anywhere. Representative
end-of-run numbers (win rate, eval at each arm's own sims, 1000 deals):

| vs | sims=1 | sims=10 | sims=100 |
|---|---|---|---|
| random | 0.898 | 0.910 | 0.93 |
| greedy | 0.633 | ~0.648 | ~0.685 |
| pimc(20) | 0.522 | ~0.546 | ~0.603 |
| typed_search | 0.420 | ~0.448 | ~0.466 |
| champion (= gen0) | 0.476 | ~0.472 | ~0.476 |

**Two conclusions:**
1. **Self-play value-grounding is inert for this method.** Across generations the net
   does not climb and never beats gen0 — at sims=1 by construction (policy self-
   distills, the option-A fix keeps it frozen), but *also* at sims=10 and sims=100
   where real MCTS visit distributions are available. The architecture has plateaued
   at teacher-bootstrap quality; more self-play generations don't help.
2. **Search depth at play/eval time is the only thing that helps.** Holding the net
   fixed, raising eval sims 1→100 buys ≈ **+5pp vs greedy** (0.633→0.685) and ≈ **+8pp
   vs pimc** (0.522→0.603). This is a larger search payoff than the ~+2pp measured
   earlier in the session (a better-calibrated net makes search worth more), but it is
   a *play-time* lever, not a training-loop lever.

This is the firm negative result the experiment was designed to find: it points to
the deferred **search-v2 GRU redesign** (better nets, history-conditioned opponent
model) as the path forward, not more self-play on the current nets.

### Self-play hang

sims=100 gen 7 self-play wedged: it logged **26009s** (vs ~210s for gens 1–6), GPU
idle at ~6%, no forward progress, until killed. **Most likely cause: the laptop was
suspended / a shutdown was attempted mid-run, freezing the CUDA context** (the process
never recovered on resume). SIGTERM let it flush its parquet (so gen7 *data* exists but
is from the wedged run — excluded from the numbers above; the gen7 `.pt` is a
half-trained orphan, ignore it). gens 1–6 at sims=100 plus the full 10+10 gens of the
sims=1/10 arms all ran fine, so there is **no evidence of an intrinsic pipeline
deadlock** — treat this as a suspend/resume casualty. (If a future unattended long run
hangs again *without* a suspend event, revisit the async CPU/GPU pipeline, Commit C.)

---

## (historical) infrastructure, calibration, and the gen1 blocker

## Goal

Test how far the *current* az_search method can go via self-play, and whether the
self-play search budget matters. Three runs distinguished by self-play sims
(**1 / 10 / 100**), each starting from the shared teacher-bootstrapped gen0 and
self-playing forward **10 generations**; plateau (closed-loop) training each gen;
per-gen **batched panel eval** (random / greedy / pimc(20) / typed_search /
current champion) reporting overall win rate + the **primary sweep-ratio metric**
(winning both games of a paired deal); final **cross-play** of the 3 best models.
The old no-search Big2Net was dropped from the panel (it's the broken `v_init`
baseline — low signal).

## Infrastructure built (committed)

- **`bd2e6c0`** — pre-encoded data path + plateau + batched eval:
  - `az_selfplay` emits **pre-encoded `enc`** (`encode_exact(hand)|encode_thermo(opp)|encode_exact(trick)`, 144 floats) + ragged `legal` (player: concrete ids; opp: head slots) + targets, replacing per-rank count columns. Train/infer feature parity is by construction (same `encode_*` as `nn_eval`).
  - `dataset.py` loads once into memory (dense `enc`; sparse `legal`/`visit` as flat+offsets) and densifies mask+policy **per-batch in a collate** — zero per-item Python, RAM-safe under mix-decay.
  - `train_az.py` `--sched plateau` (constant LR → ReduceLROnPlateau, early-stop at `--lr-floor`) + `--ckpt-metric head` (checkpoint on the policy/behavior head CE alone).
  - `eval_az_match` single-threaded **batched pool** (all `2*deals` games concurrent, az NN leaf-evals batched across slots; CPU opponents via the Player interface inline; az-vs-az via a 2nd evaluator). Matches the play path (`az_definitive_move` root-skip + `finalize(&ev)`). Adds the **sweep-ratio metric**.
- **`12d4013`** — `scripts/run_az_experiment.sh` driver; `eval_az_match --sims-b` (B's search budget for cross-play).
- **`1d9c105`** — `GpuLoader`: GPU-resident training (whole dataset on-GPU, gather + densify per batch on-device, no DataLoader/collate/H2D).

## Calibration (measured, this machine: 6 GB GPU)

- **Self-play throughput** (cuda, slots 4096): sims=1 50k games = **24s** (~2080 g/s); sims=10 ≈ 40s; sims=100 ≈ 210s. → ~46 min total across 3 runs × 10 gens.
- **Training**: GPU-resident **130s** per gen-pair (1.16M samples) vs DataLoader **509s** (**3.9×**), val bit-matching. The (B,lr) sweep showed time is **batch-invariant** (517s@2k ≈ 509s@8k) → training was **host-pipeline-bound, not compute-bound**; that's why GPU-resident (not bigger batch) is the fix. **Locked: batch 8192, lr 6e-4, gpu-resident=auto.**
- **Eval** (batched, @sims100, 1000 deals): greedy 19s, pimc(20) ~50s, typed_search ~35s, champion (az-vs-az) 42s → **~165s/gen** panel.
- **Firmed end-to-end ≈ 3 hr** (self-play ~46m + training ~65m + eval ~65m + cross-play).
- **Gotcha**: the C++ binaries (`az_selfplay`, `eval_az_match`) on cuda need
  `LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cu13/lib` (dynamic
  nvrtc load; not covered by rpath) or they abort with a kernel-source dump. The
  driver script exports it; direct launches must too.

## RESOLVED: the gen1 collapse was a sims=1 self-play degeneracy

**Root cause.** `select_leaf()`'s first call expands the *root* itself and that eval
counts against the sim budget, so at `--sims 1` the single eval is spent on the root
and **no child is ever visited**. Two downstream failures followed:
1. `root_visits()` (edges with `child->N > 0`) returned **empty** for ~74% of
   decisions (the 26% non-empty were roots pre-expanded by subtree reuse → 1 visit).
   The empty rows densified to an all-zero policy target → contributed **zero** policy
   gradient, so the player head trained only on a biased one-hot 26% subset and lost
   gen0's calibrated prior. Low train CE (0.0816) was an illusion: 74% of rows had a
   trivially-zero target.
2. `sample_from_visits` returns **kPASS** when the visit list is empty → at sims=1 the
   self-play mover *passed on every non-forced decision*, so the games (and thus the
   value targets) were garbage too. Confirmed in data: opp behavior CE was an
   artificially low 0.8354 because the opponent almost always passed.

**Fix (option A).** Added `Search::root_prior()` (root edge priors as pseudo-counts,
forced-win one-hot when present). In `az_selfplay` `complete_decision`, when
`root_visits()` has <2 moves, fall back to the prior for **both** the sampled move and
the recorded policy target. At sims=1 this is self-distillation: the policy stays at
gen0 level (frozen, no collapse), games are real, and the value head grounds on real
outcomes. sims=10/100 keep real visit distributions.

**Verified** (corrected gen1, 50k games, sims=1, plateau-trained): visit_moves per-row
mean length 0.26 → **8.41** (3.2% empty = the legit forced/value-only positions); opp
pass-frac of plays ≈ 6.9%; player policy CE 0.0816 → **0.9384**, opp behavior CE
0.8354 → **2.4021** (both now genuine distribution CEs). Eval @sims=1: **gen1 vs
greedy 0.192 → 0.634** (≈ gen0's 0.682); **gen1 vs gen0 0.200 → 0.477** (≈ tie, as
option A predicts). Changes: `src/players/az_search/az_search.{h,cpp}` (`root_prior`),
`src/datagen/az_selfplay.cpp` (degenerate-visits fallback).

## (historical) BLOCKER: the first real generation is broken

With the experiment launched, **gen1 of the sims=1 run** (trained from scratch on
1.16M gen0-self-play samples via GPU-resident plateau; train completed in 128s,
player policy val CE **0.0816**, opp behavior **0.8354**) evaluated as:

| matchup (@sims=1, batched eval) | A win rate |
|---|---|
| **gen0** (teacher champion) vs greedy | **0.682** ✓ (sound) |
| gen1 vs random | **0.441** (loses to random!) |
| gen1 vs greedy | **0.192** |
| gen1 vs gen0 champion | **0.200** |

So **the eval is correct** (gen0 plays as expected, az-vs-az ties 0.500), but the
**gen1 net plays actively badly** despite a "good" (low) training loss. The bug is
in the **new training path or the self-play data** — not the eval, and not the
search. The leading suspicion is the *delta from gen0's pipeline*: gen0 was trained
on the OLD path (Python-encoded counts) and is fine; gen1 is the first net trained
on the NEW path (C++ pre-encoded `enc` + `GpuLoader`).

### Hypotheses to test next (fan-out)

1. **GPU-resident training bug (most actionable).** Retrain gen1 from the *same*
   `data/az_s1_{player,opp}_gen1.parquet` with `--gpu-resident off` (DataLoader)
   and eval vs greedy @sims1. If `off` ≈ 0.66 and `on` ≈ 0.20 → a `GpuLoader`
   full-training bug (the 2-epoch tiny densify check matched bit-for-bit, so it
   would be something that only manifests over a long run — shuffle, an aliasing
   bug across the concatenated mix tensors, or a value/size column mixup).
2. **Self-play data corruption.** Inspect the gen1 parquet: value-target mean
   (~0.5?), `visit_moves`/`counts` validity (one-hot at sims=1? ids in 0–467?),
   `enc` ∈ {0,1} shape 144, `legal` ranges. Compare to a fresh teacher-bootstrap
   sample.
3. **Train/infer `enc` parity.** Feed gen0's net the parquet `enc` rows directly;
   if gen0 gives sane policy/value on them, the pre-encoded representation matches
   inference; if not, the `enc` layout (training) ≠ `nn_eval` (inference).
4. **From-scratch on sims=1 one-hot degeneracy.** Low CE may be the net fitting the
   dominant easy positions (forced pass) with a degenerate argmax. Check gen1's
   policy entropy / argmax distribution vs gen0 on sample positions. *Counter-
   evidence*: gen0 at sims=1 already plays 0.68, so a net imitating its one-hot
   argmax should land ~0.68, not 0.20 — which points more at a code/data bug (#1/#3)
   than at this.

### Artifacts left for debugging
- `models/az_s1_{player,opp}_gen1.pt` (the broken net), `models/az_s1_*_champion_*.pt` (= gen0).
- `data/az_s1_{player,opp}_gen1.parquet` (the gen1 self-play data).
- `logs/az_experiment/run_s1.log`, `logs/az_experiment/driver.log`.
- gen0 (good) = `models/az_player_gen0.pt` / `models/az_opp_gen0.pt`.

## Next steps

1. Run the **#1 discriminator** (gpu-resident off vs on retrain) + the **#2/#3**
   data/enc-parity checks. Fix the bug.
2. Verify a single gen reaches ~teacher level (~0.66 vs greedy) before relaunching.
3. Relaunch `scripts/run_az_experiment.sh` (~3 hr) once a gen is sane.
4. Consider **warm-starting** each gen from the champion (vs train-from-scratch) —
   standard, more sample-efficient, and a robustness lever once the bug is fixed.
