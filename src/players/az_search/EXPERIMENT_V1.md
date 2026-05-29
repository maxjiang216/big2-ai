# az_search self-play limit test (v1) — infrastructure, calibration, and the gen1 blocker

Status: **infrastructure built + calibrated; first real generation came out broken; experiment halted pending a fix.**

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

## BLOCKER: the first real generation is broken

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
