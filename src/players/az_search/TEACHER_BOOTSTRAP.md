# az_search — typed_search teacher bootstrap + opponent-model analysis

This documents (1) bootstrapping gen0 of the az_search nets by imitating the
strongest non-NN player (`typed_search`, the "Shannon" tabular search) instead of
random play, (2) the resulting strength + a search-budget (sims) sweep, and (3) a
deep dive into how well the opponent-behavior head actually models opponent moves
— which surfaced a real, localized calibration issue and pinned down its cause.

All work here sits on top of the factored player-policy head (see
`FACTORED_POLICY.md`); the player policy head is the 138-wide factored head.

## 1. Teacher bootstrap (`az_selfplay --teacher`)

New self-play mode in `src/datagen/az_selfplay.cpp`:
- `--teacher NAME [--teacher-param P]` plays self-play with a registry policy
  (e.g. `typed_search`) via `GameSimulator`, then extracts az samples from the
  `GameRecord` turns under the **same per-head filter as the search path**
  (`classify_position`): player-policy + value at real decisions, player-value-only
  at tablebase, opp behavior+value everywhere except the mover's forced-pass /
  insta-win. The policy/behavior target is the move the teacher actually played
  (one-hot imitation); value targets are the realized game outcome.
- **Start-state mixing was removed** (the `setup_game` mid-game-prefix sampler and
  `--start-frac`). It was low-value, and `GameSimulator` only deals full games, so
  teacher games start from full deals. Random/NN self-play also now start from full
  deals.

Data: 20k typed_search games → 229,850 player + 237,295 opp samples
(`data/az_{player,opp}_gen0.parquet`). Trained 10 epochs, batch 1024 → player val
1.576 (v=0.559, p=1.017), opp val 3.110 (v=0.533, b=2.577).

## 2. Strength of the teacher-bootstrapped gen0 net

Paired-deal eval (`eval_az_match`), 300 games each, seat-swapped, `OMP_NUM_THREADS=1`.

**vs greedy — search budget sweep (same 150 deals/seed):**

| sims | win rate | 95% CI |
|---|---|---|
| 100 | 69.7% | [0.642, 0.746] |
| 500 | 70.0% | [0.646, 0.749] |
| 1000 | 69.7% | [0.642, 0.746] |
| 2000 | 70.0% | [0.646, 0.749] |

**vs other opponents (sims=100):**

| opponent | win rate | 95% CI |
|---|---|---|
| greedy | 69.7% | [0.642, 0.746] |
| pimc(20) | 60.3% | [0.547, 0.657] |
| typed_search (the teacher) | 51.3% | [0.457, 0.569] (tie) |
| typed_search @ sims=1000 | 50.0% | [0.444, 0.556] (tie) |

**Findings:**
- The teacher bootstrap is a big win: **69.7% vs greedy at only sims=100**, vs the
  old random-bootstrap net's 58.3% at sims=100. Better targets front-load strength.
- **Search saturates immediately** — flat ~70% from sims 100→2000 (win counts
  209–210/300, a one-game drift). Opposite of the weak random-bootstrap net, where
  search added +10 pts over 100→1000. With a strong policy prior, the searched move
  ≈ the policy argmax (confirmed below), so extra sims rarely change the decision.
- The net is **~teacher strength**: it ties typed_search (51.3% @ 100, 50.0% @ 1000
  — more search does not let it exceed the teacher) and beats what the teacher beats
  (greedy, pimc). It has distilled the teacher but not surpassed it. Exceeding the
  teacher needs **generational self-play** (real-outcome value signal), not more sims.

## 3. Move agreement: az (net+100-search) vs typed_search

`research/az_vs_teacher_agree.cpp` replays typed_search trajectories and runs the
full az search (100 sims) at each real decision, dumping (teacher, az) move pairs.

- **35.4% disagreement** on the *searched* move (≈ the 36.8% policy-argmax
  disagreement — i.e. 100 sims of search barely overrides the raw policy here).
- Where it comes from (`analysis/az_opp/disagreement_breakdown.py`):
  - **different combo type 36.6%** and **which single (diff rank) 33.9%** dominate;
    76% of the combo-type disagreements involve a *single* (single↔combo).
    → disagreement is concentrated in **single-card management** (which loose card to
    throw, break a combo vs shed a single) — the lowest-leverage axis.
  - **High-leverage decisions are largely agreed**: all bomb-related disagreements
    combined = 3.6% of disagreements; pass-vs-play = 12.8% and roughly balanced;
    full-house/bomb aux = ~2.3% (negligible).
- This explains the flat win-rate: the players are **tactically aligned** (bombs,
  passes, trick-taking) and differ on near-indifferent dump-card style.

## 4. Opponent-behavior head: how well does it model moves?

`analysis/az_opp/behavior_vs_baselines.py` (val split):

| predictor | behavior CE | perplexity |
|---|---|---|
| uniform over legal set (avg 67 moves) | 3.733 | 41.8 |
| marginal move frequency | 3.075 | 21.6 |
| **trained model** | **2.577** | **13.2** |

Top-1 = 24.7%, top-3 = 51.1%. So it beats the marginal 1.6× (it conditions on the
position, not just memorizing frequencies), but the **ceiling is structural**: the
opponent's hand is hidden, so most of the perplexity is irreducible entropy, and
low top-1 is consistent with good *calibration*.

**Why calibration (not top-1) is what matters:** at opponent (chance) nodes the
backup is `V = Σ_a π(a)·[expanded ? child : q_a]` — a π-weighted **expectation**, so
the behavior probabilities must be *calibrated*, not accurate. (Unlike player/max
nodes, where the prior only guides exploration; here a miscalibrated π biases the
value at any sim count.)

### Pass & bomb calibration (`analysis/az_opp/pass_bomb_calibration.py`)

Both **aggregate-calibrated to the decimal** and discriminative:

| event | actual rate | mean predicted | AUC |
|---|---|---|---|
| PASS (responding positions) | 11.3% | 11.3% | 0.784 |
| BOMB (bomb possible) | 2.5% | 2.5% | 0.910 |

Bomb is well-calibrated across all bins (ECE 0.11%). Pass overall ECE 1.2%.

### The one real flaw: localized high-pass over-prediction

Pass reliability is **biased in the high-confidence region**: when the model
predicts P(pass) ∈ [0.7,0.9] the actual rate is only ~0.61 (statistically
significant). Investigation:
- **Not noise / not overfitting.** Train and val show the *identical* bias
  (`pass_calib_train_vs_val.py`): train [0.7,1.0) bin n=869, pred 0.793 → actual
  0.627 (~10σ). The regional non-pass rate 0.37 ± 0.03 is pinned down by the data;
  the model puts only 0.21 non-pass mass. It's an **in-sample underfit** of a
  learnable mean.
- **Not temperature-fixable.** Best-fit T = 1.05 (≈1.0), no improvement — so it's
  conditional miscalibration, not generic overconfidence.
- **What surprises it** (`high_pass_surprises.py`): in those positions the opponent
  occasionally drops a **hidden bomb** (the actually-played move got mean model prob
  0.035, spread over 115 distinct bomb-heavy moves). The model leans "pass" because
  the opponent's *public* max-cards can't beat the trick, missing the concealed out.
- **Cause = underfitting, not "many rare bombs."** At the optimum each bomb prob is
  unbiased and their sum is unbiased (lower variance than any single bomb), so the
  many-classes framing cannot produce a *bias* — it vanishes at the limit. The
  regional mean is well-sampled (324 positives), so this is capacity/convergence
  underfitting (the ~2–3% region contributes little to average loss; OneCycle
  annealed before fitting it; the small trunk under-weights the `opp_max` bomb-risk
  signal), **not** a data-volume problem for the aggregate.
- **Implication for search:** over-predicting pass over-weights the "opponent folds,
  we keep the lead" branch → the value is **optimistically biased exactly when we've
  committed a strong card** (high-leverage). Reducible in principle.

### "More epochs?" and the LR schedule

Training uses **OneCycleLR** (fixed cosine over `total_steps = epochs × batches`),
**no plateau annealing**; the only val-aware behavior is best-val checkpointing.
- Because the schedule depends on total epochs, a 10-epoch and 40-epoch run have
  *different* LR curves — so "10 vs 40 epochs" is confounded by schedule, not pure
  epochs.
- Best-val under a 40-epoch run: player improves to epoch ~16 (val 1.548 vs 1.576;
  policy CE 1.017→0.977) but the **value heads peak ~epoch 10 and overfit after**.
  So more epochs help the policy slightly, not the value.

## Next steps (not done)

- **gen1 self-play** with the gen0 net — the principled lever to (a) exceed the
  teacher and (b) correct the high-pass optimism via real-outcome value grounding.
- Optionally test underfitting directly: bigger trunk and/or a `--sched plateau`
  (ReduceLROnPlateau + early stop) retrain, then re-check the high-pass bin.

## Files

- `src/datagen/az_selfplay.cpp` — `--teacher` mode (`run_classic_selfplay`,
  `classify_position`), start-state mixing removed.
- `research/az_vs_teacher_agree.cpp` (+ Makefile target) — searched-move agreement dump.
- `analysis/az_opp/*.py` — behavior-vs-baselines, pass/bomb calibration, train/val
  calibration, high-pass surprise diagnosis, disagreement breakdown.
