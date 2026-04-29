# Player Strategies

All players share a common wrapper (`Player<S>`) that maintains a `PartialGame` view of the board, runs the tablebase check before every decision, and then delegates to an inner `Strategy`. The strategies are described below.

---

## Random

**`RandomStrategy`** — picks a uniformly random legal move, including pass.

This is the weakest baseline. Every legal move is equally likely regardless of its quality. Used for generating training data that covers diverse game states and for worst-case testing.

---

## Greedy (Rules-Based)

All rules-based greedy strategies score positions using a hand-crafted lexicographic evaluation function (`GreedyEval`). After simulating each non-pass move, the resulting hand is scored and the highest-scoring move is played.

### Evaluation function

The `GreedyEval` tuple is compared lexicographically — the first differing field determines the winner:

| Priority | Field | Rationale |
|----------|-------|-----------|
| 1 | `win_now` | Any move that empties the hand is immediately best |
| 2 | `bombs` | Bombs are nuclear options; keeping them is good |
| 3 | `neg_num_cards` | Fewer cards remaining is better |
| 4–15 | `num_2s`, `num_As`, `num_Ks`, … down to `num_4s` | High-value cards are sticky; we prefer to exhaust low ones |

### Variants

**`GreedyStrategy`** — pure greedy. Always plays the lexicographically best move.

**`GreedyRandomStrategy(p)`** — with probability `p`, plays a uniformly random legal move; otherwise plays greedy. Introduces exploration for training data diversity.

**`GreedyRandomPassStrategy(p)`** — with probability `p`, voluntarily passes (if legal); otherwise plays greedy. Models the scenario where occasionally doing nothing is correct. Only passes when responding to a trick (never at lead).

**`GreedyNoBombStrategy(p_bomb)`** — plays greedy, but when greedy selects a bomb, suppresses it with probability `1 - p_bomb`. Suppression preference: pass > greedy non-bomb > bomb anyway. Useful for training against weaker, less bomb-happy opponents.

---

## Greedy (Model-Based)

These strategies replace the hand-crafted `GreedyEval` tuple with a learned scalar score from an ML model trained on self-play outcomes.

### Feature extraction

`extract_tree_features` produces a 60-dimensional feature vector from a `PartialGame`:

- Card counts per rank (13 features)
- Count of cards at or above each rank (11 cumulative features)
- Count of cards at or below each rank (12 cumulative features)
- Highest rank with ≥ 1/2/3/4 copies (4 features)
- Highest rank with ≥ 1/2/3 copies ignoring bombs (3 features)
- One-hot: combination type of last move (pass/single/double/triple/full-house/bomb/straight/double-straight/triple-straight) (9 features)
- Last move card count, total bombs held, possible move count, possible non-bomb move count, trick rank (5 features)

This exact set must match `FEATURE_COLS` in `analysis/train_tree_greedy.py`.

### Tree evaluator (`TreeEvaluator`)

A decision tree trained to predict win probability. File format (text):

```
n_nodes n_features
feature_idx threshold left_child right_child leaf_value
...
```

Leaf nodes have `feature_idx == -2`. Traversal: follow left if `feature[node] <= threshold`, else right.

**`TreeGreedyStrategy`** — simulates each non-pass move, predicts win probability with the tree, plays the highest-scoring move.

### Linear evaluator (`LinearEvaluator`)

A Ridge regression model on a selected subset of the 60 features. Trained via `analysis/train_linear_rollout.py`. File format (text, 6 lines):

```
k
intercept
k feature_indices
k means
k scales
k coefficients
```

Prediction: `intercept + Σ coef[j] * (f[idx[j]] - mean[j]) / scale[j]`

**`GreedyLinearStrategy`** — same as `TreeGreedyStrategy` but uses the linear model for scoring.

**`GreedyPassStrategy`** — at each response turn, runs the Ridge pass model on the current state. If the model predicts a positive delta (passing wins more than playing), passes voluntarily; otherwise falls back to pure greedy. This is the cheapest way to exploit pass timing.

---

## PIMC (Perfect Information Monte Carlo)

PIMC addresses imperfect information by sampling possible opponent hands, then solving each as a perfect-information game.

### Core algorithm

For each of `N` determinizations:

1. **Sample** a plausible opponent hand of the known size, drawn uniformly from unseen cards (not in our hand, not discarded).
2. **Reconstruct** a full `Game` with our cards plus the sampled opponent hand.
3. **For each candidate move** (all legal moves including pass): apply it, then run a **rollout** to terminal state using the rollout policy.
4. **Tally wins** per candidate across all determinizations.

The candidate with the most wins is chosen (ties broken by lower move ID).

All candidates are scored against the **same** sampled hand in each determinization — common random numbers (CRN) — which reduces variance compared to independent sampling.

### Hoeffding early stop (adaptive mode)

After at least `n_min` determinizations, if there is a unique leader whose lower Hoeffding bound exceeds the upper Hoeffding bound of every other candidate (with Bonferroni correction over the `M` candidates), sampling stops early. This saves computation when the winner is already statistically obvious, with a bounded false-stop probability `delta`.

### Rollout policies

**Greedy rollout** (default): at each step, the tablebase is checked first; if it fires, that move is played. Otherwise greedy-eval selects the move. No ML overhead, fast.

**ML rollout** (tree/linear): same structure but the evaluator drives move selection during rollout. Slower but potentially more accurate.

**Pass-aware rollout**: same as greedy rollout but additionally checks the Ridge pass model at each response turn during the rollout. Models the opponent's pass behaviour.

**Re-determinize (`resample=true`)**: before each opponent move during a rollout, re-draw the opponent's hand from the current unseen pool. This prevents the rollout from committing to one fixed opponent hand for its full duration, but may introduce inconsistency (the opponent's moves can be inconsistent with different hands across turns).

### Variants

| Name | Rollout | Adaptive |
|------|---------|----------|
| `PimcGreedyStrategy` | Greedy | Optional |
| `PimcTreeStrategy` | Tree eval | Optional |
| `PimcLinearStrategy` | Linear eval | Optional |
| `PimcPassRolloutStrategy` | Greedy + pass model | Optional |

All variants support the re-determinize flag independently of the adaptive flag.

---

## Player registry

`make_player(name, param, seed)` constructs a `Box<dyn AnyPlayer>` by string name. The `param` meaning depends on the player:

- Greedy variants: probability in [0, 1]
- PIMC variants: number of determinizations (n_max)
- Model-based players: param ignored; models loaded from `data/`

Available names: `random`, `greedy`, `greedy_random`, `greedy_random_pass`, `greedy_no_bomb`, `greedy_linear`, `greedy_pass`, `tree_greedy`, `pimc`, `pimc_redet`, `pimc_adaptive`, `pimc_linear`, `pimc_linear_redet`, `pimc_tree`, `pimc_tree_redet`, `pimc_tree_adaptive`, `pimc_pass_rollout`.
