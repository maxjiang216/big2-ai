# Player strategies (C++)

All concrete players derive from `Player`. Each holds a `PartialGame` view of the board: after `accept_deal`, state advances through `accept_opponent_move` / `select_move`. `select_move()` checks the opponent-has-one-card tablebase first, records which case fired when applicable, then runs the subclass implementation.

**Main families:** uniform **random**, rule/tuple **greedy** (`greedy`), and **PIMC** (`pimc`, `double` parameter = determinization count / samples). Other registered names are **variants** (randomized greedy, ML evaluators, alternate PIMC rollouts); see `make_player_factory` in `player_factory_registry.h`.

---

## Random

Uniform random among **legal** moves (including pass). Weakest baseline—used for diversity in data generation and sanity checks.

---

## Greedy (rules-based)

Default greedy scores each candidate non-pass move by simulating it and evaluating the resulting hand with **`GreedyEval`** (`greedy_player.h`). The tuple is ordered lexicographically (via `std::tie`); larger is better.

| Priority | Field | Rationale |
|----------|-------|-----------|
| 1 | `win_now` | Finishing the hand wins immediately |
| 2 | `bombs` | Prefer keeping bomb material |
| 3 | `neg_num_cards` | Fewer remaining cards is better |
| 4–15 | `num_2s`, `num_as`, … `num_4s` | Shed low ranks first; high ranks are stickier |

### Variants (brief)

- **`greedy_random` / `greedy_random_pass` / `greedy_no_bomb`** — stochastic tweaks for training or weaker opponents (`player_factory_registry.h`).
- **`greedy_linear` / `greedy_pass` / `tree_greedy`** — model-based greedy (below).

---

## Greedy (model-based)

Same “simulate non-pass moves, pick best score” shell; scoring uses a learned evaluator instead of `GreedyEval`.

**Features** for tree/linear models follow `extract_tree_features` / `FEATURE_COLS` in `analysis/train_tree_greedy.py` (60-D vector from a `PartialGame`).

- **`TreeGreedyPlayer`** — decision tree on win probability; default weight path pattern `data/tree_model_d{depth}.txt` (factory passes depth).
- **`GreedyLinearPlayer`** — Ridge on a feature subset; weights `data/linear_rollout_w.txt` (see `analysis/train_linear_rollout.py`).
- **`GreedyPassPlayer`** — Ridge pass-vs-play Δ using `data/pass_ridge_w.txt`, then falls back to tuple greedy.

---

## PIMC (perfect-information Monte Carlo)

Imperfect information is handled by **determinizations**: sample a plausible opponent hand consistent with public counts, build a full `Game`, then for each legal candidate move run rollouts to the end. Count wins per candidate; pick the highest (ties broken by lower move id). Candidates share the **same** sampled hand per determinization (variance reduction).

**Adaptive / Hoeffding early-stop** (`pimc_adaptive`, `pimc_tree_adaptive`, …): optional stopping when a leader’s lower confidence bound clears all others’ upper bounds (Bonferroni over candidates).

**Rollouts:** greedy (default for `pimc`), tree (`pimc_tree`), linear (`pimc_linear`), pass-aware (`pimc_pass_rollout`). **Re-determinization** variants (`pimc_redet`, …) resample the opponent hand during rollouts.

---

## Factory API

`make_player_factory(name, param, seed)` returns a `PlayerFactory`; `param` is ignored for `random` / `greedy`, is **probability** for randomized greedy factories, **sample count** for PIMC-style names (default often 10 if you pass `0`), and **tree depth** for `tree_greedy`. Full list and semantics: comments in `player_factory_registry.h`.

---

## Empirical strength (three pairings)

From the repo root after `make eval_match`, **`bin/eval_match`** runs **paired deals** (each deal played twice with seats swapped so luck cancels; total games = 2 × `--deals`). **`pimc(20)`** is `--p0-param 20` with greedy rollouts (standard `pimc`).

| Matchup | P0 win rate | 95% Wilson CI |
|---------|-------------|----------------|
| `greedy` vs `random` | 82.30% | [81.22%, 83.33%] |
| `pimc(20)` vs `random` | 93.14% | [92.41%, 93.81%] |
| `pimc(20)` vs `greedy` | 73.66% | [72.42%, 74.86%] |

Settings: `--deals 2500`, `--seed 42`, `--threads 12`. **Ordering:** pimc(20) > greedy > random.

Reproduce:

```bash
make eval_match
./bin/eval_match --p0 greedy --p1 random --deals 2500 --seed 42 --threads 12
./bin/eval_match --p0 pimc --p0-param 20 --p1 random --deals 2500 --seed 42 --threads 12
./bin/eval_match --p0 pimc --p0-param 20 --p1 greedy --deals 2500 --seed 42 --threads 12
```
