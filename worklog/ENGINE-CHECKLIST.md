# Engine checklist

Everything the engine does that `RULES.md` does not settle: a case the rules are silent
on, a behaviour derived from a proof rather than a quotation, a divergence between two
implementations of the same idea, an edge case rare enough that no game has exercised it.

`RULES.md` is the variant description; `src/core/` is authoritative where they disagree.
This file is for the third case — where both were silent and the code still had to choose.

Tick an item off when a test or a real game confirms it. Items are not bugs; a bug goes in
an entry and gets a regression test. An item here is a place where the next person should
know an assumption is load-bearing.

---

## Two implementations of "what beats what", and they can drift

- [ ] Beat relations are computed in **three** places that must agree:
  `get_beating_moves()` (`src/core/util.cpp:305`, face rank vs face rank),
  `compute_possible_moves` (`src/core/util.cpp:461`, face rank),
  and `compute_legal_moves` (`src/core/util.cpp:222-275`, rank *index* after
  `rank_to_idx`). The third mixed the two spaces for bombs until 2026-08-08.
  **Watch for:** the browser consumes `get_beating_moves()` via `az_moves.json`, the
  search consumes `compute_legal_moves`, and the belief model consumes
  `compute_possible_moves`. Two of the three are now pinned to each other by the
  cross-check in `test/test_legal_moves.cpp` (20 000 random positions, currently zero
  disagreements). **`compute_possible_moves` is still uncovered** — it is the belief
  model, so an error there is invisible to legality and shows up only as the search
  mispredicting what the opponent can do. Extending the same cross-check to it would
  close this item.

## Who goes first, past the spades

- [ ] `RULES.md:27` says: 3♠, then 4♠, "and so on up through the spades", and if all
  spades are in the unused pile, "use 3♥, 4♥, 5♥, etc." — the "etc." is never terminated,
  and the rules do not say what happens if that also fails.
  `src/core/series.cpp:82` scans ranks 3..K only, omitting the ace and 2, and
  `series.cpp:98` falls through to `return 0` — a silent seat bias rather than an error.
  `web/cardgame.js:296` scans only 3♥ through 6♥ (`i <= 6`).
  **Watch for:** the dead pile is 16 cards and the deck holds ~12-13 spades, so
  "all spades unused" is improbable but **not impossible**. The two implementations also
  disagree with each other about how far the fallback runs.

## The opponent-one-card endgame: the browser is ahead of the C++

- [ ] `web/mcts.js:109-120` pins an exact line from the full `opp1.js` solver at **both**
  lead and response positions. `src/players/az_search/az_search.cpp:80-84` still uses the
  straight-blind `opp1_series_move`, gated on `hand_has_straight_lead`. This is deliberate
  and documented (`web/README.md:30-36`), and the C++ `opp1_solver` is currently reachable
  only from tests.
  **Watch for:** the deployed player and the C++ player make **different moves** in this
  position class, so any C++-side strength measurement of the endgame does not describe
  what ships. Porting the solver into `az_search` would close it.

- [ ] `src/core/tablebase_opp1.cpp:69` picks bombs by lowest move id, which is always the
  *bare* bomb. `src/core/opp1_solver.h:16-25` proves the outcome depends only on the count
  of **exposed** singles, and `opp1_solver.cpp:56` deliberately buries loose singles as
  bomb auxiliaries to shrink that set. The default strategy therefore contradicts the
  theory the solver is built on.
  **Watch for:** this is reached from `src/core/tablebase_peek.h:66` — the base-class root
  skip that *every* player goes through, not just tablebase users.

## Claims of exactness that rest on a predicate, not a rule

- [ ] `src/players/az_pi/pi_prune.h` decides which attachments are "dominated" and can be
  discarded, via `loose_single` (line 31) and `loose_pair` (line 42). The solver is
  described as exact; that exactness is only as good as these two predicates. Both must
  answer "could this card still be needed by a straight, sister, or triple-straight I
  hold?" and `RULES.md:63-70` makes that question wider than it looks — in particular
  `2-3-4-5-6` is valid with the 2 as the **low** card (`RULES.md:70`), so rank 2 has
  neighbours at both ends of the rank order.
  **Watch for:** a wrong answer here does not fail loudly. It prunes a winning line, the
  resulting wrong verdict is written to the **process-global, cross-game, cross-thread**
  transposition table (`pi_solver.cpp:60`), and every later query inherits it.

## The value scale under a series objective

- [ ] `src/players/az_search/az_search.cpp:265` and `:455` hardcode `0.5f` as the node
  value when a node has no edges. Under the flat win/loss objective 0.5 is the neutral
  point; under `cfg_.series` the backup values come from `series_value_after_win(...)` and
  the scale is **not** centred on 0.5.
  **Watch for:** whether this path is reachable at all. If it is unreachable it should
  assert rather than return a number; if it is reachable the constant is a systematic bias
  that only appears in series play.

## Confirmed, kept for the reasoning

- [x] **There is no bomb of 2s, and this is correct.** `RULES.md:9` removes three of the
  four 2s, so four-of-a-kind at rank 2 is impossible. The move table holds 12 bare bombs,
  face ranks 3..14 (ids 169, 182, …, 312), where rank 14 is the ace bomb — three aces, per
  `RULES.md:90`, since one ace is also removed. Any loop that indexes bombs `0..12` and
  expects 13 entries is wrong at the top end.
  Verified 2026-08-08 by enumerating `all_moves()`.
