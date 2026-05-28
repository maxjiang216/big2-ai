# az_search — factored / hierarchical player policy head

This documents the player-policy redesign (and the matching training-sample
filtering) layered on top of the base `az_search` feature. The opponent behavior
head and both value heads are unchanged.

## Motivation

The original player policy head was **flat**: one logit per engine move id
(`PLAYER_HEAD_DIM = 457`). Three families dominate that space and are
combinatorially large: full houses (132), bombs (156), straights/sisters/triple-
straights (132). Across those ids the thing that varies is mostly an *auxiliary*
(full-house pair, bomb kicker) or a *length* (straights), and the rare members
(a length-8 sister, a length-13 straight) get a near-untrained logit. A flat head
therefore (a) wastes capacity and (b) can assign a rare-but-strong move a large
negative logit, so the search never explores it.

AlphaZero policy is only a search prior — value + visits dominate near the root —
so coarseness is fine. What matters is **density** (shared parameters, so every
sample trains the shared logit) and **not suppressing rare-but-good moves**.

## The factored head (`PLAYER_HEAD_DIM = 138`)

A concrete move's policy logit is the **sum of shared component logits along its
path** through a family → rank/high → aux/low tree. Layout
(`considered_moves.h`):

| Region | Idx | Count |
|---|---|---|
| pass | 0 | 1 |
| singles (rank) | 1–13 | 13 |
| doubles (rank) | 14–25 | 12 |
| triples (rank) | 26–36 | 11 |
| full-house rank | 37–48 | 12 |
| full-house aux (pair) | 49–60 | 12 |
| bomb entry | 61 | 1 |
| bomb bare | 62 | 1 |
| bomb kicker (rank) | 63–75 | 13 |
| single-straight high (6→2) / low (2→J) | 76–95 | 10 + 10 |
| double-straight high (4→A) / low (3→K) | 96–117 | 11 + 11 |
| triple-straight high (4→K) / low (3→Q) | 118–137 | 10 + 10 |

Composition (`player_path_logits` / `player_composed_logit`):

| Move | Composed logit |
|---|---|
| pass / single / double / triple | that one rank logit |
| full house `(T,P)` | `fh_rank[T] + fh_aux[P]` |
| bomb bare `(R)` / bomb `(R,K)` | `bomb_entry + bomb_bare` / `bomb_entry + bomb_kicker[K]` |
| straight `(type, hi, lo)` | `straight_high[type][hi] + straight_low[type][lo]` |

### Per-family design decisions

- **Full house → `rank (12) + aux pair (12)`.** Rank matters (beatability) and is
  kept; the pair logits are **shared across triple ranks** (the same 12 aux logits
  serve every rank), so aux is modeled densely rather than as 132 near-unique slots.
- **Bomb → `entry (1) + {bare | kicker (13)}`.** Rank is **uniform** (no rank
  logit; the search tries lower ranks first). Bombs are the one family with no
  primary-axis logit, so they get a dedicated **`bomb_entry`** that sets the
  overall bomb-vs-other-type mass — common to every bomb leaf, so it shifts only
  the family mass and cancels within the aux. The bare/kicker logits are shared
  across bomb ranks (the same-rank kicker is masked out).
- **Straights → `high + low`, no type logit.** A straight is `(type, hi, lo)` with
  `length = hi − lo + 1` (2-wrap: 2 is low in `2-3-4-5-6`, high in `J-Q-K-A-2`), so
  **length is implied, never a logit** — rare long ones inherit the dense signal of
  common short ones, and the quadratic length×rank blowup becomes linear. The
  per-type high-card logits also carry the top-level "play this straight type" mass
  (no separate type logit). When responding, length is fixed by the trick, so the
  `low` logits are idle and train almost entirely from initiative positions.
- **DS8 / TS5 need no special handling.** The insta-win check at the top of
  `expand_player` short-circuits any hand-emptying move (DS8 = whole hand) before
  the policy is consulted; TS5 is just one concrete triple-straight a hand holds at
  most once. The old DS8-drop / 1-slot-TS5-collapse special cases are retired.

### Additive composition = nested softmax (the key identity)

The prior is the masked softmax of composed logits over the legal concrete moves.
Because every family/level entry is *derived* (full-house mass from its rank
logits, straight mass from its high logits, bomb mass from `bomb_entry`), this
additive composition is **mathematically identical** to the nested softmax
`P(family)·P(rank|family)·P(aux|family,rank)` with log-sum-exp entries. Two
consequences:

1. **Aux/low logits are genuinely separate.** The conditional distribution over
   pairs given a full-house rank is `exp(fh_rank[T]+fh_aux[P]) / Σ_P'
   exp(fh_rank[T]+fh_aux[P']) = exp(fh_aux[P]) / Σ_P' exp(fh_aux[P'])` — the rank
   term **cancels**. So the aux distribution is a pure softmax over just the aux
   logits, sums to 1 over the legal pairs, and is independent of the rank logit.
   Changing a rank logit only moves mass *between ranks*, never between pairs.
2. **Training is one masked cross-entropy** over the concrete-move space with
   composed logits (`head @ Cᵀ`) — no segmented/nested loss needed. The CE
   decomposes exactly into per-level CEs, so a composite move puts its visit mass
   on every component it satisfies (rank *and* aux).

## Hierarchical search (3-level PUCT)

CPU-side search is effectively free, and grouping prevents wasting visits on near-
identical moves (the anti-correlation we want), so player-node selection descends
the same tree the prior factors over (`build_groups` / `select_player` in
`az_search.cpp`):

- **family → subgroup (rank/high/kicker) → leaf edge.** PUCT at each level uses the
  subtree's exploitation value = **max** over its members (player nodes are max
  nodes: we pick the best in the group) and exploration term from the **summed**
  child visits and the **conditional prior** at that level (`child.prior_sum /
  parent.prior_sum`). So "going into" a full-house rank yields an aux distribution
  that sums to 1 over that rank's legal pairs — the search literally mirrors the
  conditional softmax.
- One edge per concrete legal move (no dedup; distinct aux/low carry distinct
  composed priors). Opponent nodes are unchanged: a flat 2-level most-undersampled
  (`prob/(visits+1)`) selection over response-equivalence classes.

`group_key` for the player is now the **family** (single/double/triple/full-house/
bomb/straight-single/-double/-triple), replacing the old per-length straight split.

## Single source of truth for the composition map

`considered_moves.h` (C++) is authoritative. `research/az_compose_gen.cpp` dumps
the per-move path-logit indices to `nn/az_compose.json` (committed). Python loads
it (`load_compose_matrix` in `model_az.py`) into the binary matrix
`C[468, 138]`; the training loss is `head @ Cᵀ` → masked CE. So the C++ search
(`player_composed_logit`) and the Python loss cannot drift. `az_nn_check` confirms
the raw 138-logit forward matches between C++ and Python to float precision.

## Training-sample filtering (per head)

Governing rule: **record a head's sample only where the search would actually
query that head's NN** — don't train a head on positions it resolves trivially
(it just adds noise / forces the net to cover situations it never sees in anger).
Implemented in `az_selfplay.cpp` via `classify_position` (`PosType`):

| Mover position | Player POLICY | Player VALUE | Opp behavior+value |
|---|---|---|---|
| real decision (≥2 choices) | ✓ | ✓ | ✓ |
| tablebase (opp-1; move pinned, **value used**) | ✗ | ✓ (value-only) | ✓ |
| forced-win combo (root value → 1) | ✗ | ✗ | ✓ |
| insta-win (hand-emptying; terminal before NN) | ✗ | ✗ | ✗ |
| forced pass (node fused in search) | ✗ | ✗ | ✗ |

This tracks the search exactly: `expand_player` short-circuits insta-win to
terminal-1 (no NN) and pins the tablebase move while **still using the NN value**
(value-yes / policy-no); `normalize_forced_pass` fuses passes (no NN); the
opponent net is queried whenever the observer reaches a node in-tree, which it does
for forced-win / tablebase positions (the observer can't tell from public info that
the mover's move was forced) but not for fused passes or the opponent's hand-
emptying terminal. The "opponent leads with 1 card → insta-loss" case is just the
insta-win exclusion from the observer's side.

There is **no "single forced non-pass move" category**: a size-1 legal set is
always `{pass}` (forced pass) or, on the lead with 1 card, the hand-emptying single
(insta-win) — both already classified.

Value-only player samples are recorded with an **empty visit list**; the zero
policy target contributes no policy gradient (the value head trains, the policy
head is untouched), so no `policy_valid` flag is needed.

## What is unchanged

- Opponent behavior head + per-move value head: still flat (`OPP_HEAD_DIM = 458`,
  TS5/DS8 collapses), control-variate backup, opp-takes-our-hand input.
- Player **value** head (scalar), trunk, optimizer/schedule, Parquet schema,
  `az_selfplay` async pipeline, GC, forced-move root extension.
- Existing `data/az_*_genN.parquet` stay valid inputs (move ids unchanged); only
  the player `.pt` model is incompatible with the new 138-dim head and must be
  retrained from gen0. `models/az_player_gen{1,2}.pt` are now stale (457-dim).

## Validation

- `make test_core` — all suites pass, incl. rewritten factored-head mapping tests
  (family + path-logit invariants, 2-wrap endpoints), the nested-group partition
  test, and a composed-prior test (edge priors == masked softmax of composed
  logits, sum to 1). `move_audit` passes.
- C++ ↔ Python forward parity at dim 138: values/logits match to float precision,
  argmax exact.
- Player net retrains cleanly from gen0; `eval_az_match` az(gen0) vs greedy runs
  end-to-end through the factored head + 3-level search.
- `az_selfplay` gen0 run: opp samples exceed player samples by exactly the forced-
  win positions (recorded for opp, not player); value-only rows load with a zero
  policy target and yield a finite mixed-batch loss.

## Files

- `src/players/az_search/considered_moves.h` — factored player head layout +
  `player_family_id` / `player_subgroup_key` / `player_path_logits` /
  `player_composed_logit`.
- `src/players/az_search/az_search.{h,cpp}` — nested `EdgeGroup`; composed-prior
  `expand_player`; family `group_key`; 3-level `select_player`.
- `research/az_compose_gen.cpp` (+ Makefile target) → `nn/az_compose.json`.
- `nn/model_az.py` — `PLAYER_HEAD_DIM = 138`, `load_compose_matrix`.
- `nn/dataset.py` — player targets over the 468 concrete-move space (no collapse).
- `nn/train_az.py` — masked CE on `head @ Cᵀ`.
- `src/datagen/az_selfplay.cpp` — `classify_position` + per-head sample recording.
- `test/test_az_search.cpp`, `research/move_audit.cpp` — updated for the new head.
