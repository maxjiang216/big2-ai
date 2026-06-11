# MEMORY_V1 — history-transformer rewrite of az_search

Date: 2026-06-09. Branch: `az-search`. Commits: `45c611e`, `adc6d5b`, `da394db`.

## Why

EXPERIMENT_V1 (see `EXPERIMENT_V1.md`) ended with a firm negative: self-play
never improved the old two-MLP architecture past the teacher-bootstrapped
gen0, at any search budget. The only lever that moved strength was search
depth at play time. The hypothesis going forward: the nets were
*information-starved*, not search-starved. The old inputs (hand, opp-max
thermometer, current trick, sizes) summarize the game's history losslessly
for card counting but throw away the *behavioral* content of the move
sequence — what the opponent chose to play and, just as telling, when they
passed. The head expected to gain the most is the opponent-behavior model.

This iteration adds **memory**: the net conditions on the full move history
through a causal transformer.

## Architecture

### One net instead of two

The old player net (policy + value) and opponent net (behavior + q_a) merged
into a single `Big2NetSeqAZ` (`nn/model_az_seq.py`). The key observation
enabling the merge: at *every* leaf the search evaluates, the exact hand fed
to the net is the **searcher's** — the mover's hand at our-turn leaves, the
observer's at opponent-turn leaves. So one trunk + one readout serves all
four heads, with a single scalar input `owner_to_move` distinguishing the
two regimes:

- `owner_to_move = 1`: search reads (policy 138, value) — factored player
  policy composed via the existing `C[468, 138]` matrix, value =
  P(hand-owner wins).
- `owner_to_move = 0`: search reads (behavior 458, q_a 458) — opponent
  imitation logits and per-move P(owner wins after opp plays a), used for
  the control-variate backup exactly as before.

The value head is supervised only at owner-to-move rows (the search never
reads it elsewhere; opp rows are grounded through q_a).

### Trunk: causal transformer over move tokens

- Token = the move's 13 rank counts in the standard 48-dim exact-count
  encoding (pass = all zeros). The 468×13 table is dumped from C++
  (`make az_tokens_gen` → `nn/az_token_cards.json`) — single source of
  truth; **C++ only ever sends integer move ids across the boundary**,
  which eliminates a whole class of encode-parity bugs.
- Learned BOS token at index 0 + learned positional embeddings.
  `seq_cap = 66`: the maximal game is 61 moves (winner plays 16 singles,
  loser 15, passes fill the strict alternation; proved during planning),
  capped at 64 + BOS + slack.
- Pre-LN, d_model 128, 3 layers, 4 heads, FFN 256, GELU (all configurable).
- No per-token seat tags: strict turn alternation means seat identity is a
  function of position parity, which the positional embedding carries.
  Perspective enters only at the readout (whose hand + owner_to_move).
- Readout = hidden state at the decision's sequence index, concatenated
  with the card-literacy embeddings of hand (exact 48) and opp_max
  (thermo 48) plus (sizes/16, owner_to_move) → 256 junction → 2×ResBlock →
  heads. The derived summaries stay as side inputs deliberately: making a
  small transformer re-derive 48-card counting over 60 tokens wastes
  capacity, and opp_max is free at inference (C++ already computes it).

**Index convention (the single most off-by-one-prone item):** sequence =
`[BOS, move_0, move_1, …]`; a decision taken after `hist_idx` applied moves
reads sequence index `hist_idx` (BOS when 0). `hist_idx` counts ALL applied
moves including forced passes and insta-wins — it is NOT the sample
`turn_idx`.

## Search: transpositions are gone

History-conditioned evals make state-key sharing *incorrect*: two paths
reaching the same (hand, opp_size, trick, side) carry different histories
and legitimately get different evals. The former transposition DAG
(`memo_`, `state_key`, in-edge refcounts, the out-of-tree eval cache) is
deleted outright. The search is now a plain tree:

- Edges own their children. Each simulation resolves at most one new node,
  so live nodes ≤ sims + 1 (tested).
- `select_leaf()` accumulates the move-id path from root to leaf; the
  forced-pass fusion (`normalize_forced_pass`) marks its edge `fused_pass`
  so the token suffix is `{move, kPASS}` — the token stream a leaf sees is
  exactly the move stream a real game would record (tested by replaying
  captured token streams through `az_transition`). At most one fuse per
  transition: after a fused pass the mover holds the lead, where passing is
  never forced.
- `advance_root(true_next, new_history)` walks the real-game move suffix
  edge by edge (consuming fused passes). Full match → the played-out child
  becomes the root, its subtree and visits survive, everything else is
  freed. Any divergence (opponent representative-class mismatch,
  unexpanded line) discards the whole tree — a partially reused
  history-conditioned tree would be wrong.
- The NN-valued forced-move extension threads hypothetical token lines too:
  each provably-unbeatable move appends `{m, kPASS}` for its eval.

Cost of losing transpositions: transposed states get re-evaluated and their
statistics are no longer shared — an effective-sims haircut that can be
compensated by raising `--sims`. A duplicate-eval debug counter to quantify
it precisely is deferred.

## Inference: root-prefix KV cache

The expensive part of a transformer eval is the history prefix, and within
one real turn *every* leaf eval shares it. `NNEvaluator` therefore keeps a
device-resident KV pool (`[slots, layers, 2, heads, seq_cap, head_dim]`,
fp16 by default — ~100KB/slot):

- `set_prefixes(slot_ids, histories)`: once per real turn per game slot,
  one batched `encode_prefix` writes the prefix K/V + BOS-line readout
  hidden into the pool.
- `eval_batch(slot_ids, feats)`: runs `forward_leaf` over only the in-tree
  path tokens (typically 5–20) attending to [cached prefix + path], with
  per-row prefix-length masking so mixed-length slots batch together.
- `--no-kv-cache` switches every consumer to full-sequence recompute — the
  correctness reference used by the equivalence checks.

TorchScript surface (`config / encode_prefix / forward_leaf / readout /
forward_full`) is hand-rolled SDPA (not `nn.TransformerEncoder`, which
cannot take external KV); the fp16 pool is cast inside the net.

In the async self-play pipeline, the first leaf request of each decision
carries a history snapshot; the inference thread runs all prefix refreshes
as one batch before that flush's evals. In `eval_az_match` the refresh is
synchronous at decision setup, because `finalize()`'s forced extension can
evaluate before any batched flush happens (zero-eval decisions on fully
reused subtrees) — that ordering bug was caught during implementation.

## Data + training: per-game sequences

Per-sample history prefixes would duplicate O(T²) tokens. Instead each
generation emits a parquet **triple**:

- `az_games_genN.parquet`: game_id → full move-id list (+ winner,
  first_player). The whole token store for 100k games is a few MB.
- player/opp files: as before plus `hist_idx`, `owner_seat`, raw hand
  counts; the pre-encoded 144-float `enc` column is dropped. `opp_max` is
  recomputed in Python from move list + hand (exact twin of
  `features.h::opp_max_counts`); stored sizes are cross-checked against a
  full replay of the move list at load time.

Training (`nn/train_az_seq.py` + `nn/dataset_seq.py`) batches **games**:
one causal trunk forward per game, hidden states gathered at every sample's
`hist_idx`, one readout pass for all player+opp rows — ~T× cheaper than
per-sample prefixes. The loader is device-resident (the old `--gpu-resident`
path and the CPU debug path are literally the same code on different
devices).

`Player` gained an `on_self_move` hook (tablebase root-skips must enter the
history too); `AzSearchPlayer` tracks the move history and refreshes its
evaluator prefix each turn.

## Old-vs-new evaluation

The pre-memory champion stays playable: `LegacyNNEvaluator` drives the old
two-net TorchScript pair through the new unified `Evaluator` interface
(ignoring path tokens, reading hand/opp_max/trick/sizes — its exact old
inputs), dispatching player-net vs opp-net on `owner_to_move`. So
`eval_az_match --model-a NEW --legacy-player-b OLD_P --legacy-opp-b OLD_O`
is a clean paired-deal comparison where both sides run the identical new
search core.

## Verification (all green)

| Check | Result |
|---|---|
| `make test_core` (incl. rewritten az tests: tree bound, token streams, fused-pass tokens, suffix advance_root) | PASS |
| `nn/test_model_seq.py` (causality, KV≡full w/ mixed lengths + fp16 pool, BOS alignment, scripted roundtrip) | PASS |
| `az_nn_check`: C++ KV path ≡ C++ full path ≡ Python forward (incl. empty + forced-pass histories) | agree ≤1e-5 |
| `az_selfplay --slots 1` same seed, KV vs `--no-kv-cache` | 9/10 identical move sequences (one fp16 tie-flip), winners 10/10 |
| Threaded NN self-play (slots 16), schema validation (`scripts/validate_az_seq_data.py`) | PASS |
| 1000-game typed_search teacher smoke: train 8 epochs | policy CE 1.47 (uniform-over-legal ≈2.5); 0.500 vs greedy @ sims 50 — sane for 2% of normal data |
| `eval_az_match` vs classic + vs legacy champion | runs; legacy champion crushes the tiny smoke model, as expected |

## Gen0 results (2026-06-10) — memory ablation

Full-scale gen0 ran: `az_selfplay --teacher typed_search --games 50000`
(574,517 player / 592,718 opp sample rows, max history 41), then
`train_az_seq --epochs 30 --batch-games 512` (onecycle, seed 42).

The key question — *does history-memory improve opponent-behavior
prediction?* — answered with a clean ablation: the **same** data / split /
seed / code, re-trained with `--no-memory` (zeros the trunk hidden before
the junction; identical side-input pathways, transformer gets no gradient).

| head (val loss) | memory-OFF | memory-ON | Δ |
|---|---|---|---|
| value | 0.544 | 0.522 | −0.022 |
| policy | 0.915 | 0.928 | +0.013 |
| **behavior** | 2.572 | **2.441** | **−0.131** |
| q_a | 0.520 | 0.508 | −0.012 |

Memory's payoff lands exactly where hypothesized: **opponent behavior**,
−0.131 nats (perplexity 13.1 → 11.5, ~12% fewer effective guesses). Value
and q_a improve modestly (history aids hidden-info win-prob inference).
Player **policy is flat-to-slightly-worse** (+0.013, within noise) — the
mover's own hand + legal set already carry most of what policy needs;
history adds little there. This isolates memory's contribution from any
data-source/scale confound (the old two-net opp baseline b≈2.55 was on
different, smaller self-play data and is not a clean control).

The `--no-memory` model (`models/az_seq_gen0_nomem.pt`) is a diagnostic
artifact, not a champion. Champion = memory-on gen0 (`models/az_seq.pt`).

## Next steps

1. ~~**Full-scale gen0**~~ — done (see above). Champion = gen0.
2. **The headline comparison**: new gen0 vs old champion
   (`models/az_s100_champion_{player,opp}.pt`) via `--legacy-*-b`, plus
   both vs typed_search at matched sims. This isolates what memory buys at
   equal data/teacher.
3. Self-play generations via `scripts/run_az_training_loop.sh` — the
   question EXPERIMENT_V1 answered negatively gets re-asked with memory:
   does the opponent-behavior head, now history-conditioned, finally give
   self-play something to learn?
4. Deferred: duplicate-eval counter (transposition-loss cost), KV pool
   dtype/slot tuning under real GPU load, prefix-encode latency profile.
