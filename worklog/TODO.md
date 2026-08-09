# TODO

Everything the 2026-08-08 audit found and did not fix. Ordered by tier; within a tier,
by whether something downstream is blocked on it.

Each item names the regression test it needs. Per `worklog/README.md`, a fix lands with a
test that **fails before it and passes after** — a test written against already-fixed code
proves nothing.

**Verified** = reproduced with a probe or direct read during the audit.
**Reported** = from an auditor, not independently confirmed. Confirm before fixing.

---

## Tier 1 — correctness in the authoritative engine

- [x] ~~**`src/core/util.cpp:263` — bomb responses compared a face rank to a rank index.**~~
  Fixed 2026-08-08. Test: `test/test_legal_moves.cpp` (explicit bomb cases + a
  20 000-position cross-check that `compute_legal_moves` agrees with `get_beating_moves`).

- [ ] **`src/players/az_pi/pi_prune.h:31` — `loose_single` misses the `2-3-4-5-6` wrap
  window.** *Reported.* The scan is linear (`lo+4 <= 12`) and never considers
  `{12,0,1,2,3}`, though `RULES.md:70` makes that straight legal with the 2 low. Probe in
  the report: hand `{2,3,4,5,6,9999,K}` marks all of 2/3/4/5/6 loose, so
  `prune_dominated_attachments` deletes `bomb+K` and `bomb+2` and keeps only `bomb+3` —
  the one kicker that destroys the straight. A provable WIN is pruned, and the wrong LOSS
  is written to the **process-global, cross-game, cross-thread** TT (`pi_solver.cpp:60`).
  *Test:* that exact hand, assert the straight-preserving attachments survive the prune;
  plus a solver-level assert that the position is a WIN.

- [ ] **`src/players/az_pi/pi_prune.h:42` — `loose_pair` never checks straight
  participation.** *Reported.* Only rejects a pair whose neighbour rank also holds a pair.
  A pair of 5s in a hand holding `3-4-5-6-7` is judged loose and can be eaten as a
  full-house attachment. Same unsoundness class as above, same solver.
  *Test:* same shape, with a pair.

- [ ] **`src/core/tablebase_opp1.cpp:69` — the default opp-1-card line never buries a
  loose single as a bomb auxiliary.** *Reported.* Picks bombs by lowest move id, always the
  bare bomb, contradicting the proof in `opp1_solver.h:16-25` that only the count of
  *exposed* singles matters. Reached from `tablebase_peek.h:66` — the base-class root skip
  **every** player goes through. Example: opp at 1 card, we hold four 7s + `{3,9,K}`; the
  bare bomb leaves 3 exposed singles (loss) where `bomb+3` leaves 2 (win).
  *Test:* that position, assert the chosen move is the aux bomb.

- [ ] **`src/players/az_search/az_search_player.cpp:45` — `Move(best_move())` with no `-1`
  guard.** *Reported.* `best_move()` returns `-1` on an edgeless root (`az_search.cpp:684`)
  and `Move::Move(int)` does an unchecked `all_moves()[encoded_move]` (`move.cpp:102`) —
  out-of-bounds read. Reachable at `sims == 0`.
  *Test:* construct `Move(-1)` and `Move(LEGAL_MOVES_SIZE)`, assert both throw.

- [ ] **`src/core/hint_compute.h:50` — `thread_local mt19937 rng(random_device{}())`
  destroys reproducibility.** *Reported.* Every other RNG is seeded from `--seed`; anything
  derived from `compute_hint` differs between two runs with identical seeds.
  *Test:* two runs, same seed, assert identical output.

## Tier 2 — silently invalidated experiments

- [ ] **`nn/dataset_seq.py:300-302` — `--series-v` weighting is a no-op.** *Reported,
  strong evidence.* `w = natural[a,b] / emp` divides a rate by a raw count, then clips to
  `[0.1, 10]` **before** normalising to mean 1. On `data/az_games_gen5.parquet` raw weights
  are `[7.4e-5, 7.3e-3]`, all under the floor → **100.00% of weights come out exactly
  1.0**. Every `--series` run trained unweighted while logging that it reweighted.
  *Fix:* normalise to mean 1 first, then clip, then renormalise.
  *Test:* assert the returned weights have variance > 0 and mean ≈ 1 on a fixture.

- [ ] **`models/az_seq.pt` — series input columns are a zero fixed point.** *Verified.*
  `size_embed.weight[:, 3:5]` (`mpts`, `opts`) is exactly 0.0; the ONNX `value` output is
  bit-identical for `sides=[…,0,0]` vs `[…,0.98,0.02]`. Zero input × zero weight = zero
  gradient, so no amount of training recovers them. Any series work must **re-init those
  columns with noise** before warm-start (`nn/train_az_seq.py:170-176`).
  *Test:* a check that no `size_embed` column is all-zero after init.

- [ ] **`src/players/az_search/az_search.cpp` — no Dirichlet root noise anywhere in the II
  search.** *Reported.* `az_pi` has it, `az_search` has none; `cfg_.training` only
  randomises the opponent class representative. Self-play policy targets therefore come
  from PUCT over an un-noised prior and collapse onto the argmax. **Leading suspect for the
  gen-N-ties-champion plateau**, and independent of both series bugs.
  *Test:* two searches, same seed, `training=true`, assert root visit distributions differ.

- [ ] **`src/players/az_pi/pi_search.cpp:499` — the root noise `az_pi` does have mostly
  does not fire.** *Reported.* `apply_root_noise()` is called only from `advance_root` and
  returns immediately on `!root_->expanded`. The constructor-built root — the first move of
  every game — and every re-root into a fresh node get none.
  *Fix:* also call it from `apply_eval` when `n == root_ && cfg_.root_noise`.

- [ ] **`src/players/az_search/az_search_player_factory.h:31` — one `NNEvaluator` with
  `max_slots=1` shared across threads.** *Reported.* `GameCoordinator` workers each call
  `create_player()` then write slot 0 via `nn_->set_prefix(history_)`, mutating
  `kv_pool_`/`plen_`/`raw_hist_` and a `mutable torch::jit::Module`. At `--threads 8`,
  thread A's leaf evals attend to thread B's history. Silent corruption, no crash.
  *Test:* two players from one factory, assert distinct evaluator slots.

- [ ] **`nn/train_az_seq.py:141-150` — cross-generation train/val leakage.** *Reported.*
  The val split is redrawn from the merged mixed-gen dataset each generation while
  warm-starting from a champion trained on ~90% of those same games. Val loss drives
  checkpoint selection, plateau LR **and** early stop.
  *Fix:* derive val membership from a stable hash of `game_id`.

- [ ] **`nn/dataset.py:483-495` — series columns hijack the hand-size slots.** *Reported.*
  When `my_pts` is in the schema, `opp_size`/`our_size` are overwritten with points
  unconditionally, not gated on `--series-table`. Consumers other than `train_az_pi`
  (`train_az.py:174`, `analysis/az_opp/*`) silently feed points into hand-size inputs.

- [ ] **`scripts/run_az_training_loop.sh:222` — `--seed` never forwarded.** *Reported.*
  Every generation trains at `seed=0`, so `mix_decay` keeps the identical subsample of each
  older generation on every rerun.

- [ ] **`src/players/az_pimc/pimc_player_factory.h:51` — `seed_++` on a plain `unsigned`
  from worker threads.** *Reported.* `AzSearchPlayerFactory` uses an atomic; this one races
  and can hand two seats the same seed.

- [ ] **`src/players/az_search/az_search_player.cpp:23` — `current_state()` never sets
  `my_pts`/`opp_pts`.** *Reported.* Under `SearchConfig::series` every terminal is evaluated
  at series state (0,0). Same defect in `az_pimc/det_value.h:95` (`score_worlds`).

## Tier 3 — the build, and CI that cannot see it

- [ ] **Four Makefile object lists are missing torch factories.** *Reported, with link
  errors.* `player_factory_registry.h` gained `az_ii` + `az_pimc` under `BIG2_WITH_TORCH`
  on 2026-06-17; `Makefile:484` (`az_selfplay`), `:545` (`eval_az_pi_match`), `:642`
  (`az_play_check`), `:725` (`az_vs_teacher_agree`) never got
  `$(AZ_II_OBJS) $(AZ_PIMC_OBJS) $(AZ_PI_OBJS)`. **`az_selfplay` and `eval_az_pi_match` are
  on the critical path of the active training loops** — neither runs from a clean `build/`.
- [ ] `Makefile:385` (`game_stats`) missing `$(TYPED_SEARCH_OBJS)`.
- [ ] `src/datagen/generate_nn_data.cpp:67-68` reads `t.hand_at`/`t.opp_at`; renamed to
  `hand_after` in `nn_game_runner.h` in May. Dead target — delete rather than fix.
- [ ] **11 targets missing from `.PHONY`** (`Makefile:32`); `make help` lists 16 of 39 and
  advertises **zero** `az_*` targets while still advertising the broken `generate_nn_data`.

**CI** (`.github/workflows/ci.yml`, untouched since 2026-04-03) builds 1 of 39 targets:

- [ ] Add a `make -k` compile sweep of the no-Arrow/no-Torch targets.
- [ ] Run `nn/test_model_seq.py` — it guards the shipping net's forward path (causal mask,
  KV-cache equivalence, C++ token-feature parity, `torch.jit.script` survival) and has
  **never been executed by CI**. `pytest` isn't even a declared dependency.
- [ ] Run `node web/test_opp1.mjs` — seconds, no model download, and it guards the
  `opp1.js` ↔ `src/core/opp1_solver.cpp` port directly.
- [ ] **Regen-and-diff the three committed derived artifacts** — `az_moves_gen` →
  `web/az_moves.json`, `az_compose_gen` → `nn/az_compose.json`, `az_tokens_gen` →
  `nn/az_token_cards.json`. Cheapest high-value check available; guards the exact
  C++/browser boundary this audit found a bug on the other side of.

## Tier 4 — the shipped web game

- [ ] **`web/cardgame.js:162-177` — `checkFullHouse` rejects ace-triple full houses.**
  *Reported, strong evidence.* The `sorted[2].number !== 1` guard blocks `AAA+xx`, which
  `RULES.md:60,136` explicitly allows (ids 158-168). A sweep of all 467 non-pass moves
  found these 11 are the **only** engine moves the UI rejects. When the CPU plays one,
  `playType` returns `[-1,-1,-1]`, `currType` goes garbage, and `validPlay` then rejects
  every legal higher full house — the human can only bomb or pass for the rest of the
  trick. *Test:* `web/test_play.mjs`, assert `playType(AAA+33)` is well-formed.

- [ ] **`web/worker.js:43` — no `seq_cap` guard; the shipped ONNX hard-throws at T ≥ 66.**
  *Reported, verified against the model:* T=65 OK, T=66/70/90 throw
  `Add … broadcast 66 by 67`. `mcts.js:291` builds `history.concat(pathTokens)` with no
  cap. The C++ guards this twice (`nn_eval.cpp:45` throws, `forward_leaf` clamps); the
  Python `forward_full` (`model_az_seq.py:191`) does not.
  *Fix:* clamp in both `evalLeaf` and `forward_full`. *Test:* T=70 returns a value.

- [ ] **`web/ai.js:41-44` — `init()` can never fail.** *Reported.* Polls a `ready` flag on
  `setInterval` with no error path and no `worker.onerror`; `worker.js:105` posts
  `{type:'error'}` which is only `console.error`d. Blocked CDN → loading bar stuck at 40%
  forever, `boot()`'s catch never runs.

- [ ] **`web/cardgame.js:788-793` — `runCpuTurn` has no `try/catch` around `cpuChoose()`,
  and re-checks `gamePhase`/`turn` only *before* the await.** *Reported.* Any worker error
  strands the board on "CPU is thinking…"; "New Game" mid-flight applies a stale reply to
  the fresh deal. *Fix:* try/catch + a generation counter.

- [ ] **`web/vercel.json:6` — `immutable, max-age=31536000` on unversioned `model.onnx`
  *and* `az_moves.json`, as independent cache entries.** *Reported.* A user can end up with
  a new move table against a cached old model → wrong or illegal CPU moves, no error.
  `series_v_gen5.json` is not in the pattern at all, so the 22 KB table revalidates every
  load while the 5 MB model never does. No CSP, no `X-Content-Type-Options`.

- [ ] **`web/opp1.js:18,73` — straight-family test has no upper bound.** *Reported.* JS uses
  `combo >= 6`, C++ `opp1_solver.cpp:15` uses `c >= 6 && c <= 14`. Combos 15-25 (sisters,
  triple straights — 90 moves) are pulled into straight-packing enumeration. Wasted work
  (~5 ms per `solveResponse`, called per legal response at every `oppSize===1` leaf) and
  risks truncating real packings against `NODE_CAP`.

- [ ] **`web/cardgame.js:409` + `ai.js:48-50` — unrecognised plays silently become a lead
  pass.** *Reported.* `moveIdForCards` returns `?? kPASS`; `lastMoveId = 0` reads as *lead*
  to `engine.js:47`, and a `kPASS` token enters `moveHistory` — a pass at a lead position is
  not in the C++ grammar, so the transformer gets an out-of-distribution sequence. Latent
  today; it is the amplifier that turns any future UI/table mismatch into silent corruption.
  *Fix:* throw instead of coercing.

- [ ] `web/index.html:11,22` — nav still says "vs. typed-search AI"; loader claims
  "~54 MB of trained tables" for a ~5.5 MB payload.

## Tier 5 — performance

Ranked by expected payoff.

- [ ] **`nn/export_az_seq_onnx.py:97` — `dynamic_axes` pins batch to 1**, forcing
  `worker.js` into 100 sequential single-row WASM inferences per move. *Reported as the
  dominant browser cost.* Add `{0: "B"}` to all inputs/outputs and batch leaves in
  `mcts.js`.
- [ ] **No AMP anywhere in training.** bf16 applies directly to the seq/ii transformer
  trunk, ~2× wall-clock. **Caveat:** `nn/train_az.py:48` `NEG = -1e30` overflows fp16 —
  bf16 only, or lower NEG.
- [ ] **`src/players/az_pi/pi_solver.cpp:60` — 512 MB TT allocated and zero-touched at
  static init** in every binary linking `pi_solver.o`. Measured: `./bin/test_core` peaks at
  530 MB RSS purely for this. OOMs any CI box under 1 GB. Make it lazy + configurable.
- [ ] `nn/dataset_seq.py:157-160` per-row Python gid mapping + needless `to_pandas()`;
  `:216-219` int64 `[G,64,13]` temp (~530 MB at G=80k) to cumsum into int16; `:350-355`
  whole dataset uploaded to GPU twice (train and val loaders each `.to(device)`).
- [ ] `src/players/az_search/az_search.cpp:402,208` — `compute_possible_moves` run twice
  per opponent state (once in `normalize_forced_pass`, again in `expand_opp`). Hottest
  function in the II search.
- [ ] `az_search.cpp:279-311, 318-388` — `build_groups` reallocates the nested group tree
  per expansion, and group visit sums are recomputed by scanning every edge at every
  selection step. `az_pi` already flattened both (`pi_search.h:71-93`); port it.
- [ ] `az_search.cpp:232` — uses allocating `group_opp_moves` where `group_opp_moves_into`
  exists precisely for this hot path.
- [ ] `src/core/util.cpp:289` — `compute_legal_moves_into` is `out = compute_legal_moves(…)`,
  i.e. it allocates, contradicting its own doc comment (`util.h:117`). Only tests call it.
- [ ] `src/core/game_record.cpp:46` — `get_possible_moves()` computed every turn of every
  recorded game whether or not a feature reads it (~6k ops + an allocation per turn);
  `:41-51` `TurnRecord` is copied rather than moved and the whole 200k-game batch is
  retained before Parquet.
- [ ] `src/core/game.cpp:82` — two asserts inside the per-rank loop of `apply_move`, the
  innermost operation of `pi_solver::rec`, with no `-DNDEBUG` in `CXXFLAGS`.
- [ ] `src/core/opp1_solver.cpp:263` — `solve_response` calls the full `solve_lead`
  (a 200k-node DFS over `std::set<std::array<int,13>>`) once **per legal response**.
- [ ] `web/mcts.js:258-263` — `resolveChild` calls `azTransition` twice per child, two full
  state clones, to read a value that is always `1 - parent.st.side`.
- [ ] `web/ai.js:19` + `worker.js:31-35` — `az_moves.json` (345 KB) fetched, parsed and
  built into a 468-entry `Map` **twice**, once per thread.
- [ ] `web/worker.js:44,45-58` — boxed-BigInt intermediate array and 3 typed arrays +
  4 tensors allocated per leaf, 100×/move. Only `tokens` needs a fresh tensor.

## Tier 6 — dead code, hygiene, docs

Dead with evidence (see the audit entry; ~40 files):

- [ ] Legacy `nn` line: `nn/model.py`, `nn/train.py`, `nn/benchmark_inference.py`,
  `nn/az_nn_check.py`, `scripts/run_training_loop.sh`, its four binaries
  (`generate_nn_selfplay`, `eval_nn_match`, `eval_nn_vs_classic`, `play_games`),
  `src/datagen/generate_nn_data.cpp`, `src/simulation/nn_game_runner.*`.
- [ ] WASM path: `src/wasm/bridge.cpp`, `scripts/build_wasm.sh`.
  **Keep `src/players/typed_search/*`** — it is CI-covered and is the gen0 teacher
  (`az_selfplay.cpp:574`).
- [ ] All of `tools/` — no Makefile target at all; `eval_inspect.cpp` no longer compiles
  (`forced_search_value` signature drifted).
- [ ] `src/datagen/typed_search_train.cpp` (35 KB) + both `typed_search_*.sh`.
- [ ] 13 of 24 `scripts/configs/*.json`; ~15 analysis scripts.
  **Keep `nn/train_az.py`** — dead as a CLI, live as a library (`_device`, `_make_optim`,
  `_save_scripted`, `masked_log_softmax` are imported by all three newer trainers).

Hygiene:

- [ ] **`.gitignore:126,133` — trailing comments on negation patterns.** *Verified with
  `git check-ignore -v`.* gitignore has no inline comment syntax, so
  `!assets/*.png  # Keep assets folder images` never fires. `assets/big2-ai-cards.png` is
  tracked only because it predates the rule. Move comments to their own lines.
- [ ] **`projects/standard-linter/` is an unregistered submodule** — a `.git` file pointing
  at `gitdir: ../../.git/modules/…` with no `.gitmodules`. Move it out of the tree or
  register it. `black --check .` would lint it if committed.
- [ ] `models/` (199 MB) is ignored only incidentally via `*.pt`; add an explicit rule.
  `logs/` and `samples/` are not ignored at all.
- [ ] `.gitignore:45 *.txt` is repo-wide; `research/gemini-research.txt` survives by
  force-add. Narrow it.
- [ ] `analysis/sample_games.md` is in the index despite `.gitignore:118`.

Docs:

- [ ] **`README.md` and `CLAUDE.md` describe a repo that no longer exists** — neither
  mentions `nn/`, `web/`, `models/`, or `az_search`/`az_seq`/`az_pi`/`az_ii`. The layout
  tree at `README.md:26-38` omits `nn/` and `web/`, so the deployed app and both active
  pipelines are invisible from the front door. `CLAUDE.md` also says 472 move ids; there
  are 468.
- [ ] **`src/players/az_search/EXPERIMENT_V1.md` contradicts shipped reality** — "self-play
  does NOT improve this architecture… the only lever is search depth at PLAY time", against
  `MEMORY_V1.md` (2026-06-17) whose rewrite produced the champion. Four of the six docs
  there pre-date the rewrite. Fold them in or move to `research/` with a superseded header.
- [ ] `src/README.md` lists only `random/` and `greedy/` under players, and 4 of 17
  `datagen/` sources. `src/players/README.md` documents no NN player.
- [ ] `web/README.md:38-41` claims the running points "feed the net's `sides[3]/[4]`" —
  true of the wiring, false of the effect. `web/worker.js:8` says the opposite of
  `worker.js:51`. Both need correcting alongside the series work.
