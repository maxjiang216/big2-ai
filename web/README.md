# Big 2 — play the AI (web)

A static browser game where you play Big 2 against the **AlphaZero neural-net
player** (the `az_seq` history-transformer + MCTS). Inference runs entirely in
your browser via **ONNX Runtime Web** inside a **Web Worker** — no backend.

The net beats the previous web AI (`typed_search` v5 tables) ~57% head-to-head
and the greedy baseline ~0.74 (paired), validated end-to-end in Node
(`test_play.mjs`).

## Layout

```
web/
  index.html        # markup + loading screen
  style.css
  cardgame.js       # game UI / rules (module); tracks the move-id history, calls the worker
  ai.js             # module: spawns worker.js, exposes window.Big2 (async selectMove)
  engine.js         # data-driven Big2 engine (legality, encoding, composition, grouping)
  mcts.js           # plain-tree MCTS — port of src/players/az_search/az_search.cpp
  opp1.js           # exact opp-1-card endgame solver — port of src/core/opp1_solver
  worker.js         # Web Worker: ONNX Runtime Web (WASM) + MCTS, runs the net per leaf
  model.onnx        # az_seq champion exported to ONNX (nn/export_az_seq_onnx.py)
  az_moves.json     # per-move static table (bin/az_moves_gen) — C++ single source of truth
  series_v_gen5.json # 50x50 series win-prob table v[a][b] (from data/series_v_gen5.csv)
  test_play.mjs     # Node strength harness (onnxruntime-node): MCTS vs greedy/random
  test_opp1.mjs     # Node test: opp1.js solver vs brute-force optimum (node web/test_opp1.mjs)
```

When the opponent holds exactly 1 card, the MCTS pins an **exact** endgame line
from `opp1.js` instead of searching: it brute-forces which single straights to
play, buries low loose singles in bomb auxiliaries, and plays a k-dominating
line wherever one exists (belief-free, optimal in every series state), falling
back to max expected series value otherwise. Handles both leading and responding
(when opp's last move just left them at 1 card). Validated against a brute-force
game-tree optimum (`web/test_opp1.mjs`, `test/test_opp1_solver.cpp`).

The web game is a series (first to 50 pts), so the net runs **series-aware**: the
CPU's / human's running points feed the net's `sides[3]/[4]` (`mpts`/`opts`, ÷50)
and the MCTS win/loss backup values become `series_value_after_win(...)` from
`series_v_gen5.json` instead of a flat 1/0 (port of `az_search` `cfg_.series`).

`model.onnx`, `engine.js`, `mcts.js` and `az_moves.json` are all derived from
the C++ engine, so the browser player cannot drift from `az_search`.

## Regenerating the model + tables

From the repo root:

```bash
uv run python -m nn.export_az_seq_onnx --model models/az_seq.pt --out web/model.onnx
make az_moves_gen && ./bin/az_moves_gen > web/az_moves.json
# series win-prob table (50x50 v[a][b]) -> compact JSON:
python3 -c "import csv,json; v=[[0.0]*50 for _ in range(50)]; [v.__getitem__(int(r[0])).__setitem__(int(r[1]),round(float(r[2]),6)) for r in csv.reader(open('data/series_v_gen5.csv')) if r and r[0][:1].isdigit()]; json.dump(v,open('web/series_v_gen5.json','w'),separators=(',',':'))"
```

## Run locally

```bash
npx serve web        # or: python3 -m http.server -d web 8000
```

Open the printed URL. (A static file server is required — `fetch()` of
`model.onnx` / `az_moves.json` and the worker module won't work from `file://`.)

## Strength harness (Node)

```bash
cd web && npm install        # one-time: pulls onnxruntime-node (dev only)
node test_play.mjs 40 100 greedy   # 40 paired games, 100 sims/move, vs greedy
```

## Deploy on Vercel

Import the GitHub repo, set **Root Directory = `web`**, framework **Other**, no
build command. `web/vercel.json` adds long-lived cache headers for the model +
move table. ONNX Runtime Web's WASM binaries load from the jsDelivr CDN.

## Obsolete

The old `wasm/big2ai.{js,wasm}` + `tables/*.bin` (typed_search v5) are no longer
used and can be deleted to slim the deploy.
