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
  worker.js         # Web Worker: ONNX Runtime Web (WASM) + MCTS, runs the net per leaf
  model.onnx        # az_seq champion exported to ONNX (nn/export_az_seq_onnx.py)
  az_moves.json     # per-move static table (bin/az_moves_gen) — C++ single source of truth
  test_play.mjs     # Node strength harness (onnxruntime-node): MCTS vs greedy/random
```

`model.onnx`, `engine.js`, `mcts.js` and `az_moves.json` are all derived from
the C++ engine, so the browser player cannot drift from `az_search`.

## Regenerating the model + tables

From the repo root:

```bash
uv run python -m nn.export_az_seq_onnx --model models/az_seq.pt --out web/model.onnx
make az_moves_gen && ./bin/az_moves_gen > web/az_moves.json
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
