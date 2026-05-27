# Big 2 — play the AI (web)

A static browser game where you play a 50-point series of Big 2 against the
**typed-search AI** (the v5 search + table player, ~60% vs PIMC-20). The C++
engine is compiled to WebAssembly and runs entirely in the browser — no
backend.

## Layout

```
web/
  index.html        # markup + loading screen
  style.css
  cardgame.js       # game UI / rules (pure JS); calls the WASM AI for CPU moves
  ai.js             # loads the WASM module + tables, exposes Big2.selectMove()
  wasm/big2ai.{js,wasm}   # compiled engine (built by scripts/build_wasm.sh)
  tables/*.bin            # v5 trained tables (~54 MB, downloaded on first load)
```

## Build the WASM AI

From the repo root (requires Emscripten at `$EMSDK`, default `~/emsdk`):

```bash
scripts/build_wasm.sh                 # uses data/typed_search_v5 by default
scripts/build_wasm.sh data/typed_search_v5
```

This regenerates `web/wasm/big2ai.{js,wasm}` and copies the table set into
`web/tables/`. Commit the regenerated artifacts.

## Run locally

```bash
npx serve web        # or: python3 -m http.server -d web 8000
```

Open the printed URL. (A static file server is required — `fetch()` of the
`.wasm` and `.bin` files won't work from `file://`.)

## Deploy on Vercel

Import the GitHub repo, then set **Root Directory = `web`**, framework
**Other**, no build command. `web/vercel.json` adds long-lived cache headers
for the WASM and table files so repeat visits load instantly.
