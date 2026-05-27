#!/usr/bin/env bash
# Compile the core engine + typed-search AI + JS bridge to WebAssembly, and
# stage the strongest table set (v5) into web/tables/ for the static site.
#
# Prereq: Emscripten installed at $EMSDK (default ~/emsdk).
# Usage: scripts/build_wasm.sh [TABLES_DIR]   (default data/typed_search_v5)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EMSDK_DIR="${EMSDK:-$HOME/emsdk}"
# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1

TABLES_DIR="${1:-data/typed_search_v5}"
OUT_DIR="web/wasm"
mkdir -p "$OUT_DIR" web/tables

SRCS=()
while IFS= read -r f; do SRCS+=("$f"); done < <(find src/core -name '*.cpp')
while IFS= read -r f; do SRCS+=("$f"); done < <(find src/players/typed_search -name '*.cpp')
SRCS+=("src/wasm/bridge.cpp")

echo "Compiling ${#SRCS[@]} sources to WASM…"
em++ -O3 -std=c++17 -DNDEBUG \
  -Isrc/core -Isrc/players -Isrc/players/typed_search \
  "${SRCS[@]}" \
  -s MODULARIZE=1 -s EXPORT_NAME=Big2AI \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 \
  -s FORCE_FILESYSTEM=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS","HEAP32"]' \
  -s EXPORTED_FUNCTIONS='["_ts_load_tables","_ts_select_move","_ts_legal_move_count","_malloc","_free"]' \
  -s ENVIRONMENT=web \
  -o "$OUT_DIR/big2ai.js"

echo "Staging tables from $TABLES_DIR …"
for f in eval_extended eval_main eval_fallback mp_disc mp_main mp_fallback; do
  cp -f "$TABLES_DIR/$f.bin" "web/tables/$f.bin"
done

echo "Done."
ls -la "$OUT_DIR" web/tables
