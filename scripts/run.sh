#!/usr/bin/env bash
# Run Parquet generation then analysis report. Usage:
#   ./scripts/run.sh [path/to/config.json]
# Default config: scripts/configs/greedy.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_ARG="${1:-scripts/configs/greedy.json}"
if [[ "$CONFIG_ARG" == /* ]]; then
  CONFIG="$CONFIG_ARG"
else
  CONFIG="$ROOT/$CONFIG_ARG"
fi

python3 scripts/generate_data.py "$CONFIG_ARG"

OUTPUT="$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('output_path',''))" "$CONFIG")"
if [[ -z "$OUTPUT" ]]; then
  echo "Config has no output_path; skipping analysis.py (Parquet paths unknown)." >&2
  exit 0
fi

GAME="${OUTPUT}_game.parquet"
TURN="${OUTPUT}_turn.parquet"
if [[ "$OUTPUT" != /* ]]; then
  GAME="$ROOT/$GAME"
  TURN="$ROOT/$TURN"
fi

SAMPLES_HINT="$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('samples_md_path',''))" "$CONFIG")"
if [[ -n "$SAMPLES_HINT" && "${SAMPLES_HINT:0:1}" != "/" ]]; then
  SAMPLES_HINT="$ROOT/$SAMPLES_HINT"
fi
if [[ -z "$SAMPLES_HINT" || ! -f "$SAMPLES_HINT" ]]; then
  if [[ -f "$ROOT/analysis/sample_games.md" ]]; then
    SAMPLES_HINT="$ROOT/analysis/sample_games.md"
  else
    SAMPLES_HINT=""
  fi
fi

ARGS=(--game-parquet "$GAME" --turn-parquet "$TURN" --output-dir "$ROOT/analysis/")
if [[ -n "$SAMPLES_HINT" ]]; then
  ARGS+=(--samples-md-hint "$SAMPLES_HINT")
fi

python3 analysis/analysis.py "${ARGS[@]}"
