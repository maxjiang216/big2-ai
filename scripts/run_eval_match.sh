#!/usr/bin/env bash
# Run eval_match from a JSON config, save a log, print output, then analyze (like run.sh for self-play).
# Usage:
#   ./scripts/run_eval_match.sh [path/to/config.json]
# Default config: scripts/configs/eval_match.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_ARG="${1:-scripts/configs/eval_match.json}"
shift || true

exec python3 "$ROOT/scripts/run_eval_match.py" "$CONFIG_ARG" "$@"
