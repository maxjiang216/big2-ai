#!/usr/bin/env bash
# Round-robin tournament via scripts/round_robin_eval.py (calls bin/eval_match per pair).
# Usage:
#   ./scripts/run_round_robin.sh [path/to/config.json] [extra args to round_robin_eval.py]
# Default config: scripts/configs/round_robin_example.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_ARG="${1:-scripts/configs/round_robin_example.json}"
shift || true

exec python3 "$ROOT/scripts/round_robin_eval.py" "$CONFIG_ARG" "$@"
