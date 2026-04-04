#!/usr/bin/env bash
# End-to-end pipeline: greedy self-play parquet → train tree → export C++ model → HTML report.
#
# Usage (from repo root or anywhere):
#   ./scripts/pipeline_tree_greedy.sh
#   ./scripts/pipeline_tree_greedy.sh --games 50000 --output data/greedy_50k
#   CONDA_ENV=big2 ./scripts/pipeline_tree_greedy.sh
#   CONDA_ENV= ./scripts/pipeline_tree_greedy.sh    # use `python` on PATH
#
# Options:
#   --games N          Number of self-play games (default: 200000)
#   --output PREFIX    Parquet prefix for generate_data (default: data/greedy_200k)
#   --depth N          Tree max_depth to train/export (default: 10)
#   --min-samples-leaf N   Passed to train_tree_greedy.py (default: 100)
#   --cv-folds N       CV folds for training (default: 3)
#   --compile          Run `make generate_data eval_match` first
#   --skip-data        Skip bin/generate_data
#   --skip-train       Skip train_tree_greedy.py
#   --skip-export      Skip export_tree_cpp.py
#   --skip-report      Skip generate_tree_report.py
#   -h, --help         Show this help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-big2}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "${CONDA_ENV}" --no-capture-output python "$@"
  else
    python "$@"
  fi
}

GAMES=200000
OUTPUT_PREFIX="data/greedy_200k"
DEPTH=10
MIN_SAMPLES_LEAF=100
CV_FOLDS=3
DO_COMPILE=0
SKIP_DATA=0
SKIP_TRAIN=0
SKIP_EXPORT=0
SKIP_REPORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games|-g)
      GAMES="$2"
      shift 2
      ;;
    --output|-o)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --depth|-d)
      DEPTH="$2"
      shift 2
      ;;
    --min-samples-leaf)
      MIN_SAMPLES_LEAF="$2"
      shift 2
      ;;
    --cv-folds)
      CV_FOLDS="$2"
      shift 2
      ;;
    --compile|-c)
      DO_COMPILE=1
      shift
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-export)
      SKIP_EXPORT=1
      shift
      ;;
    --skip-report)
      SKIP_REPORT=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
End-to-end: greedy self-play parquet → train tree → export C++ model → HTML report.

Usage: pipeline_tree_greedy.sh [options]

  --games N              Self-play games (default: 200000)
  --output PREFIX        Parquet prefix (default: data/greedy_200k)
  --depth N              Tree max_depth (default: 15)
  --min-samples-leaf N   (default: 100)
  --cv-folds N           (default: 3)
  --compile              Run make generate_data eval_match first
  --skip-data / --skip-train / --skip-export / --skip-report
  CONDA_ENV                Default: big2. Set empty to use system python.

Example:
  ./scripts/pipeline_tree_greedy.sh --compile --games 10000 --output data/greedy_10k
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "=== pipeline_tree_greedy (repo: $REPO_ROOT) ==="
echo "  games=$GAMES  output=$OUTPUT_PREFIX  depth=$DEPTH  min_leaf=$MIN_SAMPLES_LEAF"
echo "  CONDA_ENV=${CONDA_ENV:-<empty: use system python>}"
echo

if [[ "$DO_COMPILE" -eq 1 ]]; then
  echo ">>> make generate_data eval_match"
  make generate_data eval_match
  echo
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  echo ">>> Turn feature list (from analysis/train_tree_greedy.py)"
  FEATS="$(run_python -c "import sys; sys.path.insert(0,'analysis'); import train_tree_greedy as t; print(t.TURN_FEATURES_FOR_DATAGEN)")"
  echo ">>> bin/generate_data --player greedy --games $GAMES --output $OUTPUT_PREFIX"
  bin/generate_data \
    --player greedy \
    --games "$GAMES" \
    --output "$OUTPUT_PREFIX" \
    --turn-features "$FEATS"
  echo
else
  echo ">>> Skipping generate_data (--skip-data)"
  echo
fi

PARQUET="${OUTPUT_PREFIX}_turn.parquet"
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  if [[ ! -f "$PARQUET" ]]; then
    echo "Error: missing $PARQUET (run without --skip-data or fix --output)" >&2
    exit 1
  fi
  echo ">>> analysis/train_tree_greedy.py --depths $DEPTH --input $PARQUET..."
  run_python analysis/train_tree_greedy.py \
    --input "$PARQUET" \
    --depths "$DEPTH" \
    --min-samples-leaf "$MIN_SAMPLES_LEAF" \
    --cv-folds "$CV_FOLDS" \
    --no-plots
  echo
else
  echo ">>> Skipping train_tree_greedy.py (--skip-train)"
  echo
fi

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  echo ">>> scripts/export_tree_cpp.py --depth $DEPTH"
  run_python scripts/export_tree_cpp.py --depth "$DEPTH"
  echo
else
  echo ">>> Skipping export_tree_cpp.py (--skip-export)"
  echo
fi

REPORT_DIR="analysis/reports/tree_depth${DEPTH}"

if [[ "$SKIP_REPORT" -eq 0 ]]; then
  echo ">>> analysis/generate_tree_report.py --depth $DEPTH --parquet $PARQUET --out-dir $REPORT_DIR"
  run_python analysis/generate_tree_report.py \
    --depth "$DEPTH" \
    --parquet "$PARQUET" \
    --min-samples-leaf "$MIN_SAMPLES_LEAF" \
    --out-dir "$REPORT_DIR"
  echo
else
  echo ">>> Skipping generate_tree_report.py (--skip-report)"
  echo
fi

echo "=== Done ==="
echo "  C++ model:     data/tree_model_d${DEPTH}.txt"
echo "  Sklearn:       data/tree_model_d${DEPTH}.joblib"
echo "  HTML report:   ${REPORT_DIR}/report.html"
