#!/usr/bin/env bash
# Multi-generation training + evaluation harness for typed_search.
#
# For each generation N in [0, NUM_GENS]:
#   gen 0:  bootstrap from random self-play
#   gen N>0: typed_search self-play using gen (N-1) tables
# After training, snapshot tables to data/typed_search_v$N and run a set of
# eval_match comparisons (random, greedy, pimc(20), and previous gen).
# Also write a few sample games to data/typed_search_runs/gen$N/samples/.
#
# Usage:
#   scripts/typed_search_train_gens.sh [NUM_GENS] [GAMES_PER_GEN]
# defaults: NUM_GENS=4, GAMES_PER_GEN=50000

set -e

NUM_GENS=${1:-4}
GAMES=${2:-50000}
THREADS=${THREADS:-18}
ALPHA=${ALPHA:-0.7}
DEALS_FAST=${DEALS_FAST:-1000}     # for cheap baselines
DEALS_PIMC=${DEALS_PIMC:-200}      # pimc is slow; fewer deals
SAMPLE_GAMES=${SAMPLE_GAMES:-5}

ROOT=data/typed_search_runs
SUMMARY=$ROOT/summary.txt

mkdir -p $ROOT
: > $SUMMARY

# Helper: parse eval_match output and append a one-line summary to $SUMMARY.
parse_eval () {
  local label="$1"
  local out="$2"
  local p0_wins=$(echo "$out" | grep -E 'P0 .* wins:' | head -1 | grep -oE '[0-9]+ / [0-9]+' | head -1)
  local pct=$(echo "$out" | grep -E 'P0 .* wins:' | head -1 | grep -oE '\([0-9]+\.[0-9]+%\)' | head -1)
  local breakdown=$(echo "$out" | grep -E 'P0_sweep=|P0 sweep 2-0:' | head -1)
  printf "  %-40s %s %s\n" "$label" "$p0_wins" "$pct" | tee -a $SUMMARY
  echo "    $breakdown" | tee -a $SUMMARY
}

# Helper: clean and ensure a directory.
fresh_dir () {
  rm -rf "$1"
  mkdir -p "$1"
}

echo "Typed-search training run: $NUM_GENS gens × $GAMES games (threads=$THREADS, alpha=$ALPHA)" | tee -a $SUMMARY

for gen in $(seq 0 $NUM_GENS); do
  echo "" | tee -a $SUMMARY
  echo "=== GEN $gen ===" | tee -a $SUMMARY
  GEN_DIR=$ROOT/gen$gen
  fresh_dir $GEN_DIR
  SAMPLE_DIR=$GEN_DIR/samples
  mkdir -p $SAMPLE_DIR

  if [ $gen -eq 0 ]; then
    POLICY=random
    # Seed gen0 from empty tables. Don't decay (no prior).
    GEN_ALPHA=1.0
    # Make working tables dir empty for this run.
    fresh_dir data/typed_search
  else
    POLICY=typed_search
    GEN_ALPHA=$ALPHA
    # Copy previous gen tables into the working dir so typed_search loads them.
    PREV=$ROOT/gen$((gen - 1))
    fresh_dir data/typed_search
    cp -f $PREV/*.bin data/typed_search/ 2>/dev/null || true
  fi

  echo "Training: policy=$POLICY games=$GAMES alpha=$GEN_ALPHA" | tee -a $SUMMARY
  ./bin/typed_search_train \
    --policy $POLICY --games $GAMES --threads $THREADS --alpha $GEN_ALPHA \
    --seed $((1000 * gen + 1)) --tables-dir data/typed_search \
    --sample-games $SAMPLE_GAMES --sample-dir $SAMPLE_DIR \
    2>&1 | tee -a $SUMMARY

  # Snapshot trained tables into versioned dir AND a typed_search_vN dir
  # that the eval_match factory can pick up via --p?-param N.
  cp -f data/typed_search/*.bin $GEN_DIR/
  V_DIR=data/typed_search_v$gen
  fresh_dir $V_DIR
  cp -f data/typed_search/*.bin $V_DIR/

  # Evaluations.
  echo "" | tee -a $SUMMARY
  echo "Evaluations (gen $gen):" | tee -a $SUMMARY

  # vs random
  OUT=$(./bin/eval_match --p0 typed_search --p0-param $gen --p1 random \
                          --deals $DEALS_FAST --seed 100 --threads $THREADS 2>&1)
  parse_eval "v$gen vs random" "$OUT"

  # vs greedy
  OUT=$(./bin/eval_match --p0 typed_search --p0-param $gen --p1 greedy \
                          --deals $DEALS_FAST --seed 100 --threads $THREADS 2>&1)
  parse_eval "v$gen vs greedy" "$OUT"

  # vs pimc(20)
  OUT=$(./bin/eval_match --p0 typed_search --p0-param $gen --p1 pimc \
                          --p1-param 20 --deals $DEALS_PIMC --seed 100 \
                          --threads $THREADS 2>&1)
  parse_eval "v$gen vs pimc(20)" "$OUT"

  # vs previous typed_search gen (only meaningful for gen >= 1).
  if [ $gen -ge 1 ]; then
    PREV_GEN=$((gen - 1))
    OUT=$(./bin/eval_match --p0 typed_search --p0-param $gen \
                            --p1 typed_search --p1-param $PREV_GEN \
                            --deals $DEALS_FAST --seed 100 --threads $THREADS 2>&1)
    parse_eval "v$gen vs v$PREV_GEN" "$OUT"
  fi
done

echo "" | tee -a $SUMMARY
echo "Done. Summary saved to $SUMMARY" | tee -a $SUMMARY
echo "Sample games per gen: $ROOT/gen*/samples/"
