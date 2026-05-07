#!/usr/bin/env bash
# Warm-start training from an existing v6 snapshot — keeps main/fb/mp_*
# tables (unchanged by the eval-feature redesign) and rebuilds the extended
# table under the new opp-aware encoding.
#
# Usage:
#   scripts/typed_search_warm_train.sh [NUM_GENS] [GAMES_PER_GEN] [WARM_FROM]
# defaults: NUM_GENS=6, GAMES=200000, WARM_FROM=data/typed_search_v6_pre_eval_features

set -e

NUM_GENS=${1:-6}
GAMES=${2:-200000}
WARM_FROM=${3:-data/typed_search_v6_pre_eval_features}
THREADS=${THREADS:-18}
ALPHA=${ALPHA:-0.7}
DEALS_PIMC=${DEALS_PIMC:-200}

ROOT=data/typed_search_runs_new_features
SUMMARY=$ROOT/summary.txt

mkdir -p $ROOT
: > $SUMMARY

parse_eval () {
  local label="$1"
  local out="$2"
  local p0_wins=$(echo "$out" | grep -E 'P0 .* wins:' | head -1 | grep -oE '[0-9]+ / [0-9]+' | head -1)
  local pct=$(echo "$out" | grep -E 'P0 .* wins:' | head -1 | grep -oE '\([0-9]+\.[0-9]+%\)' | head -1)
  local breakdown=$(echo "$out" | grep -E 'P0_sweep=|P0 sweep 2-0:' | head -1)
  printf "  %-40s %s %s\n" "$label" "$p0_wins" "$pct" | tee -a $SUMMARY
  echo "    $breakdown" | tee -a $SUMMARY
}

fresh_dir () {
  rm -rf "$1"
  mkdir -p "$1"
}

echo "Warm-start training: $NUM_GENS gens × $GAMES games (threads=$THREADS, alpha=$ALPHA)" | tee -a $SUMMARY
echo "Warm-from: $WARM_FROM (excluding eval_extended.bin)" | tee -a $SUMMARY

# Bootstrap: copy everything except eval_extended.bin from WARM_FROM into the
# working dir. The new extended table will be built fresh under the new
# encoding starting from gen 1.
fresh_dir data/typed_search
for f in $(ls $WARM_FROM/*.bin); do
  base=$(basename $f)
  if [ "$base" = "eval_extended.bin" ]; then continue; fi
  cp -f "$f" data/typed_search/
done
ls -la data/typed_search/ | tee -a $SUMMARY

for gen in $(seq 1 $NUM_GENS); do
  echo "" | tee -a $SUMMARY
  echo "=== GEN $gen ===" | tee -a $SUMMARY
  GEN_DIR=$ROOT/gen$gen
  fresh_dir $GEN_DIR

  echo "Training: games=$GAMES alpha=$ALPHA" | tee -a $SUMMARY
  ./bin/typed_search_train \
    --policy typed_search --games $GAMES --threads $THREADS --alpha $ALPHA \
    --seed $((2000 * gen + 1)) --tables-dir data/typed_search \
    2>&1 | tee -a $SUMMARY

  # Snapshot trained tables into versioned dirs.
  cp -f data/typed_search/*.bin $GEN_DIR/
  V_DIR=data/typed_search_v$((100 + gen))
  fresh_dir $V_DIR
  cp -f data/typed_search/*.bin $V_DIR/

  # Eval vs pimc(20) only — keep it lean. The run is iterative; we mainly
  # care about the trajectory.
  echo "" | tee -a $SUMMARY
  echo "Eval (gen $gen):" | tee -a $SUMMARY
  # Use param=$((100+gen)) so the factory finds data/typed_search_v$((100+gen))
  # without colliding with the existing v1..v6 dirs.
  PARAM=$((100 + gen))
  OUT=$(./bin/eval_match --p0 typed_search --p0-param $PARAM --p1 pimc \
                          --p1-param 20 --deals $DEALS_PIMC --seed 100 \
                          --threads $THREADS 2>&1)
  parse_eval "v$PARAM vs pimc(20)" "$OUT"
done

echo "" | tee -a $SUMMARY
echo "Done. Summary saved to $SUMMARY" | tee -a $SUMMARY
