#!/usr/bin/env bash
# Self-play training loop: datagen → train (1 epoch) → eval → sample games → log.
#
# Usage:
#   scripts/run_training_loop.sh [options]
#
# Options (all have defaults):
#   --start N        first generation to produce     (default: auto = max existing gen + 1)
#   --end N          last  generation to produce     (default: start)
#   --games N        self-play games per gen         (default: 200000)
#   --lr-init F      starting learning rate (gen 1)  (default: 0.001)
#   --lr-decay F     LR multiplier per generation    (default: 0.9)
#   --lr-min F       minimum LR floor                (default: 0.0001)
#   --temp-init F    starting temperature (gen 1)    (default: 0.5)
#   --temp-decay F   temperature multiplier per gen  (default: 0.9)
#   --temp-min F     minimum temperature floor       (default: 0.05)
#   --mix-gens N     previous gens to blend in       (default: 3)
#   --mix-decay F    subsample fraction per older gen (default: 0.5)
#   --eval-deals N   deals for all evals             (default: 400)
#   --pimc-param N   pimc rollouts                   (default: 20)
#   --samples N      sample games to generate        (default: 5)
#   --seed N         RNG seed                        (default: 42)
#   --log FILE       append log here                 (default: logs/training.log)

set -euo pipefail
export LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cu13/lib

# ── Defaults ──────────────────────────────────────────────────────────────────
START=""
END=""
GAMES=200000
LR_INIT=0.001
LR_DECAY=0.9
LR_MIN=0.0001
TEMP_INIT=0.5
TEMP_DECAY=0.9
TEMP_MIN=0.05
MIX_GENS=3
MIX_DECAY=0.5
EVAL_DEALS=400
PIMC_PARAM=20
SAMPLES=5
SEED=42
LOG_FILE="logs/training.log"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)      START=$2;      shift 2 ;;
        --end)        END=$2;        shift 2 ;;
        --games)      GAMES=$2;      shift 2 ;;
        --lr-init)    LR_INIT=$2;    shift 2 ;;
        --lr-decay)   LR_DECAY=$2;   shift 2 ;;
        --lr-min)     LR_MIN=$2;     shift 2 ;;
        --temp-init)  TEMP_INIT=$2;  shift 2 ;;
        --temp-decay) TEMP_DECAY=$2; shift 2 ;;
        --temp-min)   TEMP_MIN=$2;   shift 2 ;;
        --mix-gens)   MIX_GENS=$2;   shift 2 ;;
        --mix-decay)  MIX_DECAY=$2;  shift 2 ;;
        --eval-deals) EVAL_DEALS=$2; shift 2 ;;
        --pimc-param) PIMC_PARAM=$2; shift 2 ;;
        --samples)    SAMPLES=$2;    shift 2 ;;
        --seed)       SEED=$2;       shift 2 ;;
        --log)        LOG_FILE=$2;   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect start generation if not specified
if [[ -z "$START" ]]; then
    LATEST=$(ls models/gen[0-9]*.pt models/gen[0-9][0-9]*.pt 2>/dev/null \
        | grep -oP 'gen\K[0-9]+(?=\.pt)' | sort -n | tail -1)
    START=$(( ${LATEST:-0} + 1 ))
fi
[[ -z "$END" ]] && END=$START

mkdir -p logs data models samples

# ── Helpers ───────────────────────────────────────────────────────────────────

tee_log() {
    tee -a "$LOG_FILE"
}

parse_win() {
    grep -E 'wins:' "$1" | grep -oP '\(\K[^)]+' | head -1
}
parse_ci() {
    grep 'Wilson CI' "$1" | grep -oP '\[.*?\]' | head -1
}
# Parse "Deals: WW=N (P%)  split=N (P%)  LL=N (P%)" line
parse_sweep() {
    grep -oP 'WW=\d+ \(\K[^%]+' "$1" | head -1
}
parse_split() {
    grep -oP 'split=\d+ \(\K[^%]+' "$1" | head -1
}
parse_ll() {
    grep -oP 'LL=\d+ \(\K[^%]+' "$1" | head -1
}
parse_result() {
    local file=$1
    printf '%s  %s  WW=%s%%  split=%s%%  LL=%s%%' \
        "$(parse_win "$file")" "$(parse_ci "$file")" \
        "$(parse_sweep "$file")" "$(parse_split "$file")" "$(parse_ll "$file")"
}

ms_to_s() { echo "scale=1; $1/1000" | bc; }

# Compute per-generation LR and temp (exponential decay with floor)
compute_lr() {
    local gen=$1
    python3 -c "print(f'{max($LR_MIN, $LR_INIT * $LR_DECAY**($gen-1)):.6f}')"
}
compute_temp() {
    local gen=$1
    python3 -c "print(f'{max($TEMP_MIN, $TEMP_INIT * $TEMP_DECAY**($gen-1)):.4f}')"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for GEN in $(seq "$START" "$END"); do
    PREV=$((GEN - 1))
    PREV_MODEL="models/gen${PREV}.pt"
    if [[ ! -f "$PREV_MODEL" ]]; then
        echo "ERROR: previous model $PREV_MODEL not found" | tee_log; exit 1
    fi

    LR=$(compute_lr "$GEN")
    TEMP=$(compute_temp "$GEN")

    {
        DIV='════════════════════════════════════════════════════════════════════'
        printf '\n%s\n' "$DIV"
        printf '  Generation %d   games=%d  epochs=1  lr=%s  temp=%s  mix=%d(x%.1f)\n' \
            "$GEN" "$GAMES" "$LR" "$TEMP" "$MIX_GENS" "$MIX_DECAY"
        printf '  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf '%s\n' "$DIV"
    } | tee_log

    # ── 1. Self-play datagen ─────────────────────────────────────────────────
    echo "  [1/5] Generating $GAMES games with gen${PREV}..." | tee_log
    T0=$(date +%s%3N)
    DATAGEN_OUT=$(mktemp)
    bin/generate_nn_selfplay \
        --model  "$PREV_MODEL" \
        --games  "$GAMES" \
        --pool   900 \
        --temp   "$TEMP" \
        --out    "data/gen${GEN}.parquet" \
        >"$DATAGEN_OUT" 2>&1
    grep 'games/s)' "$DATAGEN_OUT" | tail -1 || true
    DATAGEN_MS=$(( $(date +%s%3N) - T0 ))

    AVG_DP=$(grep  'Avg decision'  "$DATAGEN_OUT" | grep -oP '[\d.]+$' || echo "?")
    P1_WIN=$(grep  'P1 win rate'   "$DATAGEN_OUT" | grep -oP '\(\K[^)]+' || echo "?")
    GAMES_S=$(grep 'games/s'       "$DATAGEN_OUT" | grep -oP '[0-9]+ games/s' | tail -1 || echo "?")

    {
        printf '      avg decision points/game : %s\n' "$AVG_DP"
        printf '      p1 win rate              : %s\n' "$P1_WIN"
        printf '      throughput               : %s\n' "$GAMES_S"
        printf '      time                     : %ss\n' "$(ms_to_s "$DATAGEN_MS")"
    } | tee_log
    echo "      legal move count distribution:" | tee_log
    grep -A 200 'Legal move count' "$DATAGEN_OUT" \
        | grep -E '^\s+[0-9]' \
        | awk '{printf "        %s\n", $0}' \
        | tee_log
    cat "$DATAGEN_OUT" >> "$LOG_FILE"
    rm -f "$DATAGEN_OUT"

    # ── 2. Training (1 epoch, exponential generation mixing) ─────────────────
    echo "  [2/5] Training gen${GEN} (1 epoch, lr=${LR})..." | tee_log
    T0=$(date +%s%3N)

    # Build file list: current gen first, then older gens up to MIX_GENS back
    MIX_PARQUETS="data/gen${GEN}.parquet"
    for i in $(seq 1 "$MIX_GENS"); do
        prev_gen=$((GEN - i))
        if [[ "$prev_gen" -ge 1 && -f "data/gen${prev_gen}.parquet" ]]; then
            MIX_PARQUETS="$MIX_PARQUETS data/gen${prev_gen}.parquet"
        fi
    done

    TRAIN_OUT=$(mktemp)
    # shellcheck disable=SC2086
    uv run python -m nn.train $MIX_PARQUETS \
        --out         "models/gen${GEN}.pt" \
        --epochs      1 \
        --batch       2048 \
        --lr          "$LR" \
        --mix-decay   "$MIX_DECAY" \
        >"$TRAIN_OUT" 2>&1
    grep -E '^(Epoch|Loading dataset|Training complete)' "$TRAIN_OUT" || true
    TRAIN_MS=$(( $(date +%s%3N) - T0 ))

    BEST_LOSS=$(grep 'Best val_loss' "$TRAIN_OUT" | grep -oP '[\d.]+$' || echo "?")
    FINAL_EPOCH=$(grep '^Epoch' "$TRAIN_OUT" | tail -1 || echo "")
    {
        printf '      best val_loss : %s\n' "$BEST_LOSS"
        [[ -n "$FINAL_EPOCH" ]] && printf '      final epoch   : %s\n' "$FINAL_EPOCH"
        printf '      time          : %ss\n' "$(ms_to_s "$TRAIN_MS")"
    } | tee_log
    cat "$TRAIN_OUT" >> "$LOG_FILE"
    rm -f "$TRAIN_OUT"

    # ── 3. Evaluations (parallel) ────────────────────────────────────────────
    echo "  [3/5] Running evals..." | tee_log
    T0=$(date +%s%3N)
    E_RANDOM=$(mktemp) E_GREEDY=$(mktemp) E_PIMC=$(mktemp) E_PREV=$(mktemp)

    bin/eval_nn_vs_classic \
        --model "models/gen${GEN}.pt" --vs random \
        --deals "$EVAL_DEALS" --seed "$SEED" \
        >"$E_RANDOM" 2>&1 &

    bin/eval_nn_vs_classic \
        --model "models/gen${GEN}.pt" --vs greedy \
        --deals "$EVAL_DEALS" --seed "$SEED" \
        >"$E_GREEDY" 2>&1 &

    bin/eval_nn_vs_classic \
        --model "models/gen${GEN}.pt" --vs pimc --vs-param "$PIMC_PARAM" \
        --deals "$EVAL_DEALS" --seed "$SEED" \
        >"$E_PIMC" 2>&1 &

    bin/eval_nn_match \
        --model0 "models/gen${GEN}.pt" \
        --model1 "$PREV_MODEL" \
        --deals  "$EVAL_DEALS" --seed "$SEED" \
        >"$E_PREV" 2>&1 &

    wait
    EVAL_MS=$(( $(date +%s%3N) - T0 ))

    {
        printf '      vs random      : %s\n' "$(parse_result "$E_RANDOM")"
        printf '      vs greedy      : %s\n' "$(parse_result "$E_GREEDY")"
        printf '      vs pimc(%2d)    : %s\n' "$PIMC_PARAM" "$(parse_result "$E_PIMC")"
        printf '      vs gen%-2d       : %s\n' "$PREV" "$(parse_result "$E_PREV")"
        printf '      eval time      : %ss\n' "$(ms_to_s "$EVAL_MS")"
    } | tee_log
    rm -f "$E_RANDOM" "$E_GREEDY" "$E_PIMC" "$E_PREV"

    # ── 4. Sample games ──────────────────────────────────────────────────────
    echo "  [4/5] Generating $SAMPLES sample games..." | tee_log
    bin/play_games \
        --model "models/gen${GEN}.pt" \
        --games "$SAMPLES" \
        --seed  "$SEED" \
        --out   "samples/play_gen${GEN}.txt" \
        2>/dev/null
    echo "      written to samples/play_gen${GEN}.txt" | tee_log

    # ── 5. Timing summary ────────────────────────────────────────────────────
    TOTAL_MS=$(( DATAGEN_MS + TRAIN_MS + EVAL_MS ))
    {
        printf '\n  Timing:\n'
        printf '    datagen  %6.1fs\n' "$(ms_to_s "$DATAGEN_MS")"
        printf '    training %6.1fs\n' "$(ms_to_s "$TRAIN_MS")"
        printf '    evals    %6.1fs\n' "$(ms_to_s "$EVAL_MS")"
        printf '    total    %6.1fs\n' "$(ms_to_s "$TOTAL_MS")"
    } | tee_log

done

echo -e '\n══ Done ══' | tee_log
