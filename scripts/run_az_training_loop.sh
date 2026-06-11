#!/usr/bin/env bash
# az_search (seq / history-transformer) generational training loop:
#   gen0 bootstrap (random self-play) → train the unified seq net → champion.
#   gen N≥1: self-play with the champion → train (mixed over last K gens) →
#            eval genN vs champion (paired deals) → promote if the Wilson 95% CI
#            clears 50% → eval vs greedy for tracking.
#
# The champion is always models/az_seq.pt (a copy of the best promoted
# generation); each gen is also kept at models/az_seq_genN.pt. Every generation
# writes the parquet TRIPLE (player / opp / games — the games file carries the
# move sequences the transformer trains on).
#
# Usage:
#   scripts/run_az_training_loop.sh [options]
#
# Options (all have defaults):
#   --start N        first gen to produce (default: auto = max existing gen + 1, ≥1)
#   --end N          last  gen to produce (default: start)
#   --games N        self-play games per gen        (default: 20000)
#   --sims N         search sims per turn (gen≥1)   (default: 200)
#   --slots N        concurrent self-play slots     (default: 4096)
#   --epochs N       train epochs per gen           (default: 5)
#   --batch-games N  games per train batch          (default: 256)
#   --mix-gens N     previous gens to blend in      (default: 3)
#   --mix-decay F    subsample fraction per older gen(default: 0.5)
#   --eval-deals N   paired deals per eval          (default: 400)
#   --eval-sims N    search sims per turn at eval   (default: 200)
#   --device D       self-play inference device     (default: cuda; falls back to cpu)
#   --seed N         RNG seed                       (default: 42)
#   --log FILE       append log here                (default: logs/az_training.log)
#
#   --series         SERIES objective (see below)   (default: off, single-game)
#   --natural-frac F selfplay start-state mixture    (default: 0.5)
#   --shrink-k K     Markov per-state shrinkage      (default: 100)
#   --eval-pairs N   mirrored series pairs per eval   (default: eval-deals)
#
# Self-play runs the NN forward on --device (the forward dominates wall time).
#
# SERIES objective: optimise series wins (first to 50) rather than single games.
# Each generation, before self-play, the series Markov chain is (re)estimated
# from the previous champion's games + accumulated eval outcome dumps into
# data/series_v_genN.csv; self-play and training condition on it (--series-v),
# value targets become series win probabilities, and promotion is gated on the
# eval_az_series series-win CI. Bootstrap (no prior series table): the chain is
# seeded --bootstrap-global from existing az_games_gen*.parquet, and the champion
# models/az_seq.pt must already be series-aware (run
# scripts/convert_az_seq_series.py on a pre-series checkpoint first).

set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

# ── Defaults ────────────────────────────────────────────────────────────────
START=""
END=""
GAMES=20000
SIMS=200
SLOTS=4096
EPOCHS=5
BATCH_GAMES=256
MIX_GENS=3
MIX_DECAY=0.5
EVAL_DEALS=400
EVAL_SIMS=200
DEVICE=cuda
SEED=42
LOG_FILE="logs/az_training.log"
SERIES=0
NATURAL_FRAC=0.5
SHRINK_K=100
EVAL_PAIRS=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)       START=$2;       shift 2 ;;
        --end)         END=$2;         shift 2 ;;
        --games)       GAMES=$2;       shift 2 ;;
        --sims)        SIMS=$2;        shift 2 ;;
        --slots)       SLOTS=$2;       shift 2 ;;
        --epochs)      EPOCHS=$2;      shift 2 ;;
        --batch-games) BATCH_GAMES=$2; shift 2 ;;
        --mix-gens)    MIX_GENS=$2;    shift 2 ;;
        --mix-decay)   MIX_DECAY=$2;   shift 2 ;;
        --eval-deals)  EVAL_DEALS=$2;  shift 2 ;;
        --eval-sims)   EVAL_SIMS=$2;   shift 2 ;;
        --device)      DEVICE=$2;      shift 2 ;;
        --seed)        SEED=$2;        shift 2 ;;
        --log)         LOG_FILE=$2;    shift 2 ;;
        --series)      SERIES=1;       shift 1 ;;
        --natural-frac) NATURAL_FRAC=$2; shift 2 ;;
        --shrink-k)    SHRINK_K=$2;    shift 2 ;;
        --eval-pairs)  EVAL_PAIRS=$2;  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs data models

tee_log() { tee -a "$LOG_FILE"; }

PLAYER_DATA="data/az_player_gen%d.parquet"
OPP_DATA="data/az_opp_gen%d.parquet"
GAMES_DATA="data/az_games_gen%d.parquet"
MODEL="models/az_seq_gen%d.pt"
SERIES_V="data/series_v_gen%d.csv"
EVAL_OUTCOMES="data/eval_outcomes_gen%d.csv"
[[ -z "$EVAL_PAIRS" ]] && EVAL_PAIRS=$EVAL_DEALS

pf() { printf "$1" "$2"; }  # pf <fmt> <gen>

# eval output parsers (tolerant of eval_az_match "= X" and eval_az_series
# "series = X").
parse_winrate() { grep -oP 'A wins \d+ / \d+ (?:series )?= \K[0-9.]+' "$1" | head -1; }
parse_ci()      { grep -oP 'Wilson 95% CI \K\[[0-9., ]+\]' "$1" | head -1; }
parse_ci_lo()   { grep -oP 'Wilson 95% CI \[\K[0-9.]+' "$1" | head -1; }
ci_clears_half() {  # 1 if Wilson lower bound > 0.5
    local lo; lo=$(parse_ci_lo "$1")
    [[ -n "$lo" ]] && python3 -c "import sys; sys.exit(0 if $lo > 0.5 else 1)"
}

# ── gen0 bootstrap (random self-play) ─────────────────────────────────────────
ensure_gen0() {
    local m
    m=$(pf "$MODEL" 0)
    if [[ ! -f "$m" ]]; then
        echo "── gen0 bootstrap: random self-play ($GAMES games) ──" | tee_log
        bin/az_selfplay \
            --games "$GAMES" --slots "$SLOTS" --seed "$SEED" \
            --device "$DEVICE" \
            --player-out "$(pf "$PLAYER_DATA" 0)" --opp-out "$(pf "$OPP_DATA" 0)" \
            --games-out "$(pf "$GAMES_DATA" 0)" \
            2>&1 | tee_log
        uv run python -m nn.train_az_seq \
            --games-data "$(pf "$GAMES_DATA" 0)" \
            --player-data "$(pf "$PLAYER_DATA" 0)" --opp-data "$(pf "$OPP_DATA" 0)" \
            --out "$m" \
            --epochs "$EPOCHS" --batch-games "$BATCH_GAMES" 2>&1 | grep -E '\[seq\]' | tee_log
    fi
    # Initialise the champion pointer to gen0 if absent.
    if [[ ! -f models/az_seq.pt ]]; then
        cp "$m" models/az_seq.pt
        echo "  champion := gen0" | tee_log
    fi
}

# ── Auto-detect start generation ──────────────────────────────────────────────
ensure_gen0
if [[ -z "$START" ]]; then
    LATEST=$(ls models/az_seq_gen[0-9]*.pt 2>/dev/null \
        | grep -oP 'gen\K[0-9]+(?=\.pt)' | sort -n | tail -1)
    START=$(( ${LATEST:-0} + 1 ))
fi
[[ -z "$END" ]] && END=$START

# ── Main loop ──────────────────────────────────────────────────────────────────
for GEN in $(seq "$START" "$END"); do
    M=$(pf "$MODEL" "$GEN")
    PD=$(pf "$PLAYER_DATA" "$GEN"); OD=$(pf "$OPP_DATA" "$GEN")
    GD=$(pf "$GAMES_DATA" "$GEN")

    {
        DIV='════════════════════════════════════════════════════════════════════'
        printf '\n%s\n' "$DIV"
        printf '  az gen %d   games=%d sims=%d slots=%d epochs=%d mix=%d(x%.1f)\n' \
            "$GEN" "$GAMES" "$SIMS" "$SLOTS" "$EPOCHS" "$MIX_GENS" "$MIX_DECAY"
        printf '  %s\n%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DIV"
    } | tee_log

    # ── 0. (series) Re-estimate the Markov chain from the champion's data ─────
    SP_SERIES=""; TRAIN_SERIES=""; VCSV=""
    if [[ "$SERIES" == 1 ]]; then
        VCSV=$(pf "$SERIES_V" "$GEN")
        echo "  [0/4] Estimate series chain → $VCSV (shrink-k=$SHRINK_K)..." | tee_log
        CHAIN_ARGS=()
        prevGD=$(pf "$GAMES_DATA" $((GEN - 1)))
        [[ -f "$prevGD" ]] && CHAIN_ARGS+=(--games "$prevGD")
        prevOUT=$(pf "$EVAL_OUTCOMES" $((GEN - 1)))
        [[ -f "$prevOUT" ]] && CHAIN_ARGS+=(--outcomes "$prevOUT")
        # Bootstrap-global when there is no prior series table to specialise from
        # (the previous gen's games may also predate the series schema).
        BOOT=""
        [[ ! -f "$(pf "$SERIES_V" $((GEN - 1)))" ]] && BOOT="--bootstrap-global"
        if [[ ${#CHAIN_ARGS[@]} -eq 0 ]]; then
            # No champion data yet: seed from all existing games parquets.
            for f in data/az_games_gen*.parquet; do CHAIN_ARGS+=(--games "$f"); done
            BOOT="--bootstrap-global"
        fi
        uv run python analysis/series_markov.py "${CHAIN_ARGS[@]}" \
            --out "$VCSV" --shrink-k "$SHRINK_K" $BOOT --gen "$GEN" 2>&1 | tee_log
        SP_SERIES="--series-v $VCSV --natural-frac $NATURAL_FRAC"
        TRAIN_SERIES="--series-v $VCSV"
    fi

    # ── 1. Self-play with the current champion ───────────────────────────────
    echo "  [1/4] Self-play $GAMES games vs champion (sims=$SIMS)..." | tee_log
    T0=$(date +%s)
    # shellcheck disable=SC2086
    bin/az_selfplay \
        --games "$GAMES" --sims "$SIMS" --slots "$SLOTS" --device "$DEVICE" \
        --seed "$((SEED + GEN))" \
        --model models/az_seq.pt $SP_SERIES \
        --player-out "$PD" --opp-out "$OD" --games-out "$GD" \
        2>&1 | grep -E 'samples:|device=|series objective' | tee_log
    echo "      self-play time: $(( $(date +%s) - T0 ))s" | tee_log

    # ── 2. Train (mixed over the last MIX_GENS gens) ─────────────────────────
    echo "  [2/4] Train gen$GEN ($EPOCHS epochs, mix=$MIX_GENS)..." | tee_log
    GAMES_FILES="$GD"; PLAYER_FILES="$PD"; OPP_FILES="$OD"
    for i in $(seq 1 "$MIX_GENS"); do
        g=$((GEN - i))
        [[ "$g" -ge 0 && -f "$(pf "$GAMES_DATA" "$g")" ]] && \
            GAMES_FILES="$GAMES_FILES $(pf "$GAMES_DATA" "$g")" && \
            PLAYER_FILES="$PLAYER_FILES $(pf "$PLAYER_DATA" "$g")" && \
            OPP_FILES="$OPP_FILES $(pf "$OPP_DATA" "$g")"
    done
    T0=$(date +%s)
    # shellcheck disable=SC2086
    uv run python -m nn.train_az_seq \
        --games-data $GAMES_FILES --player-data $PLAYER_FILES --opp-data $OPP_FILES \
        --out "$M" \
        --epochs "$EPOCHS" --batch-games "$BATCH_GAMES" --mix-decay "$MIX_DECAY" \
        $TRAIN_SERIES \
        2>&1 | grep -E '\[seq\]' | tee_log
    echo "      train time: $(( $(date +%s) - T0 ))s" | tee_log

    # ── 3. Eval genN vs champion (promotion) + vs greedy (tracking) ──────────
    # Pin each eval to a single torch thread; the two run in parallel cheaply.
    # Series mode evaluates full series (eval_az_series) and dumps the per-game
    # outcomes so the next gen's chain build can re-use them.
    E_CHAMP=$(mktemp); E_GREEDY=$(mktemp)
    if [[ "$SERIES" == 1 ]]; then
        echo "  [3/4] Series eval (pairs=$EVAL_PAIRS, sims=$EVAL_SIMS)..." | tee_log
        EOUT=$(pf "$EVAL_OUTCOMES" "$GEN")
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_series \
            --model-a "$M" --model-b models/az_seq.pt --series-v "$VCSV" \
            --pairs "$EVAL_PAIRS" --sims "$EVAL_SIMS" --seed "$SEED" \
            --dump-outcomes "$EOUT" \
            >"$E_CHAMP" 2>&1 &
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_series \
            --model-a "$M" --classic greedy --series-v "$VCSV" \
            --pairs "$EVAL_PAIRS" --sims "$EVAL_SIMS" --seed "$SEED" \
            >"$E_GREEDY" 2>&1 &
        wait
    else
        echo "  [3/4] Eval (deals=$EVAL_DEALS, sims=$EVAL_SIMS)..." | tee_log
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_match \
            --model-a "$M" --model-b models/az_seq.pt \
            --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" --seed "$SEED" \
            >"$E_CHAMP" 2>&1 &
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_match \
            --model-a "$M" --classic greedy \
            --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" --seed "$SEED" \
            >"$E_GREEDY" 2>&1 &
        wait
    fi
    {
        printf '      vs champion : %s  %s\n' "$(parse_winrate "$E_CHAMP")"  "$(parse_ci "$E_CHAMP")"
        printf '      vs greedy   : %s  %s\n' "$(parse_winrate "$E_GREEDY")" "$(parse_ci "$E_GREEDY")"
    } | tee_log
    cat "$E_CHAMP" "$E_GREEDY" >> "$LOG_FILE"

    # ── 4. Promote if the Wilson lower bound clears 50% ──────────────────────
    if ci_clears_half "$E_CHAMP"; then
        cp "$M" models/az_seq.pt
        echo "  [4/4] PROMOTED gen$GEN → champion" | tee_log
    else
        echo "  [4/4] gen$GEN not promoted (champion unchanged)" | tee_log
    fi
    rm -f "$E_CHAMP" "$E_GREEDY"
done

echo -e '\n══ Done ══' | tee_log
