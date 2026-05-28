#!/usr/bin/env bash
# az_search generational training loop:
#   gen0 bootstrap (random self-play) → train both nets → champion.
#   gen N≥1: self-play with the champion → train (mixed over last K gens) →
#            eval genN vs champion (paired deals) → promote if the Wilson 95% CI
#            clears 50% → eval vs greedy for tracking.
#
# The champion pair is always models/az_player.pt + models/az_opp.pt (copies of
# the best promoted generation); each gen is also kept at models/az_{player,opp}_genN.pt.
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
#   --start-frac F   fraction of mid-game starts    (default: 0.2)
#   --epochs N       train epochs per gen           (default: 5)
#   --batch N        train batch size               (default: 2048)
#   --mix-gens N     previous gens to blend in      (default: 3)
#   --mix-decay F    subsample fraction per older gen(default: 0.5)
#   --eval-deals N   paired deals per eval          (default: 400)
#   --eval-sims N    search sims per turn at eval   (default: 200)
#   --device D       self-play inference device     (default: cuda; falls back to cpu)
#   --seed N         RNG seed                       (default: 42)
#   --log FILE       append log here                (default: logs/az_training.log)
#
# Self-play runs the NN forward on --device (GPU ~2.7x faster; the forward
# dominates wall time). Evals use batch-1 forwards, pinned to one torch thread.

set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

# ── Defaults ────────────────────────────────────────────────────────────────
START=""
END=""
GAMES=20000
SIMS=200
SLOTS=4096
START_FRAC=0.2
EPOCHS=5
BATCH=2048
MIX_GENS=3
MIX_DECAY=0.5
EVAL_DEALS=400
EVAL_SIMS=200
DEVICE=cuda
SEED=42
LOG_FILE="logs/az_training.log"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)      START=$2;      shift 2 ;;
        --end)        END=$2;        shift 2 ;;
        --games)      GAMES=$2;      shift 2 ;;
        --sims)       SIMS=$2;       shift 2 ;;
        --slots)      SLOTS=$2;      shift 2 ;;
        --start-frac) START_FRAC=$2; shift 2 ;;
        --epochs)     EPOCHS=$2;     shift 2 ;;
        --batch)      BATCH=$2;      shift 2 ;;
        --mix-gens)   MIX_GENS=$2;   shift 2 ;;
        --mix-decay)  MIX_DECAY=$2;  shift 2 ;;
        --eval-deals) EVAL_DEALS=$2; shift 2 ;;
        --eval-sims)  EVAL_SIMS=$2;  shift 2 ;;
        --device)     DEVICE=$2;     shift 2 ;;
        --seed)       SEED=$2;       shift 2 ;;
        --log)        LOG_FILE=$2;   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs data models

tee_log() { tee -a "$LOG_FILE"; }

PLAYER_DATA="data/az_player_gen%d.parquet"
OPP_DATA="data/az_opp_gen%d.parquet"
PLAYER_MODEL="models/az_player_gen%d.pt"
OPP_MODEL="models/az_opp_gen%d.pt"

pf() { printf "$1" "$2"; }  # pf <fmt> <gen>

# eval_az_match output parsers.
parse_winrate() { grep -oP 'A wins \d+ / \d+ = \K[0-9.]+' "$1" | head -1; }
parse_ci()      { grep -oP 'Wilson 95% CI \K\[[0-9., ]+\]' "$1" | head -1; }
parse_ci_lo()   { grep -oP 'Wilson 95% CI \[\K[0-9.]+' "$1" | head -1; }
ci_clears_half() {  # 1 if Wilson lower bound > 0.5
    local lo; lo=$(parse_ci_lo "$1")
    [[ -n "$lo" ]] && python3 -c "import sys; sys.exit(0 if $lo > 0.5 else 1)"
}

# ── gen0 bootstrap (random self-play) ─────────────────────────────────────────
ensure_gen0() {
    local pm om
    pm=$(pf "$PLAYER_MODEL" 0); om=$(pf "$OPP_MODEL" 0)
    if [[ ! -f "$pm" || ! -f "$om" ]]; then
        echo "── gen0 bootstrap: random self-play ($GAMES games) ──" | tee_log
        bin/az_selfplay \
            --games "$GAMES" --slots "$SLOTS" --start-frac "$START_FRAC" --seed "$SEED" \
            --device "$DEVICE" \
            --player-out "$(pf "$PLAYER_DATA" 0)" --opp-out "$(pf "$OPP_DATA" 0)" \
            2>&1 | tee_log
        uv run python -m nn.train_az \
            --player-data "$(pf "$PLAYER_DATA" 0)" --opp-data "$(pf "$OPP_DATA" 0)" \
            --player-out "$pm" --opp-out "$om" \
            --epochs "$EPOCHS" --batch "$BATCH" 2>&1 | grep -E '\[player\]|\[opp\]' | tee_log
    fi
    # Initialise the champion pointer to gen0 if absent.
    if [[ ! -f models/az_player.pt || ! -f models/az_opp.pt ]]; then
        cp "$pm" models/az_player.pt
        cp "$om" models/az_opp.pt
        echo "  champion := gen0" | tee_log
    fi
}

# ── Auto-detect start generation ──────────────────────────────────────────────
ensure_gen0
if [[ -z "$START" ]]; then
    LATEST=$(ls models/az_player_gen[0-9]*.pt 2>/dev/null \
        | grep -oP 'gen\K[0-9]+(?=\.pt)' | sort -n | tail -1)
    START=$(( ${LATEST:-0} + 1 ))
fi
[[ -z "$END" ]] && END=$START

# ── Main loop ──────────────────────────────────────────────────────────────────
for GEN in $(seq "$START" "$END"); do
    PM=$(pf "$PLAYER_MODEL" "$GEN"); OM=$(pf "$OPP_MODEL" "$GEN")
    PD=$(pf "$PLAYER_DATA" "$GEN");  OD=$(pf "$OPP_DATA" "$GEN")

    {
        DIV='════════════════════════════════════════════════════════════════════'
        printf '\n%s\n' "$DIV"
        printf '  az gen %d   games=%d sims=%d slots=%d epochs=%d mix=%d(x%.1f)\n' \
            "$GEN" "$GAMES" "$SIMS" "$SLOTS" "$EPOCHS" "$MIX_GENS" "$MIX_DECAY"
        printf '  %s\n%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DIV"
    } | tee_log

    # ── 1. Self-play with the current champion ───────────────────────────────
    echo "  [1/4] Self-play $GAMES games vs champion (sims=$SIMS)..." | tee_log
    T0=$(date +%s)
    bin/az_selfplay \
        --games "$GAMES" --sims "$SIMS" --slots "$SLOTS" --device "$DEVICE" \
        --start-frac "$START_FRAC" --seed "$((SEED + GEN))" \
        --player-model models/az_player.pt --opp-model models/az_opp.pt \
        --player-out "$PD" --opp-out "$OD" \
        2>&1 | grep -E 'samples:|device=' | tee_log
    echo "      self-play time: $(( $(date +%s) - T0 ))s" | tee_log

    # ── 2. Train (mixed over the last MIX_GENS gens) ─────────────────────────
    echo "  [2/4] Train gen$GEN ($EPOCHS epochs, mix=$MIX_GENS)..." | tee_log
    PLAYER_FILES="$PD"; OPP_FILES="$OD"
    for i in $(seq 1 "$MIX_GENS"); do
        g=$((GEN - i))
        [[ "$g" -ge 0 && -f "$(pf "$PLAYER_DATA" "$g")" ]] && \
            PLAYER_FILES="$PLAYER_FILES $(pf "$PLAYER_DATA" "$g")" && \
            OPP_FILES="$OPP_FILES $(pf "$OPP_DATA" "$g")"
    done
    T0=$(date +%s)
    # shellcheck disable=SC2086
    uv run python -m nn.train_az \
        --player-data $PLAYER_FILES --opp-data $OPP_FILES \
        --player-out "$PM" --opp-out "$OM" \
        --epochs "$EPOCHS" --batch "$BATCH" --mix-decay "$MIX_DECAY" \
        2>&1 | grep -E '\[player\]|\[opp\]' | tee_log
    echo "      train time: $(( $(date +%s) - T0 ))s" | tee_log

    # ── 3. Eval genN vs champion (promotion) + vs greedy (tracking) ──────────
    # eval_az_match does batch-1 forwards, so torch's default all-core intra-op
    # threading is pure overhead AND oversubscribes if two run at once. Pin each
    # to a single torch thread; then the two can run in parallel cheaply.
    echo "  [3/4] Eval (deals=$EVAL_DEALS, sims=$EVAL_SIMS)..." | tee_log
    E_CHAMP=$(mktemp); E_GREEDY=$(mktemp)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_match \
        --player-a "$PM" --opp-a "$OM" \
        --player-b models/az_player.pt --opp-b models/az_opp.pt \
        --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" --seed "$SEED" \
        >"$E_CHAMP" 2>&1 &
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bin/eval_az_match \
        --player-a "$PM" --opp-a "$OM" --classic greedy \
        --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" --seed "$SEED" \
        >"$E_GREEDY" 2>&1 &
    wait
    {
        printf '      vs champion : %s  %s\n' "$(parse_winrate "$E_CHAMP")"  "$(parse_ci "$E_CHAMP")"
        printf '      vs greedy   : %s  %s\n' "$(parse_winrate "$E_GREEDY")" "$(parse_ci "$E_GREEDY")"
    } | tee_log
    cat "$E_CHAMP" "$E_GREEDY" >> "$LOG_FILE"

    # ── 4. Promote if the Wilson lower bound clears 50% ──────────────────────
    if ci_clears_half "$E_CHAMP"; then
        cp "$PM" models/az_player.pt
        cp "$OM" models/az_opp.pt
        echo "  [4/4] PROMOTED gen$GEN → champion" | tee_log
    else
        echo "  [4/4] gen$GEN not promoted (champion unchanged)" | tee_log
    fi
    rm -f "$E_CHAMP" "$E_GREEDY"
done

echo -e '\n══ Done ══' | tee_log
