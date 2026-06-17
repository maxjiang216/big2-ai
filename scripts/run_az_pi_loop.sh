#!/usr/bin/env bash
# Perfect-information AlphaZero (az_pi) generational training loop.
#
#   gen0: export a RANDOM-INIT net (true scratch AZ — no teacher bootstrap).
#   gen N>=1: self-play with the champion -> train (mixed over last K gens) ->
#             eval genN vs champion (paired deals); promote if the Wilson 95% CI
#             lower bound clears 0.5 -> tracking evals vs greedy / typed_search.
#
# Champion pointer: models/az_pi.pt (copy of the best promoted gen). Each gen is
# also kept at models/az_pi_w{W}b{B}_genN.pt so model-size sweeps don't collide.
# One Parquet per gen (player schema): data/az_pi_genN.parquet.
#
# Usage: scripts/run_az_pi_loop.sh [options]
#   --start N        first gen to produce (default: auto = max existing + 1, >=1)
#   --end N          last gen to produce  (default: start)
#   --games N        self-play games per gen        (default: 20000)
#   --sims N         search sims per turn (gen>=1)   (default: 400)
#   --slots N        concurrent self-play slots      (default: 1024)
#   --epochs N       train epochs per gen            (default: 10)
#   --batch N        train batch size                (default: 2048)
#   --mix-gens N     previous gens to blend in       (default: 3)
#   --mix-decay F    subsample fraction per older gen (default: 0.5)
#   --eval-deals N   paired deals per eval           (default: 400)
#   --eval-sims N    search sims per turn at eval     (default: 400)
#   --width N        net trunk width                 (default: 256)
#   --blocks N       net ResBlock depth              (default: 2)
#   --embed N        card-embedding dim              (default: 64)
#   --device D       inference device                (default: cuda)
#   --seed N         RNG seed                        (default: 42)
#   --log FILE       append log here                 (default: logs/az_pi.log)

set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

START=""; END=""
GAMES=20000; SIMS=400; SLOTS=1024; THREADS=8
EPOCHS=10; BATCH=2048; MIX_GENS=3; MIX_DECAY=0.5
EVAL_DEALS=400; EVAL_SIMS=400
WIDTH=256; BLOCKS=2; EMBED=64
DEVICE=cuda; SEED=42
LOG_FILE="logs/az_pi.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START="$2"; shift 2;;
        --end) END="$2"; shift 2;;
        --games) GAMES="$2"; shift 2;;
        --sims) SIMS="$2"; shift 2;;
        --slots) SLOTS="$2"; shift 2;;
        --threads) THREADS="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --mix-gens) MIX_GENS="$2"; shift 2;;
        --mix-decay) MIX_DECAY="$2"; shift 2;;
        --eval-deals) EVAL_DEALS="$2"; shift 2;;
        --eval-sims) EVAL_SIMS="$2"; shift 2;;
        --width) WIDTH="$2"; shift 2;;
        --blocks) BLOCKS="$2"; shift 2;;
        --embed) EMBED="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --log) LOG_FILE="$2"; shift 2;;
        *) echo "unknown option: $1" >&2; exit 1;;
    esac
done

mkdir -p logs data models
TAG="w${WIDTH}b${BLOCKS}"
DATA="data/az_pi_gen%d.parquet"
MODEL="models/az_pi_${TAG}_gen%d.pt"
CHAMP="models/az_pi.pt"

tee_log() { tee -a "$LOG_FILE"; }
pf() { printf "$1" "$2"; }

# Promotion gate: sweep-share CI (decisive deals only — the skill signal;
# split deals are card-decided). Falls back to the raw win-rate CI when no
# decisive line is present.
parse_ci_lo() {
    local lo
    lo=$(grep 'sweep share' "$1" | grep -oP 'Wilson95 \[\K[0-9.]+' | head -1)
    [[ -z "$lo" ]] && lo=$(grep -oP 'Wilson95 \[\K[0-9.]+' "$1" | head -1)
    echo "$lo"
}
ci_clears_half() {
    local lo; lo=$(parse_ci_lo "$1")
    [[ -n "$lo" ]] && python3 -c "import sys; sys.exit(0 if $lo > 0.5 else 1)"
}

# ── gen0: random-init scratch net ────────────────────────────────────────────
ensure_gen0() {
    local m; m=$(pf "$MODEL" 0)
    if [[ ! -f "$m" ]]; then
        echo "── gen0: export random-init net ($TAG) ──" | tee_log
        uv run python -m nn.train_az_pi --export-init "$m" \
            --width "$WIDTH" --blocks "$BLOCKS" --embed "$EMBED" 2>&1 | tee_log
    fi
    if [[ ! -f "$CHAMP" ]]; then cp "$m" "$CHAMP"; echo "  champion := gen0" | tee_log; fi
}

ensure_gen0
if [[ -z "$START" ]]; then
    LATEST=$(ls "models/az_pi_${TAG}_gen"[0-9]*.pt 2>/dev/null \
        | grep -oP 'gen\K[0-9]+(?=\.pt)' | sort -n | tail -1)
    START=$(( ${LATEST:-0} + 1 ))
fi
[[ -z "$END" ]] && END=$START

for GEN in $(seq "$START" "$END"); do
    M=$(pf "$MODEL" "$GEN"); D=$(pf "$DATA" "$GEN")
    {
        DIV='════════════════════════════════════════════════════════════════════'
        printf '\n%s\n  az_pi gen %d  games=%d sims=%d slots=%d epochs=%d mix=%d(x%.1f)\n  %s\n%s\n' \
            "$DIV" "$GEN" "$GAMES" "$SIMS" "$SLOTS" "$EPOCHS" "$MIX_GENS" "$MIX_DECAY" \
            "$(date '+%Y-%m-%d %H:%M:%S')" "$DIV"
    } | tee_log

    # 1. Self-play with the champion. No temperature: each game's fresh deal
    # already diversifies the data; Dirichlet root noise still explores moves.
    echo "  [1/4] Self-play $GAMES games vs champion (sims=$SIMS)..." | tee_log
    bin/az_pi_selfplay --model "$CHAMP" --games "$GAMES" --sims "$SIMS" \
        --slots "$SLOTS" --threads "$THREADS" --temp-moves 0 \
        --seed "$((SEED + GEN))" --device "$DEVICE" --out "$D" 2>&1 | tee_log

    # 2. Train (mixed over last MIX_GENS gens).
    echo "  [2/4] Train gen$GEN ($EPOCHS epochs, mix=$MIX_GENS)..." | tee_log
    FILES="$D"
    for i in $(seq 1 "$MIX_GENS"); do
        g=$((GEN - i))
        [[ "$g" -ge 1 && -f "$(pf "$DATA" "$g")" ]] && FILES="$FILES $(pf "$DATA" "$g")"
    done
    # shellcheck disable=SC2086
    uv run python -m nn.train_az_pi --data $FILES --out "$M" \
        --width "$WIDTH" --blocks "$BLOCKS" --embed "$EMBED" \
        --epochs "$EPOCHS" --batch "$BATCH" --mix-decay "$MIX_DECAY" \
        --device "$DEVICE" 2>&1 | grep -E '\[pi\]' | tee_log

    # 2b. Sample games: deterministic (no noise, argmax), FIXED deals across
    # generations so play evolution is visible on identical cards.
    bin/az_pi_selfplay --model "$M" --games 10 --sims "$SIMS" --slots 10 \
        --threads "$THREADS" --temp-moves 0 --dirichlet-eps 0 --seed 777 \
        --device "$DEVICE" \
        --out "samples/az_pi_gen${GEN}_games.parquet" 2>&1 | grep -v '  games ' | tee_log
    uv run python scripts/sample_games_pi.py \
        "samples/az_pi_gen${GEN}_games.parquet" --games 10 --model "$M" \
        --out "samples/az_pi_gen${GEN}.html" 2>&1 | tee_log

    # 3. Eval genN vs champion; promote on Wilson lower bound > 0.5.
    echo "  [3/4] Eval gen$GEN vs champion ($EVAL_DEALS deals, sims=$EVAL_SIMS)..." | tee_log
    EV="logs/az_pi_eval_gen${GEN}.txt"
    bin/eval_az_pi_match --p0 "pi:$M" --p1 "pi:$CHAMP" \
        --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" --max-slots "$SLOTS" \
        --device "$DEVICE" --seed "$SEED" 2>&1 | tee "$EV" | tee_log
    PROMOTED=0
    if ci_clears_half "$EV"; then
        cp "$M" "$CHAMP"
        PROMOTED=1
        echo "  ✓ gen$GEN PROMOTED -> champion" | tee_log
    else
        echo "  · gen$GEN not promoted (champion unchanged)" | tee_log
    fi

    # 4. Tracking evals (champion vs classic baselines). Skipped when the
    # champion didn't change: same model + same seed replays identical games.
    if [[ "$PROMOTED" == 1 ]]; then
        echo "  [4/4] Tracking evals vs greedy / typed_search..." | tee_log
        # Raw rate only: sweep share vs these baselines is saturated at ~1.0;
        # the raw % tracks how bad a hand the champion can still win.
        for OPP in classic:greedy classic:typed_search; do
            bin/eval_az_pi_match --p0 "pi:$CHAMP" --p1 "$OPP" \
                --deals "$EVAL_DEALS" --sims "$EVAL_SIMS" \
                --max-slots "$SLOTS" --device "$DEVICE" \
                --seed "$SEED" 2>&1 | grep -E 'win rate' | tee_log
        done
    else
        echo "  [4/4] Tracking evals skipped (champion unchanged)" | tee_log
    fi
done
echo "az_pi loop done." | tee_log
