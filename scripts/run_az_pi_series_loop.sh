#!/usr/bin/env bash
# Series-aware perfect-information AlphaZero (az_pi series line) gen loop.
#
#   gen1 bootstrap: self-play with the OLD score-blind champion (models/az_pi.pt)
#     in --series-states mode (random scores recorded, search unchanged), build
#     the initial V table from that run's own outcomes (--bootstrap-global:
#     score-blind play is identical at every state, per-state splits would be
#     noise), then train gen1 (--series-table labels, optional --init-from
#     trunk/policy warm start).
#   gen N>=2: self-play with the series champion + current table -> refresh the
#     table from the last K gens' outcomes -> train -> paired-SERIES eval vs
#     champion; promote on the series sweep-share Wilson CI lower bound > 0.5.
#
# Champion pointer: models/az_pi_series.pt. Data data/az_pi_s_genN.parquet,
# outcomes data/az_pi_s_outcomes_genN.csv, table data/series_v_pi.csv.
#
# Usage: scripts/run_az_pi_series_loop.sh [options]
#   --start N        first gen to produce (default: auto = max existing + 1, >=1)
#   --end N          last gen to produce  (default: start)
#   --games N        self-play games per gen        (default: 20000)
#   --sims N         search sims per turn           (default: 1600)
#   --slots N        concurrent self-play slots      (default: 1024)
#   --threads N      self-play shard threads         (default: 6)
#   --epochs N       train epochs per gen            (default: 10)
#   --batch N        train batch size                (default: 2048)
#   --mix-gens N     previous gens to blend in       (default: 3)
#   --mix-decay F    subsample fraction per older gen (default: 0.5)
#   --eval-series N  paired series per eval          (default: 200)
#   --eval-sims N    search sims per turn at eval     (default: same as --sims)
#   --width/--blocks/--embed   net size (default: 256/2/64)
#   --init-from PATH warm-start for gen1 (default: models/az_pi.pt if present)
#   --device D       inference device                (default: cuda)
#   --seed N         RNG seed                        (default: 42)
#   --log FILE       append log here                 (default: logs/az_pi_s.log)

set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

START=""; END=""
GAMES=20000; SIMS=1600; SLOTS=1024; THREADS=6
EPOCHS=10; BATCH=2048; MIX_GENS=3; MIX_DECAY=0.5
EVAL_SERIES=200; EVAL_SIMS=""
WIDTH=256; BLOCKS=2; EMBED=64
INIT_FROM=""
DEVICE=cuda; SEED=42
LOG_FILE="logs/az_pi_s.log"

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
        --eval-series) EVAL_SERIES="$2"; shift 2;;
        --eval-sims) EVAL_SIMS="$2"; shift 2;;
        --width) WIDTH="$2"; shift 2;;
        --blocks) BLOCKS="$2"; shift 2;;
        --embed) EMBED="$2"; shift 2;;
        --init-from) INIT_FROM="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --log) LOG_FILE="$2"; shift 2;;
        *) echo "unknown option: $1" >&2; exit 1;;
    esac
done
[[ -z "$EVAL_SIMS" ]] && EVAL_SIMS=$SIMS
[[ -z "$INIT_FROM" && -f models/az_pi.pt ]] && INIT_FROM=models/az_pi.pt

mkdir -p logs data models samples
TAG="w${WIDTH}b${BLOCKS}"
DATA="data/az_pi_s_gen%d.parquet"
OUTC="data/az_pi_s_outcomes_gen%d.csv"
MODEL="models/az_pi_s_${TAG}_gen%d.pt"
CHAMP="models/az_pi_series.pt"
TABLE="data/series_v_pi.csv"

tee_log() { tee -a "$LOG_FILE"; }
pf() { printf "$1" "$2"; }

# Promotion gate: paired-series sweep-share CI (decisive pairs); falls back
# to the raw series win-rate CI when every pair splits.
parse_ci_lo() {
    local lo
    lo=$(grep 'sweep share' "$1" | grep -oP 'Wilson95 \[\K[0-9.]+' | head -1)
    [[ -z "$lo" ]] && lo=$(grep '^p0 win rate' "$1" \
        | grep -oP 'Wilson95 \[\K[0-9.]+' | head -1)
    echo "$lo"
}
ci_clears_half() {
    local lo; lo=$(parse_ci_lo "$1")
    [[ -n "$lo" ]] && python3 -c "import sys; sys.exit(0 if $lo > 0.5 else 1)"
}

# Refresh the V table from the last MIX_GENS+1 gens' outcome CSVs.
refresh_table() {  # $1 = gen, $2 = extra markov flags
    local files=() g
    for g in $(seq "$1" -1 $(( $1 - MIX_GENS ))); do
        [[ "$g" -ge 1 && -f "$(pf "$OUTC" "$g")" ]] && files+=("$(pf "$OUTC" "$g")")
    done
    # shellcheck disable=SC2086
    uv run python analysis/series_markov.py --outcomes "${files[@]}" \
        --out "$TABLE" --gen "$1" $2 2>&1 | tee_log
}

if [[ -z "$START" ]]; then
    LATEST=$(ls "models/az_pi_s_${TAG}_gen"[0-9]*.pt 2>/dev/null \
        | grep -oP 'gen\K[0-9]+(?=\.pt)' | sort -n | tail -1)
    START=$(( ${LATEST:-0} + 1 ))
fi
[[ -z "$END" ]] && END=$START

for GEN in $(seq "$START" "$END"); do
    M=$(pf "$MODEL" "$GEN"); D=$(pf "$DATA" "$GEN"); O=$(pf "$OUTC" "$GEN")
    {
        DIV='════════════════════════════════════════════════════════════════════'
        printf '\n%s\n  az_pi SERIES gen %d  games=%d sims=%d slots=%d epochs=%d mix=%d(x%.1f)\n  %s\n%s\n' \
            "$DIV" "$GEN" "$GAMES" "$SIMS" "$SLOTS" "$EPOCHS" "$MIX_GENS" "$MIX_DECAY" \
            "$(date '+%Y-%m-%d %H:%M:%S')" "$DIV"
    } | tee_log

    # 1. Self-play. Gen 1 bootstraps with the OLD score-blind champion
    # (--series-states: random scores recorded, search/net unconditioned);
    # later gens condition the search on the current table.
    if [[ "$GEN" -eq 1 ]]; then
        SP_MODEL="models/az_pi.pt"; SP_SERIES="--series-states"
    else
        SP_MODEL="$CHAMP"; SP_SERIES="--series-table $TABLE"
    fi
    echo "  [1/5] Self-play $GAMES games ($SP_MODEL, sims=$SIMS)..." | tee_log
    # shellcheck disable=SC2086
    bin/az_pi_selfplay --model "$SP_MODEL" --games "$GAMES" --sims "$SIMS" \
        --slots "$SLOTS" --threads "$THREADS" --temp-moves 0 $SP_SERIES \
        --outcomes "$O" --seed "$((SEED + GEN))" --device "$DEVICE" \
        --out "$D" 2>&1 | tee_log

    # 2. Refresh the V table (gen1: pooled-global only — score-blind play is
    # state-independent, per-state splits would be noise).
    echo "  [2/5] Refresh series V table..." | tee_log
    if [[ "$GEN" -eq 1 ]]; then
        refresh_table "$GEN" "--bootstrap-global"
    else
        refresh_table "$GEN" ""
    fi

    # 3. Train (mixed over last MIX_GENS gens, series targets from the table).
    echo "  [3/5] Train gen$GEN ($EPOCHS epochs, mix=$MIX_GENS)..." | tee_log
    FILES="$D"
    for i in $(seq 1 "$MIX_GENS"); do
        g=$((GEN - i))
        [[ "$g" -ge 1 && -f "$(pf "$DATA" "$g")" ]] && FILES="$FILES $(pf "$DATA" "$g")"
    done
    WARM=""
    [[ "$GEN" -eq 1 && -n "$INIT_FROM" ]] && WARM="--init-from $INIT_FROM"
    # shellcheck disable=SC2086
    uv run python -m nn.train_az_pi --data $FILES --out "$M" \
        --width "$WIDTH" --blocks "$BLOCKS" --embed "$EMBED" \
        --epochs "$EPOCHS" --batch "$BATCH" --mix-decay "$MIX_DECAY" \
        --series-table "$TABLE" $WARM \
        --device "$DEVICE" 2>&1 | grep -E '\[pi\]' | tee_log

    # 3b. Sample games: deterministic, FIXED deals across gens, at three pinned
    # series states so score-dependent play is visible on identical cards.
    echo "  [3b/5] Sample-game HTMLs (states 0,0 / 40,0 / 0,40)..." | tee_log
    for PTS in 0,0 40,0 0,40; do
        SUF="${PTS/,/_}"
        bin/az_pi_selfplay --model "$M" --games 10 --sims "$SIMS" --slots 10 \
            --threads "$THREADS" --temp-moves 0 --dirichlet-eps 0 --seed 777 \
            --series-table "$TABLE" --fixed-pts "$PTS" --device "$DEVICE" \
            --out "samples/az_pi_s_gen${GEN}_${SUF}_games.parquet" 2>&1 \
            | grep -v '  games ' | tee_log
        uv run python scripts/sample_games_pi.py \
            "samples/az_pi_s_gen${GEN}_${SUF}_games.parquet" --games 10 \
            --model "$M" \
            --out "samples/az_pi_s_gen${GEN}_${SUF}.html" 2>&1 | tee_log
    done

    # 4. Eval genN vs champion: paired full series. Gen 1 has no series
    # champion yet — it seeds the line unconditionally.
    EV="logs/az_pi_s_eval_gen${GEN}.txt"
    if [[ ! -f "$CHAMP" ]]; then
        cp "$M" "$CHAMP"
        echo "  [4/5] champion := gen$GEN (series line seeded)" | tee_log
    else
        echo "  [4/5] Eval gen$GEN vs champion ($EVAL_SERIES paired series, sims=$EVAL_SIMS)..." | tee_log
        bin/eval_az_pi_match --p0 "pi:$M" --p1 "pi:$CHAMP" \
            --series "$EVAL_SERIES" --series-table "$TABLE" \
            --sims "$EVAL_SIMS" --max-slots "$SLOTS" \
            --device "$DEVICE" --seed "$SEED" 2>&1 | tee "$EV" | tee_log
        if ci_clears_half "$EV"; then
            cp "$M" "$CHAMP"
            echo "  ✓ gen$GEN PROMOTED -> champion" | tee_log
        else
            echo "  · gen$GEN not promoted (champion unchanged)" | tee_log
        fi
    fi
    echo "  [5/5] gen$GEN done." | tee_log
done
echo "az_pi series loop done." | tee_log
