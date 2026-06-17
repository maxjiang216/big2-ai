#!/usr/bin/env bash
# Self-escalating az_pi SERIES run (clone of run_az_pi_overnight.sh).
#
# Runs one generation at a time via run_az_pi_series_loop.sh. After
# STALL_LIMIT consecutive non-promotions, sims and eval series pairs double
# (slots halve to keep tree memory flat). Resource-bounded: nice'd, cgroup
# MemoryMax so an overrun gets killed instead of freezing the machine; a
# failed gen is retried once at half slots.
#
# Progress: logs/az_pi_series_overnight.csv + logs/az_pi_s.log +
# logs/az_pi_s_eval_gen*.txt + samples/az_pi_s_gen*.html.
#
#   nohup bash scripts/run_az_pi_series_overnight.sh \
#       > logs/az_pi_series_overnight.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

START_GEN=4
MAX_GEN=40
SIMS=1600
EVAL_SERIES=200
MAX_SIMS=12800
MAX_EVAL_SERIES=800
SLOTS=1024     # halved on every escalation: live-tree memory scales with sims
MIN_SLOTS=128
THREADS=6
STALL_LIMIT=3
STALL=1        # gen 3 already stalled once at these settings
MEM_MAX=9G

CSV=logs/az_pi_series_overnight.csv
LOG=logs/az_pi_s.log
[[ -f "$CSV" ]] || echo "gen,sims,eval_series,slots,promoted,game_rate,raw_rate,sweep_share,decisive,start,end,exit" > "$CSV"

note() { echo "[series-overnight $(date '+%H:%M:%S')] $*"; }

mem_ok() {  # require 3.5 GB available before starting a phase
    local avail
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [[ "$avail" -ge 3500 ]]
}

run_gen() {  # $1 = gen, $2 = slots
    systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" \
        nice -n 10 bash scripts/run_az_pi_series_loop.sh \
        --start "$1" --end "$1" \
        --games 20000 --sims "$SIMS" --slots "$2" --threads "$THREADS" \
        --epochs 10 --batch 2048 \
        --eval-series "$EVAL_SERIES" --eval-sims "$SIMS" \
        --device cuda --log "$LOG"
}

stall=$STALL
for GEN in $(seq "$START_GEN" "$MAX_GEN"); do
    until mem_ok; do note "low memory, waiting 60s"; sleep 60; done

    T0=$(date '+%H:%M:%S')
    note "gen $GEN: sims=$SIMS eval_series=$EVAL_SERIES slots=$SLOTS (stall=$stall)"
    run_gen "$GEN" "$SLOTS"
    RC=$?
    if [[ $RC -ne 0 ]]; then
        note "gen $GEN failed (rc=$RC) — retrying once at half slots"
        until mem_ok; do sleep 60; done
        run_gen "$GEN" $((SLOTS / 2))
        RC=$?
        if [[ $RC -ne 0 ]]; then
            note "gen $GEN failed twice (rc=$RC) — stopping run"
            echo "$GEN,$SIMS,$EVAL_SERIES,$SLOTS,ERROR,,,,,$T0,$(date '+%H:%M:%S'),$RC" >> "$CSV"
            break
        fi
    fi

    EV="logs/az_pi_s_eval_gen${GEN}.txt"
    PROMOTED=0
    grep -q "gen${GEN} PROMOTED" "$LOG" && PROMOTED=1
    GRATE=$(grep -oP '^p0 game win rate: \d+/\d+ = \K[0-9.]+' "$EV" 2>/dev/null | head -1)
    RAW=$(grep -oP '^p0 win rate: \d+/\d+ = \K[0-9.]+' "$EV" 2>/dev/null | head -1)
    SHARE=$(grep -oP 'sweep share \K[0-9.]+' "$EV" 2>/dev/null | head -1)
    DECISIVE=$(grep -oP 'decisive deals: \K\d+' "$EV" 2>/dev/null | head -1)
    echo "$GEN,$SIMS,$EVAL_SERIES,$SLOTS,$PROMOTED,${GRATE:-},${RAW:-},${SHARE:-},${DECISIVE:-},$T0,$(date '+%H:%M:%S'),$RC" >> "$CSV"

    if [[ "$PROMOTED" == 1 ]]; then
        stall=0
    else
        stall=$((stall + 1))
        if [[ "$stall" -ge "$STALL_LIMIT" ]]; then
            if [[ "$SIMS" -ge "$MAX_SIMS" && "$EVAL_SERIES" -ge "$MAX_EVAL_SERIES" ]]; then
                note "stalled $stall gens at the caps (sims=$SIMS) — stopping"
                break
            fi
            [[ "$SIMS" -lt "$MAX_SIMS" ]] && SIMS=$((SIMS * 2))
            [[ "$EVAL_SERIES" -lt "$MAX_EVAL_SERIES" ]] && EVAL_SERIES=$((EVAL_SERIES * 2))
            [[ "$SLOTS" -gt "$MIN_SLOTS" ]] && SLOTS=$((SLOTS / 2))
            stall=0
            note "stalled $STALL_LIMIT gens -> escalating to sims=$SIMS eval_series=$EVAL_SERIES slots=$SLOTS"
        fi
    fi
done
note "series overnight run finished"
