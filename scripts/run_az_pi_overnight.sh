#!/usr/bin/env bash
# Overnight self-escalating az_pi run.
#
# Runs one generation at a time via run_az_pi_loop.sh. Starts at --sims 3200 /
# --eval-deals 3200 (2x the gen-16 plateau settings). After 3 consecutive
# non-promotions both are doubled again (up to caps). Resource-bounded:
# nice'd, 6 threads, 1024 slots, cgroup MemoryMax so an overrun gets killed
# instead of freezing the machine; a failed gen is retried once at half slots.
#
# Progress: logs/az_pi_overnight.csv (one line per gen) + logs/az_pi.log
# (full detail) + logs/az_pi_eval_gen*.txt + samples/az_pi_gen*.html.
#
#   nohup bash scripts/run_az_pi_overnight.sh > logs/az_pi_overnight.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

START_GEN=17
MAX_GEN=60
SIMS=3200
EVAL_DEALS=3200
MAX_SIMS=25600
MAX_EVAL_DEALS=12800
SLOTS=1024     # halved on every escalation: live-tree memory scales with sims
MIN_SLOTS=128
THREADS=6
STALL_LIMIT=3
MEM_MAX=9G     # cgroup cap: an overrun is killed, never freezes the machine

CSV=logs/az_pi_overnight.csv
LOG=logs/az_pi.log
[[ -f "$CSV" ]] || echo "gen,sims,eval_deals,slots,promoted,raw_rate,sweep_share,decisive,start,end,exit" > "$CSV"

note() { echo "[overnight $(date '+%H:%M:%S')] $*"; }

mem_ok() {  # require 3.5 GB available before starting a phase
    local avail
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [[ "$avail" -ge 3500 ]]
}

run_gen() {  # $1 = gen, $2 = slots; returns loop exit code
    systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" \
        nice -n 10 bash scripts/run_az_pi_loop.sh \
        --start "$1" --end "$1" \
        --games 20000 --sims "$SIMS" --slots "$2" --threads "$THREADS" \
        --epochs 10 --batch 2048 \
        --eval-deals "$EVAL_DEALS" --eval-sims "$SIMS" \
        --device cuda --log "$LOG"
}

stall=0
for GEN in $(seq "$START_GEN" "$MAX_GEN"); do
    until mem_ok; do note "low memory, waiting 60s"; sleep 60; done

    T0=$(date '+%H:%M:%S')
    note "gen $GEN: sims=$SIMS eval_deals=$EVAL_DEALS slots=$SLOTS (stall=$stall)"
    run_gen "$GEN" "$SLOTS"
    RC=$?
    if [[ $RC -ne 0 ]]; then
        note "gen $GEN failed (rc=$RC) — retrying once at half slots"
        until mem_ok; do sleep 60; done
        run_gen "$GEN" $((SLOTS / 2))
        RC=$?
        if [[ $RC -ne 0 ]]; then
            note "gen $GEN failed twice (rc=$RC) — stopping overnight run"
            echo "$GEN,$SIMS,$EVAL_DEALS,$SLOTS,ERROR,,,,$T0,$(date '+%H:%M:%S'),$RC" >> "$CSV"
            break
        fi
    fi

    # Harvest results for the CSV from the loop log / eval file.
    EV="logs/az_pi_eval_gen${GEN}.txt"
    PROMOTED=0
    grep -q "gen${GEN} PROMOTED" "$LOG" && PROMOTED=1
    RAW=$(grep -oP 'p0 win rate: \d+/\d+ = \K[0-9.]+' "$EV" 2>/dev/null | head -1)
    SHARE=$(grep -oP 'sweep share \K[0-9.]+' "$EV" 2>/dev/null | head -1)
    DECISIVE=$(grep -oP 'decisive deals: \K\d+' "$EV" 2>/dev/null | head -1)
    echo "$GEN,$SIMS,$EVAL_DEALS,$SLOTS,$PROMOTED,${RAW:-},${SHARE:-},${DECISIVE:-},$T0,$(date '+%H:%M:%S'),$RC" >> "$CSV"

    if [[ "$PROMOTED" == 1 ]]; then
        stall=0
    else
        stall=$((stall + 1))
        if [[ "$stall" -ge "$STALL_LIMIT" ]]; then
            if [[ "$SIMS" -ge "$MAX_SIMS" && "$EVAL_DEALS" -ge "$MAX_EVAL_DEALS" ]]; then
                note "stalled $stall gens at the caps (sims=$SIMS) — stopping"
                break
            fi
            [[ "$SIMS" -lt "$MAX_SIMS" ]] && SIMS=$((SIMS * 2))
            [[ "$EVAL_DEALS" -lt "$MAX_EVAL_DEALS" ]] && EVAL_DEALS=$((EVAL_DEALS * 2))
            # Tree memory scales with sims: halve concurrency to compensate.
            [[ "$SLOTS" -gt "$MIN_SLOTS" ]] && SLOTS=$((SLOTS / 2))
            stall=0
            note "stalled $STALL_LIMIT gens -> escalating to sims=$SIMS eval_deals=$EVAL_DEALS slots=$SLOTS"
        fi
    fi
done
note "overnight run finished"
