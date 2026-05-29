#!/usr/bin/env bash
# az_search self-play LIMIT TEST.
#
# Three independent runs distinguished by self-play search budget (sims = 1, 10,
# 100), each starting from the SHARED teacher-bootstrapped gen0 and self-playing
# forward for GENS generations. Training uses the closed-loop plateau schedule.
# After each generation the current gen is evaluated (batched) against a panel
# (random, greedy, pimc(20), typed_search, current champion); we report overall
# win rate + Wilson CI AND the primary sweep-ratio metric. A gen is promoted to
# champion when it clears the champion's Wilson 50% line. Finally the best model
# of each run cross-plays the others (each at its own sims via --sims-b).
#
# Usage: scripts/run_az_experiment.sh [--sims-list "1 10 100"] [--games N]
#          [--gens N] [--slots N] [--eval-deals N] [--eval-slots N]
#          [--batch N] [--lr F] [--epochs N] [--mix-gens N] [--mix-decay F]
#          [--device cuda|cpu] [--seed N] [--g0-player PATH] [--g0-opp PATH]

set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

SIMS_LIST="1 10 100"
GAMES=50000
GENS=10
SLOTS=4096
EVAL_DEALS=1000
EVAL_SLOTS=2000
BATCH=8192
LR=6e-4              # ~sqrt(BATCH/2048)*3e-4; calibration (T36) tunes this
EPOCHS=80           # plateau early-stops well before this
MIX_GENS=3
MIX_DECAY=0.5
DEVICE=cuda
SEED=42
G0_PLAYER=models/az_player_gen0.pt   # shared teacher-bootstrapped gen0
G0_OPP=models/az_opp_gen0.pt
LOG_DIR=logs/az_experiment

while [[ $# -gt 0 ]]; do
  case $1 in
    --sims-list)  SIMS_LIST=$2; shift 2 ;;
    --games)      GAMES=$2; shift 2 ;;
    --gens)       GENS=$2; shift 2 ;;
    --slots)      SLOTS=$2; shift 2 ;;
    --eval-deals) EVAL_DEALS=$2; shift 2 ;;
    --eval-slots) EVAL_SLOTS=$2; shift 2 ;;
    --batch)      BATCH=$2; shift 2 ;;
    --lr)         LR=$2; shift 2 ;;
    --epochs)     EPOCHS=$2; shift 2 ;;
    --mix-gens)   MIX_GENS=$2; shift 2 ;;
    --mix-decay)  MIX_DECAY=$2; shift 2 ;;
    --device)     DEVICE=$2; shift 2 ;;
    --seed)       SEED=$2; shift 2 ;;
    --g0-player)  G0_PLAYER=$2; shift 2 ;;
    --g0-opp)     G0_OPP=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" data models
[[ -f "$G0_PLAYER" && -f "$G0_OPP" ]] || { echo "missing gen0 models $G0_PLAYER / $G0_OPP" >&2; exit 1; }

# ── output parsers ────────────────────────────────────────────────────────────
pw()       { grep -oP 'A wins \d+ / \d+ = \K[0-9.]+' "$1" | head -1; }
pci()      { grep -oP 'Wilson 95% CI \K\[[0-9., ]+\]' "$1" | head -1; }
pci_lo()   { grep -oP 'overall:.*|A wins.*Wilson 95% CI \[\K[0-9.]+' "$1" | grep -oP '^[0-9.]+' | head -1; }
psweep()   { grep -oP 'sweeps: \KA=\d+  B=\d+  split=\d+' "$1" | head -1; }
pswline()  { grep -E 'A SWEEPS|B sweeps|sweep ratio not' "$1" | head -1; }
overall_lo() { grep -oP 'A wins \d+ / \d+ = [0-9.]+  \(Wilson 95% CI \[\K[0-9.]+' "$1" | head -1; }

# batched eval of (player A, opp A) at sims SA vs a panel member -> prints summary line
eval_vs() {  # eval_vs <pa> <oa> <sims> <label> <opp-args...>  (writes to global $OUTF)
  local pa=$1 oa=$2 sa=$3 label=$4; shift 4
  bin/eval_az_match --player-a "$pa" --opp-a "$oa" --sims "$sa" \
      --deals "$EVAL_DEALS" --slots "$EVAL_SLOTS" --device "$DEVICE" --seed "$SEED" \
      "$@" >"$OUTF" 2>/dev/null || true
  printf '      vs %-14s win=%s %s | sweeps %s\n' \
      "$label" "$(pw "$OUTF")" "$(pci "$OUTF")" "$(psweep "$OUTF")"
}

run_one() {  # run_one <sims>
  local S=$1 tag="s${1}"
  local cp="models/az_${tag}_champion_player.pt" co="models/az_${tag}_champion_opp.pt"
  local LOG="$LOG_DIR/run_${tag}.log"
  : >"$LOG"
  cp "$G0_PLAYER" "models/az_${tag}_player_gen0.pt"
  cp "$G0_OPP"    "models/az_${tag}_opp_gen0.pt"
  cp "$G0_PLAYER" "$cp"; cp "$G0_OPP" "$co"
  echo "════ RUN sims=$S  (games=$GAMES gens=$GENS batch=$BATCH lr=$LR) ════" | tee -a "$LOG"

  OUTF=$(mktemp)
  for GEN in $(seq 1 "$GENS"); do
    local pd="data/az_${tag}_player_gen${GEN}.parquet" od="data/az_${tag}_opp_gen${GEN}.parquet"
    local pm="models/az_${tag}_player_gen${GEN}.pt"     om="models/az_${tag}_opp_gen${GEN}.pt"
    printf '\n── [%s] gen %d  %s ──\n' "$tag" "$GEN" "$(date '+%H:%M:%S')" | tee -a "$LOG"

    local T0=$(date +%s)
    bin/az_selfplay --games "$GAMES" --sims "$S" --slots "$SLOTS" --device "$DEVICE" \
        --seed "$((SEED + GEN))" \
        --player-model "$cp" --opp-model "$co" \
        --player-out "$pd" --opp-out "$od" 2>&1 | grep -E 'samples:' | tee -a "$LOG"
    echo "    self-play ${S}-sims: $(( $(date +%s) - T0 ))s" | tee -a "$LOG"

    local PF="$pd" OF="$od" i g
    for i in $(seq 1 "$MIX_GENS"); do
      g=$((GEN - i))
      if [[ "$g" -ge 1 && -f "data/az_${tag}_player_gen${g}.parquet" ]]; then
        PF="$PF data/az_${tag}_player_gen${g}.parquet"
        OF="$OF data/az_${tag}_opp_gen${g}.parquet"
      fi
    done
    T0=$(date +%s)
    # shellcheck disable=SC2086
    uv run python -m nn.train_az --player-data $PF --opp-data $OF \
        --player-out "$pm" --opp-out "$om" \
        --sched plateau --ckpt-metric head --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
        --mix-decay "$MIX_DECAY" 2>&1 | grep -E 'done\.' | tee -a "$LOG"
    echo "    train: $(( $(date +%s) - T0 ))s" | tee -a "$LOG"

    echo "    panel eval (deals=$EVAL_DEALS, sims=$S):" | tee -a "$LOG"
    eval_vs "$pm" "$om" "$S" random       --classic random        | tee -a "$LOG"
    eval_vs "$pm" "$om" "$S" greedy        --classic greedy        | tee -a "$LOG"
    eval_vs "$pm" "$om" "$S" "pimc(20)"    --classic pimc --classic-param 20 | tee -a "$LOG"
    eval_vs "$pm" "$om" "$S" typed_search  --classic typed_search  | tee -a "$LOG"
    eval_vs "$pm" "$om" "$S" champion      --player-b "$cp" --opp-b "$co" | tee -a "$LOG"

    # promotion: gen vs champion, overall Wilson lower bound > 0.5
    local lo; lo=$(overall_lo "$OUTF")
    if [[ -n "$lo" ]] && python3 -c "import sys;sys.exit(0 if $lo>0.5 else 1)"; then
      cp "$pm" "$cp"; cp "$om" "$co"
      echo "    -> PROMOTED gen$GEN to ${tag} champion (vs-champ lo=$lo)" | tee -a "$LOG"
    else
      echo "    -> gen$GEN not promoted (vs-champ lo=${lo:-NA})" | tee -a "$LOG"
    fi
  done
  rm -f "$OUTF"
  echo "════ RUN sims=$S done. champion = $cp ════" | tee -a "$LOG"
}

# ── run the three sims settings ────────────────────────────────────────────────
for S in $SIMS_LIST; do run_one "$S"; done

# ── final cross-play of the best (champion) model of each run ──────────────────
XLOG="$LOG_DIR/crossplay.log"; : >"$XLOG"
echo "════ CROSS-PLAY (each at its own sims) ════" | tee -a "$XLOG"
read -ra SARR <<<"$SIMS_LIST"
for ((i=0;i<${#SARR[@]};i++)); do
  for ((j=i+1;j<${#SARR[@]};j++)); do
    SA=${SARR[$i]} SB=${SARR[$j]}
    bin/eval_az_match \
      --player-a "models/az_s${SA}_champion_player.pt" --opp-a "models/az_s${SA}_champion_opp.pt" \
      --player-b "models/az_s${SB}_champion_player.pt" --opp-b "models/az_s${SB}_champion_opp.pt" \
      --sims "$SA" --sims-b "$SB" --deals "$EVAL_DEALS" --slots "$EVAL_SLOTS" \
      --device "$DEVICE" --seed "$SEED" 2>/dev/null | tee -a "$XLOG"
    echo "  (A=sims$SA  B=sims$SB)" | tee -a "$XLOG"
  done
done
echo "══ experiment done ══" | tee -a "$XLOG"
