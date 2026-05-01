#!/usr/bin/env bash
# PIMC(20) vs PIMC(20) eval_match profiling helper.
#
# Uses a fixed seed (default 42) so deal order and positions match across builds.
#
# Usage:
#   ./scripts/profile_pimc_selfplay.sh --mode perf
#   ./scripts/profile_pimc_selfplay.sh --mode callgrind --target-seconds 180
#   ./scripts/profile_pimc_selfplay.sh --mode perf --mode plain --perf-record
#   ./scripts/profile_pimc_selfplay.sh --mode perf --deals 1200    # skip calibration
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_REL="bin/eval_match"
EXE="${ROOT}/${BIN_REL}"

DEFAULT_SEED=42
DEFAULT_TARGET_SEC=150
DEFAULT_THREADS=1
DEFAULT_PROBE_PERF=5
DEFAULT_PROBE_CALLGRIND=1

MODES=()
SEED="$DEFAULT_SEED"
TARGET_SEC="$DEFAULT_TARGET_SEC"
THREADS="$DEFAULT_THREADS"
FIXED_DEALS=""
PROBE_PERF="$DEFAULT_PROBE_PERF"
PROBE_CALLGRIND="$DEFAULT_PROBE_CALLGRIND"
DO_BUILD=true
PERF_RECORD=false
PERF_DATA="${ROOT}/prof/perf.data"
CALLGRIND_OUT="${ROOT}/prof/callgrind_pimc20.out"
CALL_GRAPH="dwarf"

usage() {
  cat <<'HDR'
Symmetric PIMC(20) self-play profiler (runs bin/eval_match). Fixed default seed keeps
deal order repeatable across optimizations.

HDR
  cat <<EOF
Required (at least once):
  --mode perf|--mode callgrind|--mode plain

Optional:
  --seed N                   RNG base seed for eval_match (default: ${DEFAULT_SEED})
  --target-seconds N         Calibration wall-time target seconds (default: ${DEFAULT_TARGET_SEC})
  --threads N                Worker threads (default: ${DEFAULT_THREADS}; use 1 for profiling)
  --deals N                  Skip calibration; run exactly N unique deals per mode
  --probe-perf-deals N       Probe size for perf/plain calibration (default: ${DEFAULT_PROBE_PERF})
  --probe-callgrind-deals N  Probe size for callgrind (default: ${DEFAULT_PROBE_CALLGRIND})
  --no-build                 Skip 'make eval_match'
  --perf-record              Run 'perf record' (-g --call-graph) instead of perf stat
  --perf-data PATH           Output for perf.data (default: repo/prof/perf.data)
  --callgrind-out PATH       Callgrind outfile (default: repo/prof/callgrind_pimc20.out)
  --call-graph STYLE         dwarf|fp when using --perf-record (default: ${CALL_GRAPH})

Rebuild tip (better stacks):  make clean && CXXFLAGS='-O2 -g' make eval_match

perf may need:  sudo sysctl kernel.perf_event_paranoid=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { echo "error: --mode requires a value"; exit 1; }
      MODES+=("$2")
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "error: --seed requires a value"; exit 1; }
      SEED="$2"
      shift 2
      ;;
    --target-seconds)
      [[ $# -ge 2 ]] || { echo "error: --target-seconds requires a value"; exit 1; }
      TARGET_SEC="$2"
      shift 2
      ;;
    --threads)
      [[ $# -ge 2 ]] || { echo "error: --threads requires a value"; exit 1; }
      THREADS="$2"
      shift 2
      ;;
    --deals)
      [[ $# -ge 2 ]] || { echo "error: --deals requires a value"; exit 1; }
      FIXED_DEALS="$2"
      shift 2
      ;;
    --probe-perf-deals)
      [[ $# -ge 2 ]] || { echo "error: --probe-perf-deals requires a value"; exit 1; }
      PROBE_PERF="$2"
      shift 2
      ;;
    --probe-callgrind-deals)
      [[ $# -ge 2 ]] || { echo "error: --probe-callgrind-deals requires a value"; exit 1; }
      PROBE_CALLGRIND="$2"
      shift 2
      ;;
    --no-build) DO_BUILD=false; shift ;;
    --perf-record) PERF_RECORD=true; shift ;;
    --perf-data)
      [[ $# -ge 2 ]] || { echo "error: --perf-data requires a value"; exit 1; }
      PERF_DATA="$2"
      shift 2
      ;;
    --callgrind-out)
      [[ $# -ge 2 ]] || { echo "error: --callgrind-out requires a value"; exit 1; }
      CALLGRIND_OUT="$2"
      shift 2
      ;;
    --call-graph)
      [[ $# -ge 2 ]] || { echo "error: --call-graph requires a value"; exit 1; }
      CALL_GRAPH="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ ${#MODES[@]} -gt 0 ]] || {
  echo "error: specify at least one --mode (perf | callgrind | plain)" >&2
  usage >&2
  exit 1
}

if "$DO_BUILD"; then
  make -C "$ROOT" eval_match
fi

if [[ ! -x "$EXE" ]]; then
  echo "error: missing executable ${BIN_REL}; run make eval_match from repo root." >&2
  exit 1
fi

mkdir -p "${ROOT}/prof"

elapsed_sec_plain() {
  local deals="$1"
  if [[ "${BASH_VERSINFO[0]}" -ge 5 ]]; then
    local start="$EPOCHREALTIME"
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
      --deals "$deals" --seed "$SEED" --threads "$THREADS" \
      >/dev/null 2>&1
    local end="$EPOCHREALTIME"
    awk -v a="$start" -v b="$end" 'BEGIN { printf "%f\n", (b > a ? b - a : 1e-9) }'
  else
    perl -e '
      use Time::HiRes qw(time);
      my ($exe, @args) = @ARGV;
      open(my $bak, ">&", \*STDOUT) or die $!;
      open(STDOUT, ">", "/dev/null") or die $!;
      open(STDERR, ">", "/dev/null") or die $!;
      my $t0 = time();
      system $exe, @args and exit($? >> 8);
      open(STDOUT, ">&", $bak);
      printf "%f\n", time() - $t0;
    ' "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
      --deals "$deals" --seed "$SEED" --threads "$THREADS"
  fi
}

# negligible vs plain; extrapolate perf-sized runs from plain probe time
elapsed_sec_perf_stat() {
  elapsed_sec_plain "$1"
}

elapsed_sec_callgrind() {
  local deals="$1"
  valgrind --tool=callgrind --instr-atstart=yes --quiet \
    --callgrind-out-file=/dev/null -- \
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
      --deals "$deals" --seed "$SEED" --threads "$THREADS" \
    >/dev/null 2>&1
}

scale_deals_from_probe() {
  local probe="$1"
  local t_probe="$2"
  awk -v p="$probe" -v targ="$TARGET_SEC" -v t="$t_probe" '
    BEGIN {
      if (t <= 0) t = 1e-9;
      x = int(p * targ / t + 0.5);
      if (x < 1) x = 1;
      print x;
    }
  ' </dev/null
}

resolve_deals() {
  local kind="$1"
  local probe="$2"

  if [[ "$FIXED_DEALS" != "" ]]; then
    echo "$FIXED_DEALS"
    return
  fi

  if [[ "$probe" -lt 1 ]]; then
    echo "error: probe deals must be >= 1 ($kind)" >&2
    exit 1
  fi

  echo "Calib ($kind): probe deals=${probe}, seed=${SEED}, threads=${THREADS}" >&2
  local t_probe
  case "$kind" in
    plain) t_probe="$(elapsed_sec_plain "$probe")" ;;
    perf) t_probe="$(elapsed_sec_perf_stat "$probe")" ;;
    callgrind)
      command -v valgrind >/dev/null 2>&1 || {
        echo "error: callgrind requested but valgrind not in PATH" >&2
        exit 1
      }
      t_probe="$(elapsed_sec_callgrind "$probe")"
      ;;
  esac
  echo "(probe wall ${t_probe}s) -> extrapolating to ~${TARGET_SEC}s target ..." >&2
  scale_deals_from_probe "$probe" "$t_probe"
}

require_perf() {
  command -v perf >/dev/null 2>&1 || {
    echo "error: perf not in PATH" >&2
    exit 1
  }
}

run_mode_plain() {
  local deals="$1"
  echo "plain: deals=$deals  total_games=$((2 * deals))  seed=$SEED  threads=$THREADS" >&2
  SECONDS=0
  "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
    --deals "$deals" --seed "$SEED" --threads "$THREADS"
  echo "(plain wall-clock ~ ${SECONDS}s via bash \$SECONDS; see eval_match games/s)" >&2
}

run_mode_perf() {
  local deals="$1"
  echo "perf: deals=$deals  total_games=$((2 * deals))  seed=$SEED  threads=$THREADS" >&2
  mkdir -p "$(dirname "${PERF_DATA}")"
  require_perf

  if "$PERF_RECORD"; then
    echo "Recording to ${PERF_DATA} (--call-graph ${CALL_GRAPH})" >&2
    perf record --output="${PERF_DATA}" -g --call-graph "${CALL_GRAPH}" -- \
      "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
      --deals "$deals" --seed "$SEED" --threads "$THREADS"
    echo "perf report -i '${PERF_DATA}'" >&2
  else
    perf stat -r 1 -- \
      "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
      --deals "$deals" --seed "$SEED" --threads "$THREADS"
  fi
}

run_mode_callgrind() {
  local deals="$1"
  echo "callgrind: deals=$deals  total_games=$((2 * deals))  seed=$SEED  threads=$THREADS" >&2
  mkdir -p "$(dirname "${CALLGRIND_OUT}")"
  command -v valgrind >/dev/null 2>&1 || {
    echo "error: valgrind not in PATH" >&2
    exit 1
  }
  echo "Writing ${CALLGRIND_OUT}" >&2
  valgrind --tool=callgrind --instr-atstart=yes \
    --callgrind-out-file="${CALLGRIND_OUT}" -- \
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 \
    --deals "$deals" --seed "$SEED" --threads "$THREADS"
  echo "KCachegrind / qcachegrind: ${CALLGRIND_OUT}" >&2
}

declare -A DID=()

for m in "${MODES[@]}"; do
  case "$m" in
    perf | callgrind | plain)
      ;;
    *)
      echo "error: unknown mode '${m}' (use perf | callgrind | plain)" >&2
      exit 1
      ;;
  esac
done

for m in "${MODES[@]}"; do
  [[ "${DID[$m]:-}" ]] && continue
  DID["$m"]=1

  deals=""
  echo ""
  echo "================ ${m} ================="
  case "$m" in
    plain)
      deals="$(resolve_deals plain "${PROBE_PERF}")"
      run_mode_plain "$deals"
      ;;
    perf)
      require_perf
      deals="$(resolve_deals perf "${PROBE_PERF}")"
      run_mode_perf "$deals"
      ;;
    callgrind)
      deals="$(resolve_deals callgrind "${PROBE_CALLGRIND}")"
      run_mode_callgrind "$deals"
      ;;
  esac
done

