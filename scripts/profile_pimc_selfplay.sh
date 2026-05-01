#!/usr/bin/env bash
# PIMC(20) vs PIMC(20) eval_match profiling helper.
#
# Default: independent games (--games) — each index uses RNG seed+N for more
# position diversity than paired deals. Optional: --paired-deals for classical
# eval_match paired mode (--deals, 2*N total games).
#
# Usage:
#   ./scripts/profile_pimc_selfplay.sh --mode perf
#   ./scripts/profile_pimc_selfplay.sh --mode callgrind --target-seconds 180
#   ./scripts/profile_pimc_selfplay.sh --mode perf --paired-deals 600   # paired mode
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
# Fixed workload count (--games or --paired-deals overrides calibration)
FIXED=""
USE_PAIRED_DEALS=false
WORKLOAD_CLI_COUNT=0
PROBE_PERF="$DEFAULT_PROBE_PERF"
PROBE_CALLGRIND="$DEFAULT_PROBE_CALLGRIND"
DO_BUILD=true
PERF_RECORD=false
PERF_DATA="${ROOT}/prof/perf.data"
CALLGRIND_OUT="${ROOT}/prof/callgrind_pimc20.out"
CALL_GRAPH="dwarf"

declare -a WAL=()

# eval_match workload: --games (default) or --paired-deals → --deals
populate_workload_argv() {
  if "$USE_PAIRED_DEALS"; then
    WAL=(--deals "$1")
  else
    WAL=(--games "$1")
  fi
}

usage() {
  cat <<'HDR'
Symmetric PIMC(20) eval_match profiler.

Default workload: independent full games (--games), seed+N per game — more move
variety than paired deals. Fixed default seed (--seed 42) keeps work repeatable.

HDR
  cat <<EOF
Required (at least once):
  --mode perf|--mode callgrind|--mode plain

Workload (default independent games):
  (no workload flag)           Calibrated --games count from probe timing
  --games N                    Skip calibration; exactly N independent games
  --paired-deals N             Paired-eval mode (--deals N ⇒ 2*N total games)

Additional options:
  --seed N                     Base seed (default: ${DEFAULT_SEED})
  --target-seconds N           Calibration target seconds (default: ${DEFAULT_TARGET_SEC})
  --threads N                  (default: ${DEFAULT_THREADS}; use 1 when profiling)
  --probe-perf N               Probe size for perf/plain calibration (default: ${DEFAULT_PROBE_PERF})
  --probe-callgrind N          Probe size for callgrind (default: ${DEFAULT_PROBE_CALLGRIND})
  --no-build                   Skip make eval_match
  --perf-record                perf record (-g --call-graph) instead of perf stat
  --perf-data PATH             (default: ${PERF_DATA})
  --callgrind-out PATH         (default: ${CALLGRIND_OUT})
  --call-graph STYLE           dwarf|fp when using --perf-record (default: ${CALL_GRAPH})

Rebuild tip:  make clean && CXXFLAGS='-O2 -g' make eval_match
perf sysctl:   sudo sysctl kernel.perf_event_paranoid=1
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
    --games)
      [[ $# -ge 2 ]] || { echo "error: --games requires a value"; exit 1; }
      FIXED="$2"
      USE_PAIRED_DEALS=false
      WORKLOAD_CLI_COUNT=$((WORKLOAD_CLI_COUNT + 1))
      shift 2
      ;;
    --paired-deals)
      [[ $# -ge 2 ]] || { echo "error: --paired-deals requires a value"; exit 1; }
      FIXED="$2"
      USE_PAIRED_DEALS=true
      WORKLOAD_CLI_COUNT=$((WORKLOAD_CLI_COUNT + 1))
      shift 2
      ;;
    --probe-perf|--probe-perf-deals)
      [[ $# -ge 2 ]] || { echo "error: $1 requires a value"; exit 1; }
      PROBE_PERF="$2"
      shift 2
      ;;
    --probe-callgrind|--probe-callgrind-deals)
      [[ $# -ge 2 ]] || { echo "error: $1 requires a value"; exit 1; }
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

if [[ "$WORKLOAD_CLI_COUNT" -gt 1 ]]; then
  echo "error: use at most one of --games or --paired-deals" >&2
  exit 1
fi

if "$DO_BUILD"; then
  make -C "$ROOT" eval_match
fi

if [[ ! -x "$EXE" ]]; then
  echo "error: missing executable ${BIN_REL}; run make eval_match." >&2
  exit 1
fi

mkdir -p "${ROOT}/prof"

# shellcheck disable=SC2086
elapsed_sec_plain() {
  local n="$1"
  populate_workload_argv "$n"

  if [[ "${BASH_VERSINFO[0]}" -ge 5 ]]; then
    local start="$EPOCHREALTIME"
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
      --seed "$SEED" --threads "$THREADS" \
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
    ' "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
      --seed "$SEED" --threads "$THREADS"
  fi
}

elapsed_sec_perf_stat() {
  elapsed_sec_plain "$1"
}

elapsed_sec_callgrind() {
  local n="$1"
  populate_workload_argv "$n"
  valgrind --tool=callgrind --instr-atstart=yes --quiet \
    --callgrind-out-file=/dev/null -- \
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
    --seed "$SEED" --threads "$THREADS" \
    >/dev/null 2>&1
}

scale_from_probe() {
  awk -v p="$1" -v targ="$TARGET_SEC" -v t="$2" '
    BEGIN {
      if (t <= 0) t = 1e-9;
      x = int(p * targ / t + 0.5);
      if (x < 1) x = 1;
      print x;
    }
  ' </dev/null
}

workload_descr() {
  if "$USE_PAIRED_DEALS"; then
    echo "paired deals (total games=$((2 * $1)))"
  else
    echo "independent games ($(($1)))"
  fi
}

resolve_workload_units() {
  local kind="$1"
  local probe="$2"

  if [[ "$FIXED" != "" ]]; then
    echo "$FIXED"
    return
  fi

  if [[ "$probe" -lt 1 ]]; then
    echo "error: probe must be >= 1 ($kind)" >&2
    exit 1
  fi

  echo "Calib ($kind): probe units=${probe} ($(workload_descr "$probe")), seed=${SEED}, threads=${THREADS}" >&2
  local t_probe
  case "$kind" in
    plain) t_probe="$(elapsed_sec_plain "$probe")" ;;
    perf) t_probe="$(elapsed_sec_perf_stat "$probe")" ;;
    callgrind)
      command -v valgrind >/dev/null 2>&1 || {
        echo "error: valgrind not in PATH" >&2
        exit 1
      }
      t_probe="$(elapsed_sec_callgrind "$probe")"
      ;;
  esac
  echo "(probe wall ${t_probe}s) -> extrapolating to ~${TARGET_SEC}s ..." >&2
  scale_from_probe "$probe" "$t_probe"
}

require_perf() {
  command -v perf >/dev/null 2>&1 || {
    echo "error: perf not in PATH" >&2
    exit 1
  }
}

populate_wald() {
  populate_workload_argv "$1"
}

run_mode_plain() {
  local units="$1"
  populate_wald "$units"
  echo "plain: $(workload_descr "$units"), seed=${SEED} threads=${THREADS}" >&2
  SECONDS=0
  "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
    --seed "$SEED" --threads "$THREADS"
  echo "(plain wall-clock ~ ${SECONDS}s via bash \$SECONDS; see eval_match games/s)" >&2
}

run_mode_perf() {
  local units="$1"
  populate_wald "$units"
  echo "perf: $(workload_descr "$units"), seed=${SEED} threads=${THREADS}" >&2
  mkdir -p "$(dirname "${PERF_DATA}")"
  require_perf

  if "$PERF_RECORD"; then
    echo "Recording to ${PERF_DATA} (--call-graph ${CALL_GRAPH})" >&2
    perf record --output="${PERF_DATA}" -g --call-graph "${CALL_GRAPH}" -- \
      "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
      --seed "$SEED" --threads "$THREADS"
    echo "Inspect: perf report -i '${PERF_DATA}'" >&2
  else
    perf stat -r 1 -- \
      "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
      --seed "$SEED" --threads "$THREADS"
  fi
}

run_mode_callgrind() {
  local units="$1"
  populate_wald "$units"
  echo "callgrind: $(workload_descr "$units"), seed=${SEED} threads=${THREADS}" >&2
  mkdir -p "$(dirname "${CALLGRIND_OUT}")"
  command -v valgrind >/dev/null 2>&1 || {
    echo "error: valgrind not in PATH" >&2
    exit 1
  }
  echo "Writing ${CALLGRIND_OUT}" >&2
  valgrind --tool=callgrind --instr-atstart=yes \
    --callgrind-out-file="${CALLGRIND_OUT}" -- \
    "$EXE" --p0 pimc --p0-param 20 --p1 pimc --p1-param 20 "${WAL[@]}" \
    --seed "$SEED" --threads "$THREADS"
  echo "Open in KCachegrind: ${CALLGRIND_OUT}" >&2
}

declare -A DID=()

for m in "${MODES[@]}"; do
  case "$m" in
    perf | callgrind | plain)
      ;;
    *)
      echo "error: unknown mode '${m}'" >&2
      exit 1
      ;;
  esac
done

for m in "${MODES[@]}"; do
  [[ "${DID[$m]:-}" ]] && continue
  DID["$m"]=1

  echo ""
  echo "================ ${m} ================="
  case "$m" in
    plain)
      units="$(resolve_workload_units plain "${PROBE_PERF}")"
      run_mode_plain "$units"
      ;;
    perf)
      require_perf
      units="$(resolve_workload_units perf "${PROBE_PERF}")"
      run_mode_perf "$units"
      ;;
    callgrind)
      units="$(resolve_workload_units callgrind "${PROBE_CALLGRIND}")"
      run_mode_callgrind "$units"
      ;;
  esac
done
