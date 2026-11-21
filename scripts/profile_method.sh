#!/usr/bin/env bash
# Profiles ONE method executable with Intel VTune, one analysis per run, pinned to one performance core.
#
#   hotspots            which functions take the time, and the CPU utilization histogram
#   uarch-exploration   CPI, retiring, front-end, bad speculation, memory bound, core bound
#   memory-access       loads, last-level cache misses, DRAM bandwidth and its histogram
#
# VTune starts paused and the executable resumes it only for its iterations (apps/methods/method_main.hpp),
# so reading the matrix file and building the preconditioner are not in the numbers.
# While each analysis runs, the core's frequency and the package temperature are logged once a second
# to NAME-ANALYSIS-thermal.csv, so a slowdown from heat would show.
#
# Usage: scripts/profile_method.sh METHOD MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]
#   METHOD      one of: cg_jacobi cg_sell_jacobi cg_sgs cg_ic cg_ic_fixed cg_ic0 cg_ic0_rcm eigen_e1 eigen_e4
#   REPORT_DIR  receives METHOD-ANALYSIS-summary.txt, -functions.csv, -run.log and -thermal.csv
#   RAW_DIR     receives the VTune result folders, to open in vtune-gui (default ~/p08/vtune-methods)
#   CPU         logical CPU to pin to, a performance core (default 0)
#
# Needs: source /opt/intel/oneapi/setvars.sh, the sampling driver loaded (sepdk/src/insmod-sep),
#        cmake --preset profile && cmake --build build/profile --target methods
set -u

usage="usage: profile_method.sh METHOD MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]"
method=${1:?$usage}
matrix=${2:?$usage}
reports=${3:?$usage}
raw=${4:-$HOME/p08/vtune-methods}
cpu=${5:-0}
build=${BUILD:-build/profile}
analyses=${ANALYSES:-"hotspots uarch-exploration memory-access"}
maxit=${MAXIT:-30000}

executable="$build/methods/$method"
[ -x "$executable" ] || { echo "no executable $executable: build the 'methods' target first" >&2; exit 1; }
mkdir -p "$reports" "$raw"

temperature_file=
for zone in /sys/class/thermal/thermal_zone*; do
  [ "$(cat "$zone/type" 2>/dev/null)" = x86_pkg_temp ] && temperature_file="$zone/temp"
done

# Logs "seconds,MHz of the pinned CPU,package °C" once a second until killed.
log_thermal() {
  local start=$SECONDS
  echo "seconds,mhz,package_celsius"
  while true; do
    local mhz temp=""
    mhz=$(awk -v cpu="$cpu" '/^processor/ {p = $3} /^cpu MHz/ && p == cpu {print int($4); exit}' /proc/cpuinfo)
    [ -n "$temperature_file" ] && temp=$(( $(cat "$temperature_file") / 1000 ))
    echo "$((SECONDS - start)),$mhz,$temp"
    sleep 1
  done
}

failed=0
for analysis in $analyses; do
  result="$raw/$method-$analysis"
  rm -rf "$result"
  # A sample every millisecond makes a hotspots result of 1 GB for a 50 s solve. Hotspots and memory-access
  # only need function shares and bandwidth, so they sample less often; uarch-exploration keeps 1 ms,
  # because it rotates many counters through the samples and needs them all for its percentages.
  case "$analysis" in
    hotspots) knobs=(-knob sampling-mode=hw -knob sampling-interval=5) ;;
    memory-access) knobs=(-knob sampling-interval=2) ;;
    *) knobs=() ;;
  esac
  echo "== $method: $analysis"
  log_thermal > "$reports/$method-$analysis-thermal.csv" &
  logger=$!
  if ! vtune -collect "$analysis" "${knobs[@]}" -start-paused -result-dir "$result" -quiet -- \
      taskset -c "$cpu" "$executable" --matrix "$matrix" --maxit "$maxit" > "$reports/$method-$analysis-run.log" 2>&1; then
    echo "FAILED: see $reports/$method-$analysis-run.log"
    failed=1
  fi
  kill "$logger" 2>/dev/null
  wait "$logger" 2>/dev/null
  [ -d "$result" ] || continue
  vtune -report summary -result-dir "$result" -report-knob show-issues=false \
      -report-output "$reports/$method-$analysis-summary.txt" > /dev/null 2>&1
  vtune -report hotspots -result-dir "$result" -format csv -csv-delimiter comma \
      -report-output "$reports/$method-$analysis-functions.csv" > /dev/null 2>&1
  rm -rf "$result"/data.*  # raw samples, not needed once finalized: vtune-gui opens sqlite-db
done
exit $failed
