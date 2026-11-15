#!/usr/bin/env bash
# Profiles every solver on one matrix with Intel VTune and saves text reports.
#
# For each solver it collects three analyses, each in its own run pinned to one core:
#   hotspots (hardware sampling)   which functions take the time
#   uarch-exploration              where the pipeline stalls: memory, core, front end, bad speculation
#   memory-access                  loads, last-level cache misses and memory bandwidth per function
#
# Hotspots and memory-access need Intel's sampling driver. Without it, on the hybrid i5-1335U, both lost
# most of their samples (0.1 s of CPU time reported for a 28 s solve). The VTune installer builds the
# driver; load it until the next reboot with
#   sudo /opt/intel/oneapi/vtune/latest/sepdk/src/insmod-sep -g "$(id -gn)"
# If it cannot be loaded, run with HOTSPOTS_MODE=sw ANALYSES="hotspots uarch-exploration".
#
# Usage: scripts/profile_vtune.sh MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]
#   REPORT_DIR  receives NAME-ANALYSIS-summary.txt and NAME-ANALYSIS-functions.csv
#   RAW_DIR     receives the VTune result folders, for vtune-gui (default ~/p08/vtune)
#   CPU         logical CPU to pin to; pick a performance core from lscpu --all --extended (default 0)
#
# Needs: source /opt/intel/oneapi/setvars.sh
#        sudo sysctl kernel.perf_event_paranoid=0 kernel.kptr_restrict=0 kernel.yama.ptrace_scope=0
#        cmake --preset profile && cmake --build build/profile --target solve bench_cg
set -u

matrix=${1:?usage: profile_vtune.sh MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]}
reports=${2:?usage: profile_vtune.sh MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]}
raw=${3:-$HOME/p08/vtune}
cpu=${4:-0}
build=${BUILD:-build/profile}
analyses=${ANALYSES:-"hotspots uarch-exploration memory-access"}
only=${ONLY:-}

mkdir -p "$reports" "$raw"

solve="$build/solve --matrix $matrix --problem random --maxit 30000"
bench="$build/bench_cg --matrix $matrix --maxit 30000 --repeat 1 --warmup 0"

names=(jacobi sell-jacobi sgs ic ic-fixed ic0 ic0-rcm eigen-E1 eigen-E4)
commands=(
  "$solve --precond jacobi"
  "$solve --precond jacobi --format sell"
  "$solve --precond sgs"
  "$solve --precond ic"
  "$solve --precond ic-fixed"
  "$solve --precond ic0"
  "$solve --precond ic0 --order rcm"
  "$bench --solver eigen-E1"
  "$bench --solver eigen-E4"
)

failed=0
for i in "${!names[@]}"; do
  name=${names[$i]}
  if [ -n "$only" ] && [[ ",$only," != *",$name,"* ]]; then continue; fi
  for analysis in $analyses; do
    result="$raw/$name-$analysis"
    rm -rf "$result"
    knobs=()
    [ "$analysis" = hotspots ] && knobs=(-knob "sampling-mode=${HOTSPOTS_MODE:-hw}")
    echo "== $name: $analysis"
    # shellcheck disable=SC2086
    if ! vtune -collect "$analysis" "${knobs[@]}" -result-dir "$result" -quiet -- \
        taskset -c "$cpu" ${commands[$i]} > "$reports/$name-$analysis-run.log" 2>&1; then
      echo "FAILED: see $reports/$name-$analysis-run.log"
      failed=1
      continue
    fi
    vtune -report summary -result-dir "$result" -report-knob show-issues=false \
        -report-output "$reports/$name-$analysis-summary.txt" > /dev/null 2>&1
    vtune -report hotspots -result-dir "$result" -format csv -csv-delimiter comma \
        -report-output "$reports/$name-$analysis-functions.csv" > /dev/null 2>&1
  done
done
exit $failed
