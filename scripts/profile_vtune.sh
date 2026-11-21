#!/usr/bin/env bash
# Profiles the nine methods one after another, each its own executable in its own VTune runs
# (scripts/profile_method.sh does one method), then writes the table of the key metrics.
#
# Usage: scripts/profile_vtune.sh MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]
#   ONLY=cg_ic0,cg_jacobi   profiles only the named methods
#   ANALYSES="uarch-exploration"   runs only some analyses
#
# Needs: source /opt/intel/oneapi/setvars.sh, the sampling driver loaded,
#        cmake --preset profile && cmake --build build/profile --target methods
set -u

usage="usage: profile_vtune.sh MATRIX.mtx REPORT_DIR [RAW_DIR] [CPU]"
matrix=${1:?$usage}
reports=${2:?$usage}
raw=${3:-$HOME/p08/vtune-methods}
cpu=${4:-0}
only=${ONLY:-}
here=$(dirname "$0")

methods=(cg_jacobi cg_sell_jacobi cg_sgs cg_ic cg_ic_fixed cg_ic0 cg_ic0_rcm eigen_e1 eigen_e4)

failed=0
for method in "${methods[@]}"; do
  if [ -n "$only" ] && [[ ",$only," != *",$method,"* ]]; then continue; fi
  "$here/profile_method.sh" "$method" "$matrix" "$reports" "$raw" "$cpu" || failed=1
done
python3 "$here/vtune_table.py" "$reports" > "$reports/metrics.md" || failed=1
exit $failed
