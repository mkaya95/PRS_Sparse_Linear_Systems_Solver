#!/usr/bin/env bash
# Fetches Eigen master at a pinned commit, for the secondary baseline.
# Then configure with: cmake --preset release -DSSOLVE_EIGEN_DIR=third_party/eigen-master
set -euo pipefail

COMMIT=1a5e11c199535ac0ff4d4d57d086f4aa105de495
DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/eigen-master"

mkdir -p "$DIR"
cd "$DIR"
if [ "$(git rev-parse HEAD 2>/dev/null || true)" = "$COMMIT" ]; then
  echo "Eigen master already at $COMMIT"
  exit 0
fi
git init -q .
git remote remove origin 2>/dev/null || true
git remote add origin https://gitlab.com/libeigen/eigen.git
git fetch -q --depth 1 origin "$COMMIT"
git checkout -q FETCH_HEAD
echo "Eigen master at $COMMIT in $DIR"
