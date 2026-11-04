# P08: Sparse Linear Systems Solver

A sparse linear systems solver built to match Eigen 5.0.1 in sequential runs, then beat it with SELL-C-σ and OpenMP. CUDA is Part 2.

## Setup

Clone with the pinned Eigen submodule:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without it, run `git submodule update --init`.

### Linux or WSL2

```bash
sudo apt install g++ cmake ninja-build python3
```

Without `sudo`, use a private virtual environment:

```bash
python3 -m venv ~/.local/share/p08-tools
~/.local/share/p08-tools/bin/pip install cmake ninja
export PATH="$HOME/.local/share/p08-tools/bin:$PATH"
```

### Build and test

```bash
cmake --preset debug             # address and undefined-behaviour sanitizers
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure

cmake --preset release           # -O3 -march=native, used for all benchmarks
cmake --build build/release
```

On WSL2, keep the build tree and matrices in the Linux filesystem, because `/mnt/c` file access is slow. Add `-B ~/p08/build/debug` to the preset command and use that path in the next two commands.

## Solve a system

```bash
build/release/solve --matrix poisson3d:100 --precond jacobi
build/release/solve --matrix path/to/ldoor.mtx --tol 1e-8 --maxit 5000 --out x.txt
```

The right-hand side is b = A·1, so the exact solution is the vector of ones. The exit status is 0 if it converged, 2 if it did not, and 1 for an input error.

## Rerun the benchmarks on your machine

Results are only comparable on a quiet machine, so plug in the charger, close other applications and use a high-performance power plan.

1. Get the real matrices. The collection's file host refuses scripted downloads from some networks, so if this prints `FAILED`, download the Matrix Market file from the printed page by browser and save it as `DATA_DIR/NAME.mtx`:

    ```bash
    python3 scripts/download_matrices.py --data-dir ~/p08/data
    ```

2. Run the G1 comparison, replacing `my-machine` with a short name for your computer:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --repeat 15 --maxit 100 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out results/my-machine/g1.csv
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --repeat 1 --maxit 30000 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out results/my-machine/g1-converge.csv
    ```

3. Read the result:

    ```bash
    python3 scripts/summarize.py results/my-machine/g1.csv --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
    python3 scripts/summarize.py results/my-machine/g1-converge.csv --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
    ```

4. Commit `results/my-machine/`. Gate G1 closes when every team machine passes.

To rerun the optimization ladder or compare single kernels, see the commands in `docs/results/`.

## Project layout

| Path | Contents |
| --- | --- |
| `include/ssolve/`, `src/` | The solver library: matrices, reader, generators, backend, CG |
| `bench/`, `apps/` | Eigen baselines, the optimization ladder, and the command-line programs |
| `tests/` | Unit and behaviour tests, run with `ctest` |
| `scripts/` | Matrix download, benchmark runner and result summaries |
| `results/` | Raw benchmark CSVs, one folder per machine |
| `docs/` | Plans and written-up results |
