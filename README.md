# P08: Sparse Linear Systems Solver

This delivery is a sequential FP64 CSR CG/Jacobi solver. SELL and OpenMP solver
parallelism belong to the next delivery; `omp simd` is not thread parallelism.

## Install

Clone with the Eigen submodule:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without it, run `git submodule update --init`.

Install the tools on Linux or WSL2:

```bash
sudo apt install g++ cmake ninja-build python3
```

Without `sudo`, use a private virtual environment:

```bash
python3 -m venv ~/.local/share/p08-tools
~/.local/share/p08-tools/bin/pip install cmake ninja
export PATH="$HOME/.local/share/p08-tools/bin:$PATH"
```

## Build and test

```bash
cmake --preset debug             # address and undefined-behaviour sanitizers
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure

cmake --preset release -DCMAKE_CXX_COMPILER=g++  # use the actual GCC binary, e.g. g++-14 on macOS
cmake --build build/release
```

On WSL2, keep the build tree and matrices in the Linux filesystem, because `/mnt/c` file access is slow. Add `-B ~/p08/build/debug` to the preset command and use that path in the next two commands.

## Run the solver

```bash
build/release/solve --matrix poisson3d:100 --precond jacobi
build/release/solve --matrix path/to/ldoor.mtx --tol 1e-8 --maxit 5000 --out x.txt
build/release/solve --matrix path/to/A.mtx --rhs b.txt --out x.txt
```

Without `--rhs`, the manufactured/demo RHS is `b = A*1`. An RHS file contains
exactly `n` whitespace-separated finite numbers. Input matrices must be known
SPD: finite values, positive diagonals and a symmetry diagnostic are checked,
but these necessary checks do not prove positive definiteness. Jacobi rejects
missing, zero, negative or non-finite diagonals/inverses.

Exit status is 0 only after independent true-residual verification, 2 for
nonconvergence/breakdown/`ResidualFailure`, and 1 for invalid input. With zero
RHS the verification criterion is absolute residual <= tolerance.
Preparation, setup, solve, verification and end-to-end times are reported
separately. Solve time includes initialization and workspace allocation, but
excludes independent verification, printing and solution-file output.
`cmake --build build/release --target solve` builds the production executable
without compiling Eigen or the benchmark fixtures.

## Run the benchmarks

Plug in the charger and close other applications first, because timings on battery are about twice as slow and far noisier.

1. Download the test matrices. If this prints `FAILED`, download the Matrix Market file from the printed page by browser and save it as `~/p08/data/NAME.mtx`:

    ```bash
    python3 scripts/download_matrices.py --data-dir ~/p08/data
    ```

2. Run the comparison with Eigen, replacing `my-machine` with a short name for your computer:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --warmup 1 --repeat 10 --maxit 30000 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out results/my-machine/seq-solve.csv
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --mode fixed --maxit 100 --warmup 1 --repeat 10 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out results/my-machine/seq-fixed.csv
    ```

3. Print the results:

    ```bash
    python3 scripts/summarize.py results/my-machine/seq-solve.csv --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
    python3 scripts/summarize.py results/my-machine/seq-fixed.csv --metric iteration
    ```

Warm-ups run the same workload and are not recorded as samples. Each solve
starts from zero; each kernel sample starts from identical deterministic
vectors. `bench_cg` and `bench_kernels` both accept `--warmup` (default 1) and
`--repeat` (default 10). Execution order still rotates between measured rounds.

The default summary uses median solve time. `--metric end-to-end` includes
shared input preparation plus that sample's storage construction/setup, solve
and verification; it is not a cold-process measurement. Storage copies are
included in setup for both Eigen and our backend. `per_spmv_s` is retained in
CSV for compatibility but means time per CG iteration, not isolated SpMV time.
Kernel GB/s is a stated minimum-byte model, not measured DRAM traffic.

Gates require matching preconditioners, successful verification and solve mode.
Without a performance threshold they report `VERIFIED` and a runtime ratio;
add `--time-ratio 1.05` only if that limit is explicitly desired. Fixed-iteration
diagnostics cannot pass a solve gate. Failed solves/gates return nonzero.
Incompatible hardware, input, numerical or unlabeled build configurations
are rejected rather than pooled. Explicit build labels allow intentional
build comparisons on the same problem/machine.

Use new CSV files for this schema; existing result files are not migrated or
overwritten. Historical CSVs remain readable as diagnostics, but missing
accuracy/timing evidence cannot pass a verified solve gate.
