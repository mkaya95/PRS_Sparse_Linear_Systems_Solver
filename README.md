# P08: Sparse Linear Systems Solver

A sequential FP64 CSR CG/Jacobi solver, benchmarked against Eigen. How it works: [solver flow](docs/solver-flow.md) and [benchmark flow](docs/benchmark-flow.md). Measurements: [docs/results](docs/results).

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

To build only the solver, without Eigen and the benchmarks, run `cmake --build build/release --target solve`.

## Run the solver

```bash
build/release/solve --matrix poisson3d:100 --precond jacobi
build/release/solve --matrix path/to/ldoor.mtx --tol 1e-8 --maxit 5000 --out x.txt
build/release/solve --matrix path/to/A.mtx --rhs b.txt --out x.txt
```

The matrix must be symmetric positive definite. Without `--rhs`, the solver uses b = A·1. A `--rhs` file holds exactly n numbers separated by whitespace.

The exit status is 0 if the solve converged and its true residual was verified, 2 if it did not converge, and 1 for invalid input.

## Run the benchmarks

Plug in the charger and close other applications first. `run_bench.py` refuses to run on battery unless you pass `--allow-battery`.

1. Download the test matrices. If this prints `FAILED`, download the Matrix Market file from the printed page by browser and save it as `~/p08/data/NAME.mtx`:

    ```bash
    python3 scripts/download_matrices.py --data-dir ~/p08/data
    ```

2. Set the output folder, replacing `my-machine` with a short name for your computer:

    ```bash
    R=results/my-machine
    ```

3. Measure speed, which takes about 30 minutes. Each solver runs 100 iterations, and every matrix must come in at 1.05 times Eigen's time or less:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --mode fixed --maxit 100 --repeat 11 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out $R/g1-real.csv
    python3 scripts/summarize.py $R/g1-real.csv --metric iteration --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
    ```

4. Measure accuracy, which takes about 30 minutes. Each solver runs until it converges or reaches the limit, on the six smaller matrices, and must reach the same answer as Eigen in the same number of iterations:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --maxit 30000 --repeat 3 --warmup 0 \
        --matrices ecology2,apache2,tmt_sym,G3_circuit,thermal2,hood \
        --solvers eigen-E1,ours-jacobi --out $R/g1-real-solve.csv
    python3 scripts/summarize.py $R/g1-real-solve.csv --gate eigen-E1=ours-jacobi
    ```

5. Check the answers against SciPy, which needs `pip install scipy`:

    ```bash
    python3 scripts/check_against_scipy.py --build build/release --matrix ~/p08/data/apache2.mtx
    ```

Use a new CSV file for each measurement, because a CSV from a different commit or machine is rejected rather than pooled. The full set of commands is in [the results document](docs/results/g1-i5-1335U-wsl2.md).
