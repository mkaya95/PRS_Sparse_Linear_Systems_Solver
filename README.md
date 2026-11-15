# P08: Sparse Linear Systems Solver

A sequential FP64 conjugate gradient solver with CSR or SELL-C-σ storage and Jacobi, symmetric Gauss-Seidel or incomplete Cholesky preconditioning, benchmarked against Eigen. How it works: [solver flow](docs/solver-flow.md) and [benchmark flow](docs/benchmark-flow.md). Measurements: [docs/results](docs/results).

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

For a profiler such as Intel VTune or `perf`, use the `profile` preset: the release flags plus symbols and frame pointers, so the profiler can name functions and source lines.

```bash
cmake --preset profile -DCMAKE_CXX_COMPILER=g++
cmake --build build/profile --target solve bench_cg
```

With VTune installed, one script profiles every solver on a matrix and saves the reports as text. The kernel must first let VTune read the processor's counters and load its sampling driver, both of which last until the next reboot:

```bash
source /opt/intel/oneapi/setvars.sh
sudo sysctl kernel.perf_event_paranoid=0 kernel.kptr_restrict=0 kernel.yama.ptrace_scope=0
sudo /opt/intel/oneapi/vtune/latest/sepdk/src/insmod-sep -g "$(id -gn)"   # Intel's sampling driver
scripts/profile_vtune.sh ~/p08/data/thermal2.mtx results/my-machine/vtune
```

The VTune result folders go to `~/p08/vtune`; open one with `vtune-gui ~/p08/vtune/ic0-hotspots`. What it showed on `thermal2` is in [the VTune results](docs/results/vtune-i5-1335U-linux.md).

## Run the solver

```bash
build/release/solve --matrix poisson3d:100 --precond jacobi
build/release/solve --matrix poisson3d:100 --precond ic
build/release/solve --matrix path/to/ldoor.mtx --tol 1e-8 --maxit 5000 --out x.txt
build/release/solve --matrix path/to/A.mtx --rhs b.txt --out x.txt
build/release/solve --matrix poisson3d:100 --format sell --chunk 8 --sigma 256
build/release/solve --matrix path/to/ldoor.mtx --precond ic0 --order rcm
```

The matrix must be symmetric positive definite. `--precond` is `none` (the default), `jacobi`, `sgs` (symmetric Gauss-Seidel), `ic` (incomplete Cholesky, Eigen's operation for operation), `ic-fixed` (the same with Eigen's two bugs corrected) or `ic0` (incomplete Cholesky that keeps exactly the matrix's own pattern). Without `--rhs`, the solver uses b = A·1. A `--rhs` file holds exactly n numbers separated by whitespace. `--problem random` instead uses the right-hand side the benchmarks use. `--history h.txt` writes ‖r‖ / ‖b‖ before the first iteration and after each one, one value per line. `--format sell` stores the matrix as SELL-C-σ instead of CSR: chunks of C rows (4, 8, 16 or 32), sorted by length inside windows of σ rows; the output shows how much padding that adds. `--order rcm` first renumbers the unknowns by reverse Cuthill-McKee, which pulls the non-zeros towards the diagonal; the answer is written in the file's original order.

The exit status is 0 if the solve converged and its true residual was verified, 2 if it did not converge, and 1 for invalid input.

## Run everything from the browser

Start the page where the solver builds, on Linux, WSL2 or macOS, and open the address it prints, http://localhost:8008/ by default. On WSL2, open it in your Windows browser.

```bash
python3 ui/server.py
```

The Run tab runs the steps below with your settings and shows each exact command. Its One matrix box runs the speed or accuracy step on a single matrix and saves the CSV as `results/NAME/one-MATRIX-TEST.csv`, apart from the full results, and an accuracy run also saves its solution pictures in `results/NAME/one-solutions/`. The Results tab charts every machine in `results/`, the Matrices tab draws the non-zero pattern of each test matrix, and the Demo tab solves one problem with our solver and with Eigen. Use `--port` to pick another port. The page answers only on this computer.

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

3. Measure speed, which takes about 20 minutes. Each solver runs 100 iterations, and every matrix must come in at 1.05 times Eigen's time or less:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --mode fixed --maxit 100 --repeat 11 \
        --solvers eigen-E0,ours,eigen-E1,ours-jacobi --out $R/g1-real.csv
    python3 scripts/summarize.py $R/g1-real.csv --metric iteration --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
    ```

4. Measure accuracy, which takes about an hour. Each solver runs until it converges or reaches the limit, on the six smaller matrices, with every preconditioner. Each of ours that has an Eigen counterpart must meet the tolerance in Eigen's iteration count within 1%, or fewer:

    ```bash
    python3 scripts/run_bench.py --build build/release --data-dir ~/p08/data --maxit 30000 --repeat 3 --warmup 0 \
        --matrices ecology2,apache2,tmt_sym,G3_circuit,thermal2,hood \
        --solvers eigen-E1,ours-jacobi,ours-sell-jacobi,ours-sgs,eigen-E4,ours-ic,ours-ic-fixed,ours-ic0 \
        --solutions $R/solutions --compare same-precond --out $R/g1-real-solve.csv
    python3 scripts/summarize.py $R/g1-real-solve.csv --solutions $R/solutions \
        --gate eigen-E1=ours-jacobi --gate eigen-E1=ours-sell-jacobi --gate eigen-E4=ours-ic
    ```

    `--solutions` saves, for each matrix, how far our answer is from its Eigen counterpart's and from the true answer, and how the residual fell iteration by iteration, as `$R/solutions/MATRIX.json`. The Results tab draws the differences. The gate reads the residuals: more iterations than Eigen still pass as `VERIFIED (plateau)` when our residual at Eigen's stopping iteration was within 5% of the tolerance, because rounding then decides who crosses the line first.

5. Check the answers against SciPy, which needs `pip install scipy`:

    ```bash
    python3 scripts/check_against_scipy.py --build build/release --matrix ~/p08/data/apache2.mtx
    ```

6. Sweep the SELL-C-σ settings, which takes about 15 minutes. It times the matrix-vector product for each chunk size C and sorting window σ against CSR, and prints the padding, the time ratio and the number of products after which converting to SELL has paid off:

    ```bash
    python3 scripts/sweep_sell.py --build build/release --data-dir ~/p08/data --out $R/sell-sweep.csv
    ```

Use a new CSV file for each measurement, because a CSV from a different commit or machine is rejected rather than pooled. The full set of commands is in [the results document](docs/results/g1-i5-1335U-wsl2.md).
