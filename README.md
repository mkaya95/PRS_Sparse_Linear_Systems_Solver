# P08: Sparse Linear Systems Solver

A standalone sequential FP64 sparse linear systems solver using CSR storage and
conjugate gradient, with optional Jacobi preconditioning. How it works:
[solver flow](docs/solver-flow.md).

This branch keeps the original solver directory layout and implementation.
It contains no benchmark applications, Eigen dependency, or threaded solver.
SIMD vectorization stays enabled without the OpenMP runtime.

## Install

Requirements: a C++17 GCC or Clang compiler, CMake 3.21+, and Ninja.
Install the tools on Linux or WSL2:

```bash
sudo apt install g++ cmake ninja-build
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

To build only the solver executable, run `cmake --build build/release --target solve`.

## Run the solver

```bash
build/release/solve --matrix poisson3d:100 --precond jacobi
build/release/solve --matrix path/to/ldoor.mtx --tol 1e-8 --maxit 5000 --out x.txt
build/release/solve --matrix path/to/A.mtx --rhs b.txt --out x.txt
```

The matrix must be symmetric positive definite. Without `--rhs`, the solver uses b = A·1. A `--rhs` file holds exactly n numbers separated by whitespace.

The exit status is 0 if the solve converged and its true residual was verified, 2 if it did not converge, and 1 for invalid input.

Generated matrix specifications are `poisson2d:N`, `poisson3d:N`, and `point27:N`,
where `N` is the number of grid points per side. Matrix Market files are also supported.
