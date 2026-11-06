#!/usr/bin/env python3
"""Checks the solve command's answer against SciPy, on Matrix Market files.

Each matrix gets b = A x for a fixed random x. The solve command runs Jacobi CG on it. SciPy then
checks the residual on its own, and computes its own answer: a direct solve for small matrices,
and Jacobi CG with its iteration count for large ones, where a direct solve would need too much memory.
Passes when solve succeeds, the residual is within tol by SciPy's arithmetic, the two answers agree,
and, against SciPy's CG, the iteration count agrees within 1% or 1 iteration.

Example: python3 scripts/check_against_scipy.py --build build/release --matrix ~/p08/data/apache2.mtx
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io
import scipy.sparse.linalg as sla

DIRECT_NNZ_LIMIT = 2_000_000


def scipy_cg(a, b, tol, maxit):
    """Returns SciPy's Jacobi CG answer and iteration count."""
    inverse_diagonal = 1.0 / a.diagonal()
    preconditioner = sla.LinearOperator(a.shape, matvec=lambda v: inverse_diagonal * v)
    iterations = [0]

    def count(_):
        iterations[0] += 1

    options = dict(atol=0.0, maxiter=maxit, M=preconditioner, callback=count)
    try:
        x, _ = sla.cg(a, b, rtol=tol, **options)
    except TypeError:  # SciPy before 1.12 calls rtol "tol"
        x, _ = sla.cg(a, b, tol=tol, **options)
    return x, iterations[0]


def check(build, path, tol, maxit, max_difference, reference_kind):
    a = scipy.io.mmread(path).tocsr()
    b = a @ np.random.default_rng(20261106).uniform(-1.0, 1.0, a.shape[0])
    with tempfile.TemporaryDirectory() as work:
        rhs_path = os.path.join(work, "b.txt")
        out_path = os.path.join(work, "x.txt")
        np.savetxt(rhs_path, b, fmt="%.17e")
        done = subprocess.run(
            [os.path.join(build, "solve"), "--matrix", path, "--rhs", rhs_path, "--precond", "jacobi",
             "--tol", repr(tol), "--maxit", str(maxit), "--out", out_path],
            capture_output=True, text=True,
        )
        if not os.path.exists(out_path):
            print(f"| {os.path.basename(path)} | no answer: {done.stderr.strip()} | | | | | FAIL |")
            return False
        x = np.loadtxt(out_path)
    found = re.search(r"spmv count\s+(\d+)", done.stdout)
    our_count = int(found.group(1)) if found else -1

    residual = np.linalg.norm(b - a @ x) / np.linalg.norm(b)
    if reference_kind == "direct" or (reference_kind == "auto" and a.nnz <= DIRECT_NNZ_LIMIT):
        reference, scipy_count = sla.spsolve(a.tocsc(), b), None
    else:
        reference, scipy_count = scipy_cg(a, b, tol, maxit)
    difference = np.linalg.norm(x - reference) / np.linalg.norm(reference)

    counts_agree = scipy_count is None or abs(our_count - scipy_count) <= max(1, 0.01 * scipy_count)
    passed = done.returncode == 0 and residual <= tol and difference <= max_difference and counts_agree
    method = "direct" if scipy_count is None else f"CG, {scipy_count} iterations"
    print(f"| {os.path.basename(path)} | {our_count} | {method} | {residual:.2e} | {difference:.2e} | "
          f"{'PASS' if passed else 'FAIL'} |", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="build directory holding solve")
    parser.add_argument("--matrix", action="append", required=True, help="Matrix Market file; may repeat")
    parser.add_argument("--reference", choices=("auto", "direct", "cg"), default="auto",
                        help=f"SciPy's answer; auto is direct up to {DIRECT_NNZ_LIMIT} entries, then CG")
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxit", type=int, default=30000)
    parser.add_argument("--max-difference", type=float, default=1e-2,
                        help="largest accepted relative difference between the two answers")
    args = parser.parse_args()

    print("| Matrix | Our SpMVs | SciPy reference | Residual by SciPy | Difference from SciPy | Result |")
    print("| --- | --- | --- | --- | --- | --- |")
    results = [check(args.build, path, args.tol, args.maxit, args.max_difference, args.reference)
               for path in args.matrix]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
