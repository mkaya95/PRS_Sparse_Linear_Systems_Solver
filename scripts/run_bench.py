#!/usr/bin/env python3
"""Runs bench_cg across the matrix suite and appends every repeat to one CSV.

Solvers run interleaved inside one process per matrix (A, B, A, B, ...), so a machine
that slows down mid-run affects them equally. A missing matrix file prints SKIP and a
failed or killed run prints FAILED. Either way the remaining matrices still run. The
exit status is non-zero if any run failed or nothing ran.

Example:
  python3 scripts/run_bench.py --build ~/p08/build/release --solvers eigen-E0,ours \\
      --data-dir ~/p08/data --maxit 200 --out results/my-laptop/g1.csv
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def matrix_specs(suite, data_dir):
    """Yields (name, spec) for each suite entry, with spec None if the file is missing."""
    for entry in suite.get("generated", []):
        for factor in entry.get("llc", []):
            spec = f"{entry['stencil']}:llc={factor}"
            yield spec, spec
        for side in entry.get("n", []):
            spec = f"{entry['stencil']}:{side}"
            yield spec, spec
    for entry in suite.get("files", []):
        path = os.path.join(data_dir, entry["name"] + ".mtx")
        yield entry["name"], (path if os.path.exists(path) else None)


def pin_prefix(pin):
    if pin == "none":
        return []
    if shutil.which("taskset") is None:
        print("NOTE: taskset is not available, so runs are not pinned to a core")
        return []
    return ["taskset", "-c", pin]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="build directory holding bench_cg")
    parser.add_argument("--solvers", required=True, help="comma-separated solver names")
    parser.add_argument("--out", required=True, help="CSV file to append to")
    parser.add_argument("--suite", default=str(ROOT / "scripts" / "suite.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"), help="directory holding NAME.mtx files")
    parser.add_argument("--pin", default="2", help="logical CPU for taskset, or 'none'")
    parser.add_argument("--repeat", default="10")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--mode", choices=("solve", "fixed"), default="solve")
    parser.add_argument("--maxit", default=None)
    parser.add_argument("--tol", default=None)
    parser.add_argument("--matrices", default=None, help="comma-separated names; default is the whole suite")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per matrix")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    with open(args.suite, encoding="utf-8") as f:
        suite = json.load(f)
    binary = os.path.join(args.build, "bench_cg")
    wanted = set(args.matrices.split(",")) if args.matrices else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    prefix = pin_prefix(args.pin)

    ran = skipped = failed = 0
    for name, spec in matrix_specs(suite, args.data_dir):
        if wanted is not None and name not in wanted:
            continue
        if spec is None:
            print(f"SKIP {name}: no {name}.mtx in {args.data_dir}. Run scripts/download_matrices.py")
            skipped += 1
            continue
        command = [*prefix, binary, "--matrix", spec, "--solver", args.solvers, "--repeat", args.repeat, "--csv", args.out]
        command += ["--warmup", args.warmup, "--mode", args.mode]
        if args.maxit is not None:
            command += ["--maxit", args.maxit]
        if args.tol is not None:
            command += ["--tol", args.tol]
        if args.label is not None:
            command += ["--label", args.label]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            # Finished rows are already in the CSV, because bench_cg flushes after each one.
            output = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            ran += len(output.strip().splitlines())
            print(f"FAILED {name}: timed out after {args.timeout} s")
            failed += 1
            continue
        lines = done.stdout.strip().splitlines()
        ran += len(lines)
        for line in lines[-len(args.solvers.split(",")):]:
            print(line, flush=True)
        if done.returncode != 0:
            reason = done.stderr.strip() or f"exit status {done.returncode}, possibly out of memory"
            print(f"FAILED {name}: {reason}", flush=True)
            failed += 1

    print(f"Finished: {ran} ran, {skipped} skipped, {failed} failed. Results: {args.out}")
    return 0 if ran > 0 and failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
