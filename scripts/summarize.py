#!/usr/bin/env python3
"""Summarizes bench_cg CSV files as Markdown tables and checks gates.

Example:
  python3 scripts/summarize.py results/my-laptop/g1.csv --gate eigen-E0=ours --gate eigen-E1=ours-jacobi

A gate BASE=OURS passes on a matrix when OURS takes at most 1.05x BASE per SpMV and
their SpMV counts differ by at most 1% or 1 iteration, whichever is larger.
Medians and 95% nonparametric confidence intervals follow Hoefler and Belli (SC15).
"""
import argparse
import collections
import csv
import math
import sys

TIME_RATIO_LIMIT = 1.05
NOISY_CI_FRACTION = 0.10


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def ci_ranks(n):
    """1-based order statistics giving at least 95% coverage of the median.

    The largest k with P(Binomial(n, 1/2) < k) <= 0.025, and its mirror n + 1 - k.
    Below 6 samples no interval reaches 95%, so the full range is returned.
    """
    tail = 0.0
    lower = 0
    for k in range(1, n // 2 + 1):
        tail += math.comb(n, k - 1) / 2**n
        if tail > 0.025:
            break
        lower = k
    if lower == 0:
        return 1, n
    return lower, n + 1 - lower


def ci_bounds(values):
    low, high = ci_ranks(len(values))
    ordered = sorted(values)
    return ordered[low - 1], ordered[high - 1]


def time_passes(ratio):
    return ratio <= TIME_RATIO_LIMIT


def spmv_passes(base, ours):
    return abs(ours - base) <= max(1, 0.01 * base)


def load_groups(paths):
    """Returns {(matrix, maxit): {solver: {"times": [...], "spmv": [...], "status": set}}}."""
    groups = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"times": [], "spmv": [], "status": set()})
    )
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                entry = groups[(row["matrix"], row.get("maxit", ""))][row["solver"]]
                entry["times"].append(float(row["per_spmv_s"]))
                entry["spmv"].append(int(row["spmv_count"]))
                entry["status"].add(row["status"])
    return groups


def print_gate(groups, base, ours):
    print(f"\n### Gate: {ours} against {base}\n")
    print("| Matrix | Limit | Base (us/SpMV) | Ours (us/SpMV) | Ours 95% CI | Ratio | Base SpMVs | Our SpMVs | Result |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    passed = total = 0
    for (matrix, maxit), solvers in sorted(groups.items()):
        if base not in solvers or ours not in solvers:
            continue
        base_time = median(solvers[base]["times"])
        our_time = median(solvers[ours]["times"])
        low, high = ci_bounds(solvers[ours]["times"])
        ratio = our_time / base_time if base_time > 0 else math.inf
        base_spmv = median(solvers[base]["spmv"])
        our_spmv = median(solvers[ours]["spmv"])
        ok = time_passes(ratio) and spmv_passes(base_spmv, our_spmv)
        total += 1
        passed += 1 if ok else 0
        limit = maxit if maxit not in ("", "0") else "default"
        print(
            f"| {matrix} | {limit} | {base_time * 1e6:.2f} | {our_time * 1e6:.2f} | "
            f"{low * 1e6:.2f} to {high * 1e6:.2f} | {ratio:.3f} | {base_spmv:g} | {our_spmv:g} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )
    print(f"\n{passed} of {total} matrices pass.")


def print_all(groups):
    print("| Matrix | Limit | Solver | Median us/SpMV | 95% CI | SpMVs | Status | Runs |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (matrix, maxit), solvers in sorted(groups.items()):
        limit = maxit if maxit not in ("", "0") else "default"
        for solver, entry in sorted(solvers.items()):
            low, high = ci_bounds(entry["times"])
            print(
                f"| {matrix} | {limit} | {solver} | {median(entry['times']) * 1e6:.2f} | "
                f"{low * 1e6:.2f} to {high * 1e6:.2f} | {median(entry['spmv']):g} | "
                f"{'/'.join(sorted(entry['status']))} | {len(entry['times'])} |"
            )


def print_noise_warnings(groups):
    for (matrix, _), solvers in sorted(groups.items()):
        for solver, entry in sorted(solvers.items()):
            middle = median(entry["times"])
            low, high = ci_bounds(entry["times"])
            if middle > 0 and (high - low) / middle > NOISY_CI_FRACTION:
                print(
                    f"WARNING: {matrix} {solver}: confidence interval is {100 * (high - low) / middle:.0f}% "
                    "of the median, so this comparison is noisy"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="CSV files from bench_cg")
    parser.add_argument("--gate", action="append", default=[], metavar="BASE=OURS")
    args = parser.parse_args()

    groups = load_groups(args.files)
    if not groups:
        print("No benchmark rows found.")
        return 1
    if args.gate:
        for gate in args.gate:
            if "=" not in gate:
                parser.error(f"--gate needs BASE=OURS, got '{gate}'")
            base, ours = gate.split("=", 1)
            print_gate(groups, base, ours)
    else:
        print_all(groups)
    print()
    print_noise_warnings(groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
