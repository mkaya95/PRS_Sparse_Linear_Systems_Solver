#!/usr/bin/env python3
"""Summarizes bench_cg CSV files as Markdown tables and checks gates.

Examples:
  python3 scripts/summarize.py results/my-laptop/g1.csv --gate eigen-E0=ours --gate eigen-E1=ours-jacobi
  python3 scripts/summarize.py results/my-laptop/kernels.csv --kernel-baseline eigen

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


def paired_ratios(base, ours):
    """Ratio ours/base within each round both ran. Each argument maps a round to a time.

    Solvers take turns, so a round's two measurements are adjacent in time. Comparing them
    cancels slow drift in machine speed, which a ratio of overall medians does not.
    """
    return [ours[key] / base[key] for key in base if key in ours and base[key] > 0]


def record_round(entry, row, value):
    """Stores a time under (repeat, occurrence), so appended runs do not overwrite each other."""
    if "repeat" not in row:
        return
    occurrence = entry["seen"][row["repeat"]]
    entry["seen"][row["repeat"]] += 1
    entry["rounds"][(row["repeat"], occurrence)] = value


def new_entry():
    return {"times": [], "spmv": [], "status": set(), "rounds": {}, "seen": collections.Counter()}


def load_groups(paths):
    """Returns {(matrix, maxit): {solver: entry}}."""
    groups = collections.defaultdict(lambda: collections.defaultdict(new_entry))
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # A label separates the same solver measured from different builds, as solver@label.
                solver = row["solver"] + (f"@{row['label']}" if row.get("label") else "")
                entry = groups[(row["matrix"], row.get("maxit", ""))][solver]
                time = float(row["per_spmv_s"])
                entry["times"].append(time)
                entry["spmv"].append(int(row["spmv_count"]))
                entry["status"].add(row["status"])
                record_round(entry, row, time)
    return groups


def ratio_with_ci(base_entry, our_entry):
    """Returns (ratio, ci_low, ci_high), with the interval None when rounds cannot be paired."""
    ratios = paired_ratios(base_entry["rounds"], our_entry["rounds"])
    if ratios:
        low, high = ci_bounds(ratios)
        return median(ratios), low, high
    reference = median(base_entry["times"])
    return (median(our_entry["times"]) / reference if reference > 0 else math.inf), None, None


def print_gate(groups, base, ours):
    print(f"\n### Gate: {ours} against {base}\n")
    print("| Matrix | Limit | Base (us/SpMV) | Ours (us/SpMV) | Ratio | Ratio 95% CI | Base SpMVs | Our SpMVs | Result |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    passed = total = 0
    noisy = []
    for (matrix, maxit), solvers in sorted(groups.items()):
        if base not in solvers or ours not in solvers:
            continue
        base_time = median(solvers[base]["times"])
        our_time = median(solvers[ours]["times"])
        ratio, low, high = ratio_with_ci(solvers[base], solvers[ours])
        interval = f"{low:.3f} to {high:.3f}" if low is not None else "not paired"
        if low is not None and high - low > NOISY_CI_FRACTION:
            noisy.append(matrix)
        base_spmv = median(solvers[base]["spmv"])
        our_spmv = median(solvers[ours]["spmv"])
        ok = time_passes(ratio) and spmv_passes(base_spmv, our_spmv)
        total += 1
        passed += 1 if ok else 0
        limit = maxit if maxit not in ("", "0") else "default"
        print(
            f"| {matrix} | {limit} | {base_time * 1e6:.2f} | {our_time * 1e6:.2f} | {ratio:.3f} | "
            f"{interval} | {base_spmv:g} | {our_spmv:g} | {'PASS' if ok else 'FAIL'} |"
        )
    print(f"\n{passed} of {total} matrices pass.")
    if noisy:
        print(f"WARNING: ratio interval wider than 0.10 on: {', '.join(sorted(set(noisy)))}. Rerun on a quiet machine.")


def geometric_mean_ratio(groups, base, ours):
    """Geometric mean across matrices of the time ratio ours/base, or None if never compared."""
    ratios = [
        ratio_with_ci(solvers[base], solvers[ours])[0]
        for solvers in groups.values()
        if base in solvers and ours in solvers
    ]
    if not ratios:
        return None, 0
    return math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)), len(ratios)


def print_ladder(groups, rungs):
    """One row per rung: time per SpMV against the first rung and against the previous one."""
    print(f"| Rung | Time / {rungs[0]} | Time / previous rung | Matrices |")
    print("| --- | --- | --- | --- |")
    print(f"| {rungs[0]} | 1.000 | | |")
    for previous, rung in zip(rungs, rungs[1:]):
        to_first, count = geometric_mean_ratio(groups, rungs[0], rung)
        to_previous, _ = geometric_mean_ratio(groups, previous, rung)
        first_text = f"{to_first:.3f}" if to_first else "no data"
        previous_text = f"{to_previous:.3f}" if to_previous else "no data"
        print(f"| {rung} | {first_text} | {previous_text} | {count} |")


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


def print_kernels(paths, baseline):
    """Prints median time per kernel and the ratio to the baseline implementation."""
    groups = collections.defaultdict(lambda: collections.defaultdict(new_entry))
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                entry = groups[(row["matrix"], row["kernel"])][row["impl"]]
                time = float(row["seconds"])
                entry["times"].append(time)
                record_round(entry, row, time)
    print(f"| Matrix | Kernel | Implementation | Median us | Time / {baseline} | Ratio 95% CI |")
    print("| --- | --- | --- | --- | --- | --- |")
    for (matrix, kernel), implementations in sorted(groups.items()):
        for impl, entry in sorted(implementations.items()):
            if baseline in implementations:
                ratio, low, high = ratio_with_ci(implementations[baseline], entry)
                ratio_text = f"{ratio:.3f}"
                interval = f"{low:.3f} to {high:.3f}" if low is not None else "not paired"
            else:
                ratio_text, interval = "no baseline", ""
            print(f"| {matrix} | {kernel} | {impl} | {median(entry['times']) * 1e6:.2f} | {ratio_text} | {interval} |")


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
    parser.add_argument("--kernel-baseline", metavar="IMPL", help="treat the files as bench_kernels output")
    parser.add_argument("--ladder", metavar="A,B,C", help="print one row per rung of the optimization ladder")
    args = parser.parse_args()

    if args.kernel_baseline:
        print_kernels(args.files, args.kernel_baseline)
        return 0

    groups = load_groups(args.files)
    if not groups:
        print("No benchmark rows found.")
        return 1
    if args.ladder:
        print_ladder(groups, args.ladder.split(","))
    elif args.gate:
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
