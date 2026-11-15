#!/usr/bin/env python3
"""Times the SELL-C-sigma matrix-vector product for several chunk sizes C and sorting windows sigma,
against the CSR product, on each suite matrix.

Example:
  python3 scripts/sweep_sell.py --build build/release --data-dir ~/p08/data --out results/my-laptop/sell-sweep.csv

Writes the timings to --out and each layout's padding and conversion time to OUT-layout.csv, then prints
one table per matrix:
  Stored / nnz   entries stored per non-zero; above 1 is padding, which is read and multiplied for nothing
  Time / CSR     median over the rounds of the SELL time divided by the CSR time of the same round
  Break-even     products after which the time saved has paid for converting CSR to SELL
A sigma that is neither 1 nor a multiple of C is skipped, because the format does not allow it.
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys

import run_bench
import summarize

BASELINE = "L3-acc2"  # the CSR product of our shipping solver


def implementations(chunks, sigmas):
    """Every allowed sell-c<C>-s<sigma>, in the order given."""
    return [f"sell-c{c}-s{s}" for c in chunks for s in sigmas if s == 1 or s % c == 0]


def summary(timing_rows, layout_rows):
    """Returns {matrix: [(impl, stored_per_nnz, ratio, break_even)]}; break_even is None when SELL is not faster."""
    times = collections.defaultdict(lambda: collections.defaultdict(dict))  # matrix -> impl -> round -> seconds
    for row in timing_rows:
        if row["kernel"] == "spmv":
            times[row["matrix"]][row["impl"]][(row.get("run_id", ""), row["repeat"])] = float(row["seconds"])
    layouts = {(row["matrix"], row["impl"]): (float(row["stored_per_nnz"]), float(row["build_s"]))
               for row in layout_rows}
    out = {}
    for matrix, by_impl in times.items():
        base = by_impl.get(BASELINE, {})
        lines = []
        for impl, rounds in by_impl.items():
            ratios = summarize.paired_ratios(base, rounds)
            if impl == BASELINE or not ratios or (matrix, impl) not in layouts:
                continue
            stored, build = layouts[(matrix, impl)]
            # Seconds saved per product, from the paired ratio, so the two columns can never disagree.
            ratio = summarize.median(ratios)
            saved = summarize.median(list(base.values())) * (1.0 - ratio)
            lines.append((impl, stored, ratio, build / saved if saved > 0 else None))
        out[matrix] = lines
    return out


def print_summary(tables):
    for matrix, lines in tables.items():
        print(f"\n### {matrix}\n")
        print("| Layout | Stored / nnz | Time / CSR | Break-even products |")
        print("| --- | --- | --- | --- |")
        for impl, stored, ratio, break_even in lines:
            print(f"| {impl} | {stored:.3f} | {ratio:.3f} | {'never' if break_even is None else f'{break_even:.0f}'} |")
        if lines:
            best = min(lines, key=lambda line: line[2])
            print(f"\nFastest: {best[0]} at {best[2]:.3f} of the CSR time.")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="build directory holding bench_kernels")
    parser.add_argument("--out", required=True, help="CSV file for the timings; the layouts go to OUT-layout.csv")
    parser.add_argument("--suite", default=str(run_bench.ROOT / "scripts" / "suite.json"))
    parser.add_argument("--data-dir", default=str(run_bench.ROOT / "data"), help="directory holding NAME.mtx files")
    parser.add_argument("--matrices", default=None, help="comma-separated names; default is the whole suite")
    parser.add_argument("--chunks", default="4,8,16,32", help="values of C")
    parser.add_argument("--sigmas", default="1,256,4096,65536", help="values of sigma")
    parser.add_argument("--repeat", default="7")
    parser.add_argument("--min-ms", default="50")
    parser.add_argument("--pin", default="2", help="logical CPU for taskset, or 'none'")
    parser.add_argument("--allow-battery", action="store_true", help="measure even on battery")
    args = parser.parse_args()

    try:
        chunks = [int(c) for c in args.chunks.split(",")]
        sigmas = [int(s) for s in args.sigmas.split(",")]
    except ValueError:
        parser.error("--chunks and --sigmas take comma-separated whole numbers")
    impls = implementations(chunks, sigmas)
    if not impls:
        parser.error("no allowed combination of --chunks and --sigmas")
    out, layout = args.out, os.path.splitext(args.out)[0] + "-layout.csv"
    for path in (out, layout):
        if os.path.exists(path):
            parser.error(f"{path} exists; use a new file for each measurement")

    if run_bench.detect_power() == "battery" and not args.allow_battery:
        print("REFUSED: this machine is on battery. Plug in the charger, or pass --allow-battery.")
        return 3

    with open(args.suite, encoding="utf-8") as f:
        suite = json.load(f)
    wanted = set(args.matrices.split(",")) if args.matrices else None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    failed = ran = 0
    for name, spec in run_bench.matrix_specs(suite, args.data_dir):
        if wanted is not None and name not in wanted:
            continue
        if spec is None:
            print(f"SKIP {name}: no {name}.mtx in {args.data_dir}. Run scripts/download_matrices.py")
            continue
        command = [*run_bench.pin_prefix(args.pin), os.path.join(args.build, "bench_kernels"), "--matrix", spec,
                   "--impl", ",".join([BASELINE] + impls), "--kernel", "spmv", "--repeat", args.repeat,
                   "--min-ms", args.min_ms, "--csv", out, "--layout", layout]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(f"FAILED {name}: {done.stderr.strip() or f'exit status {done.returncode}'}", flush=True)
            failed += 1
            continue
        print(f"measured {name}", flush=True)
        ran += 1
    if ran:
        print_summary(summary(read_csv(out), read_csv(layout)))
    return 1 if failed or not ran else 0


if __name__ == "__main__":
    sys.exit(main())
