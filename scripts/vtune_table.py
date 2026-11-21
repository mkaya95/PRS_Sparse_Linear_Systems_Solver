#!/usr/bin/env python3
"""Tables of the VTune metrics the professor asked about, from the reports scripts/profile_method.sh writes.

    vtune_table.py REPORT_DIR                one table per analysis, one row per method
    vtune_table.py BEFORE_DIR AFTER_DIR      the same metrics side by side, before -> after

Metrics, all from the performance core the run was pinned to:
    CPI           cycles per instruction (uarch-exploration); lower is better
    Bad spec.     pipeline slots thrown away after a wrong branch guess; lower is better
    Memory bound  pipeline slots waiting for data from caches or memory; lower is better
    Retiring      pipeline slots doing useful work; higher is better
    DRAM GB/s     average memory bandwidth (memory-access), and the share of time at high bandwidth
"""
import csv
import re
import sys
from pathlib import Path

METHODS = ["cg_jacobi", "cg_sell_jacobi", "cg_sgs", "cg_ic", "cg_ic_fixed", "cg_ic0", "cg_ic0_rcm", "eigen_e1",
           "eigen_e4"]


def first(pattern, text, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    return cast(match.group(1).replace(",", "")) if match else None


def read(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def metrics(directory, method):
    d = Path(directory)
    m = {}
    run = read(d / f"{method}-uarch-exploration-run.log") or read(d / f"{method}-hotspots-run.log")
    m["iterations"] = first(r"(\d+) iterations", run, int)
    m["solve_s"] = first(r"solve ([\d.]+) s", run)
    m["ms_per_iteration"] = first(r"([\d.]+) ms per iteration", run)
    m["status"] = first(r"run 1\s+(\w+),", run, str)

    uarch = read(d / f"{method}-uarch-exploration-summary.txt")
    m["cpi"] = first(r"^\s*CPI Rate: ([\d.]+)", uarch)
    for key, label in [("retiring", "Retiring"), ("front_end", "Front-End Bound"), ("bad_spec", "Bad Speculation"),
                       ("back_end", "Back-End Bound"), ("memory_bound", "Memory Bound"), ("core_bound", "Core Bound")]:
        m[key] = first(rf"^\s*{label}: ([\d.]+)% of Pipeline Slots", uarch)
    m["instructions"] = first(r"^\s*Instructions Retired: ([\d,]+)", uarch)
    m["fp_vector"] = first(r"^\s*FP Vector: ([\d.]+)% of uOps", uarch)
    m["fp_scalar"] = first(r"^\s*FP Scalar: ([\d.]+)% of uOps", uarch)

    hot = read(d / f"{method}-hotspots-summary.txt")
    m["cpu_time_s"] = first(r"^\s*CPU Time: ([\d.]+)s", hot)
    m["elapsed_s"] = first(r"^Elapsed Time: ([\d.]+)s", hot)
    m["top_function"] = None
    rows = re.search(r"Top Hotspots\n.*\n-+.*\n((?:.+\n)+?)\[Others\]", hot)
    if rows:
        line = rows.group(1).splitlines()[0]
        m["top_function"] = re.split(r"\s{2,}", line.strip())[0]
        m["top_share"] = first(r"([\d.]+)%\s*$", line)

    mem = read(d / f"{method}-memory-access-summary.txt")
    m["dram_bound"] = first(r"^\s*DRAM Bound: ([\d.]+)%", mem)
    m["llc_misses"] = first(r"^\s*LLC Miss Count: ([\d,]+)", mem)
    m["load_latency"] = first(r"^\s*Average Latency \(cycles\): ([\d.]+)", mem)
    bw = re.search(r"^DRAM, GB/sec\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", mem, re.MULTILINE)
    if bw:
        m["bw_max_platform"], m["bw_peak"], m["bw_avg"], m["bw_high_share"] = (float(g) for g in bw.groups())
    m["dram_bw_bound"] = first(r"^\s*DRAM Bandwidth Bound: ([\d.]+)%", mem)

    temps = []
    for analysis in ("uarch-exploration", "hotspots", "memory-access"):
        path = d / f"{method}-{analysis}-thermal.csv"
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get("package_celsius"):
                    temps.append(float(row["package_celsius"]))
    ghz = first(r"^\s*Average CPU Frequency: ([\d.]+) GHz", uarch)
    m["avg_mhz"] = ghz * 1000 if ghz else None
    m["max_celsius"] = max(temps) if temps else None
    return m


def fmt(value, digits=1, suffix=""):
    if value is None:
        return "–"
    if isinstance(value, str):
        return value
    if isinstance(value, int) or digits == 0:
        return f"{value:,.0f}{suffix}"
    return f"{value:.{digits}f}{suffix}"


COLUMNS = [
    ("Iterations", "iterations", 0, ""),
    ("Solve (s)", "solve_s", 2, ""),
    ("ms / iteration", "ms_per_iteration", 2, ""),
    ("CPI", "cpi", 2, ""),
    ("Bad speculation", "bad_spec", 1, "%"),
    ("Memory bound", "memory_bound", 1, "%"),
    ("Retiring", "retiring", 1, "%"),
    ("Front-end", "front_end", 1, "%"),
    ("Core bound", "core_bound", 1, "%"),
    ("FP vector", "fp_vector", 1, "%"),
    ("DRAM GB/s avg", "bw_avg", 1, ""),
    ("DRAM GB/s peak", "bw_peak", 1, ""),
    ("Time at high BW", "bw_high_share", 1, "%"),
    ("LLC misses (M)", "llc_misses_m", 0, ""),
    ("Core MHz", "avg_mhz", 0, ""),
    ("Max °C", "max_celsius", 0, ""),
]


def short_name(function):
    """Drops template arguments and namespaces: Eigen's SpMV is a 300-character name otherwise."""
    depth, out = 0, []
    for ch in function:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    name = "".join(out)
    parts = [p for p in name.split("::") if p and not p.startswith("(anonymous")]
    return "::".join(parts[-2:]) if len(parts) > 1 and parts[-2][0].isupper() else parts[-1]


def present(directory):
    return [m for m in METHODS if any(Path(directory).glob(f"{m}-*-summary.txt"))]


def table_one(directory):
    methods = present(directory)
    data = {m: metrics(directory, m) for m in methods}
    for m in data.values():
        m["llc_misses_m"] = m["llc_misses"] / 1e6 if m.get("llc_misses") else None
    out = ["| Method | " + " | ".join(c[0] for c in COLUMNS) + " | Hottest function |",
           "| --- | " + " | ".join("---:" for _ in COLUMNS) + " | --- |"]
    for name in methods:
        m = data[name]
        cells = [fmt(m.get(key), digits, suffix) for _, key, digits, suffix in COLUMNS]
        top = f"`{short_name(m['top_function'])}` {m.get('top_share', 0):.0f}%" if m.get("top_function") else "–"
        out.append(f"| `{name}` | " + " | ".join(cells) + f" | {top} |")
    return "\n".join(out)


COMPARE = [
    ("ms / iteration", "ms_per_iteration", 2, ""),
    ("Solve (s)", "solve_s", 2, ""),
    ("CPI", "cpi", 2, ""),
    ("Bad speculation", "bad_spec", 1, "%"),
    ("Memory bound", "memory_bound", 1, "%"),
    ("Retiring", "retiring", 1, "%"),
    ("DRAM GB/s avg", "bw_avg", 1, ""),
    ("Time at high BW", "bw_high_share", 1, "%"),
]


def table_compare(before, after):
    methods = [m for m in METHODS if m in present(before) and m in present(after)]
    out = ["| Method | " + " | ".join(c[0] for c in COMPARE) + " |", "| --- | " + " | ".join("---:" for _ in COMPARE) + " |"]
    for name in methods:
        b, a = metrics(before, name), metrics(after, name)
        cells = [f"{fmt(b.get(k), d, s)} → **{fmt(a.get(k), d, s)}**" for _, k, d, s in COMPARE]
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    if len(sys.argv) == 2:
        print(table_one(sys.argv[1]))
    elif len(sys.argv) == 3:
        print(table_compare(sys.argv[1], sys.argv[2]))
    else:
        sys.exit("usage: vtune_table.py REPORT_DIR | BEFORE_DIR AFTER_DIR")


if __name__ == "__main__":
    main()
