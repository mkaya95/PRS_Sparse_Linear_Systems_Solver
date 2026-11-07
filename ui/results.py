"""Reads every results/<machine>/ folder into chart data, using summarize.py's own statistics.

The gates here are the README's: speed from g1-real.csv, accuracy from g1-real-solve.csv, so the
page and summarize.py can never disagree. g1-ladder.csv, when present, gives the optimization ladder.
"""
import csv
import math
import sys

from steps import ACCURACY_CSV, ROOT, SPEED_CSV

sys.path.insert(0, str(ROOT / "scripts"))
import summarize  # noqa: E402

LADDER_CSV = "g1-ladder.csv"
SPEED_PAIRS = (("eigen-E0", "ours", "Plain CG"), ("eigen-E1", "ours-jacobi", "Jacobi"))
ACCURACY_PAIR = ("eigen-E1", "ours-jacobi")
# (solver, what the rung adds, a short axis label), as in the results document's ladder tables.
LADDER = {
    "eigen-E0": [("L0-naive", "textbook CG, no tuning", "textbook"),
                 ("L1-restrict", "restrict and local array pointers", "restrict"),
                 ("L2-simd", "omp simd on the vector kernels", "simd"),
                 ("L3-acc2", "two accumulators in the CSR row loop", "2-acc"),
                 ("L4-fused", "p·q inside SpMV; x, r and r·r in one sweep", "fused"),
                 ("L5-nopass", "no initial SpMV when x0 = 0", "no-spmv0"),
                 ("L6-rowptr32", "32-bit row pointers", "rowptr32")],
    "eigen-E1": [("J0-passes", "Jacobi with separate passes and a stored z", "passes"),
                 ("J1-fused", "fused sweeps, z never stored", "fused"),
                 ("ours-jacobi", "32-bit row pointers", "rowptr32")],
}


def number(value):
    """JSON has no infinity or NaN, so those become None."""
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if row is None:
        return {}
    return {key: row.get(key, "") for key in ("git_commit", "cpu_model", "compiler", "flags", "hostname",
                                              "eigen_version", "logical_cpus")}


def powers(groups):
    return sorted({p for solvers in groups.values() for entry in solvers.values() for p in entry["power"]})


def gate_rows(groups, base, ours, speed):
    rows, limit = summarize.evaluate_gate(groups, base, ours, speed=speed)
    out = []
    for row in rows:
        clean = {"matrix": row["matrix"], "generated": ":" in row["matrix"], "ok": bool(row["ok"]),
                 "missing": row["missing"]}
        if not row["missing"]:
            clean.update(ratio=number(row["ratio"]), ci_low=number(row["ci_low"]), ci_high=number(row["ci_high"]),
                         base_time=number(row["base_time"]), our_time=number(row["our_time"]),
                         base_spmv=number(row["base_spmv"]), our_spmv=number(row["our_spmv"]),
                         noisy=bool(row["noisy"]))
        out.append(clean)
    return out, limit


def speed_section(path):
    groups = summarize.load_groups([str(path)], "per_spmv_s")
    pairs = []
    for base, ours, name in SPEED_PAIRS:
        rows, limit = gate_rows(groups, base, ours, speed=True)
        pairs.append({"name": name, "base": base, "ours": ours, "limit": limit, "rows": rows,
                      "passed": bool(rows) and all(r["ok"] for r in rows)})
    return {"passed": all(p["passed"] for p in pairs), "pairs": pairs, "power": powers(groups),
            "manifest": manifest(path)}


def accuracy_section(path):
    groups = summarize.load_groups([str(path)], "solve_s")
    base, ours = ACCURACY_PAIR
    rows, _ = gate_rows(groups, base, ours, speed=False)
    return {"passed": bool(rows) and all(r["ok"] for r in rows), "base": base, "ours": ours, "rows": rows,
            "power": powers(groups), "manifest": manifest(path)}


def ladder_section(path):
    groups = summarize.load_groups([str(path)], "per_spmv_s")
    present = {solver for solvers in groups.values() for solver in solvers}
    ladders = []
    for base, rungs in LADDER.items():
        points = []
        for rung, label, short in rungs:
            if base in present and rung in present:
                ratio, count = summarize.geometric_mean_ratio(groups, base, rung)
                points.append({"rung": rung, "label": label, "short": short, "ratio": number(ratio), "matrices": count})
        if points:
            ladders.append({"base": base, "points": points})
    return {"ladders": ladders} if ladders else None


def section(loader, path):
    """A missing file is "not run"; a file summarize.py rejects reports its reason."""
    if not path.exists():
        return None
    try:
        return loader(path)
    except (ValueError, KeyError, OSError) as error:
        return {"error": str(error)}


def machine(folder):
    return {
        "name": folder.name,
        "speed": section(speed_section, folder / SPEED_CSV),
        "accuracy": section(accuracy_section, folder / ACCURACY_CSV),
        "ladder": section(ladder_section, folder / LADDER_CSV),
    }


def all_machines(results_dir=None):
    base = results_dir or ROOT / "results"
    if not base.is_dir():
        return []
    return [machine(folder) for folder in sorted(base.iterdir())
            if folder.is_dir() and any((folder / name).exists() for name in (SPEED_CSV, ACCURACY_CSV, LADDER_CSV))]
