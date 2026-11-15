"""Reads every results/<machine>/ folder into chart data, using summarize.py's own statistics.

The gates here are the README's: speed from g1-real.csv, accuracy from g1-real-solve.csv, so the
page and summarize.py can never disagree. g1-ladder.csv, when present, gives the optimization ladder.
"""
import csv
import json
import math
import sys

from steps import ACCURACY_CSV, ONE_SOLUTIONS_DIR, ROOT, SOLUTIONS_DIR, SPEED_CSV

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


def check_picture(entry):
    """Raises ValueError unless entry has every field the page draws, as bench_cg writes them."""
    if not isinstance(entry, dict):
        raise ValueError("not a solution picture")
    for key, kind in (("matrix", str), ("n", int), ("grid", int), ("x_scale", (int, float)), ("comparisons", list)):
        if not isinstance(entry.get(key), kind) or isinstance(entry.get(key), bool):
            raise ValueError(f"not a solution picture: missing or wrong '{key}'")
    for comparison in entry["comparisons"]:
        if not (isinstance(comparison, dict) and isinstance(comparison.get("a"), str)
                and isinstance(comparison.get("b"), str) and isinstance(comparison.get("cells"), list)
                and isinstance(comparison.get("equal"), int)):
            raise ValueError("not a solution picture: a comparison lacks a, b, equal or cells")


def pictures_in(folder):
    """Every solution picture file in folder, sorted by matrix name. A file that cannot be read, or lacks a
    field the page draws, says why instead, so one bad file cannot stop the page from drawing the rest."""
    found = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
            check_picture(entry)
        except (OSError, ValueError) as error:
            entry = {"error": str(error)}
        entry["file"] = path.name
        found.append(entry)
    return sorted(found, key=lambda e: str(e.get("matrix", e["file"])))


def solutions(folder):
    """The Accuracy step's solution pictures: our answer against Eigen's and against the true one."""
    return pictures_in(folder / SOLUTIONS_DIR)


def machine(folder):
    return {
        "name": folder.name,
        "speed": section(speed_section, folder / SPEED_CSV),
        "accuracy": section(accuracy_section, folder / ACCURACY_CSV),
        "ladder": section(ladder_section, folder / LADDER_CSV),
        "solutions": solutions(folder),
    }


def all_machines(results_dir=None):
    base = results_dir or ROOT / "results"
    if not base.is_dir():
        return []
    return [machine(folder) for folder in sorted(base.iterdir())
            if folder.is_dir() and any((folder / name).exists() for name in (SPEED_CSV, ACCURACY_CSV, LADDER_CSV))]


def one_matrix_run(path):
    """A one-MATRIX-TEST.csv from the Run tab, judged by the same gate as the full step."""
    test = path.stem.rsplit("-", 1)[1]
    if test == "speed":
        groups = summarize.load_groups([str(path)], "per_spmv_s")
        pairs = [(name, base, ours, True) for base, ours, name in SPEED_PAIRS]
    else:
        groups = summarize.load_groups([str(path)], "solve_s")
        pairs = [("Jacobi", *ACCURACY_PAIR, False)]
    out = []
    for name, base, ours, speed in pairs:
        rows, _ = gate_rows(groups, base, ours, speed)
        if rows:
            out.append({"name": name, "ratio": rows[0].get("ratio"), "ok": rows[0]["ok"]})
    matrices = [m for m, _ in groups]
    return {"matrix": matrices[0] if matrices else path.stem, "pairs": out,
            "passed": bool(out) and all(p["ok"] for p in out)}


def one_matrix_solutions(results_dir=None):
    """Solution pictures from one-matrix accuracy runs in every results/<name>/one-solutions/ folder.
    Like the one-matrix CSVs, they are listed apart from any machine's G1 results."""
    base = results_dir or ROOT / "results"
    if not base.is_dir():
        return []
    found = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        for entry in pictures_in(folder / ONE_SOLUTIONS_DIR):
            entry["results"] = folder.name
            found.append(entry)
    return found


def one_matrix_runs(results_dir=None):
    """Every results/<name>/one-*.csv, newest first. These never count towards the G1 gates."""
    base = results_dir or ROOT / "results"
    runs = []
    for path in sorted(base.glob("*/one-*-*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        test = path.stem.rsplit("-", 1)[1]
        if test not in ("speed", "accuracy"):
            continue
        entry = {"results": path.parent.name, "file": path.name, "test": test, "matrix": path.stem}
        try:
            entry.update(one_matrix_run(path))
        except (ValueError, KeyError, OSError, IndexError) as error:
            entry.update(error=str(error), pairs=[], passed=False)
        runs.append(entry)
    return runs
