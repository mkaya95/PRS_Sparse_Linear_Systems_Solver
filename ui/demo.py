"""The live demo: our solver and Eigen on the same problem, then our convergence curve.

The timed part is bench_cg exactly as the benchmarks use it: one warm-up, then two measured rounds
that take turns going first, compared by summarize.py's paired ratio. Its --progress lines, printed
outside the timed regions, tell the page which solver is running. The curve comes from a separate,
untimed solve run on the same right-hand side, so recording it never touches the timing.
"""
import csv
import json
import math
import pathlib
import shutil
import sys
import tempfile

from steps import ROOT, command

sys.path.insert(0, str(ROOT / "scripts"))
import run_bench  # noqa: E402
import summarize  # noqa: E402

GENERATED = [
    ("poisson3d:llc=4", "3-D Poisson, 4x the cache", "a few seconds"),
    ("point27:llc=4", "27-point 3-D stencil, 4x the cache", "a few seconds"),
    ("poisson2d:llc=4", "2-D Poisson, 4x the cache", "about a minute"),
    ("poisson3d:llc=0.5", "3-D Poisson, half the cache", "under a second"),
]
PAIRS = {"jacobi": ("ours-jacobi", "eigen-E1"), "none": ("ours", "eigen-E0")}
TOL = "1e-8"
MAX_POINTS = 1500


def choices(data_dir):
    """Generated problems first, then any suite matrix already downloaded."""
    found = [{"spec": spec, "label": label, "takes": takes} for spec, label, takes in GENERATED]
    with open(ROOT / "scripts" / "suite.json", encoding="utf-8") as f:
        suite = json.load(f)
    for entry in suite.get("files", []):
        path = pathlib.Path(data_dir) / f"{entry['name']}.mtx"
        if path.exists():
            found.append({"spec": str(path), "label": f"{entry['name']} (real matrix, n = {entry['rows']:,})",
                          "takes": "a minute or more"})
    return found


def thin(values, limit=MAX_POINTS):
    """Keeps at most limit points, always the first and the last, with their iteration numbers."""
    if len(values) <= limit:
        return list(enumerate(values))
    step = (len(values) - 1) / (limit - 1)
    picked = sorted({round(i * step) for i in range(limit)})
    return [(i, values[i]) for i in picked]


def plan(release, spec, precond):
    """Returns (steps, finish, cleanup). The caller has already checked spec and precond."""
    ours, eigen = PAIRS[precond]
    work = pathlib.Path(tempfile.mkdtemp(prefix="p08-demo-"))
    csv_path, history_path = work / "demo.csv", work / "history.txt"
    steps = [
        {"id": "race", "title": "Head to head", "commands": [command([
            release / "bench_cg", "--matrix", spec, "--solver", f"{ours},{eigen}", "--tol", TOL,
            "--warmup", "1", "--repeat", "2", "--problem", "random", "--csv", csv_path, "--progress", "1"])]},
        {"id": "curve", "title": "Error curve", "commands": [command([
            release / "solve", "--matrix", spec, "--precond", precond, "--problem", "random", "--tol", TOL,
            "--history", history_path])]},
    ]

    def finish(job):
        job.result = summarize_run(csv_path, history_path, spec, precond)
        # Battery once doubled our run times, so the page says so rather than refusing the demo.
        job.result["power"] = run_bench.detect_power()

    def cleanup(job):
        shutil.rmtree(work, ignore_errors=True)

    return steps, finish, cleanup


def summarize_run(csv_path, history_path, spec, precond):
    ours, eigen = PAIRS[precond]
    groups = summarize.load_groups([str(csv_path)], "solve_s")
    rows, _ = summarize.evaluate_gate(groups, eigen, ours)
    if len(rows) != 1 or rows[0]["missing"]:
        raise ValueError("the demo run did not produce both solvers' results")
    row = rows[0]
    (matrix, _), solvers = next(iter(groups.items()))
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    with open(history_path, encoding="ascii") as f:
        history = [float(line) for line in f if line.strip()]

    def side(name):
        entry = solvers[name]
        return {"solver": name, "time": summarize.median(entry["times"]), "spmv": summarize.median(entry["spmv"]),
                "converged": all(entry["verified"])}

    ratio = row["ratio"]
    return {
        "spec": spec, "matrix": matrix, "precond": precond, "n": int(first["n"]), "nnz": int(first["nnz"]),
        "tol": float(TOL), "ours": side(ours), "eigen": side(eigen),
        "ratio": ratio if math.isfinite(ratio) else None,
        "speedup": 1.0 / ratio if math.isfinite(ratio) and ratio > 0 else None,
        "same_answer": bool(row["ok"]),
        "history": [[i, v] for i, v in thin(history) if math.isfinite(v) and v > 0],
        "history_iterations": len(history) - 1,
        "commit": first.get("git_commit", ""), "cpu": first.get("cpu_model", ""),
    }
