"""Tests for the benchmark scripts. Usage: test_scripts.py BUILD_DIR"""
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import summarize  # noqa: E402

BUILD_DIR = None  # set in __main__


def run_bench(*extra, suite):
    """Runs run_bench.py and returns the finished process."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_bench.py"), "--build", BUILD_DIR, "--suite", suite, *extra],
        capture_output=True,
        text=True,
    )


class Statistics(unittest.TestCase):
    def test_median_of_even_count_averages_the_middle(self):
        self.assertEqual(summarize.median(list(range(1, 11))), 5.5)

    def test_median_of_odd_count_is_the_middle(self):
        self.assertEqual(summarize.median([3, 1, 2]), 2)

    def test_confidence_interval_uses_exact_binomial_ranks(self):
        self.assertEqual(summarize.ci_ranks(10), (2, 9))
        self.assertEqual(summarize.ci_ranks(20), (6, 15))

    def test_confidence_interval_falls_back_to_the_range_when_too_few_runs(self):
        self.assertEqual(summarize.ci_ranks(3), (1, 3))
        self.assertEqual(summarize.ci_bounds([5, 1, 3]), (1, 5))

    def test_confidence_interval_bounds_come_from_sorted_values(self):
        self.assertEqual(summarize.ci_bounds(list(range(10, 0, -1))), (2, 9))


class Gate(unittest.TestCase):
    def test_time_ratio_limit_is_1_05(self):
        self.assertTrue(summarize.time_passes(1.04))
        self.assertTrue(summarize.time_passes(1.05))
        self.assertFalse(summarize.time_passes(1.06))

    def test_spmv_count_rule_is_1_percent_or_1_iteration(self):
        self.assertTrue(summarize.spmv_passes(1000, 1009))
        self.assertFalse(summarize.spmv_passes(1000, 1011))
        self.assertTrue(summarize.spmv_passes(10, 11))
        self.assertFalse(summarize.spmv_passes(10, 12))
        self.assertTrue(summarize.spmv_passes(1009, 1000))


class RunBench(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = os.path.join(self.dir.name, "out.csv")
        self.suite = os.path.join(self.dir.name, "suite.json")

    def write_suite(self, generated=(), files=()):
        with open(self.suite, "w", encoding="utf-8") as f:
            json.dump({"generated": list(generated), "files": list(files)}, f)

    def rows(self):
        with open(self.out, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_writes_one_row_per_repeat_with_the_manifest(self):
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "eigen-E0", "--repeat", "2", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["matrix"], "poisson2d:8")
        self.assertEqual(rows[0]["solver"], "eigen-E0")
        self.assertEqual(rows[0]["status"], "Converged")
        self.assertTrue(rows[0]["cpu_model"] or rows[0]["logical_cpus"])
        self.assertLess(float(rows[0]["rel_res_true"]), 1e-7)

    def test_solvers_are_interleaved_round_by_round(self):
        # A machine that slows down mid-run must affect every solver equally.
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "eigen-E0,eigen-E1", "--repeat", "2", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        order = [(row["solver"], row["repeat"]) for row in self.rows()]
        self.assertEqual(order, [("eigen-E0", "0"), ("eigen-E1", "0"), ("eigen-E0", "1"), ("eigen-E1", "1")])

    def test_a_failing_solver_is_reported_once_not_every_round(self):
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "no-such-solver,eigen-E0", "--repeat", "3", "--out", self.out, suite=self.suite)
        self.assertEqual(done.stdout.count("FAILED"), 1, done.stdout)
        self.assertEqual(len(self.rows()), 3)

    def test_missing_matrix_file_is_skipped_and_the_rest_still_run(self):
        self.write_suite(
            generated=[{"stencil": "poisson2d", "n": [8]}],
            files=[{"name": "not_downloaded", "group": "None", "rows": 1, "nnz": 1}],
        )
        done = run_bench(
            "--solvers", "eigen-E0", "--repeat", "1", "--data-dir", self.dir.name, "--out", self.out, suite=self.suite
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("SKIP", done.stdout)
        self.assertIn("not_downloaded", done.stdout)
        self.assertEqual(len(self.rows()), 1)

    def test_a_failing_run_is_reported_and_the_rest_still_run(self):
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "no-such-solver,eigen-E0", "--repeat", "1", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("FAILED", done.stdout)
        self.assertIn("no-such-solver", done.stdout)
        self.assertEqual(len(self.rows()), 1)

    def test_exit_is_nonzero_when_nothing_ran(self):
        self.write_suite(files=[{"name": "not_downloaded", "group": "None", "rows": 1, "nnz": 1}])
        done = run_bench(
            "--solvers", "eigen-E0", "--repeat", "1", "--data-dir", self.dir.name, "--out", self.out, suite=self.suite
        )
        self.assertNotEqual(done.returncode, 0)


class Summarize(unittest.TestCase):
    def test_gate_table_reports_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "status"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for matrix, ours in (("good", 1.04), ("bad", 1.5)):
                    for _ in range(3):
                        writer.writerow(dict(matrix=matrix, solver="base", spmv_count=100, per_spmv_s=1.0, status="Converged"))
                        writer.writerow(dict(matrix=matrix, solver="ours", spmv_count=100, per_spmv_s=ours, status="Converged"))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--gate", "base=ours"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            lines = done.stdout.splitlines()
            good = next(line for line in lines if line.startswith("| good"))
            bad = next(line for line in lines if line.startswith("| bad"))
            self.assertIn("PASS", good)
            self.assertIn("FAIL", bad)


if __name__ == "__main__":
    BUILD_DIR = sys.argv.pop(1)
    unittest.main()
