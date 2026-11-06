"""Tests for the benchmark scripts. Usage: test_scripts.py BUILD_DIR"""
import csv
import json
import os
import pathlib
import random
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

    def test_solvers_are_interleaved_and_rotate_who_goes_first(self):
        # A machine that slows down mid-run must affect every solver equally, and the
        # first solver in a round was measured to run about 20% slower than the second.
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "eigen-E0,eigen-E1", "--repeat", "2", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        order = [(row["solver"], row["repeat"]) for row in self.rows()]
        self.assertEqual(order, [("eigen-E0", "0"), ("eigen-E1", "0"), ("eigen-E1", "1"), ("eigen-E0", "1")])

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


class BenchKernels(unittest.TestCase):
    def run_kernels(self, impl, out):
        return subprocess.run(
            [os.path.join(BUILD_DIR, "bench_kernels"), "--matrix", "poisson2d:50", "--impl", impl,
             "--repeat", "2", "--min-ms", "1", "--csv", out],
            capture_output=True,
            text=True,
        )

    def test_each_implementation_writes_a_row_per_kernel_and_repeat(self):
        for impl in ("L0", "eigen"):
            with self.subTest(impl=impl), tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "k.csv")
                done = self.run_kernels(impl, out)
                self.assertEqual(done.returncode, 0, done.stderr)
                with open(out, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 10)
                self.assertEqual({r["kernel"] for r in rows}, {"spmv", "dot", "axpy", "norm2", "xpby"})
                self.assertTrue(all(float(r["seconds"]) > 0 for r in rows))

    def test_implementations_take_turns(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "k.csv")
            done = self.run_kernels("L0,eigen", out)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 20)
            self.assertEqual([r["impl"] for r in rows[:2]], ["L0", "eigen"])
            self.assertEqual([r["impl"] for r in rows[10:12]], ["eigen", "L0"])  # rotated next round

    def test_spmv_samples_always_take_several_calls(self):
        # With one call per sample the input and output vectors would never swap.
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "k.csv")
            done = self.run_kernels("L0", out)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["kernel"] == "spmv"]
            self.assertTrue(all(int(r["calls"]) >= 4 for r in rows), rows)

    def test_an_unknown_implementation_names_the_valid_ones(self):
        with tempfile.TemporaryDirectory() as d:
            done = self.run_kernels("no-such-impl", os.path.join(d, "k.csv"))
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("eigen", done.stderr)

    def test_summarize_prints_kernel_ratios_against_a_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["matrix", "impl", "kernel", "seconds", "gb_per_s"])
                writer.writeheader()
                for _ in range(3):
                    writer.writerow(dict(matrix="m", impl="eigen", kernel="dot", seconds=2.0, gb_per_s=1))
                    writer.writerow(dict(matrix="m", impl="L0", kernel="dot", seconds=3.0, gb_per_s=1))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--kernel-baseline", "eigen"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            line = next(l for l in done.stdout.splitlines() if l.startswith("| m | dot | L0"))
            self.assertIn("1.500", line)


class PairedRatios(unittest.TestCase):
    def test_each_round_is_compared_with_itself(self):
        # The machine doubles in speed between rounds. Ours is always 2% slower than base
        # within a round, but a ratio of overall medians would report 0.51.
        base = {0: 1.0, 1: 2.0, 2: 2.0}
        ours = {0: 1.02, 1: 2.04, 2: 1.02}
        ratios = summarize.paired_ratios(base, ours)
        self.assertEqual(sorted(round(r, 3) for r in ratios), [0.51, 1.02, 1.02])
        self.assertAlmostEqual(summarize.median(ratios), 1.02)

    def test_rounds_present_on_only_one_side_are_ignored(self):
        self.assertEqual(summarize.paired_ratios({0: 1.0, 1: 1.0}, {1: 2.0, 2: 9.0}), [2.0])

    def test_gate_uses_paired_ratios_when_repeats_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "status", "repeat"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for repeat, (base, ours) in enumerate([(1.0, 1.02), (2.0, 2.04), (4.0, 4.08)]):
                    writer.writerow(dict(matrix="m", solver="base", spmv_count=100, per_spmv_s=base, status="Converged", repeat=repeat))
                    writer.writerow(dict(matrix="m", solver="ours", spmv_count=100, per_spmv_s=ours, status="Converged", repeat=repeat))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--gate", "base=ours"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            line = next(l for l in done.stdout.splitlines() if l.startswith("| m "))
            self.assertIn("1.020", line)
            self.assertIn("PASS", line)


class Labels(unittest.TestCase):
    def test_a_label_separates_the_same_solver_from_two_builds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "status", "label"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for label, time in (("o2", 2.0), ("o3", 1.0)):
                    writer.writerow(dict(matrix="m", solver="L0", spmv_count=100, per_spmv_s=time, status="Converged", label=label))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--gate", "L0@o2=L0@o3"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            line = next(l for l in done.stdout.splitlines() if l.startswith("| m "))
            self.assertIn("0.500", line)


class Ladder(unittest.TestCase):
    def test_ladder_table_reports_each_rung_against_the_previous_one(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "status", "repeat"]
            times = {"a": {"L0": 4.0, "L1": 2.0, "L2": 2.0}, "b": {"L0": 1.0, "L1": 1.0, "L2": 0.5}}
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for matrix, rungs in times.items():
                    for solver, time in rungs.items():
                        writer.writerow(dict(matrix=matrix, solver=solver, spmv_count=10, per_spmv_s=time, status="Converged", repeat=0))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--ladder", "L0,L1,L2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            lines = done.stdout.splitlines()
            # Geometric means across matrices: L1/L0 = sqrt(0.5 * 1), L2/L0 = sqrt(0.5 * 0.5).
            l1 = next(l for l in lines if l.startswith("| L1 "))
            self.assertIn("0.707", l1)
            l2 = next(l for l in lines if l.startswith("| L2 "))
            self.assertIn("0.500", l2)  # against L0
            self.assertIn("0.707", l2)  # against L1


class SolveCommand(unittest.TestCase):
    def run_solve(self, *args):
        return subprocess.run([os.path.join(BUILD_DIR, "solve"), *args], capture_output=True, text=True)

    def test_a_converged_solve_exits_0_and_reports_the_true_residual(self):
        done = self.run_solve("--matrix", "poisson2d:30")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Converged", done.stdout)
        line = next(l for l in done.stdout.splitlines() if l.startswith("true residual"))
        self.assertLess(float(line.split()[-1]), 1e-7)

    def test_jacobi_is_selectable(self):
        done = self.run_solve("--matrix", "poisson2d:30", "--precond", "jacobi")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("jacobi", done.stdout)

    def test_running_out_of_iterations_exits_2(self):
        done = self.run_solve("--matrix", "poisson2d:30", "--maxit", "1")
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("MaxIterations", done.stdout)

    def test_a_missing_file_exits_1_and_names_it(self):
        done = self.run_solve("--matrix", "missing.mtx")
        self.assertEqual(done.returncode, 1)
        self.assertIn("missing.mtx", done.stderr)

    def test_an_unknown_preconditioner_exits_1(self):
        done = self.run_solve("--matrix", "poisson2d:5", "--precond", "ilu")
        self.assertEqual(done.returncode, 1)
        self.assertIn("ilu", done.stderr)

    def test_recursive_convergence_is_not_accepted_without_true_accuracy(self):
        rng = random.Random(31)
        n = 20
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                a[i][j] = a[j][i] = -rng.uniform(0.5, 2.0)
        for i in range(n):
            a[i][i] = -sum(a[i]) + 1e-9  # symmetric, positive, strictly diagonally dominant
        with tempfile.TemporaryDirectory() as d:
            matrix = pathlib.Path(d) / "spd.mtx"
            matrix.write_text(
                f"%%MatrixMarket matrix coordinate real general\n{n} {n} {n*n}\n"
                + "".join(f"{i+1} {j+1} {a[i][j]:.17g}\n" for i in range(n) for j in range(n)),
                encoding="ascii")
            done = self.run_solve("--matrix", str(matrix))
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertIn("ResidualFailure", done.stdout)

    def test_a_nonsymmetric_matrix_file_is_rejected_before_iterating(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.mtx")
            with open(path, "w", encoding="ascii") as f:
                f.write("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 2\n1 2 1\n2 2 2\n")
            done = self.run_solve("--matrix", path)
            self.assertEqual(done.returncode, 1)
            self.assertIn("symmetric", done.stderr)

    def test_the_solution_can_be_written_to_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.txt")
            done = self.run_solve("--matrix", "poisson2d:5", "--out", path)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            with open(path, encoding="ascii") as f:
                values = [float(line) for line in f]
            self.assertEqual(len(values), 25)
            self.assertTrue(all(abs(v - 1.0) < 1e-5 for v in values))


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
