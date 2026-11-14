"""Tests for the benchmark scripts. Usage: test_scripts.py BUILD_DIR"""
import csv
import http.server
import io
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import download_matrices  # noqa: E402
import run_bench as run_bench_script  # noqa: E402
import summarize  # noqa: E402
import sweep_sell  # noqa: E402

BUILD_DIR = None  # set in __main__


def run_bench(*extra, suite):
    """Runs run_bench.py and returns the finished process."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_bench.py"), "--build", BUILD_DIR, "--suite", suite,
         "--power", "ac", *extra],
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

    def test_an_accuracy_gate_accepts_equal_or_fewer_iterations(self):
        self.assertEqual(summarize.iterations_verdict(1000, 1009, 1e-8, None), (True, ""))
        self.assertEqual(summarize.iterations_verdict(1000, 700, 1e-8, None), (True, "fewer iterations"))

    def test_more_iterations_pass_only_on_a_recorded_plateau(self):
        # history[k] is the residual after k iterations; the base stopped after 3.
        near = [1.0, 1e-4, 2e-8, 1.04e-8, 1.2e-8, 1.1e-8, 0.99e-8]
        far = [1.0, 1e-2, 1e-4, 1.06e-8, 1e-7, 2e-8, 0.99e-8]
        self.assertEqual(summarize.iterations_verdict(3, 6, 1e-8, near), (True, "plateau"))
        self.assertEqual(summarize.iterations_verdict(3, 6, 1e-8, far), (False, ""))
        self.assertEqual(summarize.iterations_verdict(3, 6, 1e-8, None), (False, ""))
        self.assertEqual(summarize.iterations_verdict(3, 6, None, near), (False, ""))
        self.assertEqual(summarize.iterations_verdict(30, 60, 1e-8, near), (False, ""))  # history too short
        self.assertEqual(summarize.iterations_verdict(3, 6, 1e-8, [1.0, 1e-4, 2e-8, None]), (False, ""))


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
        done = run_bench("--solvers", "eigen-E0", "--warmup", "2", "--repeat", "2", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["matrix"], "poisson2d:8")
        self.assertEqual(rows[0]["solver"], "eigen-E0")
        self.assertEqual(rows[0]["status"], "Converged")
        self.assertEqual(rows[0]["warmup"], "2")
        self.assertEqual([row["repeat"] for row in rows], ["0", "1"])
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
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("FAILED", done.stdout)
        self.assertIn("no-such-solver", done.stdout)
        self.assertEqual(len(self.rows()), 1)

    def test_exit_is_nonzero_when_nothing_ran(self):
        self.write_suite(files=[{"name": "not_downloaded", "group": "None", "rows": 1, "nnz": 1}])
        done = run_bench(
            "--solvers", "eigen-E0", "--repeat", "1", "--data-dir", self.dir.name, "--out", self.out, suite=self.suite
        )
        self.assertNotEqual(done.returncode, 0)

    def test_solution_pictures_go_one_file_per_matrix(self):
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8, 9]}])
        pictures = os.path.join(self.dir.name, "pictures")
        done = run_bench("--solvers", "ours,eigen-E0", "--repeat", "1", "--out", self.out, "--solutions", pictures,
                         suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(sorted(os.listdir(pictures)), ["poisson2d_8.json", "poisson2d_9.json"])
        with open(os.path.join(pictures, "poisson2d_9.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["n"], 81)

    def test_fixed_iteration_mode_is_explicit_and_not_a_solve_gate(self):
        self.write_suite(generated=[{"stencil": "poisson2d", "n": [8]}])
        done = run_bench("--solvers", "eigen-E0,ours", "--mode", "fixed", "--maxit", "1",
                         "--repeat", "1", "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(all(row["status"] == "MaxIterations" for row in self.rows()))
        report = subprocess.run(
            [sys.executable, str(ROOT / "scripts/summarize.py"), self.out, "--gate", "eigen-E0=ours"],
            capture_output=True, text=True)
        self.assertEqual(report.returncode, 2, report.stdout + report.stderr)
        self.assertNotIn("PASS", report.stdout)


class Solutions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = os.path.join(self.dir.name, "solutions.json")

    def run_cg(self, solvers, *extra):
        return subprocess.run(
            [os.path.join(BUILD_DIR, "bench_cg"), "--matrix", "poisson2d:20", "--solver", solvers, "--repeat", "1",
             "--warmup", "0", *extra],
            capture_output=True, text=True, cwd=self.dir.name)

    def pictures(self):
        with open(self.file, encoding="utf-8") as f:
            return json.load(f)

    def test_each_pair_and_each_solver_against_the_true_answer_is_written(self):
        done = self.run_cg("ours-jacobi,eigen-E1", "--solutions", self.file)
        self.assertEqual(done.returncode, 0, done.stderr)
        data = self.pictures()
        self.assertEqual((data["matrix"], data["n"], data["grid"], data["problem"]), ("poisson2d:20", 400, 20, "random"))
        self.assertEqual([(c["a"], c["b"]) for c in data["comparisons"]],
                         [("ours-jacobi", "eigen-E1"), ("ours-jacobi", "true"), ("eigen-E1", "true")])
        for comparison in data["comparisons"]:
            self.assertLess(comparison["rel_norm"], 1e-6)
            self.assertEqual(len(comparison["cells"]), 400)
        self.assertGreater(data["x_scale"], 0.5)
        self.assertLessEqual(data["x_scale"], 1.0)

    def test_the_unit_problem_has_no_true_answer(self):
        done = self.run_cg("ours-jacobi,eigen-E1", "--problem", "unit", "--solutions", self.file)
        self.assertEqual(done.returncode, 0, done.stderr)
        data = self.pictures()
        self.assertEqual([(c["a"], c["b"]) for c in data["comparisons"]], [("ours-jacobi", "eigen-E1")])
        self.assertGreater(data["x_scale"], 0)

    def test_a_dropped_solver_is_left_out(self):
        done = self.run_cg("no-such-solver,ours", "--solutions", self.file)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertEqual([(c["a"], c["b"]) for c in self.pictures()["comparisons"]], [("ours", "true")])

    def test_a_solver_that_did_not_converge_is_left_out(self):
        # The Accuracy step runs without a warm-up, so a failed solve must be dropped from its measured round too.
        done = self.run_cg("ours,ours-ic", "--maxit", "30", "--solutions", self.file)
        self.assertEqual(done.returncode, 1, done.stdout)  # ours needs 60 iterations here, ours-ic 21
        self.assertEqual([(c["a"], c["b"]) for c in self.pictures()["comparisons"]], [("ours-ic", "true")])

    def test_a_run_where_no_solver_converged_still_writes_its_file(self):
        done = self.run_cg("ours,eigen-E0", "--maxit", "1", "--solutions", self.file)
        self.assertEqual(done.returncode, 1, done.stdout)
        data = self.pictures()
        self.assertEqual((data["n"], data["comparisons"]), (400, []))

    def test_our_solvers_get_a_convergence_history_and_eigen_none(self):
        done = self.run_cg("ours-jacobi,eigen-E1,ours-ic0", "--solutions", self.file)
        self.assertEqual(done.returncode, 0, done.stderr)
        histories = {h["solver"]: h["residuals"] for h in self.pictures()["histories"]}
        self.assertEqual(sorted(histories), ["ours-ic0", "ours-jacobi"])
        counts = {line.split()[1]: int(line.split("spmv=")[1].split()[0]) for line in done.stdout.splitlines()}
        for solver, residuals in histories.items():
            self.assertEqual(len(residuals), counts[solver] + 1)
            self.assertEqual(residuals[0], 1)
            self.assertLess(residuals[-1], 1e-8)
            self.assertGreaterEqual(residuals[-2], 1e-8)
        self.assertEqual(summarize.load_histories(self.dir.name)["poisson2d:20"], histories)

    def test_compare_same_precond_draws_only_counterparts(self):
        done = self.run_cg("eigen-E1,ours-jacobi,eigen-E4,ours-ic,ours-sgs", "--solutions", self.file,
                           "--compare", "same-precond")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual([(c["a"], c["b"]) for c in self.pictures()["comparisons"]],
                         [("eigen-E1", "ours-jacobi"), ("eigen-E4", "ours-ic"), ("eigen-E1", "true"),
                          ("ours-jacobi", "true"), ("eigen-E4", "true"), ("ours-ic", "true"), ("ours-sgs", "true")])

    def test_compare_needs_solutions_and_a_known_value(self):
        for extra in (("--compare", "same-precond"), ("--solutions", self.file, "--compare", "some")):
            done = self.run_cg("ours", *extra)
            self.assertEqual(done.returncode, 1, done.stdout)
            self.assertIn("--compare", done.stderr)

    def test_without_the_option_no_file_is_written(self):
        done = self.run_cg("ours")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(os.listdir(self.dir.name), [])


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = os.path.join(self.dir.name, "out.csv")

    def run_cg(self, *extra):
        return subprocess.run(
            [os.path.join(BUILD_DIR, "bench_cg"), "--matrix", "poisson2d:12", "--solver", "ours", "--repeat", "1",
             "--csv", self.out, *extra],
            capture_output=True,
            text=True,
        )

    def rows(self):
        with open(self.out, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_the_default_problem_is_a_random_solution(self):
        done = self.run_cg()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.rows()[0]["problem"], "random")

    def test_each_problem_is_a_different_right_hand_side_and_is_recorded(self):
        counts = {}
        for problem in ("a1", "random", "unit"):
            with self.subTest(problem=problem):
                if os.path.exists(self.out):
                    os.remove(self.out)
                done = self.run_cg("--problem", problem)
                self.assertEqual(done.returncode, 0, done.stderr)
                row = self.rows()[0]
                self.assertEqual(row["problem"], problem)
                self.assertLessEqual(float(row["rel_res_true"]), 1e-8)
                counts[problem] = int(row["spmv_count"])
        self.assertGreater(len(set(counts.values())), 1, counts)

    def test_an_unknown_problem_is_rejected(self):
        done = self.run_cg("--problem", "nonsense")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--problem", done.stderr)

    def test_run_bench_passes_the_problem_on(self):
        suite = os.path.join(self.dir.name, "suite.json")
        with open(suite, "w", encoding="utf-8") as f:
            json.dump({"generated": [{"stencil": "poisson2d", "n": [8]}]}, f)
        done = run_bench("--solvers", "ours", "--repeat", "1", "--problem", "unit", "--out", self.out, suite=suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(self.rows()[0]["problem"], "unit")

    def test_results_from_different_problems_are_not_pooled(self):
        self.assertEqual(self.run_cg("--problem", "a1").returncode, 0)
        self.assertEqual(self.run_cg("--problem", "random").returncode, 0)
        done = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), self.out],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("incompatible", done.stderr)


class PowerGuard(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = os.path.join(self.dir.name, "out.csv")
        self.suite = os.path.join(self.dir.name, "suite.json")
        with open(self.suite, "w", encoding="utf-8") as f:
            json.dump({"generated": [{"stencil": "poisson2d", "n": [8]}]}, f)

    def test_it_refuses_to_measure_on_battery(self):
        done = run_bench("--solvers", "ours", "--repeat", "1", "--power", "battery", "--out", self.out, suite=self.suite)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("battery", done.stdout)
        self.assertFalse(os.path.exists(self.out))

    def test_battery_can_be_allowed_and_is_recorded(self):
        done = run_bench("--solvers", "ours", "--repeat", "1", "--power", "battery", "--allow-battery",
                         "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        with open(self.out, newline="", encoding="utf-8") as f:
            self.assertEqual(next(csv.DictReader(f))["power"], "battery")

    def test_windows_battery_status_codes(self):
        self.assertEqual(run_bench_script.power_from_windows("1\r\n"), "battery")
        self.assertEqual(run_bench_script.power_from_windows("2\r\n"), "ac")
        self.assertEqual(run_bench_script.power_from_windows("3\r\n"), "ac")
        self.assertEqual(run_bench_script.power_from_windows("5\r\n"), "battery")
        self.assertEqual(run_bench_script.power_from_windows(""), "ac")  # a desktop has no battery

    def test_macos_power_source(self):
        self.assertEqual(run_bench_script.power_from_pmset("Now drawing from 'AC Power'\n"), "ac")
        self.assertEqual(run_bench_script.power_from_pmset("Now drawing from 'Battery Power'\n"), "battery")
        self.assertEqual(run_bench_script.power_from_pmset("nothing useful"), "unknown")

    def test_linux_power_supply_directories(self):
        base = pathlib.Path(self.dir.name) / "supply"
        (base / "AC").mkdir(parents=True)
        (base / "AC" / "type").write_text("Mains\n", encoding="ascii")
        (base / "AC" / "online").write_text("0\n", encoding="ascii")
        self.assertEqual(run_bench_script.power_from_sysfs(base), "battery")
        (base / "AC" / "online").write_text("1\n", encoding="ascii")
        self.assertEqual(run_bench_script.power_from_sysfs(base), "ac")
        self.assertEqual(run_bench_script.power_from_sysfs(base / "missing"), "unknown")

    def test_a_linux_mains_supply_without_an_online_file_is_unknown(self):
        base = pathlib.Path(self.dir.name) / "supply"
        (base / "AC").mkdir(parents=True)
        (base / "AC" / "type").write_text("Mains\n", encoding="ascii")
        self.assertEqual(run_bench_script.power_from_sysfs(base), "unknown")

    def test_a_linux_battery_decides_only_when_it_is_discharging(self):
        base = pathlib.Path(self.dir.name) / "supply"
        (base / "BAT0").mkdir(parents=True)
        (base / "BAT0" / "type").write_text("Battery\n", encoding="ascii")
        (base / "BAT0" / "status").write_text("Discharging\n", encoding="ascii")
        self.assertEqual(run_bench_script.power_from_sysfs(base), "battery")
        (base / "BAT0" / "status").write_text("Charging\n", encoding="ascii")
        self.assertEqual(run_bench_script.power_from_sysfs(base), "unknown")

    def test_the_summary_warns_about_battery_results(self):
        done = run_bench("--solvers", "ours", "--repeat", "1", "--power", "battery", "--allow-battery",
                         "--out", self.out, suite=self.suite)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        report = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), self.out],
                                capture_output=True, text=True)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn("battery", report.stdout)


class ScipyCheck(unittest.TestCase):
    def setUp(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("needs SciPy")
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.matrix = os.path.join(self.dir.name, "poisson.mtx")
        side = 12
        entries = []
        for i in range(side * side):
            entries.append((i, i, 4.0))
            if (i + 1) % side != 0:
                entries.append((i + 1, i, -1.0))
            if i + side < side * side:
                entries.append((i + side, i, -1.0))
        with open(self.matrix, "w", encoding="ascii") as f:
            f.write("%%MatrixMarket matrix coordinate real symmetric\n")
            f.write(f"{side * side} {side * side} {len(entries)}\n")
            for row, col, value in entries:
                f.write(f"{row + 1} {col + 1} {value}\n")

    def check(self, build, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_against_scipy.py"), "--build", build, "--matrix", self.matrix,
             *extra],
            capture_output=True,
            text=True,
        )

    def test_the_solver_agrees_with_scipy(self):
        done = self.check(BUILD_DIR)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASS", done.stdout)
        self.assertIn("direct", done.stdout)

    def test_the_iteration_count_matches_scipy_cg(self):
        done = self.check(BUILD_DIR, "--reference", "cg")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertRegex(done.stdout, r"CG, \d+ iterations")
        self.assertIn("PASS", done.stdout)

    def test_a_different_iteration_count_is_caught(self):
        fake = pathlib.Path(self.dir.name) / "solve"
        fake.write_text(
            f"#!/bin/sh\n{os.path.abspath(BUILD_DIR)}/solve \"$@\" | sed 's/spmv count .*/spmv count 100000/'\n",
            encoding="ascii",
        )
        fake.chmod(0o755)
        done = self.check(self.dir.name, "--reference", "cg")
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("FAIL", done.stdout)

    def test_older_scipy_spelling_of_the_tolerance_works(self):
        # SciPy before 1.12 calls rtol "tol", and Ubuntu 24.04 packages 1.11.
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_against_scipy as script
        import scipy.sparse
        real_cg = script.sla.cg

        def old_cg(a, b, tol, atol, maxiter, M, callback):
            return real_cg(a, b, rtol=tol, atol=atol, maxiter=maxiter, M=M, callback=callback)

        script.sla.cg = old_cg
        self.addCleanup(setattr, script.sla, "cg", real_cg)
        x, count = script.scipy_cg(scipy.sparse.identity(3, format="csr"), [1.0, 2.0, 3.0], 1e-8, 10)
        self.assertEqual(list(x), [1.0, 2.0, 3.0])
        self.assertEqual(count, 1)

    def test_a_wrong_answer_is_caught(self):
        fake = pathlib.Path(self.dir.name) / "solve"
        fake.write_text(
            "#!/bin/sh\n"
            "while [ $# -gt 0 ]; do [ \"$1\" = --out ] && out=$2; shift; done\n"
            "i=0; while [ $i -lt 144 ]; do echo 0.5; i=$((i+1)); done > \"$out\"\n"
            "echo 'spmv count     1'\n",
            encoding="ascii",
        )
        fake.chmod(0o755)
        done = self.check(self.dir.name)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("FAIL", done.stdout)


class SellSweep(unittest.TestCase):
    def test_only_allowed_layouts_are_swept(self):
        self.assertEqual(sweep_sell.implementations([4, 8], [1, 4, 8]),
                         ["sell-c4-s1", "sell-c4-s4", "sell-c4-s8", "sell-c8-s1", "sell-c8-s8"])

    def test_the_summary_pairs_rounds_and_finds_the_break_even(self):
        def timing(impl, repeat, seconds):
            return dict(matrix="m", impl=impl, kernel="spmv", repeat=str(repeat), seconds=str(seconds))
        timings = [timing("L3-acc2", 0, 10.0), timing("L3-acc2", 1, 20.0),
                   timing("sell-c8-s1", 0, 8.0), timing("sell-c8-s1", 1, 16.0),
                   timing("sell-c4-s1", 0, 11.0), timing("sell-c4-s1", 1, 22.0),
                   dict(timing("sell-c8-s1", 0, 99.0), kernel="dot")]
        layouts = [dict(matrix="m", impl="sell-c8-s1", stored_per_nnz="1.25", build_s="30"),
                   dict(matrix="m", impl="sell-c4-s1", stored_per_nnz="1.5", build_s="30")]
        lines = sweep_sell.summary(timings, layouts)["m"]
        self.assertEqual(lines[0], ("sell-c8-s1", 1.25, 0.8, 10.0))  # medians 15 and 12: 30 s / 3 s saved
        self.assertEqual(lines[1][0], "sell-c4-s1")
        self.assertAlmostEqual(lines[1][2], 1.1)
        self.assertIsNone(lines[1][3])  # slower than CSR, so the conversion never pays

    def test_a_sweep_measures_every_layout_and_prints_a_table(self):
        with tempfile.TemporaryDirectory() as d:
            suite = os.path.join(d, "suite.json")
            with open(suite, "w", encoding="utf-8") as f:
                json.dump({"generated": [{"stencil": "poisson2d", "n": [30]}], "files": []}, f)
            out = os.path.join(d, "sweep.csv")
            command = [sys.executable, str(ROOT / "scripts" / "sweep_sell.py"), "--build", BUILD_DIR, "--suite", suite,
                       "--out", out, "--chunks", "4,8", "--sigmas", "1,8", "--repeat", "2", "--min-ms", "1",
                       "--pin", "none", "--allow-battery"]
            done = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["kernel"] for r in rows}, {"spmv"})
            self.assertEqual(len(rows), 2 * 5)  # CSR and four layouts, two rounds
            with open(os.path.join(d, "sweep-layout.csv"), newline="", encoding="utf-8") as f:
                layouts = list(csv.DictReader(f))
            self.assertEqual([r["impl"] for r in layouts], ["sell-c4-s1", "sell-c4-s8", "sell-c8-s1", "sell-c8-s8"])
            self.assertTrue(all(float(r["stored_per_nnz"]) >= 1 for r in layouts))
            self.assertIn("| sell-c8-s8 |", done.stdout)
            self.assertIn("Fastest: sell-c", done.stdout)
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("exists", again.stderr)


class BenchKernels(unittest.TestCase):
    def test_an_unknown_kernel_is_refused(self):
        done = subprocess.run([os.path.join(BUILD_DIR, "bench_kernels"), "--matrix", "poisson2d:10", "--impl", "L0",
                               "--kernel", "spmv,gemm"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("gemm", done.stderr)

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

    def test_a_sell_implementation_is_timed_like_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "k.csv")
            done = self.run_kernels("L3-acc2,sell-c8-s256", out)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual({r["impl"] for r in rows}, {"L3-acc2", "sell-c8-s256"})
        self.assertEqual({r["kernel"] for r in rows if r["impl"] == "sell-c8-s256"},
                         {"spmv", "dot", "axpy", "norm2", "xpby"})

    def test_a_bad_sell_name_lists_the_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            done = self.run_kernels("sell-c6-s1", os.path.join(d, "k.csv"))
        self.assertEqual(done.returncode, 1)
        self.assertIn("chunk must be 4, 8, 16 or 32", done.stderr)

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


class Stream(unittest.TestCase):
    def test_it_reports_a_read_only_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.csv")
            done = subprocess.run(
                [os.path.join(BUILD_DIR, "stream"), "--array-mb", "8", "--repeat", "2", "--csv", out],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                rates = {r["kernel"]: float(r["best_gb_per_s"]) for r in csv.DictReader(f)}
        self.assertEqual(set(rates), {"read", "copy", "scale", "add", "triad"})
        self.assertGreater(rates["read"], 0.0)


class DownloadFallback(unittest.TestCase):
    """A local server stands in for the matrix hosts: /denied refuses, as the main host does."""

    ENTRY = {"name": "tiny", "group": "Test", "rows": 2, "nnz": 2}
    MTX = b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 1\n"

    def setUp(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            info = tarfile.TarInfo("tiny/tiny.mtx")
            info.size = len(self.MTX)
            tar.addfile(info, io.BytesIO(self.MTX))
        archive = buffer.getvalue()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/mirror/"):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(archive)
                else:
                    self.send_error(403)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(setattr, download_matrices, "BASE_URLS", download_matrices.BASE_URLS)

    def test_a_refused_download_falls_back_to_the_next_host(self):
        download_matrices.BASE_URLS = [
            self.base + "/denied/{group}/{name}.tar.gz",
            self.base + "/mirror/{group}/{name}.tar.gz",
        ]
        with tempfile.TemporaryDirectory() as d:
            download_matrices.fetch(self.ENTRY, d)
            self.assertEqual(pathlib.Path(d, "tiny.mtx").read_bytes(), self.MTX)

    def test_it_fails_when_every_host_refuses(self):
        download_matrices.BASE_URLS = [self.base + "/denied/{group}/{name}.tar.gz"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(urllib.error.URLError):
                download_matrices.fetch(self.ENTRY, d)
            self.assertFalse(pathlib.Path(d, "tiny.mtx").exists())


class BuildInfo(unittest.TestCase):
    """Runs cmake/write_build_info.cmake on a small git repository."""

    def setUp(self):
        if shutil.which("git") is None or shutil.which("cmake") is None:
            self.skipTest("needs git and cmake")
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.repo = pathlib.Path(work.name)
        (self.repo / "tracked.txt").write_text("one\n", encoding="utf-8")
        self.git("init", "-q")
        self.git("add", "tracked.txt")
        self.git("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "first")

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def is_dirty(self):
        output = self.repo / "info.h"
        subprocess.run(
            ["cmake", f"-DSOURCE_DIR={self.repo}", f"-DOUTPUT={output}", "-DFLAGS=x", "-P",
             str(ROOT / "cmake" / "write_build_info.cmake")],
            check=True,
            capture_output=True,
        )
        text = output.read_text(encoding="utf-8")
        output.unlink()
        return "-dirty" in text

    def test_an_untracked_file_does_not_mark_the_build_dirty(self):
        (self.repo / "lab_notes.txt").write_text("not part of the build\n", encoding="utf-8")
        self.assertFalse(self.is_dirty())

    def test_a_changed_tracked_file_marks_the_build_dirty(self):
        (self.repo / "tracked.txt").write_text("two\n", encoding="utf-8")
        self.assertTrue(self.is_dirty())


class PreconditionerColumn(unittest.TestCase):
    def test_the_csv_names_the_preconditioner_each_solver_really_uses(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.csv")
            done = subprocess.run(
                [os.path.join(BUILD_DIR, "bench_cg"), "--matrix", "poisson2d:5", "--solver",
                 "ours,ours-jacobi,eigen-E0,eigen-E1,eigen-E3", "--repeat", "1", "--csv", out],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                labels = {r["solver"]: r["precond"] for r in csv.DictReader(f)}
        self.assertEqual(
            labels,
            {
                "ours": "none",
                "ours-jacobi": "jacobi",
                "eigen-E0": "none",
                "eigen-E1": "jacobi",
                "eigen-E3": "incomplete-cholesky",
            },
        )


class Progress(unittest.TestCase):
    def bench(self, *extra):
        return subprocess.run(
            [os.path.join(BUILD_DIR, "bench_cg"), "--matrix", "poisson2d:12", "--solver", "ours,eigen-E0",
             "--warmup", "1", "--repeat", "2", *extra],
            capture_output=True,
            text=True,
        )

    def test_progress_names_each_stage_before_it_starts(self):
        done = self.bench("--progress", "1")
        self.assertEqual(done.returncode, 0, done.stderr)
        stages = [line for line in done.stderr.splitlines() if line.startswith("progress:")]
        self.assertEqual(stages, ["progress: read",
                                  "progress: solve ours warmup", "progress: solve eigen-E0 warmup",
                                  "progress: solve ours round 1 of 2", "progress: solve eigen-E0 round 1 of 2",
                                  "progress: solve eigen-E0 round 2 of 2", "progress: solve ours round 2 of 2"])
        self.assertEqual(len(done.stdout.splitlines()), 4)

    def test_without_the_flag_the_output_is_unchanged(self):
        done = self.bench()
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertEqual(len(done.stdout.splitlines()), 4)

    def test_the_flag_is_zero_or_one(self):
        self.assertEqual(self.bench("--progress", "2").returncode, 1)


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
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "solve_s", "status", "repeat", "tol", "rel_res_true"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for repeat, (base, ours) in enumerate([(1.0, 1.02), (2.0, 2.04), (4.0, 4.08)]):
                    writer.writerow(dict(matrix="m", solver="base", spmv_count=100, per_spmv_s=base, solve_s=base,
                                         status="Converged", repeat=repeat, tol=1e-8, rel_res_true=1e-10))
                    writer.writerow(dict(matrix="m", solver="ours", spmv_count=100, per_spmv_s=ours, solve_s=ours,
                                         status="Converged", repeat=repeat, tol=1e-8, rel_res_true=1e-10))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--gate", "base=ours"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            line = next(l for l in done.stdout.splitlines() if l.startswith("| m "))
            self.assertIn("1.020", line)
            self.assertIn("VERIFIED", line)


class Labels(unittest.TestCase):
    def test_a_label_separates_the_same_solver_from_two_builds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "solve_s", "status", "label", "tol", "rel_res_true"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for label, time in (("o2", 2.0), ("o3", 1.0)):
                    writer.writerow(dict(matrix="m", solver="L0", spmv_count=100, per_spmv_s=time, solve_s=time,
                                         status="Converged", label=label, tol=1e-8, rel_res_true=1e-10))
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

    def test_sell_storage_is_selectable(self):
        done = self.run_solve("--matrix", "poisson3d:10", "--format", "sell", "--chunk", "4", "--sigma", "8")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("format         sell (C = 4, sigma = 8, stored/nnz = ", done.stdout)
        csr = self.run_solve("--matrix", "poisson3d:10")
        self.assertIn("format         csr", csr.stdout)

    def test_sell_settings_need_the_sell_format(self):
        done = self.run_solve("--matrix", "poisson3d:10", "--chunk", "4")
        self.assertEqual(done.returncode, 1)
        self.assertIn("--chunk and --sigma need --format sell", done.stderr)
        done = self.run_solve("--matrix", "poisson3d:10", "--format", "ell")
        self.assertEqual(done.returncode, 1)
        self.assertIn("csr or sell", done.stderr)

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
        self.assertIn("none, jacobi, sgs, ic, ic-fixed or ic0", done.stderr)

    def test_gauss_seidel_and_incomplete_cholesky_are_selectable(self):
        done = self.run_solve("--matrix", "poisson2d:5", "--order", "amd")
        self.assertEqual(done.returncode, 1)
        self.assertIn("natural or rcm", done.stderr)

    def test_reordering_solves_the_same_system(self):
        with tempfile.TemporaryDirectory() as d:
            answers = []
            for order in ("natural", "rcm"):
                out = os.path.join(d, f"{order}.txt")
                done = self.run_solve("--matrix", "poisson3d:9", "--problem", "random", "--precond", "ic0",
                                      "--order", order, "--out", out)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ordering       {order}", done.stdout)
                with open(out, encoding="utf-8") as f:
                    answers.append([float(line) for line in f])
            # The same unknowns in the same places: a solution left in the new order would differ everywhere.
            self.assertEqual(len(answers[0]), 729)
            self.assertLess(max(abs(x - y) for x, y in zip(*answers)), 1e-6)
            self.assertGreater(max(answers[0]) - min(answers[0]), 0.1)

    def test_gauss_seidel_and_incomplete_cholesky_are_selectable(self):
        for precond in ("sgs", "ic", "ic-fixed", "ic0"):
            with self.subTest(precond=precond):
                done = self.run_solve("--matrix", "poisson3d:10", "--precond", precond)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"preconditioner {precond}", done.stdout)

    def test_invalid_options_are_rejected(self):
        for option, value in (("--tol", "inf"), ("--tol", "nan"), ("--tol", "-1"),
                              ("--maxit", "-1"), ("--maxit", "999999999999999999999999")):
            done = self.run_solve("--matrix", "poisson2d:5", option, value)
            self.assertEqual(done.returncode, 1, done.stdout + done.stderr)

    def test_supplied_rhs_and_its_dimension_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            rhs = pathlib.Path(d) / "b.txt"
            out = pathlib.Path(d) / "x.txt"
            rhs.write_text("8\n", encoding="ascii")
            done = self.run_solve("--matrix", "poisson2d:1", "--rhs", str(rhs), "--out", str(out))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertAlmostEqual(float(out.read_text()), 2.0)
            rhs.write_text("8\n9\n", encoding="ascii")
            self.assertEqual(self.run_solve("--matrix", "poisson2d:1", "--rhs", str(rhs)).returncode, 1)

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

    def test_history_records_the_residual_before_and_after_each_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.txt")
            done = self.run_solve("--matrix", "poisson2d:30", "--precond", "jacobi", "--history", path)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            spmv = int(next(l for l in done.stdout.splitlines() if l.startswith("spmv count")).split()[-1])
            with open(path, encoding="ascii") as f:
                values = [float(line) for line in f]
        self.assertEqual(len(values), spmv + 1)
        self.assertEqual(values[0], 1.0)
        self.assertLess(values[-1], 1e-8)

    def test_the_random_problem_takes_the_same_iterations_as_bench_cg(self):
        # The demo times bench_cg and plots solve's history, so both must solve the same system.
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.csv")
            bench = subprocess.run(
                [os.path.join(BUILD_DIR, "bench_cg"), "--matrix", "poisson2d:30", "--solver", "ours-jacobi",
                 "--repeat", "1", "--warmup", "0", "--problem", "random", "--csv", out],
                capture_output=True, text=True)
            self.assertEqual(bench.returncode, 0, bench.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                expected = next(csv.DictReader(f))["spmv_count"]
        done = self.run_solve("--matrix", "poisson2d:30", "--precond", "jacobi", "--problem", "random")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"spmv count     {expected}\n", done.stdout)

    def test_a_problem_and_an_rhs_file_together_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rhs = pathlib.Path(d) / "b.txt"
            rhs.write_text("1\n" * 25, encoding="ascii")
            done = self.run_solve("--matrix", "poisson2d:5", "--rhs", str(rhs), "--problem", "random")
        self.assertEqual(done.returncode, 1)
        self.assertIn("--problem", done.stderr)

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
            fields = ["matrix", "solver", "spmv_count", "per_spmv_s", "solve_s", "status", "tol", "rel_res_true"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for matrix, ours in (("good", 1.04), ("bad", 1.5)):
                    for _ in range(3):
                        writer.writerow(dict(matrix=matrix, solver="base", spmv_count=100, per_spmv_s=1.0,
                                             solve_s=1.0, status="Converged", tol=1e-8, rel_res_true=1e-10))
                        writer.writerow(dict(matrix=matrix, solver="ours", spmv_count=100, per_spmv_s=ours,
                                             solve_s=ours, status="Converged", tol=1e-8, rel_res_true=1e-10))
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "summarize.py"), path, "--gate", "base=ours",
                 "--time-ratio", "1.05"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 2, done.stderr)
            lines = done.stdout.splitlines()
            good = next(line for line in lines if line.startswith("| good"))
            bad = next(line for line in lines if line.startswith("| bad"))
            self.assertIn("PASS", good)
            self.assertIn("FAIL", bad)

    def test_nonconverged_runs_cannot_pass_and_incompatible_runs_are_rejected(self):
        def row(solver, status="MaxIterations", tol="1e-8"):
            return dict(matrix="m", solver=solver, spmv_count="1", per_spmv_s="1", solve_s="1",
                        status=status, tol=tol, rel_res_true="0.5")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "r.csv"
            for rows in ([row("base"), row("ours")],
                         [row("base"), row("ours", tol="1e-6")]):
                with path.open("w", newline="", encoding="ascii") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                done = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/summarize.py"), str(path), "--gate", "base=ours"],
                    capture_output=True, text=True)
                self.assertNotEqual(done.returncode, 0)
                self.assertNotIn("PASS", done.stdout)

    def test_a_verified_solve_with_a_different_iteration_count_fails(self):
        def row(solver, spmv):
            return dict(matrix="m", solver=solver, spmv_count=spmv, per_spmv_s="1", solve_s="1",
                        status="Converged", tol="1e-8", rel_res_true="1e-9")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "r.csv"
            rows = [row("base", 100), row("ours", 110)]
            with path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize.py"), str(path), "--gate", "base=ours"],
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertNotIn("VERIFIED", done.stdout)


class PlateauGate(unittest.TestCase):
    """The accuracy gate with the convergence histories bench_cg --solutions saved."""

    def gate(self, residual_at_base_stop, extra=(), mode="solve", with_solutions=True):
        def row(solver, spmv):
            return dict(matrix="m", solver=solver, spmv_count=spmv, per_spmv_s="1", solve_s="1",
                        status="Converged" if mode == "solve" else "MaxIterations", tol="1e-8",
                        rel_res_true="1e-9", mode=mode)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "r.csv"
            rows = [row("base", 3), row("ours", 6)]
            with path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            pictures = pathlib.Path(d) / "solutions"
            pictures.mkdir()
            history = [1.0, 1e-3, 1e-5, residual_at_base_stop, 2e-8, 1.5e-8, 9e-9]
            (pictures / "m.json").write_text(json.dumps(
                {"matrix": "m", "comparisons": [], "histories": [{"solver": "ours", "residuals": history}]}),
                encoding="utf-8")
            (pictures / "broken.json").write_text("{not json", encoding="utf-8")
            options = ["--solutions", str(pictures)] if with_solutions else []
            return subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize.py"), str(path), *options, "--gate", "base=ours", *extra],
                capture_output=True, text=True)

    def test_a_residual_at_the_line_when_the_base_stopped_passes_as_a_plateau(self):
        done = self.gate(1.03e-8)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("VERIFIED (plateau)", done.stdout)

    def test_a_residual_well_above_the_line_fails(self):
        done = self.gate(3e-8)
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertNotIn("VERIFIED", done.stdout)

    def test_without_the_histories_more_iterations_fail(self):
        done = self.gate(1.03e-8, with_solutions=False)
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)

    def test_the_speed_gate_ignores_histories(self):
        done = self.gate(1.03e-8, extra=("--metric", "iteration"), mode="fixed")
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertNotIn("plateau", done.stdout)


class SpeedGate(unittest.TestCase):
    """The gate real matrices can pass: time per iteration on fixed-iteration runs."""

    def gate(self, ours_time=1.0, ours_spmv=100, ours_status="MaxIterations", mode="fixed", extra=()):
        def row(solver, time, spmv, status):
            return dict(matrix="m", solver=solver, spmv_count=spmv, per_spmv_s=time, solve_s=time * spmv,
                        status=status, tol="1e-8", rel_res_true="0.5", mode=mode, precond="jacobi", repeat=0)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "r.csv"
            rows = [row("base", 1.0, 100, "MaxIterations"), row("ours", ours_time, ours_spmv, ours_status)]
            with path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize.py"), str(path), "--metric", "iteration",
                 "--gate", "base=ours", *extra],
                capture_output=True, text=True)

    def test_a_run_within_the_limit_passes(self):
        done = self.gate(ours_time=1.04)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASS", done.stdout)

    def test_a_run_over_the_limit_fails(self):
        done = self.gate(ours_time=1.5)
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("FAIL", done.stdout)

    def test_an_explicit_ratio_replaces_the_limit(self):
        self.assertEqual(self.gate(ours_time=1.5, extra=("--time-ratio", "2")).returncode, 0)

    def test_different_iteration_counts_fail(self):
        self.assertEqual(self.gate(ours_spmv=110).returncode, 2)

    def test_a_breakdown_fails_however_fast_it_was(self):
        self.assertEqual(self.gate(ours_status="Breakdown").returncode, 2)

    def test_complete_solves_are_not_accepted_as_speed_evidence(self):
        self.assertEqual(self.gate(mode="solve").returncode, 2)


if __name__ == "__main__":
    BUILD_DIR = sys.argv.pop(1)
    unittest.main()
