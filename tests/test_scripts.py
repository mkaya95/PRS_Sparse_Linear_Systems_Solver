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
