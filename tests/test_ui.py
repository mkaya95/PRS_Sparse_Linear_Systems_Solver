"""Tests for the browser UI's server side. Usage: test_ui.py BUILD_DIR"""
import csv
import http.client
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))

import demo  # noqa: E402
import jobs  # noqa: E402
import results  # noqa: E402
import server  # noqa: E402
import steps  # noqa: E402

BUILD_DIR = None  # set in __main__


def settings_in(folder, machine="test-box"):
    return {"machine": machine, "build": str(folder), "data": str(folder / "data"), "compiler": "g++"}


def py(code, gate=False):
    return steps.command([sys.executable, "-c", code], gate=gate)


def run(job, timeout=60):
    jobs.Runner().start(job)
    deadline = time.time() + timeout
    while job.state == "running" and time.time() < deadline:
        time.sleep(0.05)
    while job.finished is None and time.time() < deadline:
        time.sleep(0.01)
    return job


class Steps(unittest.TestCase):
    def test_the_steps_are_the_readme_commands_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            every = steps.steps(settings_in(pathlib.Path(d)))
        self.assertEqual([s["id"] for s in every], ["build", "test", "download", "speed", "accuracy", "scipy"])
        speed, gate = [c["argv"] for c in every[3]["commands"]]
        self.assertEqual(speed[speed.index("--mode"):speed.index("--out")],
                         ["--mode", "fixed", "--maxit", "100", "--repeat", "11",
                          "--solvers", "eigen-E0,ours,eigen-E1,ours-jacobi"])
        self.assertEqual(speed[-1], str(ROOT / "results" / "test-box" / "g1-real.csv"))
        self.assertEqual(gate[-4:], ["--gate", "eigen-E0=ours", "--gate", "eigen-E1=ours-jacobi"])
        self.assertTrue(every[3]["commands"][1]["gate"])
        accuracy = every[4]["commands"][0]["argv"]
        self.assertIn("ecology2,apache2,tmt_sym,G3_circuit,thermal2,hood", accuracy)
        self.assertEqual(every[4]["replaces"], [ROOT / "results" / "test-box" / "g1-real-solve.csv"])

    def test_the_compiler_is_named_only_when_the_release_folder_is_new(self):
        # Naming it again makes CMake wipe the cache and drop the preset's settings.
        with tempfile.TemporaryDirectory() as d:
            folder = pathlib.Path(d)
            first = steps.steps(settings_in(folder))
            (folder / "release").mkdir()
            (folder / "release" / "CMakeCache.txt").write_text("", encoding="ascii")
            again = steps.steps(settings_in(folder))
        self.assertIn("-DCMAKE_CXX_COMPILER=g++", first[0]["commands"][0]["argv"])
        self.assertFalse(any("COMPILER" in a for a in again[0]["commands"][0]["argv"]))
        self.assertFalse(any("COMPILER" in a for a in first[1]["commands"][0]["argv"]))

    def test_settings_are_checked(self):
        good = settings_in(pathlib.Path("/tmp"))
        self.assertEqual(steps.validate(good)["machine"], "test-box")
        for machine in ("../up", "two words", "", "x;rm", "-dash"):
            with self.subTest(machine=machine), self.assertRaises(steps.SettingsError):
                steps.validate(dict(good, machine=machine))
        with self.assertRaises(steps.SettingsError):
            steps.validate(dict(good, compiler="g++; rm -rf ~"))
        with self.assertRaises(steps.SettingsError):
            steps.validate(dict(good, data="a\nb"))

    def test_commands_are_shown_as_a_person_would_type_them(self):
        argv = [steps.PYTHON, str(ROOT / "scripts" / "summarize.py"), os.path.expanduser("~/p08/x.csv"), "a b"]
        self.assertEqual(steps.display(argv), "python3 scripts/summarize.py ~/p08/x.csv 'a b'")

    def test_the_suggested_machine_name_is_a_valid_one(self):
        machine = steps.suggested_settings()["machine"]
        self.assertTrue(machine == "" or steps.MACHINE_PATTERN.match(machine), machine)


class Runner(unittest.TestCase):
    def test_a_failing_command_stops_the_job(self):
        job = run(jobs.Job("run", [{"id": "a", "title": "A", "commands": [py("raise SystemExit(3)")]},
                                   {"id": "b", "title": "B", "commands": [py("print('never')")]}]))
        self.assertEqual(job.state, "failed")
        self.assertEqual([s["state"] for s in job.steps], ["failed", "waiting"])
        self.assertIn("exit status 3", job.lines)
        self.assertNotIn("never", job.lines)

    def test_a_failed_gate_is_marked_and_the_next_step_still_runs(self):
        job = run(jobs.Job("run", [{"id": "a", "title": "A", "commands": [py("raise SystemExit(2)", gate=True)]},
                                   {"id": "b", "title": "B", "commands": [py("print('ran')")]}]))
        self.assertEqual([s["state"] for s in job.steps], ["gate-failed", "done"])
        self.assertEqual(job.state, "gate-failed")
        self.assertIn("ran", job.lines)

    def test_stop_ends_the_command_and_what_it_started(self):
        runner = jobs.Runner()
        code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
        job = runner.start(jobs.Job("run", [{"id": "a", "title": "A", "commands": [py(code)]}]))
        time.sleep(0.5)
        with self.assertRaises(jobs.Busy):
            runner.start(jobs.Job("run", [{"id": "b", "title": "B", "commands": [py("pass")]}]))
        started = time.time()
        self.assertTrue(runner.stop())
        while job.finished is None and time.time() - started < 15:
            time.sleep(0.05)
        self.assertEqual(job.state, "stopped")
        self.assertLess(time.time() - started, 10)

    def test_preparation_and_cleanup_run_around_the_steps(self):
        order = []
        job = run(jobs.Job("run", [{"id": "a", "title": "A", "commands": [py("print('step')")]}],
                           before=lambda j: order.append("before"), finish=lambda j: order.append("finish"),
                           cleanup=lambda j: order.append("cleanup")))
        self.assertEqual(job.state, "done")
        self.assertEqual(order, ["before", "finish", "cleanup"])

    def test_output_is_read_from_an_offset(self):
        job = run(jobs.Job("run", [{"id": "a", "title": "A", "commands": [py("print('one'); print('two')")]}]))
        everything = job.snapshot(0)
        later = job.snapshot(everything["offset"] - 1)
        self.assertEqual(later["lines"], ["two"])


class Results(unittest.TestCase):
    def write_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_the_committed_results_match_the_results_document(self):
        folder = ROOT / "results" / "i5-1335U-wsl2"
        if not folder.is_dir():
            self.skipTest("no committed i5-1335U results")
        machine = results.machine(folder)
        self.assertTrue(machine["speed"]["passed"])
        self.assertTrue(machine["accuracy"]["passed"])
        plain = next(l for l in machine["ladder"]["ladders"] if l["base"] == "eigen-E0")["points"]
        self.assertEqual(round(plain[0]["ratio"], 3), 0.971)
        self.assertEqual(round(plain[-1]["ratio"], 3), 0.781)
        g3 = next(r for r in machine["speed"]["pairs"][0]["rows"] if r["matrix"] == "G3_circuit")
        self.assertEqual(round(g3["ratio"], 3), 0.649)

    def test_missing_files_are_not_run_and_rejected_files_say_why(self):
        with tempfile.TemporaryDirectory() as d:
            folder = pathlib.Path(d) / "box"
            folder.mkdir()
            row = dict(matrix="m", solver="eigen-E1", spmv_count=10, per_spmv_s=1, solve_s=1, status="Converged",
                       tol="1e-8", rel_res_true="1e-9")
            self.write_csv(folder / steps.ACCURACY_CSV, [row, dict(row, solver="ours-jacobi", tol="1e-6")])
            machine = results.machine(folder)
        self.assertIsNone(machine["speed"])
        self.assertIn("incompatible", machine["accuracy"]["error"])

    def test_a_failing_matrix_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            folder = pathlib.Path(d) / "box"
            folder.mkdir()
            rows = []
            for base, ours in (("eigen-E0", "ours"), ("eigen-E1", "ours-jacobi")):
                for solver, time_s in ((base, 1.0), (ours, 1.5)):
                    rows.append(dict(matrix="m", solver=solver, spmv_count=100, per_spmv_s=time_s, solve_s=time_s,
                                     status="MaxIterations", tol="1e-8", rel_res_true="0.5", mode="fixed", repeat=0))
            self.write_csv(folder / steps.SPEED_CSV, rows)
            speed = results.machine(folder)["speed"]
        self.assertFalse(speed["passed"])
        self.assertEqual(speed["pairs"][0]["rows"][0]["ratio"], 1.5)


class Demo(unittest.TestCase):
    def test_a_demo_compares_both_solvers_on_one_problem_and_records_the_curve(self):
        plan_steps, finish, cleanup = demo.plan(pathlib.Path(BUILD_DIR), "poisson2d:30", "jacobi")
        work = pathlib.Path(plan_steps[0]["commands"][0]["argv"][-1]).parent
        job = run(jobs.Job("demo", plan_steps, finish=finish, cleanup=cleanup))
        self.assertEqual(job.state, "done", "\n".join(job.lines))
        r = job.result
        self.assertTrue(r["same_answer"])
        self.assertEqual(r["ours"]["spmv"], r["eigen"]["spmv"])
        self.assertEqual(r["history"][0], [0, 1.0])
        self.assertEqual(r["history"][-1][0], r["ours"]["spmv"])
        self.assertLess(r["history"][-1][1], 1e-8)
        self.assertGreater(r["speedup"], 0)
        self.assertEqual((r["n"], r["ours"]["solver"], r["eigen"]["solver"]), (900, "ours-jacobi", "eigen-E1"))
        self.assertFalse(work.exists())

    def test_a_long_curve_is_thinned_but_keeps_both_ends(self):
        points = demo.thin(list(range(10000)), 100)
        self.assertLessEqual(len(points), 100)
        self.assertEqual((points[0], points[-1]), ((0, 0), (9999, 9999)))

    def test_choices_include_downloaded_suite_matrices_only(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "hood.mtx").write_text("", encoding="ascii")
            specs = [c["spec"] for c in demo.choices(d)]
        self.assertIn("poisson3d:llc=4", specs)
        self.assertIn(os.path.join(d, "hood.mtx"), specs)
        self.assertFalse(any("ecology2" in s for s in specs))


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = server.make_server(0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        all_headers = {"Host": f"localhost:{self.port}"}
        if body is not None:
            all_headers.update({"Content-Type": "application/json", "X-P08": "1"})
        all_headers.update(headers or {})
        connection.request(method, path, None if body is None else json.dumps(body), all_headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def test_changes_need_the_header_the_page_sends(self):
        with tempfile.TemporaryDirectory() as d:
            body = {"settings": settings_in(pathlib.Path(d))}
            status, _ = self.request("POST", "/api/preview", body, {"X-P08": "0"})
            self.assertEqual(status, 403)
            status, data = self.request("POST", "/api/preview", body)
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(data)["steps"]), 6)

    def test_other_host_names_are_refused(self):
        status, _ = self.request("GET", "/api/state", headers={"Host": f"attacker.example:{self.port}"})
        self.assertEqual(status, 403)

    def test_existing_results_are_only_replaced_when_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            original = steps.ROOT
            steps.ROOT = pathlib.Path(d)
            self.addCleanup(setattr, steps, "ROOT", original)
            target = pathlib.Path(d) / "results" / "test-box" / steps.SPEED_CSV
            target.parent.mkdir(parents=True)
            target.write_text("kept\n", encoding="ascii")
            status, data = self.request("POST", "/api/run", {"settings": settings_in(pathlib.Path(d)), "steps": ["speed"]})
            self.assertEqual(status, 409)
            self.assertEqual(len(json.loads(data)["existing"]), 1)
            self.assertEqual(target.read_text(encoding="ascii"), "kept\n")

    def test_unknown_steps_and_bad_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            status, _ = self.request("POST", "/api/run", {"settings": settings_in(pathlib.Path(d)), "steps": ["rm"]})
            self.assertEqual(status, 400)
            status, data = self.request("POST", "/api/run", {"settings": settings_in(pathlib.Path(d), "../x"),
                                                             "steps": ["build"]})
        self.assertEqual(status, 400)
        self.assertIn("machine name", json.loads(data)["error"])

    def test_files_outside_the_static_folder_are_not_served(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/charts.js")[0], 200)
        for path in ("/../server.py", "/%2e%2e/server.py", "/fonts/../../server.py"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_results_are_served_as_json(self):
        status, data = self.request("GET", "/api/results")
        self.assertEqual(status, 200)
        self.assertIsInstance(json.loads(data)["machines"], list)


if __name__ == "__main__":
    BUILD_DIR = sys.argv.pop(1)
    unittest.main()
