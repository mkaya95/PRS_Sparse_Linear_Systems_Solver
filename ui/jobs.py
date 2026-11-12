"""Runs one job at a time: a list of steps, each a list of commands, with output kept for the page.

A step fails at the first command that exits non-zero, except that summarize.py exiting 2 only
means its gate failed: the step is marked that way and the job goes on to the next step.
"""
import os
import signal
import subprocess
import threading
import time

from steps import ROOT, display

MAX_LINES = 20000


class Busy(RuntimeError):
    pass


class Job:
    def __init__(self, kind, steps, finish=None, before=None, cleanup=None):
        """steps: [{"id", "title", "commands": [{"argv", "gate"}]}]. finish(job) may set job.result.
        before(job) runs first, inside the job, for preparation such as removing a replaced file.
        cleanup(job) runs last, however the job ended."""
        self.id = f"{int(time.time() * 1000)}"
        self.kind = kind
        self.steps = [dict(id=s["id"], title=s["title"], commands=s["commands"], state="waiting",
                           started=None, finished=None) for s in steps]
        self.finish = finish
        self.before = before
        self.cleanup = cleanup
        self.state = "running"
        self.started = time.time()
        self.finished = None
        self.result = None
        self.error = None
        self.lines = []
        self.dropped = 0
        self.stop_requested = False
        self.process = None
        self.lock = threading.Lock()

    def say(self, text):
        with self.lock:
            for line in text.splitlines() or [""]:
                self.lines.append(line)
            if len(self.lines) > MAX_LINES:
                extra = len(self.lines) - MAX_LINES
                del self.lines[:extra]
                self.dropped += extra

    def snapshot(self, offset=0):
        with self.lock:
            start = max(offset - self.dropped, 0)
            lines = self.lines[start:]
            now = time.time()
            return {
                "id": self.id, "kind": self.kind, "state": self.state, "error": self.error,
                "elapsed": (self.finished or now) - self.started,
                "steps": [{"id": s["id"], "title": s["title"], "state": s["state"],
                           "elapsed": None if s["started"] is None else (s["finished"] or now) - s["started"]}
                          for s in self.steps],
                "lines": lines, "offset": self.dropped + len(self.lines), "result": self.result,
            }


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = None

    def busy(self):
        return self.job is not None and self.job.state == "running"

    def start(self, job):
        with self.lock:
            if self.busy():
                raise Busy("another run is still going; stop it first")
            self.job = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def stop(self):
        job = self.job
        if job is None or job.state != "running":
            return False
        job.stop_requested = True
        process = job.process
        if process is not None and process.poll() is None:
            kill_tree(process, signal.SIGTERM)
            threading.Thread(target=_kill_later, args=(process,), daemon=True).start()
        return True

    def _run(self, job):
        try:
            if job.before is not None:
                job.before(job)
            for step in job.steps:
                if job.stop_requested:
                    break
                step["state"], step["started"] = "running", time.time()
                step["state"] = self._run_step(job, step)
                step["finished"] = time.time()
                if step["state"] in ("failed", "stopped"):
                    break
            if job.stop_requested:
                job.state = "stopped"
                job.say("Stopped.")
            elif any(s["state"] == "failed" for s in job.steps):
                job.state = "failed"
            else:
                if job.finish is not None:
                    job.finish(job)
                job.state = "gate-failed" if any(s["state"] == "gate-failed" for s in job.steps) else "done"
        except Exception as error:  # a broken job must still end, and say why
            job.error = str(error)
            job.say(f"ERROR: {error}")
            job.state = "failed"
        finally:
            for step in job.steps:
                if step["state"] == "running":
                    step["state"], step["finished"] = ("stopped" if job.stop_requested else "failed"), time.time()
            if job.cleanup is not None:
                try:
                    job.cleanup(job)
                except Exception as error:
                    job.say(f"cleanup failed: {error}")
            job.finished = time.time()

    def _run_step(self, job, step):
        gate_failed = False
        for cmd in step["commands"]:
            if job.stop_requested:
                return "stopped"
            job.say(f"$ {display(cmd['argv'])}")
            code = self._run_command(job, cmd["argv"])
            if job.stop_requested:
                return "stopped"
            if code == 0:
                continue
            if cmd.get("gate") and code == 2:
                gate_failed = True
                continue
            job.say(f"exit status {code}")
            return "failed"
        return "gate-failed" if gate_failed else "done"

    def _run_command(self, job, argv):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, text=True, errors="replace", bufsize=1,
                                       env=env, start_new_session=True)
        except OSError as error:
            job.say(f"cannot start {argv[0]}: {error.strerror or error}")
            return 127
        job.process = process
        with process.stdout:
            for line in process.stdout:
                job.say(line.rstrip("\n"))
        process.wait()
        job.process = None
        return process.returncode


def kill_tree(process, sig):
    """Signals the command and everything it started, which share its session."""
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _kill_later(process):
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        kill_tree(process, signal.SIGKILL)
