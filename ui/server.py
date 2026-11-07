#!/usr/bin/env python3
"""Browser front end for the P08 solver: run the README's steps, read the results, run the live demo.

Run it where the solver builds (Linux, WSL2 or macOS), then open the printed address:
  python3 ui/server.py [--port 8008]

It listens on this computer only, because it runs build and benchmark commands. Requests that
change anything must carry the X-P08 header the page sends, so other websites cannot trigger them.
"""
import argparse
import json
import mimetypes
import pathlib
import sys
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import demo  # noqa: E402
import results  # noqa: E402
import steps  # noqa: E402
from jobs import Busy, Job, Runner  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent / "static"
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
RUNNER = Runner()


class ClientError(ValueError):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST, extra=None):
        super().__init__(message)
        self.status = status
        self.extra = extra or {}


def settings_from(data):
    try:
        return steps.validate(data.get("settings"))
    except steps.SettingsError as error:
        raise ClientError(str(error)) from None


def preview(data):
    settings = settings_from(data)
    out = []
    for step in steps.steps(settings):
        out.append({"id": step["id"], "title": step["title"], "about": step["about"], "minutes": step["minutes"],
                    "commands": [steps.display(c["argv"]) for c in step["commands"]],
                    "existing": [steps.display([str(p)]) for p in step["replaces"] if p.exists()]})
    return {"steps": out}


def start_run(data):
    settings = settings_from(data)
    wanted = data.get("steps")
    every = steps.steps(settings)
    ids = [s["id"] for s in every]
    if not isinstance(wanted, list) or not wanted or any(w not in ids for w in wanted):
        raise ClientError("unknown step")
    chosen = [s for s in every if s["id"] in wanted]
    existing = [p for s in chosen for p in s["replaces"] if p.exists()]
    if existing and not data.get("replace"):
        raise ClientError("these results already exist", HTTPStatus.CONFLICT,
                          {"existing": [steps.display([str(p)]) for p in existing]})

    def before(job):
        for path in existing:
            job.say(f"Removing the earlier result {steps.display([str(path)])}, as confirmed")
            path.unlink()

    return RUNNER.start(Job("run", chosen, before=before if existing else None))


def start_demo(data):
    settings = settings_from(data)
    paths = steps.paths(settings)
    options = {c["spec"]: c for c in demo.choices(paths["data"])}
    spec, precond = data.get("matrix"), data.get("precond")
    if spec not in options:
        raise ClientError("pick a matrix from the list")
    if precond not in demo.PAIRS:
        raise ClientError("preconditioner must be jacobi or none")
    for binary in ("bench_cg", "solve"):
        if not (paths["release"] / binary).exists():
            raise ClientError(f"{binary} is not built in {steps.display([str(paths['release'])])}. "
                              "Run the Build step first.")
    plan_steps, finish, cleanup = demo.plan(paths["release"], spec, precond)

    def finish_and_record(job):
        finish(job)
        job.result["label"] = options[spec]["label"]
        job.result["when"] = time.strftime("%Y-%m-%d %H:%M")
        try:
            paths["last_demo"].write_text(json.dumps(job.result), encoding="utf-8")
        except OSError as error:
            job.say(f"Could not save this run as the fallback: {error}")

    return RUNNER.start(Job("demo", plan_steps, finish=finish_and_record, cleanup=cleanup))


def last_demo(query):
    settings = settings_from({"settings": {k: query.get(k, [""])[0] for k in ("machine", "build", "data", "compiler")}})
    path = steps.paths(settings)["last_demo"]
    try:
        return {"last": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"last": None}


def demo_choices(query):
    data_dir = query.get("data", ["~/p08/data"])[0]
    return {"choices": demo.choices(steps.resolve(data_dir))}


class Handler(BaseHTTPRequestHandler):
    server_version = "p08-ui"

    def log_message(self, fmt, *args):
        pass  # the terminal shows the address, not every poll

    def host_ok(self):
        """Refuses names other than localhost, so a site that rebinds its DNS to 127.0.0.1 gets nothing."""
        host = self.headers.get("Host") or ""
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return name in LOOPBACK

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status, extra=None):
        self.send_json({"error": message, **(extra or {})}, status)

    def do_GET(self):
        if not self.host_ok():
            return self.send_error_json("this server only answers on localhost", HTTPStatus.FORBIDDEN)
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        try:
            if url.path == "/api/state":
                job = RUNNER.job
                return self.send_json({"suggested": steps.suggested_settings(), "busy": RUNNER.busy(),
                                       "job": job.snapshot() if job else None})
            if url.path == "/api/job":
                job = RUNNER.job
                offset = int(query.get("offset", ["0"])[0])
                return self.send_json({"job": job.snapshot(offset) if job else None})
            if url.path == "/api/results":
                return self.send_json({"machines": results.all_machines()})
            if url.path == "/api/demo/choices":
                return self.send_json(demo_choices(query))
            if url.path == "/api/demo/last":
                return self.send_json(last_demo(query))
        except ClientError as error:
            return self.send_error_json(str(error), error.status, error.extra)
        except ValueError as error:
            return self.send_error_json(str(error), HTTPStatus.BAD_REQUEST)
        self.send_static(url.path)

    def do_POST(self):
        if not self.host_ok() or self.headers.get("X-P08") != "1":
            return self.send_error_json("request refused", HTTPStatus.FORBIDDEN)
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send_error_json("expected JSON", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(min(length, 1_000_000)) or b"{}")
            if not isinstance(data, dict):
                raise ClientError("expected a JSON object")
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/preview":
                return self.send_json(preview(data))
            if path == "/api/run":
                job = start_run(data)
                return self.send_json({"job": job.snapshot()})
            if path == "/api/demo":
                job = start_demo(data)
                return self.send_json({"job": job.snapshot()})
            if path == "/api/stop":
                return self.send_json({"stopping": RUNNER.stop()})
            return self.send_error_json("no such endpoint", HTTPStatus.NOT_FOUND)
        except Busy as error:
            return self.send_error_json(str(error), HTTPStatus.CONFLICT)
        except ClientError as error:
            return self.send_error_json(str(error), error.status, error.extra)
        except ValueError as error:
            return self.send_error_json(f"bad request: {error}", HTTPStatus.BAD_REQUEST)

    def send_static(self, path):
        relative = "index.html" if path in ("", "/") else urllib.parse.unquote(path).lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        body = target.read_bytes()
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".woff2":
            kind = "font/woff2"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith("text/") or kind.endswith("javascript") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def make_server(port):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except OSError as error:
        print(f"Cannot listen on port {args.port}: {error.strerror}. Try --port {args.port + 1}.")
        return 1
    print(f"P08 UI running. Open http://localhost:{server.server_address[1]}/ in your browser. Ctrl+C stops it.",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        RUNNER.stop()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
