"""The README's run steps as commands, built from the settings on the Run tab.

Every command is an argument list, never a shell string, and runs from the repository root.
"""
import os
import pathlib
import platform
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PYTHON = sys.executable or "python3"

ACCURACY_MATRICES = "ecology2,apache2,tmt_sym,G3_circuit,thermal2,hood"
SPEED_CSV = "g1-real.csv"
ACCURACY_CSV = "g1-real-solve.csv"

MACHINE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COMPILER_PATTERN = re.compile(r"^[A-Za-z0-9._+/-]{1,200}$")


class SettingsError(ValueError):
    pass


def cpu_name():
    """A short CPU name such as i5-1335U, or an empty string."""
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if not model and platform.system() == "Darwin":
        try:
            import subprocess
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=5).stdout.strip()
        except (OSError, ValueError):
            model = ""
    words = [w for w in re.split(r"\s+", model.replace("(R)", "").replace("(TM)", "")) if w]
    for word in reversed(words):
        if any(c.isdigit() for c in word) and not word.endswith("Hz") and word.lower() != "gen":
            return re.sub(r"[^A-Za-z0-9._-]", "", word)
    return re.sub(r"[^A-Za-z0-9._-]", "-", "-".join(words[-2:])) if words else ""


def system_suffix():
    if platform.system() == "Darwin":
        return "macos"
    if "microsoft" in platform.release().lower():
        return "wsl2"
    return platform.system().lower() or "unknown"


def suggested_settings():
    """Defaults for a first visit. On WSL2 with the repository on a Windows drive, builds go to
    the Linux filesystem, as the README recommends, because /mnt/c is slow."""
    on_windows_drive = str(ROOT).startswith("/mnt/")
    cpu = cpu_name()
    return {
        "machine": f"{cpu}-{system_suffix()}" if cpu else "",
        "build": "~/p08/build" if on_windows_drive else "build",
        "data": "~/p08/data",
        "compiler": "g++",
    }


def resolve(path_text):
    """Expands ~ and makes a relative path relative to the repository root."""
    path = pathlib.Path(os.path.expanduser(path_text))
    return path if path.is_absolute() else ROOT / path


def validate(settings):
    """Returns clean settings or raises SettingsError naming the field."""
    if not isinstance(settings, dict):
        raise SettingsError("settings are missing")
    clean = {}
    for key in ("machine", "build", "data", "compiler"):
        value = settings.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise SettingsError(f"{key} is empty")
        value = value.strip()
        if "\x00" in value or "\n" in value:
            raise SettingsError(f"{key} contains a control character")
        clean[key] = value
    if not MACHINE_PATTERN.match(clean["machine"]):
        raise SettingsError("machine name may use letters, digits, '.', '_' and '-' only")
    if not COMPILER_PATTERN.match(clean["compiler"]):
        raise SettingsError("compiler must be a program name or path, such as g++ or g++-14")
    return clean


def paths(settings):
    build = resolve(settings["build"])
    return {
        "release": build / "release",
        "debug": build / "debug",
        "data": resolve(settings["data"]),
        "results": ROOT / "results" / settings["machine"],
        "last_demo": build / "ui-last-demo.json",
    }


def command(argv, gate=False):
    """gate=True marks summarize.py: exit 2 means the gate failed, not that the step broke."""
    return {"argv": [str(a) for a in argv], "gate": gate}


def steps(settings):
    """The README's steps in order. Each has commands and the result files it would replace."""
    p = paths(settings)
    release, debug, data, results = p["release"], p["debug"], p["data"], p["results"]
    # The compiler is chosen once, when a build folder is first configured. Naming it again, even as
    # g++ for a cached /usr/bin/g++, makes CMake wipe the cache and drop the preset's settings.
    # The debug build uses the default compiler, as in the README.
    configure_release = ["cmake", "--preset", "release", "-B", release]
    if not (release / "CMakeCache.txt").exists():
        configure_release.append(f"-DCMAKE_CXX_COMPILER={settings['compiler']}")
    speed_csv, accuracy_csv = results / SPEED_CSV, results / ACCURACY_CSV
    return [
        {"id": "build", "title": "Build", "about": "release build of the solver and benchmarks", "minutes": 2,
         "commands": [command(configure_release),
                      command(["cmake", "--build", release])],
         "replaces": []},
        {"id": "test", "title": "Test", "about": "debug build with sanitizers, then every test", "minutes": 3,
         "commands": [command(["cmake", "--preset", "debug", "-B", debug]),
                      command(["cmake", "--build", debug]),
                      command(["ctest", "--test-dir", debug, "--output-on-failure"])],
         "replaces": []},
        {"id": "download", "title": "Download matrices", "about": "the ten SuiteSparse matrices", "minutes": 5,
         "commands": [command([PYTHON, SCRIPTS / "download_matrices.py", "--data-dir", data])],
         "replaces": []},
        {"id": "speed", "title": "Speed", "about": "100 iterations per solver, at most 1.05 times Eigen", "minutes": 20,
         "commands": [command([PYTHON, SCRIPTS / "run_bench.py", "--build", release, "--data-dir", data,
                               "--mode", "fixed", "--maxit", "100", "--repeat", "11",
                               "--solvers", "eigen-E0,ours,eigen-E1,ours-jacobi", "--out", speed_csv]),
                      command([PYTHON, SCRIPTS / "summarize.py", speed_csv, "--metric", "iteration",
                               "--gate", "eigen-E0=ours", "--gate", "eigen-E1=ours-jacobi"], gate=True)],
         "replaces": [speed_csv]},
        {"id": "accuracy", "title": "Accuracy", "about": "complete solves to 1e-8 on the six smaller matrices",
         "minutes": 15,
         "commands": [command([PYTHON, SCRIPTS / "run_bench.py", "--build", release, "--data-dir", data,
                               "--maxit", "30000", "--repeat", "3", "--warmup", "0",
                               "--matrices", ACCURACY_MATRICES, "--solvers", "eigen-E1,ours-jacobi",
                               "--out", accuracy_csv]),
                      command([PYTHON, SCRIPTS / "summarize.py", accuracy_csv, "--gate", "eigen-E1=ours-jacobi"],
                              gate=True)],
         "replaces": [accuracy_csv]},
        {"id": "scipy", "title": "SciPy check", "about": "our answer on apache2, judged by SciPy", "minutes": 1,
         "commands": [command([PYTHON, SCRIPTS / "check_against_scipy.py", "--build", release,
                               "--matrix", data / "apache2.mtx"])],
         "replaces": []},
    ]


def display(argv):
    """A command as a person would type it: repository paths relative, the home folder as ~."""
    home = os.path.expanduser("~")
    shown = []
    for i, arg in enumerate(argv):
        if i == 0 and arg == PYTHON:
            arg = "python3"
        elif arg.startswith(str(ROOT) + os.sep):
            arg = os.path.relpath(arg, ROOT)
        elif home not in ("", "/") and (arg == home or arg.startswith(home + os.sep)):
            arg = "~" + arg[len(home):]
        shown.append(arg if arg.startswith("~") and " " not in arg else shlex.quote(arg))
    return " ".join(shown)
