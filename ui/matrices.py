"""The Matrices tab: every suite matrix, what it models, and a drawing of its non-zero pattern.

matrix_preview draws each matrix once into <build>/ui-drawings/NAME.json. A drawing of a downloaded
file is redrawn after the file changes. Drawing runs as a job, so it never overlaps a benchmark.
"""
import json
import re

import steps

# What each matrix models: SuiteSparse's problem kind for the real ones (each links to its page there).
ABOUT = {
    "ecology2": "landscape ecology, modelled as an electrical circuit on a 2-D grid",
    "apache2": "3-D structural problem",
    "tmt_sym": "electromagnetics problem",
    "G3_circuit": "circuit simulation problem",
    "thermal2": "steady-state thermal problem on an unstructured mesh",
    "hood": "structural problem",
    "inline_1": "structural problem",
    "Emilia_923": "structural problem from geomechanics",
    "ldoor": "structural problem",
    "audikw_1": "structural problem",
}
STENCILS = {
    "poisson2d": "2-D Poisson equation, 5-point stencil: each unknown is tied to 4 neighbours",
    "poisson3d": "3-D Poisson equation, 7-point stencil: 6 neighbours",
    "point27": "3-D 27-point stencil as in the HPCG benchmark: 26 neighbours",
}
SIZES = {"0.5": "sized to half the last-level cache, so it stays in cache",
         "4": "sized to 4 times the last-level cache, so every iteration streams it from memory"}
GRID, ZOOM = "160", "40"


def about(name):
    if name in ABOUT:
        return ABOUT[name]
    stencil, _, size = name.partition(":llc=")
    if stencil in STENCILS:
        return STENCILS[stencil] + (f", {SIZES[size]}" if size in SIZES else "")
    return ""


def drawings_dir(settings):
    return steps.resolve(settings["build"]) / "ui-drawings"


def drawing_path(settings, name):
    return drawings_dir(settings) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name)}.json"


def suite_groups():
    with open(steps.SCRIPTS / "suite.json", encoding="utf-8") as f:
        return {entry["name"]: entry for entry in json.load(f).get("files", [])}


def current_drawing(settings, choice):
    """The cached drawing, or None if there is none or the matrix file is newer than it."""
    path = drawing_path(settings, choice["name"])
    try:
        if not choice["generated"] and path.stat().st_mtime < steps.resolve(choice["spec"]).stat().st_mtime:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def listing(settings):
    files = suite_groups()
    out = []
    for choice in steps.single_choices(settings):
        entry = files.get(choice["name"], {})
        group = entry.get("group")
        out.append({
            "name": choice["name"], "generated": choice["generated"], "available": choice["available"],
            "tests": choice["tests"], "about": about(choice["name"]), "group": group,
            "page": f"https://sparse.tamu.edu/{group}/{choice['name']}" if group else None,
            "drawing": current_drawing(settings, choice) if choice["available"] else None,
        })
    return out


def plan(settings, names=None):
    """One step per matrix to draw: the named ones, or every available one without a current drawing."""
    release = steps.paths(settings)["release"]
    chosen = []
    for choice in steps.single_choices(settings):
        if not choice["available"]:
            continue
        if names is None and current_drawing(settings, choice) is not None:
            continue
        if names is not None and choice["name"] not in names:
            continue
        chosen.append({"id": choice["name"], "title": f"Draw {choice['name']}", "commands": [steps.command([
            release / "matrix_preview", "--matrix", choice["spec"], "--out", drawing_path(settings, choice["name"]),
            "--grid", GRID, "--zoom", ZOOM])]})
    return chosen
