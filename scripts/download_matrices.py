#!/usr/bin/env python3
"""Downloads the benchmark matrices from the SuiteSparse Matrix Collection.

Each matrix ends up as DATA_DIR/NAME.mtx. If an automated download is refused, this
prints a FAILED line with the page to download the file from by browser.

Example: python3 scripts/download_matrices.py --data-dir ~/p08/data --only thermal2,G3_circuit
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def declared_rows(path):
    """Reads the row count from a Matrix Market size line."""
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            return int(line.split()[0])
    return None


def fetch(entry, data_dir):
    name, group = entry["name"], entry["group"]
    target = os.path.join(data_dir, name + ".mtx")
    url = BASE_URL.format(group=group, name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "p08-matrix-download/1.0"})
    with tempfile.TemporaryDirectory(dir=data_dir) as work:
        archive = os.path.join(work, name + ".tar.gz")
        with urllib.request.urlopen(request, timeout=300) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        # Copy the one expected member by name, so paths inside the archive are never used.
        member_name = f"{name}/{name}.mtx"
        extracted = os.path.join(work, name + ".mtx")
        with tarfile.open(archive, "r:gz") as tar:
            source = tar.extractfile(member_name)
            if source is None:
                raise ValueError(f"{member_name} is not a regular file in the archive")
            with source, open(extracted, "wb") as out:
                shutil.copyfileobj(source, out)
        rows = declared_rows(extracted)
        if rows != entry["rows"]:
            raise ValueError(f"size line says {rows} rows, expected {entry['rows']}")
        os.replace(extracted, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--suite", default=str(ROOT / "scripts" / "suite.json"))
    parser.add_argument("--only", default=None, help="comma-separated matrix names")
    args = parser.parse_args()

    with open(args.suite, encoding="utf-8") as f:
        entries = json.load(f).get("files", [])
    wanted = set(args.only.split(",")) if args.only else None
    os.makedirs(args.data_dir, exist_ok=True)

    failures = 0
    for entry in entries:
        name = entry["name"]
        if wanted is not None and name not in wanted:
            continue
        if os.path.exists(os.path.join(args.data_dir, name + ".mtx")):
            print(f"OK {name}: already downloaded")
            continue
        try:
            fetch(entry, args.data_dir)
            print(f"OK {name}", flush=True)
        except (urllib.error.URLError, OSError, ValueError, tarfile.TarError, KeyError) as error:
            failures += 1
            print(
                f"FAILED {name}: {error}\n"
                f"  Download by browser from https://sparse.tamu.edu/{entry['group']}/{name} (Matrix Market format),\n"
                f"  extract it, and save the file as {os.path.join(args.data_dir, name + '.mtx')}",
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
