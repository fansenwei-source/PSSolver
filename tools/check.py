"""Changed-file lint, scoped type ratchet and a small CPU regression entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "e32ac213be3ad994a95ec71c53212123530be05c"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=capture, check=False)


def changed_files(base: str) -> tuple[list[str], list[str]]:
    def git(*args: str) -> set[str]:
        return set(subprocess.check_output(["git", *args], cwd=ROOT, text=True).splitlines())

    changed = git("diff", "--name-only", base, "--")
    new = git("diff", "--name-only", "--diff-filter=A", base, "--")
    untracked = git("ls-files", "--others", "--exclude-standard")
    changed |= untracked
    new |= untracked
    # S-24 moves these historical artifacts without changing a byte. Formatting
    # them would invalidate qualification hashes; only verified moves are exempt.
    manifest_path = ROOT / "qualification/stages/manifest.json"
    frozen = set()
    if manifest_path.is_file():
        for item in json.loads(manifest_path.read_text()):
            destination = item["destination"]
            source = item["source"]
            assert destination == "qualification/stages/" + Path(source).name
            assert source.startswith("pssolver/experimental/")
            original = subprocess.check_output(
                ["git", "show", "4c12228aba01e8483e80d28a1471ab72384d5fda:" + source],
                cwd=ROOT,
            )
            assert (ROOT / destination).read_bytes() == original
            assert hashlib.sha256(original).hexdigest() == item["sha256"]
            frozen.add(destination)
    return tuple(
        sorted(p for p in paths if p.endswith(".py") and (ROOT / p).is_file() and p not in frozen)
        for paths in (changed, new)
    )


def check_types(executable: str) -> bool:
    """Expose the old diagnostics; accept them only in byte-identical baseline files."""
    baseline = json.loads((ROOT / "qualification/type_baseline.json").read_text())
    result = run(
        [
            executable,
            "--follow-imports=skip",
            "--ignore-missing-imports",
            "--no-incremental",
            "--show-error-codes",
            "--no-error-summary",
            "pssolver/api",
            "pssolver/core",
        ],
        capture=True,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode not in (0, 1):
        return False
    actual = result.stdout.splitlines()
    allowed = set(baseline["diagnostics"])
    for diagnostic in actual:
        relative = diagnostic.split(":", 1)[0]
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if diagnostic not in allowed or digest != baseline["sources"].get(relative):
            return False
    errors = sum(": error:" in line for line in actual)
    print(
        f"Scoped mypy: {errors} unchanged baseline errors, {len(actual) - errors} notes; "
        "no new diagnostics (not a zero-error type qualification)."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    changed, new = changed_files(args.base)
    ruff = os.environ.get("RUFF", "ruff")
    mypy = os.environ.get("MYPY", "mypy")
    if changed and run([ruff, "check", "--no-cache", *changed]).returncode:
        return 1
    if new and run([ruff, "format", "--check", "--no-cache", *new]).returncode:
        return 1
    if run([sys.executable, "-B", "tools/check_imports.py"]).returncode:
        return 1
    if not check_types(mypy):
        return 1
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_v0_5_engineering.py",
            "tests/test_v0_5_foundation.py",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
