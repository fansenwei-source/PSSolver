"""Changed-file lint, scoped type ratchet and a small CPU regression entry point."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "e32ac213be3ad994a95ec71c53212123530be05c"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=capture, check=False)


def changed_files(base: str) -> tuple[list[str], list[str], dict[str, str]]:
    def git(*args: str) -> set[str]:
        return set(subprocess.check_output(["git", *args], cwd=ROOT, text=True).splitlines())

    fields = (
        subprocess.check_output(
            ["git", "diff", "--name-status", "-M100%", "-z", base, "--"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    changed, renames = set(), {}
    cursor = 0
    while cursor < len(fields) - 1:
        status, source = fields[cursor : cursor + 2]
        cursor += 2
        target = source
        if status.startswith("R"):
            target = fields[cursor]
            cursor += 1
            if status == "R100":
                original = subprocess.check_output(["git", "show", base + ":" + source], cwd=ROOT)
                if original != (ROOT / target).read_bytes():
                    raise ValueError(f"non-exact rename: {source} -> {target}")
                renames[target] = source
        changed.add(target)
    baseline_paths = git("ls-tree", "-r", "--name-only", base)
    new = changed - baseline_paths - renames.keys()
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
    selected = tuple(
        sorted(p for p in paths if p.endswith(".py") and (ROOT / p).is_file() and p not in frozen)
        for paths in (changed, new)
    )
    return *selected, {p: renames[p] for p in selected[0] if p in renames}


def check_types(executable: str) -> bool:
    """Bind old diagnostics to unchanged source lines, including deletion-only shifts."""
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
    mapped_lines = {}
    for diagnostic in actual:
        relative = diagnostic.split(":", 1)[0]
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if diagnostic in allowed and digest == baseline["sources"].get(relative):
            continue
        if relative not in baseline["sources"]:
            return False
        if "commit" not in baseline:
            return False
        if relative not in mapped_lines:
            original = subprocess.check_output(
                ["git", "show", baseline["commit"] + ":" + relative], cwd=ROOT
            )
            if hashlib.sha256(original).hexdigest() != baseline["sources"][relative]:
                return False
            mapped_lines[relative] = unchanged_line_map(
                original.decode(), (ROOT / relative).read_text()
            )
        _, line, message = diagnostic.split(":", 2)
        old_line = mapped_lines[relative].get(int(line))
        if old_line is None or f"{relative}:{old_line}:{message}" not in allowed:
            return False
    errors = sum(": error:" in line for line in actual)
    print(
        f"Scoped mypy: {errors} unchanged baseline errors, {len(actual) - errors} notes; "
        "no new diagnostics (not a zero-error type qualification)."
    )
    return True


def unchanged_line_map(original: str, current: str) -> dict[int, int]:
    """Only byte-identical lines may inherit an exact historical diagnostic."""
    result = {}
    for block in difflib.SequenceMatcher(
        a=original.splitlines(keepends=True),
        b=current.splitlines(keepends=True),
        autojunk=False,
    ).get_matching_blocks():
        for offset in range(block.size):
            result[block.b + offset + 1] = block.a + offset + 1
    return result


def lint_diagnostics(executable: str, relative: str, source: bytes) -> list[dict]:
    """Use one configuration and filename for baseline and candidate diagnostics."""
    command = [
        executable,
        "check",
        "--no-cache",
        "--output-format=json",
        "--stdin-filename",
        relative,
        "-",
    ]
    completed = subprocess.run(command, cwd=ROOT, input=source, capture_output=True, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.decode())
    return json.loads(completed.stdout)


def lint_counts(executable: str, relative: str, source: bytes) -> Counter:
    return Counter(item["code"] for item in lint_diagnostics(executable, relative, source))


def unit_span(source: str, name: str) -> tuple[int, int]:
    units = [node for node in ast.parse(source).body if getattr(node, "name", None) == name]
    if len(units) != 1:
        raise ValueError(f"missing or ambiguous exact unit: {name}")
    node = units[0]
    return min([node.lineno, *(d.lineno for d in node.decorator_list)]), node.end_lineno


def exact_assemblies() -> dict[str, list[dict]]:
    """Validate every registered span against its reachable, byte-identical source."""
    manifest = json.loads((ROOT / "qualification/s25_exact_moves.json").read_text())
    result = {}
    for entry in manifest["exact_moves"]:
        original = subprocess.check_output(
            ["git", "show", manifest["source_commit"] + ":" + entry["source"]], cwd=ROOT
        ).decode()
        current = (ROOT / entry["target"]).read_text()
        old_start, old_end = unit_span(original, entry["unit"])
        start, end = unit_span(current, entry["unit"])
        before = "".join(original.splitlines(keepends=True)[old_start - 1 : old_end])
        after = "".join(current.splitlines(keepends=True)[start - 1 : end])
        if (
            [start, end] != entry["target_lines"]
            or before != after
            or hashlib.sha256(before.encode()).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"altered exact move: {entry['target']}::{entry['unit']}")
        result.setdefault(entry["target"], []).append(
            dict(
                entry, start=start, end=end, original=original, old_start=old_start, old_end=old_end
            )
        )
    for path, entries in result.items():
        ordered = sorted(entries, key=lambda row: row["start"])
        if any(a["end"] >= b["start"] for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError(f"overlapping exact moves: {path}")
    return result


def assembly_lint(executable: str, relative: str, entries: list[dict]) -> bool:
    """Only the same rule at the same line within the same old unit is allowed."""
    expected = Counter()
    for entry in entries:
        for item in lint_diagnostics(executable, entry["source"], entry["original"].encode()):
            row = item["location"]["row"]
            if entry["old_start"] <= row <= entry["old_end"]:
                expected[(entry["unit"], item["code"], row - entry["old_start"])] += 1
    actual, outside = Counter(), []
    for item in lint_diagnostics(executable, relative, (ROOT / relative).read_bytes()):
        row = item["location"]["row"]
        containing = [entry for entry in entries if entry["start"] <= row <= entry["end"]]
        if not containing:
            outside.append(item)
        else:
            entry = containing[0]
            actual[(entry["unit"], item["code"], row - entry["start"])] += 1
    passed = not outside and actual == expected
    print(json.dumps(dict(path=relative, assembly_lint=passed, outside=outside), sort_keys=True))
    return passed


def assembly_format_changes(original: str, formatted: str, entries: list[dict]) -> list[dict]:
    """Diff context is not an edit; every changed hunk must be inside one old unit."""
    violations = []
    matcher = difflib.SequenceMatcher(
        a=original.splitlines(keepends=True), b=formatted.splitlines(keepends=True), autojunk=False
    )
    for tag, start, end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        # A pure insertion must be between two lines of that same unit, never
        # before a decorator/definition or after the end of its original body.
        if not any(
            entry["start"] <= start + 1
            and end <= entry["end"]
            and (start != end or entry["start"] <= start < entry["end"])
            for entry in entries
        ):
            violations.append(dict(kind=tag, old=[start, end], new=[new_start, new_end]))
    return violations


def format_new_files(executable: str, new: list[str], assemblies: dict) -> bool:
    passed = True
    for relative in new:
        if relative.startswith("tests/"):
            continue  # Df7: no test formatting, including new tests.
        if relative not in assemblies:
            passed = (
                run([executable, "format", "--check", "--no-cache", relative]).returncode == 0
                and passed
            )
            continue
        diff = run([executable, "format", "--diff", "--no-cache", relative], capture=True)
        print(diff.stdout, end="")
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr)
        source = (ROOT / relative).read_text()
        formatted = subprocess.run(
            [executable, "format", "--no-cache", "--stdin-filename", relative, "-"],
            cwd=ROOT,
            input=source,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        violations = assembly_format_changes(source, formatted, assemblies[relative])
        print(json.dumps(dict(path=relative, assembly_format_violations=violations)))
        passed = not violations and passed
    return passed


def lint_ratchet(
    executable: str,
    changed: list[str],
    new: list[str],
    base: str,
    *,
    renames: dict | None = None,
    assemblies: dict | None = None,
) -> bool:
    """Existing files cannot gain any per-rule diagnostics; new files must be clean."""
    passed = True
    for relative in changed:
        if relative in new and assemblies and relative in assemblies:
            passed = assembly_lint(executable, relative, assemblies[relative]) and passed
            continue
        actual = lint_counts(executable, relative, (ROOT / relative).read_bytes())
        previous = Counter()
        if relative not in new:
            old_path = (renames or {}).get(relative, relative)
            original = subprocess.check_output(["git", "show", base + ":" + old_path], cwd=ROOT)
            previous = lint_counts(executable, relative, original)
        increased = actual - previous
        print(
            json.dumps(
                {"path": relative, "baseline": previous, "actual": actual, "increased": increased},
                sort_keys=True,
            )
        )
        passed = passed and not increased
    return bool(passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    changed, new, renames = changed_files(args.base)
    ruff = os.environ.get("RUFF", "ruff")
    mypy = os.environ.get("MYPY", "mypy")
    assemblies = exact_assemblies()
    if not lint_ratchet(ruff, changed, new, args.base, renames=renames, assemblies=assemblies):
        return 1
    if not format_new_files(ruff, new, assemblies):
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
