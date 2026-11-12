"""Audit distribution boundaries and exact wheel payloads without installing."""

from __future__ import annotations

import argparse
import difflib
import functools
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TOKENS = (b"/home/", b"fansenwei", b"/scratch", b"caspar")
CONSUMER_TOKENS = (b"PSSolver-Control", b"pssolver_control")


@functools.cache
def decided_source(commit: str, path: str) -> tuple[bytes, ...]:
    return tuple(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT).splitlines())


def decided_line(content: bytes, number: int, item: dict) -> bool:
    """R-0121: the exception covers the decided line while it is unchanged since its decision.

    The current line is mapped to the decided commit by byte-identical lines only; additions elsewhere in
    the file do not move the exception, an edit of the line itself ends it.
    """
    original = decided_source(item["original_commit"], item["path"])
    matcher = difflib.SequenceMatcher(a=original, b=tuple(content.splitlines()), autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.b < number <= block.b + block.size:
            return block.a + (number - block.b) == item["original_line"]
    return False


def members(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
    with tarfile.open(path) as archive:
        result = {}
        for item in archive.getmembers():
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError(f"Non-regular archive member: {item.name}")
            name = item.name.partition("/")[2]
            if not name or ".." in Path(name).parts or name.startswith("/"):
                raise ValueError(f"Unsafe archive member: {item.name}")
            result[name] = archive.extractfile(item).read()
        return result


def scan(files: dict[str, bytes], exceptions: list[dict]) -> dict:
    failures, historical_exceptions, consumer_mentions = [], [], []
    for name, content in sorted(files.items()):
        for number, line in enumerate(content.splitlines(), 1):
            private = [token.decode() for token in PRIVATE_TOKENS if token in line]
            if private:
                matched = [
                    item
                    for item in exceptions
                    if item["path"] == name
                    and item["line_sha256"] == hashlib.sha256(line).hexdigest()
                    and decided_line(content, number, item)
                ]
                row = {
                    "file": name,
                    "line": number,
                    "tokens": private,
                    "line_sha256": hashlib.sha256(line).hexdigest(),
                }
                (historical_exceptions if len(matched) == 1 else failures).append(row)
            consumer = [token.decode() for token in CONSUMER_TOKENS if token in line]
            if consumer:
                allowed = name in {"README.md", "CHANGELOG.md", "PKG-INFO"} or (
                    name.endswith(".dist-info/METADATA") or name.endswith(".egg-info/PKG-INFO")
                )
                row = {
                    "file": name,
                    "line": number,
                    "tokens": consumer,
                    "text": line.decode(errors="replace"),
                }
                (consumer_mentions if allowed else failures).append(row)
    if "CHANGELOG.md" in files:
        expected = {(item["path"], item["line_sha256"]) for item in exceptions}
        actual = [(item["file"], item["line_sha256"]) for item in historical_exceptions]
        if set(actual) != expected or len(actual) != len(expected):
            failures.append({"reason": "Historical exceptions must occur exactly once each"})
    return {
        "passed": not failures,
        "failures": failures,
        "historical_exceptions": historical_exceptions,
        "consumer_mentions": consumer_mentions,
    }


def inventory(files: dict[str, bytes]) -> dict:
    return {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    }


def audit(wheel: Path, sdist: Path, rebuilt: Path, baseline: Path) -> dict:
    policy = json.loads((ROOT / "qualification/package_policy.json").read_text())
    current, source, again, old = map(members, (wheel, sdist, rebuilt, baseline))
    scans = {
        name: scan(files, policy["private_line_exceptions"])
        for name, files in (("wheel", current), ("sdist", source))
    }
    source_allowlist = {
        line.removeprefix("include ")
        for line in (ROOT / "MANIFEST.in").read_text().splitlines()
        if line.startswith("include ")
    }
    # These files are emitted by setuptools, not additional source inputs.
    metadata = {
        "PKG-INFO",
        "setup.cfg",
        "pssolver.egg-info/PKG-INFO",
        "pssolver.egg-info/SOURCES.txt",
        "pssolver.egg-info/dependency_links.txt",
        "pssolver.egg-info/entry_points.txt",
        "pssolver.egg-info/requires.txt",
        "pssolver.egg-info/top_level.txt",
    }
    forbidden_source = sorted(set(source) - source_allowlist - metadata)
    missing_source = sorted(source_allowlist - set(source))

    def package(files):
        return {k: v for k, v in files.items() if ".dist-info/" not in k}

    before, after = package(old), package(current)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    allowed = {"pssolver/_version.py", "pssolver/visualizer.py"}
    runtime_payload_ok = set(changed) == allowed
    rebuild_ok = current == again
    return {
        "passed": (
            all(item["passed"] for item in scans.values())
            and not forbidden_source
            and not missing_source
            and runtime_payload_ok
            and rebuild_ok
        ),
        "scans": scans,
        "unexpected_sdist_members": forbidden_source,
        "missing_sdist_members": missing_source,
        "baseline_payload_changes": changed,
        "rebuild_member_bytes_equal": rebuild_ok,
        "inventories": {
            label: inventory(files)
            for label, files in (
                ("wheel", current),
                ("sdist", source),
                ("rebuilt_wheel", again),
                ("v040_wheel", old),
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "sdist", "rebuilt", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.wheel, args.sdist, args.rebuilt, args.baseline)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inventories"}, indent=2))
    return not report["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
