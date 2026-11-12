"""Check new caller breakage against exact frozen byte and diagnostic records."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from qualification.s28a_import_delta import changes


IDENTITY_DRIVERS = {
    "runtime_case",
    "runtime_case_v2",
    "channel_comparison",
    "checkpoint_fixtures",
    "historical_continuations",
    "retained_v020_fixtures",
    "trajectory_hashes",
    "vjp",
}
RECORD = "qualification/s28b_frozen_callers.json"


def scope(path, basis_scan, module, stage_paths):
    if path == "Plane_beris_edwards_stokes.py":
        return None
    if "/" not in path and path.endswith(".py") and path != "setup.py":
        return "r2.1 §7.5: frozen root script; S-30"
    if path.startswith(("benchmarks/", "jobs/", "scripts_plane/", "scripts_channel/")):
        return "r2.1 §7.5: frozen caller; S-30"
    if path.startswith("qualification/identity/") and Path(path).stem in IDENTITY_DRIVERS:
        return "r2.1 §7.5 Q1: unchanged v0.4 qualification driver"
    # The S-24 manifest independently freezes these non-live artifacts. They must already
    # have failed to import at the baseline; a new runtime failure is never authorized here.
    if path in stage_paths or path == "qualification/generate_s24_retired_fixtures.py":
        if basis_scan["actual"].get(module, {}).get("exit_code"):
            return "R-0075: already historical at baseline; static-only delta"
    return None


def register(root, baseline, candidate, basis_commit):
    stages = json.loads((root / "qualification/stages/manifest.json").read_bytes())
    stage_paths = {row["destination"]: row["sha256"] for row in stages}
    rows, invalid = {}, []
    for module, delta in changes(baseline, candidate).items():
        path = delta["path"]
        authority = scope(path, baseline, module, stage_paths)
        if authority is None:
            invalid.append(module)
            continue
        source = subprocess.check_output(["git", "show", f"{basis_commit}:{path}"], cwd=root)
        digest = hashlib.sha256(source).hexdigest()
        if (root / path).read_bytes() != source or digest != stage_paths.get(path, digest):
            invalid.append(module)
            continue
        rows[module] = {**delta, "sha256": digest, "authority": authority}
    return rows, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads((args.root / RECORD).read_bytes())
    baseline = json.loads(args.baseline.read_bytes())
    candidate = json.loads(args.candidate.read_bytes())
    actual, invalid = register(args.root, baseline, candidate, record["basis"])
    expected = record["modules"]
    mismatched = [
        name
        for name in sorted(actual.keys() | expected.keys())
        if actual.get(name) != expected.get(name)
    ]
    report = dict(
        passed=not (invalid or mismatched),
        invalid=invalid,
        mismatched=mismatched,
        caller_count=len(actual),
        actual=actual,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "actual"}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
