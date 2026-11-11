"""Exact S-28a import dispositions: only unchanged callers of retired modules."""

import hashlib
import json


def changes(baseline, candidate):
    """Return new static and isolated import failures, preserving call chains."""
    rows = {}
    for module, current in candidate["modules"].items():
        old = baseline["modules"].get(module, {})
        previous = {(r["target"], r["name"], r["kind"]) for r in old.get("errors", [])}
        static = [
            error
            for error in current["errors"]
            if (error["target"], error["name"], error["kind"]) not in previous
        ]
        actual = candidate["actual"].get(module, {})
        code = actual.get("exit_code", 0)
        new_error = code and not baseline["actual"].get(module, {}).get("exit_code", 0)
        tail = actual.get("stderr", "").strip().splitlines()
        actual_error = (
            dict(exit_code=code, last_line=tail[-1].replace(candidate["root"], "<ROOT>"))
            if new_error and tail
            else None
        )
        if new_error and not tail:
            actual_error = dict(exit_code=code, last_line="")
        if static or actual_error:
            rows[module] = {
                "path": current["path"],
                "static": static,
                "actual": actual_error,
            }
    return rows


def check(root, baseline, candidate):
    """An added, missing, changed or unregistered exception all fail closed."""
    path = root / "qualification/s28a_expected_unimportable.json"
    record = json.loads(path.read_bytes())
    actual = changes(baseline, candidate)
    expected = {
        module: {key: row[key] for key in ("path", "static", "actual")}
        for module, row in record["modules"].items()
    }
    hash_mismatches = [
        row["path"]
        for row in record["modules"].values()
        if not row["path"].startswith("benchmarks/")
        or hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]
    ]
    discrepancies = [
        module
        for module in sorted(actual.keys() | expected.keys())
        if actual.get(module) != expected.get(module)
    ]
    return {
        "passed": not (hash_mismatches or discrepancies),
        "hash_mismatches": hash_mismatches,
        "discrepancies": discrepancies,
        "new_failures": actual,
        "recorded_frozen_callers": len(expected),
    }
