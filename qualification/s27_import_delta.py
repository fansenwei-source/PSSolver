"""Exact frozen-caller exceptions for the baseline-relative S-27 import scan."""

import hashlib
import json


def compare(root, baseline, candidate):
    rows = json.loads((root / "qualification/s27_expected_unimportable.json").read_text())
    registry = {r["module"]: r for r in rows}
    invalid_hashes = [
        r["path"]
        for r in rows
        if hashlib.sha256((root / r["path"]).read_bytes()).hexdigest() != r["sha256"]
    ]
    static, actual, accepted_static, accepted_actual = [], [], [], []

    def key(edge):
        return edge["target"], edge.get("name"), edge["kind"]

    for module, row in candidate["modules"].items():
        old = {key(e) for e in baseline["modules"].get(module, {}).get("errors", [])}
        expected = registry.get(module, {}).get("static", [])
        for edge in row["errors"]:
            if key(edge) in old:
                continue
            entry = dict(module=module, **edge)
            (accepted_static if edge in expected else static).append(entry)
    for module, row in candidate["actual"].items():
        if not row["exit_code"] or baseline["actual"].get(module, {}).get("exit_code", 0):
            continue
        expected = registry.get(module, {}).get("actual_error")
        tail = row["stderr"].strip().splitlines()[-1] if row["stderr"].strip() else ""
        entry = dict(module=module, **row)
        (accepted_actual if expected is not None and tail == expected else actual).append(entry)
    return dict(
        passed=not (static or actual or invalid_hashes),
        unexpected_static=static,
        unexpected_actual=actual,
        accepted_static=accepted_static,
        accepted_actual=accepted_actual,
        invalid_hashes=invalid_hashes,
    )
