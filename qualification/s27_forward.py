"""Current-source half of historical record tests forwarded by R-0095 F-C."""

import hashlib
import json
import subprocess


def assert_forward_binding(root, nodeid):
    """Require the unchanged historical assertion and every new source binding."""
    history = json.loads((root / "qualification/historical_s27_tests.json").read_text())
    selected = history["selected_nodeids"]
    assert nodeid in selected or nodeid.split("::")[0] in selected
    relative = nodeid.split("::")[0]
    frozen = next(row for row in history["files"] if row["path"] == relative)
    source = subprocess.check_output(
        ["git", "show", history["source_commit"] + ":" + relative],
        cwd=root,
    )
    assert hashlib.sha256(source).hexdigest() == frozen["sha256"]
    record = json.loads(
        (root / "notes/PSSolver_v0_3_0_s03_successor_146_s27_plane.json").read_text()
    )
    assert record["historical_records_rewritten"] is False
    bindings = record["historical_successor_sources"]
    assert relative in bindings
    for path, entry in bindings.items():
        current = root / entry.get("current_path", path)
        assert hashlib.sha256(current.read_bytes()).hexdigest() == entry["current_sha256"], path
    for path, digest in record["introduced_sources"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
