"""Current-source half of historical record tests forwarded by R-0095 F-C."""

import hashlib
import json
import subprocess

from tests.v0_3_s03_successor import read_registered_source, resolve_s03_successor


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
        current = entry.get("current_path", path)
        expected = resolve_s03_successor(current, entry["current_sha256"])
        assert hashlib.sha256(read_registered_source(current, root)).hexdigest() == expected, path
    for path, digest in record["introduced_sources"].items():
        expected = resolve_s03_successor(path, digest)
        assert hashlib.sha256(read_registered_source(path, root)).hexdigest() == expected, path
