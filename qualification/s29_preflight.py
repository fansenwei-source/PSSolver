"""Check the actual successor and collection inventory against the S-29 increment."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"
NOTE = "notes/PSSolver_v0_3_0_s03_successor_149_s29_interfaces.json"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(root, collection=None):
    note = json.loads((root / NOTE).read_bytes())
    existing = note["historical_successor_sources"]
    introduced = note["introduced_sources"]
    changed = set(git(root, "diff", "--name-only", BASE).decode().splitlines())
    changed.update(git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    assert changed == set(existing) | set(introduced) | {NOTE}
    assert not existing.keys() & introduced.keys()
    for path, entry in existing.items():
        assert entry["historical_sha256"] == digest(git(root, "show", f"{BASE}:{path}"))
        assert entry["current_sha256"] == digest((root / path).read_bytes())
    for path, value in introduced.items():
        assert value == digest((root / path).read_bytes())
    sys.path.insert(0, str(root))
    from tests._categories import load_categories

    paths, gpu = load_categories(root)
    table = json.loads((root / "qualification/s29_authority.json").read_bytes())
    assert gpu == set(table["six_cuda_deselect"]) and len(gpu) == 6
    new_tests = sorted(name for name in introduced if name in paths)
    assert set(note["test_disposition"]["new"]) == set(new_tests)
    if collection is not None:
        doc = json.loads(collection.read_bytes())
        assert set(doc["deselected"]) == gpu
        assert not doc["collection_errors"]
        nodes = {item.split("[", 1)[0] for item in doc["nodeids"]}
        registry = json.loads((root / "qualification/s29_optimizations.json").read_bytes())
        for row in registry.values():
            assert set(row["collected_nodes"]) <= nodes
    return {
        "passed": True,
        "old_sources": len(existing),
        "new_sources": len(introduced),
        "new_test_files": new_tests,
        "deselect": sorted(gpu),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.collection)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
