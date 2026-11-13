"""Keep the current and historical S-03 chains independent of directory order."""

import hashlib
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from tests import v0_3_s03_successor as live


ROOT = Path(__file__).resolve().parents[1]
BASE = "notes/PSSolver_v0_3_0_s03_public_errors_metadata.json"
TARGET = "notes/pssolver_v0_2_x_open_issues.json"
ALLOWED_TIES = {
    17: {"17_r0016_import", "17_s07_packaging_metadata_docs"},
    134: {"134_s22_foundation", "134_v040_p861_reader"},
}
HISTORICAL = (
    "4c12228aba01e8483e80d28a1471ab72384d5fda",
    "63f19383ec2d05aedc8b0354f058302f74f4a751",
    "327a192ba879da0991110eca99f4b69b9a45c84e",
    "49d1c32b66ef0088fed4ea13ab71df4d1a1bf802",
)


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _number(path):
    return int(path.stem.split("_successor_", 1)[1].split("_", 1)[0])


def _terminal(sequence, records, start):
    for path in sequence:
        row = records[path]["historical_successor_sources"].get(TARGET)
        if row and start in row.get("accepted_predecessor_sha256", [row["historical_sha256"]]):
            start = row["current_sha256"]
    return start


def test_live_resolver_sorts_even_when_directory_iteration_is_reversed(monkeypatch):
    notes = ROOT / "notes"
    paths = list(notes.glob(live.SUCCESSOR_PATTERN))
    expected = sorted(paths, key=lambda path: (_number(path), path.name))
    real_glob = Path.glob

    def descending(path, pattern):
        if path == notes and pattern == live.SUCCESSOR_PATTERN:
            return iter(reversed(expected))
        return real_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", descending)
    assert live._records() == [live.BASE_RECORD, *expected]


@pytest.mark.parametrize("commit", HISTORICAL)
def test_only_file_name_order_reaches_the_frozen_registry_bytes(commit):
    listing = _git("ls-tree", "-z", "--name-only", commit, "notes/").decode().split("\0")
    names = [Path(name) for name in listing
             if re.fullmatch(r"notes/PSSolver_v0_3_0_s03_successor_\d+_.*\.json", name)]
    ordered = sorted(names, key=lambda path: (_number(path), path.name))
    pair = [index for index, path in enumerate(ordered) if _number(path) == 17]
    first, second = pair
    assert first + 1 == second
    reverse_pair = ordered.copy()
    reverse_pair[first], reverse_pair[second] = reverse_pair[second], reverse_pair[first]
    base = Path(BASE)
    records = {path: json.loads(_git("show", f"{commit}:{path}")) for path in (base, *ordered)}
    start = records[base]["historical_successor_sources"][TARGET]["historical_sha256"]
    accepted = hashlib.sha256(_git("show", f"{commit}:{TARGET}")).hexdigest()
    assert _terminal([base, *ordered], records, start) == accepted
    assert _terminal([base, *reverse_pair], records, start) != accepted


def test_historical_pin_rejects_unknown_resolver_before_importing_it(tmp_path):
    path = ROOT / "qualification/historical_successor_order.py"
    spec = importlib.util.spec_from_file_location("_historical_successor_order", path)
    assert spec is not None and spec.loader is not None
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    resolver = tmp_path / order.RESOLVER
    resolver.parent.mkdir()
    resolver.write_text("raise AssertionError('must not execute an unverified resolver')\n")
    with pytest.raises(RuntimeError, match="unexpected S-03 resolver bytes"):
        order.pin(tmp_path)


def test_only_the_two_exact_historical_duplicate_groups_are_allowed():
    groups = defaultdict(set)
    for path in (ROOT / "notes").glob(live.SUCCESSOR_PATTERN):
        groups[_number(path)].add(path.stem.split("_successor_", 1)[1])
    assert {number: names for number, names in groups.items() if len(names) > 1} == ALLOWED_TIES
