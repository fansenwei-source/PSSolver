"""Check the written architecture against the complete S-30 import guard."""

import re
from pathlib import Path

import pytest

from qualification.s30_guard import (
    PILOT_EXPORTS,
    PILOT_OWNER,
    PILOT_RESOLVER,
    PILOT_TARGET,
    import_contract,
)
from tools._source import read_toml


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "pssolver/operators/__init__.py"


def test_import_contract_matches_architecture():
    section = (ROOT / "ARCHITECTURE.md").read_text().split("| Package | Must not import |", 1)[1]
    table = section.split("\n\n", 1)[0]
    documented = {}
    for line in table.splitlines():
        names = re.findall(r"`([^`]+)`", line)
        if names:
            column, values = line.split("|")[1:3]
            for owner in re.findall(r"`([^`]+)`", column):
                documented[owner] = re.findall(r"`([^`]+)`", values)
    config = read_toml(ROOT / "pyproject.toml")["tool"]["pssolver"]
    assert documented == config["architecture-contracts"]
    report = import_contract(ROOT)
    assert not report["violations"] and not report["cycles"], report
    assert not report["frozen_cycles"]
    rendered = section.split("How `planning`", 1)[0]
    for owner, targets in report["exact_edges"]:
        assert owner.replace(":", ".") in rendered
        target, _, symbol = targets.partition(":")
        assert target in rendered and symbol in rendered
    for item in (PILOT_OWNER.replace(":", "."), PILOT_RESOLVER.replace(":", "."), PILOT_TARGET):
        assert item in rendered
    assert [targets for owner, _, _, targets in report["dynamic"]
            if owner == PILOT_OWNER and f"{PILOT_TARGET}:{PILOT_EXPORTS[0]}" in targets] == [
                sorted(f"{PILOT_TARGET}:{name}" for name in PILOT_EXPORTS)]


@pytest.mark.parametrize("mutation", ["add_name", "replace_name", "target", "static", "dynamic", "unresolved"])
def test_compatibility_contract_rejects_unreviewed_dependencies(mutation):
    operators = OPERATORS.read_text()
    changes = {}
    if mutation == "add_name":
        changes["pssolver.operators"] = operators.replace('"STRICT_NO_AUTOMATIC_CONDITIONING",',
            '"STRICT_NO_AUTOMATIC_CONDITIONING", "unreviewed_name",', 1)
    elif mutation == "replace_name":
        changes["pssolver.operators"] = operators.replace(PILOT_EXPORTS[0], "other_name", 1)
    elif mutation == "target":
        changes["pssolver.operators"] = operators.replace(PILOT_TARGET, "pssolver.experimental.other", 1)
    else:
        relative = ROOT / "pssolver/core/errors.py"
        additional = {
            "static": "from pssolver import experimental\n",
            "dynamic": "from importlib import import_module as load\nload('pssolver.experimental.boundary_pilots')\n",
            "unresolved": "from importlib import import_module as load\nload(configured_target)\n",
        }
        changes["pssolver.core.errors"] = relative.read_text() + "\n" + additional[mutation]
    assert import_contract(ROOT, replacements=changes)["violations"]
