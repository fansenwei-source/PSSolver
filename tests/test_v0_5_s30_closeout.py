"""Check the written architecture against the complete S-30 import guard."""

import ast
import json
import re
import subprocess
from hashlib import sha256
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
BASIS = "7161608913809466b0f9dc85e419fe9f3fec5a9e"
RELEASE = "4c12228aba01e8483e80d28a1471ab72384d5fda"
DOCS = (
    "ARCHITECTURE.md", "README.md", "docs/migration_0_4_to_0_5.md",
    "docs/adding_a_model.md", "AGENTS.md", "CLAUDE.md", "CHANGELOG.md",
)
SYMBOL = re.compile(r"\bpssolver(?:\.[A-Za-z_]\w*)+\b(?![\w-])")


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


@pytest.mark.parametrize("mutation", ["add_name", "replace_name", "target", "static", "dynamic", "unresolved", "resolver"])
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
    elif mutation == "resolver":
        path = ROOT / "pssolver/core/_compatibility.py"
        original = path.read_text()
        changes["pssolver.core._compatibility"] = original.replace(
            "import_module(module_name)", f"import_module('{PILOT_TARGET}')", 1
        )
        assert changes["pssolver.core._compatibility"] != original
    else:
        relative = ROOT / "pssolver/core/errors.py"
        additional = {
            "static": "from pssolver import experimental\n",
            "dynamic": "from importlib import import_module as load\nload('pssolver.experimental.boundary_pilots')\n",
            "unresolved": "from importlib import import_module as load\nload(configured_target)\n",
        }
        changes["pssolver.core.errors"] = relative.read_text() + "\n" + additional[mutation]
    assert import_contract(ROOT, replacements=changes)["violations"]


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _names(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _names(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            result.update({target.id: {} for target in targets if isinstance(target, ast.Name)})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            result.update({alias.asname or alias.name.split('.')[0]: {} for alias in node.names})
    return result


def _exists(reference):
    parts = reference.split(".")
    for index in range(len(parts), 0, -1):
        prefix = ROOT.joinpath(*parts[:index])
        paths = (prefix.with_suffix(".py"), prefix / "__init__.py")
        for path in paths:
            if not path.is_file():
                continue
            names = _names(ast.parse(path.read_bytes()))
            if path == ROOT / "pssolver/__init__.py":
                names.update({name: {} for name in _exports(path.read_bytes())})
            for part in parts[index:]:
                if part not in names:
                    return False
                names = names[part]
            return True
    return False


def _exports(contents):
    tree = ast.parse(contents)
    return ast.literal_eval(next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ))


def _heading_body(text, heading):
    return text.split(heading + "\n", 1)[1].split("\n## ", 1)[0]


def test_architecture_anchors():
    text = (ROOT / "ARCHITECTURE.md").read_text()
    assert not re.search(r"^#{1,3} S-\d+ .*(?:draft|intermediate)", text, re.M | re.I)
    table = _heading_body(text, "## 6. Exactly one way to do each thing")
    rows = [line for line in table.splitlines() if line.startswith("| ")][1:]
    assert len(rows) == 10
    for line in rows:
        anchors = SYMBOL.findall(line)
        paths = re.findall(r"`(qualification/[^`]+\.py)`", line)
        assert anchors or paths, line
        assert all(_exists(name) for name in anchors), line
        assert all((ROOT / path).is_file() for path in paths), line


def test_current_document_references_resolve():
    missing = []
    for name in DOCS:
        doc = ROOT / name
        lines = doc.read_text().splitlines()
        for lineno, line in enumerate(lines, 1):
            if name == "CHANGELOG.md" and line.startswith("## 0.4.0 —"):
                break  # The remaining sections explicitly describe released versions.
            for symbol in SYMBOL.findall(line):
                if not symbol.endswith((".py", ".md")) and not _exists(symbol):
                    missing.append((name, lineno, symbol))
            for link, literal in re.findall(r"\]\(([^)]+)\)|`([^`]+)`", line):
                token = (link or literal).split("#", 1)[0]
                if token in {"metadata.json", "checkpoint.json"}:
                    continue  # Output filenames, not source paths.
                if re.fullmatch(r"[\w./-]+\.(?:py|md|json|sh|toml|sbatch)", token):
                    if not ((ROOT / token).is_file() or (doc.parent / token).is_file()):
                        missing.append((name, lineno, token))
    assert not missing, missing


def test_migration_rows_are_exhaustive():
    migration = (ROOT / "docs/migration_0_4_to_0_5.md").read_text()
    names = re.findall(r"^\| `([^`]+)` \|", _heading_body(migration, "## v0.4.0 package-root names"), re.M)
    assert len(names) == len(set(names)) == 66
    assert set(names) == set(_exports(_git("show", f"{RELEASE}:pssolver/__init__.py")))
    flags = re.findall(r"^\| `(-[^`]+)` \|", _heading_body(migration, "## v0.4.0 Plane CLI options"), re.M)
    assert len(flags) == len(set(flags)) == 57
    assert set(flags) == set(_disposition()["legacy_cli_options"])
    commands = re.findall(r"^\| `([^`]+)` \|", _heading_body(migration, "## Retired v0.4.0 root commands"), re.M)
    assert set(commands) == {
        ("bash " if name.endswith(".sh") else "python ") + name
        for name, row in _disposition()["root"].items() if row["disposition"] == "delete"
    }


def _disposition():
    return json.loads((ROOT / "qualification/s30_disposition.json").read_bytes())


def test_root_and_retirement_are_exhaustive():
    report = _disposition()
    root_rows = report["root"]
    assert set(root_rows) == {
        name for name in _git("ls-tree", "--name-only", BASIS).decode().splitlines()
        if name not in report["root_whitelist"] and name.endswith((".py", ".sh"))
    }
    assert {path for path in _git("ls-files").decode().splitlines() if "/" not in path} == (
        set(report["root_whitelist"]) | {p for p, row in root_rows.items() if row["disposition"] == "historical"}
    )
    assert len(report["paths"]) == 202
    for name, row in report["paths"].items():
        if row["disposition"] == "delete":
            assert not (ROOT / name).exists(), name
            assert sha256(_git("show", f"{row['retrieval_commit']}:{name}")).hexdigest() == row["sha256"], name
        else:
            assert row["disposition"] == "historical" and row["reason"], name
            assert sha256((ROOT / name).read_bytes()).hexdigest() == row["sha256"], name


def test_unreleased_changelog_scope():
    current = (ROOT / "CHANGELOG.md").read_text()
    baseline = _git("show", f"{BASIS}:CHANGELOG.md").decode()
    assert current.split("## 0.4.0 —", 1)[1] == baseline.split("## 0.4.0 —", 1)[1]
    unreleased = _heading_body(current, "## 0.5.0.dev0 — Unreleased")
    for text in ("v0.4.0", "S-23", "restart", "Functional API 1.1", "qualification/s30_disposition.json"):
        assert text in unreleased
