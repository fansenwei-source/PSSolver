"""Completion guards for the S-16 R7 canonical-JSON consolidation."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "pssolver"


def _is_exact_shared_encoding(node: ast.Call) -> bool:
    function = node.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "json"
        and function.attr == "dumps"
        and len(node.args) == 1
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if set(keywords) != {"allow_nan", "separators", "sort_keys"}:
        return False
    separators = keywords["separators"]
    return (
        isinstance(keywords["allow_nan"], ast.Constant)
        and keywords["allow_nan"].value is False
        and isinstance(keywords["sort_keys"], ast.Constant)
        and keywords["sort_keys"].value is True
        and isinstance(separators, ast.Tuple)
        and [getattr(item, "value", None) for item in separators.elts]
        == [",", ":"]
    )


def test_only_the_shared_helper_implements_the_exact_compact_encoding():
    observed: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_exact_shared_encoding(node):
                observed.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert observed == [
        "pssolver/core/_canonical.py:23",
        "pssolver/io/checkpoint_identity.py:110",
    ]


def test_checkpoint_identity_keeps_its_schema_independent_local_encoder():
    path = PACKAGE_ROOT / "io" / "checkpoint_identity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any(name.startswith("pssolver") for name in imports)


def test_tensor_headers_keep_their_distinct_noncanonical_wire_format():
    expected = {
        "pssolver/operators/lifting.py": "_tensor_sha256",
        "pssolver/operators/robin.py": "_tensor_sha256",
    }
    for relative, function_name in expected.items():
        path = PROJECT_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        function = functions[function_name]
        dumps = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr == "dumps"
        ]
        assert len(dumps) == 1
        assert not _is_exact_shared_encoding(dumps[0])
