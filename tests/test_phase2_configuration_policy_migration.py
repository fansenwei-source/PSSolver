"""P2.6 oracles for retiring configuration numerical-policy import debt."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pssolver.configuration.plane_beris_edwards as facade
import pssolver.operators.projection as projection
import pssolver.plane as plane
import pssolver.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACADE_PATH = (
    PROJECT_ROOT / "pssolver" / "configuration" / "plane_beris_edwards.py"
)

PLANE_POLICY_NAMES = {
    "DEFAULT_PLANE_SPECTRAL_STORAGE",
    "PLANE_HERMITIAN_AXIS",
}
TRANSFORM_POLICY_NAMES = {
    "DEALIAS_RULE_FRACTIONS",
    "DEFAULT_DEALIAS_RULE",
    "DEFAULT_PROJECTED_TRANSFORM_EXECUTION",
    "DEFAULT_TRANSFORM_EXECUTION_ORDER",
    "PROJECTED_TRANSFORM_EXECUTION_MODES",
    "SPECTRAL_STORAGE_MODES",
}


def _absolute_imported_names(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        result.setdefault(node.module or "", set()).update(
            alias.name for alias in node.names
        )
    return result


def test_configuration_policy_import_debt_is_retired_to_tensor_free_policy():
    imports = _absolute_imported_names(FACADE_PATH)
    assert "pssolver.plane" not in imports
    assert "pssolver.transforms" not in imports
    assert imports["pssolver.geometries.plane_numerics"] == (
        PLANE_POLICY_NAMES
    )
    assert imports["pssolver.core.numerics"] == TRANSFORM_POLICY_NAMES


def test_numerical_policy_values_and_legacy_facade_aliases_are_frozen():
    assert plane.DEFAULT_PLANE_SPECTRAL_STORAGE == "hermitian_half"
    assert plane.PLANE_HERMITIAN_AXIS == 1
    assert transforms.DEFAULT_DEALIAS_RULE == "cubic_half"
    assert transforms.DEFAULT_PROJECTED_TRANSFORM_EXECUTION == "truncated"
    assert transforms.DEFAULT_TRANSFORM_EXECUTION_ORDER == "real_first"
    assert transforms.PROJECTED_TRANSFORM_EXECUTION_MODES == (
        "full",
        "truncated",
    )
    assert transforms.SPECTRAL_STORAGE_MODES == (
        "full_complex",
        "hermitian_half",
    )
    assert transforms.DEALIAS_RULE_FRACTIONS == {
        "none": None,
        "two_thirds": 2.0 / 3.0,
        "cubic_half": 0.5,
    }

    assert facade.DEFAULT_PLANE_SPECTRAL_STORAGE is (
        plane.DEFAULT_PLANE_SPECTRAL_STORAGE
    )
    assert facade.PLANE_HERMITIAN_AXIS is plane.PLANE_HERMITIAN_AXIS
    for name in TRANSFORM_POLICY_NAMES:
        assert getattr(facade, name) is getattr(transforms, name)
    assert transforms.DEALIAS_RULE_FRACTIONS is (
        projection.DEALIAS_RULE_FRACTIONS
    )


def test_factory_defaults_remain_the_frozen_policy_values():
    parameters = inspect.signature(
        facade.create_plane_beris_edwards_run_spec
    ).parameters
    assert parameters["dealias_rule"].default == "cubic_half"
    assert parameters["projected_transform_execution"].default == "truncated"
    assert parameters["transform_execution_order"].default == "real_first"
    assert parameters["spectral_storage"].default == "hermitian_half"
