"""Tensor-free assembly, exact capability matching and rejection boundaries."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pssolver.core.terms import (
    ConstrainedField,
    Constraint,
    EvolvedField,
    Explicit,
    FieldRef,
    Linear,
    Model,
    PackageTerm,
    Parameter,
    lap,
)
from pssolver.planning.assemble import (
    AssemblyProblem,
    AssemblyRejection,
    EulerInput,
    ExecutionInput,
    GeometryInput,
    NumericsInput,
    OutputInput,
    assemble,
)
from pssolver.planning.capabilities import CAPABILITIES, basis_signature, lookup


EXPECTED_CAPABILITIES = [
    ("constraint", ("stokes", "periodic^2×free_slip"), "pssolver.models.active_nematics.engine_terms:plane_stokes_term", "autograd"),
    ("explicit", ("beris_edwards_q", "periodic^2×neumann"), "pssolver.models.active_nematics.engine_terms:plane_explicit_term", None),
    ("constraint", ("stokes", "periodic^3"), "pssolver.models.active_nematics.engine_terms:periodic_stokes_term", "autograd"),
    ("explicit", ("beris_edwards_q", "periodic^3"), "pssolver.models.active_nematics.engine_terms:periodic_explicit_term", None),
    ("explicit", ("pointwise", "periodic^3"), "pssolver.engine.terms:pointwise_explicit_term", None),
    ("linear_recipe", ("beris_edwards", "any"), "pssolver.models.active_nematics:beris_edwards_linear_operator", None),
    ("linear_recipe", ("expression", "periodic^3"), "pssolver.engine.terms:expression_symbol", None),
]


class ScalarModel(Model):
    phi = EvolvedField()
    epsilon = Parameter(.05)

    def terms(self, f, p):
        return (Linear(f.phi, p.epsilon**2 * lap(f.phi)),
                Explicit(f.phi, lambda phi: phi - phi**3))


def problem(model=None, **changes):
    model = ScalarModel() if model is None else model
    return replace(AssemblyProblem(
        model, GeometryInput("periodic_box", (8, 8, 8), (1., 1., 1.)),
        {name: ("periodic",) * 3 for name in model.fields}, EulerInput(.001),
        NumericsInput("hermitian_half", 2, "real_first", "none", "full"),
        ExecutionInput(), OutputInput(("phi",))), **changes)


class RejectionModel(ScalarModel):
    other = EvolvedField()
    auxiliary = ConstrainedField()
    control = Parameter(1., controllable=True)

    def __init__(self, code):
        super().__init__()
        self.code = code

    def terms(self, f, p):
        cases = {
            "M1": lambda: [Constraint("absent", inputs=(f.phi,), outputs=(f.auxiliary,))],
            "M2": lambda: [Explicit(f.phi, PackageTerm("absent"))],
            "M3": lambda: [Linear(f.phi, lap(f.other))],
            "M4": lambda: [Linear(f.phi, p.control * lap(f.phi))],
            "M6": lambda: [Linear(FieldRef("missing", ("missing",)), f.phi)],
            "M7": lambda: [Linear(f.phi, f.phi * f.phi)],
            "M8": lambda: [Explicit(f.phi, lambda phi: phi, execution_options=("absent",))],
        }
        return cases[self.code]()


REJECTIONS = {
    "M1": "missing_constraint_solver", "M2": "missing_explicit_implementation",
    "M3": "linear_requires_modal_block", "M4": "linear_requires_general_solver",
    "M5": "incompatible_boundaries", "M6": "invalid_field_reference",
    "M7": "invalid_linear_expression", "M8": "unused_or_missing_execution_option",
    "S1": "geometry_not_migrated", "S2": "nonzero_dirichlet_lifting",
    "S3": "robin_or_anchoring", "S4": "integrator_not_available",
    "S5": "compiled_model_kernel", "S6": "refresh_not_supported",
    "S7": "boundary_data_reserved",
}


def rejected_problem(code):
    if code.startswith("M") and code != "M5":
        return problem(RejectionModel(code))
    changes = {
        "M5": dict(boundaries={"phi": ("neumann",) * 3}),
        "S1": dict(geometry=GeometryInput("rectangular_channel", (8, 8, 8), (1., 1., 1.))),
        "S2": dict(boundaries={"phi": ("periodic", "periodic", "nonzero_dirichlet")}),
        "S3": dict(boundaries={"phi": ("periodic", "periodic", "robin")}),
        "S4": dict(integrator="sbdf2"),
        "S5": dict(execution=ExecutionInput(pointwise_execution="compile")),
        "S6": dict(integrator=EulerInput(.001, refresh_interval=1)),
    }
    if code == "S7":
        base = problem()
        return SimpleNamespace(**{name: getattr(base, name) for name in base.__dataclass_fields__}, boundary_data=(1,))
    return problem(**changes[code])


@pytest.mark.parametrize("code", REJECTIONS)
def test_rejections_have_exact_codes_and_no_tensor_operations(code):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"rejection evaluated a tensor operation: {func}")

    with NoTensorOperations(), pytest.raises(AssemblyRejection) as error:
        assemble(rejected_problem(code))
    assert error.value.code == REJECTIONS[code]
    assert error.value.kind == ("scope" if code.startswith("S") else "mathematical")
    assert bool(error.value.source) == code.startswith("S")
    assert bool(error.value.opens_in) == code.startswith("S")


@pytest.mark.parametrize("code", REJECTIONS)
def test_rejections_do_not_construct_tensors(code):
    root = Path(__file__).resolve().parents[1]
    script = "\n".join([
        "import sys", f"sys.path.insert(0, {str(root)!r})",
        "from tests.test_v0_5_s26_assembly import rejected_problem, REJECTIONS",
        "from pssolver.planning.assemble import assemble, AssemblyRejection",
        "from qualification.s26_guards import tensor_constructors",
        "with tensor_constructors() as counts:",
        "    try:", f"        assemble(rejected_problem({code!r}))",
        "    except AssemblyRejection as e:", f"        assert e.code == REJECTIONS[{code!r}]",
        "    else: raise AssertionError('accepted')",
        "assert counts == {'tensor_new': 0, 'base_new': 0}, counts",
    ])
    run = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_constructor_hooks_observe_both_entry_points():
    import torch

    from qualification.s26_guards import tensor_constructors

    with tensor_constructors() as counts:
        torch.Tensor([1.])
    assert counts["tensor_new"] == 1
    assert counts["base_new"] == 1
    with tensor_constructors() as counts:
        torch._C._TensorBase.__new__(torch.Tensor, [1.])
    assert counts == {"tensor_new": 0, "base_new": 1}


def test_planning_transitive_import_closure_is_tensor_free():
    from qualification.s26_guards import planning_closure

    report = planning_closure(Path(__file__).resolve().parents[1])
    assert report["violations"] == []
    assert len(report["exceptions"]) == 1
    assert report["exceptions"][0]["owner"] == "AssembledProblem.engine"


def test_default_controls_are_the_same_readonly_mapping(tmp_path):
    from tests.test_v0_5_s25_engine_core import _setup

    _, engine, _ = _setup(tmp_path)
    assert engine.default_controls is engine.default_controls
    assert engine.default_controls is engine._default_controls
    with pytest.raises(TypeError):
        engine.default_controls["alpha"] = 0
    with pytest.raises(AttributeError):
        engine.default_controls = {}


def test_parameter_order_linear_normalization_and_local_identity():
    assembled = assemble(problem())
    plan = assembled.plan
    assert [item.name for item in plan.parameters] == ["epsilon"]
    assert plan.parameters[0].default == .05
    assert plan.linear[0].arguments["monomials"] == (
        (.05**2, (0, 0, 2)), (.05**2, (0, 2, 0)), (.05**2, (2, 0, 0)))
    assert set(assembled.kernels) == {plan.explicit[0].kernel}
    assert plan.explicit[0].arguments["model_version"] == "1"
    with pytest.raises(TypeError, match="unknown model parameters"):
        ScalarModel(unknown=1.)


def test_capability_table_is_exact_and_resolves_function_identity():
    from pssolver.engine.constraints import PeriodicStokesConstraint
    from pssolver.engine.engine import _resolve
    from pssolver.engine.terms import expression_symbol, pointwise_explicit_term
    from pssolver.models.active_nematics import beris_edwards_linear_operator
    from pssolver.models.active_nematics.engine_terms import (
        periodic_explicit_term,
        periodic_stokes_term,
        plane_explicit_term,
        plane_stokes_term,
    )

    assert [(item.kind, item.key, item.implementation, item.adjoint_mode) for item in CAPABILITIES] == EXPECTED_CAPABILITIES
    assert basis_signature(("periodic",) * 3) == "periodic^3"
    assert lookup("constraint", PeriodicStokesConstraint.key) is CAPABILITIES[2]
    assert [_resolve(item) for item in CAPABILITIES] == [
        plane_stokes_term, plane_explicit_term, periodic_stokes_term, periodic_explicit_term, pointwise_explicit_term,
        beris_edwards_linear_operator, expression_symbol]
    with pytest.raises(ValueError):
        _resolve(None)


def test_complete_stress_preset_remains_qualified_in_legacy_catalog():
    import pssolver
    from pssolver.presets.beris_edwards import CompleteStressBerisEdwards

    assert [(item.kind, item.key, item.implementation, item.adjoint_mode) for item in CAPABILITIES] == EXPECTED_CAPABILITIES
    catalog = json.dumps(pssolver.capability_catalog().to_metadata())
    assert "beris_edwards" in catalog and "periodic" in catalog and "qualified" in catalog
    assert CompleteStressBerisEdwards.Q.reference("Q").components == ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")


def test_all_nineteen_periodic_plans_match_installed_s25(tmp_path):
    from pssolver.configuration.legacy_periodic import translate_periodic
    from qualification.identity.runtime_case import complete_simulation
    from qualification.identity.scenarios import scenarios

    baseline = json.loads((Path(__file__).parent / "baselines/s26/periodic_plans.json").read_text())
    expected = {row["scene"]: row for row in baseline["scenes"]}
    actual = {}
    for scene in scenarios():
        if scene["family"] not in {"periodic", "functional_periodic"}:
            continue
        try:
            declaration = complete_simulation(scene, tmp_path).specification
            translation = translate_periodic(declaration, flavour=(
                "functional_1_1" if scene["family"] == "functional_periodic" else "production"),
                load_initial=False)
            row = dict(scene=scene["id"], identity_sha256=translation.plan.identity_sha256,
                       metadata=translation.plan.to_metadata())
        except (TypeError, ValueError) as error:
            row = dict(scene=scene["id"], rejected=type(error).__name__, message=str(error))
        actual[scene["id"]] = row
    assert len(actual) == 19
    assert actual == expected
