"""Derive public capability discovery from real, tensor-free assembly probes."""

from dataclasses import asdict, dataclass
from itertools import product

from pssolver.boundaries import (
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
    prescribed_dirichlet,
    robin,
)
from pssolver.geometries.public import PeriodicBox, PlaneSlab, RectangularChannel

from .declarations import (
    GeneratedInitialCondition,
    Output,
    SemiImplicitEuler,
    SnapshotInitialCondition,
    SpectralNumerics,
    TorchSpectralExecution,
)
from .errors import PublicSimulationCompilationError
from .models import CompleteStressBerisEdwards, ShendrukPlaneBerisEdwards
from .runner import compile_simulation
from .simulation import Simulation


@dataclass(frozen=True, slots=True)
class DeclarationCapability:
    """One declaration and whether any assembly probe accepts it."""

    kind: str
    key: str
    constructor: str
    executable: bool

    def to_metadata(self) -> dict[str, object]:
        return {**asdict(self), "declarable": True}


@dataclass(frozen=True, slots=True)
class QualifiedCombination:
    """The assembly verdict for a model, geometry and boundary policy set."""

    equation_variant: str
    geometry_name: str
    boundary_policies: tuple[str, ...]
    qualified: bool
    kind: str | None = None
    code: str | None = None
    reason: str | None = None
    opens_in: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {**asdict(self), "boundary_policies": list(self.boundary_policies)}


@dataclass(frozen=True, slots=True)
class PublicCapabilityCatalog:
    """Public vocabulary and the combinations determined by the assembler."""

    models: tuple[DeclarationCapability, ...]
    geometries: tuple[DeclarationCapability, ...]
    boundary_policies: tuple[DeclarationCapability, ...]
    qualified_combinations: tuple[QualifiedCombination, ...]
    rejected_combinations: tuple[QualifiedCombination, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            **{
                name: [item.to_metadata() for item in getattr(self, name)]
                for name in (
                    "models", "geometries", "boundary_policies",
                    "qualified_combinations", "rejected_combinations",
                )
            },
        }


def _model_vocab():
    return {
        "complete_stress_beris_edwards": CompleteStressBerisEdwards(
            ldg_a=0.0, ldg_b=-0.3, ldg_c=0.3, ldg_l1=1 / 81, gamma=2.94,
            flow_alignment=0.3, activity=0.01, beta=-1.0, viscosity=2 / 3,
        ),
        "shendruk_plane_beris_edwards": ShendrukPlaneBerisEdwards(activity_number=12.0),
    }


def _boundary_vocab():
    face = ("Qxx", 2, "lower")
    return {
        "neumann_q": neumann_q(),
        "prescribed_q": prescribed_dirichlet("Q", {face: 1.0}),
        "robin": robin("Q", {face: (1.0, 1.0, 1.0)}),
        "free_slip_velocity": free_slip_velocity(),
        "no_slip_velocity": no_slip_velocity(),
        "neumann_pressure_compatibility": neumann_pressure_compatibility(),
    }


def _probe_inputs(model, geometry, boundaries):
    kind = geometry.name
    channel = kind == "rectangular_channel"
    pressure = {
        "algorithm": "preconditioned_conjugate_gradient", "fixed_iterations": 12,
        "max_iterations": 40, "relative_tolerance": 1e-10, "warm_start": True,
    } if channel else None
    initial = (
        GeneratedInitialCondition("extruded_defect_gas")
        if kind == "plane_slab"
        else SnapshotInitialCondition(".", step=0)
    )
    return Simulation(
        model, geometry, boundaries,
        SpectralNumerics(dtype="float64", spectral_storage="full_complex", pressure_solver=pressure),
        SemiImplicitEuler(0.01), initial,
        TorchSpectralExecution(options={
            "stress_divergence_sum_space": "physical" if channel else "spectral",
        }),
        Output(directory=".", steps=1, save_interval=1, diagnostic_interval=1, save_start_step=0,
               diagnostics=True, save_hydrodynamics=False, dry_run=True),
    )


def capability_catalog() -> PublicCapabilityCatalog:
    """Probe the Cartesian product, retaining structured assembly rejections."""
    models = _model_vocab()
    geometries = {
        g.name: g for g in (
            factory(shape=(16, 16, 16), lengths=(16.0, 16.0, 16.0))
            for factory in (PeriodicBox, PlaneSlab, RectangularChannel)
        )
    }
    policies = _boundary_vocab()
    by_field = [
        tuple(key for key, policy in policies.items() if policy.field_name == field)
        for field in ("Q", "velocity", "pressure")
    ]
    accepted, rejected = [], []
    for (model_key, model), geometry, keys in product(
        models.items(), geometries.values(), tuple(product(*by_field)),
    ):
        boundaries = {policies[key].field_name: policies[key] for key in keys}
        declaration = _probe_inputs(model, geometry, boundaries)
        try:
            compile_simulation(declaration)
        except PublicSimulationCompilationError as exc:
            rejected.append(QualifiedCombination(
                model_key, geometry.name, keys, False, exc.kind, exc.code.value, exc.reason, exc.opens_in,
            ))
        else:
            accepted.append(QualifiedCombination(model_key, geometry.name, keys, True))
    return PublicCapabilityCatalog(
        tuple(DeclarationCapability(
            "model", key, f"pssolver.{type(model).__name__}",
            any(row.equation_variant == key for row in accepted),
        ) for key, model in models.items()),
        tuple(DeclarationCapability(
            "geometry", key, f"pssolver.{type(geometry).__name__}",
            any(row.geometry_name == key for row in accepted),
        ) for key, geometry in geometries.items()),
        tuple(DeclarationCapability(
            "boundary_policy", key, f"pssolver.boundaries.{key}",
            any(key in row.boundary_policies for row in accepted),
        ) for key in policies),
        tuple(accepted), tuple(rejected),
    )


def available_models() -> tuple[DeclarationCapability, ...]:
    return capability_catalog().models


def available_geometries() -> tuple[DeclarationCapability, ...]:
    return capability_catalog().geometries


def available_boundary_policies() -> tuple[DeclarationCapability, ...]:
    return capability_catalog().boundary_policies


def available_combinations() -> tuple[QualifiedCombination, ...]:
    return capability_catalog().qualified_combinations


__all__ = [
    "DeclarationCapability", "PublicCapabilityCatalog", "QualifiedCombination",
    "available_boundary_policies", "available_combinations", "available_geometries",
    "available_models", "capability_catalog",
]
