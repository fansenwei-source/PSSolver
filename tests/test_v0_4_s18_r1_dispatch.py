"""Exact old/new selection, rejection order and construction contracts."""

from copy import copy
from dataclasses import replace

from pssolver.configuration import application_registry as registry
from pssolver.configuration import public_simulation_runner as compiler
from pssolver.configuration.simulation import TimeIntegrationSpec
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.core.integrators import IntegratorSpec


def _outcome(call):
    try:
        result = call()
    except Exception as exc:
        record = {"type": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "rejection"):
            record["rejection"] = exc.rejection.to_metadata()
        if hasattr(exc, "context"):
            record["context"] = exc.context
        if hasattr(exc, "code"):
            record["code"] = exc.code
        return "raised", record
    return "returned", result.to_metadata() if hasattr(result, "to_metadata") else result


def _mutate(spec, case):
    if case == "valid":
        return spec
    if case == "type":
        return None
    if case == "model":
        return replace(spec, equation_system=replace(spec.equation_system, variant="unknown"))
    if case in {"runtime", "backend"}:
        return replace(spec, execution=replace(spec.execution, **{
            "runtime_path" if case == "runtime" else "backend": "unknown",
        }))
    if case == "capability":
        return replace(spec, equation_system=replace(spec.equation_system,
            evolution_laws=(replace(spec.equation_system.evolution_laws[0], capability="unknown"),)))
    if case == "time":
        return replace(spec, time_integration=TimeIntegrationSpec(
            integrator=IntegratorSpec.sbdf2(dt=spec.time_integration.integrator.dt)))
    assert case == "topology"
    bad = copy(spec)
    geometry = spec.geometry
    object.__setattr__(bad, "geometry", GeometrySpec(geometry.name, geometry.domain,
        tuple(AxisTopology.BOUNDED if item is AxisTopology.PERIODIC
              else AxisTopology.PERIODIC for item in geometry.axis_topologies)))
    return bad


def test_registry_discovery_matches_parent_byte_order():
    import json
    from dataclasses import asdict

    from tests.v0_5_s24_registry_reference import (
        retained_application_metadata,
        retained_application_records,
    )

    assert json.dumps(compiler.public_compiler_capabilities()) == json.dumps(
        retained_application_metadata()
    )
    assert [asdict(item) for item in registry.APPLICATIONS] == retained_application_records()
