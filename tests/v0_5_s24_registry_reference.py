"""Read the immutable, tensor-free release registry without importing old runtime."""

import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType


S25_IDENTITY_MAP = {
    "pssolver.configuration.public_periodic_simulation_compiler.compile_periodic_public_simulation": "pssolver.configuration.legacy_periodic.compile_periodic_public_simulation",
    "pssolver.runtime.periodic_beris_edwards.PeriodicRuntimeBuildRequest": "pssolver.configuration.legacy_periodic.PeriodicTranslation",
    "pssolver.runtime.periodic_beris_edwards.build_periodic_beris_edwards_runtime": "pssolver.engine.engine.Engine",
    "pssolver.runtime.periodic_beris_edwards.PeriodicRuntimeAdapterProtocol": "pssolver.engine.engine.Engine",
}

S25_DELETED_REGISTRY_MODULES = (
    "pssolver.configuration.public_periodic_simulation_compiler.",
    "pssolver.runtime.periodic_beris_edwards.",
)

S27_IDENTITY_MAP = {
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeBuildRequest": "pssolver.planning.engine_plan.EnginePlan",
    "pssolver.runtime.plane_beris_edwards.build_plane_beris_edwards_runtime": "pssolver.engine.engine.Engine",
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeAdapterProtocol": "pssolver.engine.engine.Engine",
}

_S28_RUNTIME = ".".join(("pssolver", "runtime", "channel_beris_edwards")) + "."
_S28_COMPILER = ".".join(("pssolver", "configuration", "public_channel_beris_edwards_compiler")) + "."
S28_IDENTITY_MAP = {
    _S28_COMPILER + "compile_channel_beris_edwards_public_simulation":
        "pssolver.configuration.legacy_channel.compile_channel_beris_edwards_public_simulation",
    _S28_RUNTIME + "ChannelBerisEdwardsRuntimeBuildRequest":
        "pssolver.configuration.legacy_channel.ChannelBerisEdwardsRunSpec",
    _S28_RUNTIME + "build_channel_beris_edwards_runtime": "pssolver.engine.engine.Engine",
    _S28_RUNTIME + "ChannelBerisEdwardsRuntimeAdapterProtocol": "pssolver.engine.engine.Engine",
}


def _forward_channel_value(value, *, channel):
    """Forward only the four reviewed identities of complete-stress Channel."""
    if isinstance(value, str) and value.startswith((_S28_RUNTIME, _S28_COMPILER)):
        assert channel, "deleted Channel identity in another application"
        assert value in S28_IDENTITY_MAP, f"unmapped Channel identity: {value}"
        return S28_IDENTITY_MAP[value]
    if isinstance(value, dict):
        return {key: _forward_channel_value(item, channel=channel) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_forward_channel_value(item, channel=channel) for item in value)
    return value


def _forward_plane_value(value, *, plane):
    """Closed S-27 mapping; unknown retired identities and wrong families fail."""
    if isinstance(value, str):
        if value.startswith("pssolver.runtime.plane_beris_edwards."):
            assert plane, "deleted Plane identity in another application"
            assert value in S27_IDENTITY_MAP, f"unmapped Plane identity: {value}"
            return S27_IDENTITY_MAP[value]
        return value
    if isinstance(value, dict):
        result = {key: _forward_plane_value(item, plane=plane) for key, item in value.items()}
        if plane and "runtime_accepts_builders" in result:
            assert result["runtime_accepts_builders"] is True
            assert result["runtime_request_error"] == "Plane construction requires PlaneRuntimeBuildRequest"
            result["runtime_accepts_builders"] = False
            result["runtime_request_error"] = "Plane construction requires EnginePlan"
        return result
    if isinstance(value, tuple):
        result = tuple(_forward_plane_value(item, plane=plane) for item in value)
        if plane and result and result[0] == "plane_compiled_v2":
            assert result[-2:] == ("caller", "compiled_builder")
            result = (*result[:-2], "package", None)
        return result
    return value


def _forward_value(value, application):
    value = _forward_plane_value(
        _forward_periodic_value(value, periodic=application.geometry_name == "periodic_box"),
        plane=application.geometry_name == "plane_slab",
    )
    return _forward_channel_value(value, channel=application.lowering_kind == "channel_complete")


def _forward_periodic_value(value, *, periodic):
    """Translate only the closed Periodic identity set; never pass a missing key."""
    if isinstance(value, str):
        if value.startswith(S25_DELETED_REGISTRY_MODULES):
            assert periodic, "deleted Periodic identity in another application"
            assert value in S25_IDENTITY_MAP, f"unmapped Periodic identity: {value}"
            return S25_IDENTITY_MAP[value]
        return value
    if isinstance(value, dict):
        return {key: _forward_periodic_value(item, periodic=periodic) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_forward_periodic_value(item, periodic=periodic) for item in value)
    return value


def _validated_registry():
    registry = release_registry()
    for item in registry.APPLICATIONS:
        _forward_value(asdict(item), item)
    return registry


@lru_cache(maxsize=1)
def release_registry():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        [
            "git",
            "show",
            "4c12228aba01e8483e80d28a1471ab72384d5fda:"
            "pssolver/configuration/application_registry.py",
        ],
        cwd=root,
        text=True,
    )
    name = "_s24_released_registry_reference"
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "<v0.4.0-registry>", "exec"), vars(module))
    return module


def retained_application_metadata():
    return tuple(
        _forward_value(item.compiler_metadata(), item)
        for item in _validated_registry().APPLICATIONS
        if item.lowering_kind != "channel_legacy"
    )


def retained_runtime_bindings():
    return {
        item.key: _forward_value(item.binding_fields(), item.application)
        for item in _validated_registry().RUNTIMES
        if item.application.lowering_kind != "channel_legacy"
        and item.runtime_path != "separated_canary"
    }


def retained_application_records():
    records = []
    for item in _validated_registry().APPLICATIONS:
        if item.lowering_kind == "channel_legacy":
            continue
        record = _forward_value(asdict(item), item)
        record["runtime_paths"] = tuple(
            name for name in record["runtime_paths"] if name != "separated_canary"
        )
        records.append(record)
    return records
