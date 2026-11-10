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
        _forward_periodic_value(asdict(item), periodic=item.geometry_name == "periodic_box")
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
        _forward_periodic_value(item.compiler_metadata(), periodic=item.geometry_name == "periodic_box")
        for item in _validated_registry().APPLICATIONS
        if item.lowering_kind != "channel_legacy"
    )


def retained_runtime_bindings():
    return {
        item.key: _forward_periodic_value(item.binding_fields(), periodic=item.application.geometry_name == "periodic_box")
        for item in _validated_registry().RUNTIMES
        if item.application.lowering_kind != "channel_legacy"
        and item.runtime_path != "separated_canary"
    }


def retained_application_records():
    records = []
    for item in _validated_registry().APPLICATIONS:
        if item.lowering_kind == "channel_legacy":
            continue
        record = _forward_periodic_value(asdict(item), periodic=item.geometry_name == "periodic_box")
        record["runtime_paths"] = tuple(
            name for name in record["runtime_paths"] if name != "separated_canary"
        )
        records.append(record)
    return records
