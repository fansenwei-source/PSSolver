"""Read the immutable, tensor-free release registry without importing old runtime."""

import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType


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
        item.compiler_metadata()
        for item in release_registry().APPLICATIONS
        if item.lowering_kind != "channel_legacy"
    )


def retained_runtime_bindings():
    return {
        item.key: item.binding_fields()
        for item in release_registry().RUNTIMES
        if item.application.lowering_kind != "channel_legacy"
        and item.runtime_path != "separated_canary"
    }


def retained_application_records():
    records = []
    for item in release_registry().APPLICATIONS:
        if item.lowering_kind == "channel_legacy":
            continue
        record = asdict(item)
        record["runtime_paths"] = tuple(
            name for name in record["runtime_paths"] if name != "separated_canary"
        )
        records.append(record)
    return records
