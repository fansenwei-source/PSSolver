"""Test-only negative control from the reachable, frozen R1 parent.

No alternate production entry or fallback is installed.  All source bytes are
read from the local Git history that a single-branch clone actually contains.
"""

from functools import lru_cache
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PARENT = "9e8b8809ca1154436993d921527a46e919b630a2"
MODULES = (
    "pssolver.configuration.simulation_lowering",
    "pssolver.configuration.simulation_binding",
    "pssolver.configuration.public_simulation_runner",
    "pssolver.runtime.simulation_construction",
    "pssolver.runtime.package_construction",
    "pssolver.api.capabilities",
)


@lru_cache(None)
def source(module):
    assert module in MODULES
    path = module.replace(".", "/") + ".py"
    return subprocess.check_output(
        ["git", "--no-optional-locks", "show", f"{PARENT}:{path}"],
        cwd=ROOT, text=True,
    )


@lru_cache(None)
def legacy(module):
    live = importlib.import_module(module)
    copy = ModuleType(module)
    copy.__package__ = module.rsplit(".", 1)[0]
    copy.__file__ = f"<R1-parent:{PARENT}:{module}>"
    # dataclasses consult sys.modules while definitions execute. Restore the
    # live import immediately; only the returned test namespace is retained.
    sys.modules[module] = copy
    try:
        exec(compile(source(module), copy.__file__, "exec"), vars(copy))
    finally:
        sys.modules[module] = live
    for name in (
        "LoweringRejectionCode", "LoweringRejection", "SimulationLoweringError",
        "BindingRejectionCode", "BindingRejection", "SimulationBindingError",
        "PublicCompilationRejectionCode", "PublicSimulationCompilationError",
        "PublicApplicationCompilerRegistration", "PublicSimulationCompilation",
        "PackageRuntimeConstructionInput", "QualifiedCombination",
    ):
        if hasattr(live, name):
            setattr(copy, name, getattr(live, name))
    if module.endswith("simulation_binding"):
        copy.lower_simulation_spec = legacy(MODULES[0]).lower_simulation_spec
    if module.endswith("public_simulation_runner"):
        copy.lower_simulation_spec = legacy(MODULES[0]).lower_simulation_spec
    return copy
