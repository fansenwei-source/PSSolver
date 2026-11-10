"""Explicit S-25 source relocation for forward historical hash checks."""

from pathlib import Path


SOURCE_PATHS = {
    "pssolver/configuration/periodic_beris_edwards.py": "pssolver/configuration/legacy_periodic.py",
    "pssolver/configuration/public_periodic_simulation_compiler.py": "pssolver/configuration/legacy_periodic.py",
    "pssolver/applications/periodic_beris_edwards.py": "pssolver/runtime/legacy_entry.py",
    "pssolver/runtime/periodic_beris_edwards.py": "pssolver/models/active_nematics/engine_terms.py",
    "pssolver/workflows/periodic_beris_edwards.py": "pssolver/runtime/runner.py",
    "pssolver/workflows/periodic_checkpoint.py": "pssolver/io/legacy/periodic_workflow.py",
    "pssolver/execution/state.py": "pssolver/engine/progress.py",
    "pssolver/execution/workspace.py": "pssolver/engine/workspace.py",
    "pssolver/workflows/diagnostic_io.py": "pssolver/io/diagnostic_io.py",
    "pssolver/workflows/state_validation.py": "pssolver/runtime/state_validation.py",
}


def source_path(root: Path, relative: str) -> Path:
    """Resolve only the reviewed exact paths, not arbitrary missing files."""
    return root / SOURCE_PATHS.get(relative, relative)
