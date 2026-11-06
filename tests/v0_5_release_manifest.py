"""Historical distribution assertions read the released Git object, never HEAD."""

import subprocess
from pathlib import Path


RELEASE = "4c12228aba01e8483e80d28a1471ab72384d5fda"


def released_manifest(root: Path) -> str:
    """Fail explicitly when the immutable historical prerequisite is absent."""
    result = subprocess.run(
        ["git", "show", f"{RELEASE}:MANIFEST.in"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("PSSOLVER_TEST_PREREQUISITE: full Git history (v0.4.0 manifest)")
    return result.stdout
