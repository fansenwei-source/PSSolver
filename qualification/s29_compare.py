"""Common-writer comparisons for unchanged S-29 execution contracts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(root, basis, old, new):
    authorized = json.loads((root / "qualification/s29_authority.json").read_bytes())
    changes = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", BASE],
            cwd=root,
            text=True,
        ).splitlines()
    )
    assert old.keys() == new.keys() == {"files", "schema_version"}
    assert old["schema_version"] == new["schema_version"]
    paths = sorted(old["files"])
    assert sorted(new["files"]) == paths
    assert old["files"] == {name: sha(basis / name) for name in paths}
    assert new["files"] == {name: sha(root / name) for name in paths}
    actual_diff = sorted(name for name in paths if sha(root / name) != sha(basis / name))
    assert actual_diff == sorted(set(paths) & changes & authorized["changed_units"].keys())
    return {"paths": paths, "changed": actual_diff}


def _hash(tensor):
    return [
        str(tensor.dtype),
        list(tensor.shape),
        hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest(),
    ]


def generic(root):
    sys.path.insert(0, str(root))
    import torch

    import pssolver
    from examples.allen_cahn import build
    from pssolver.planning.assemble import assemble
    from tests.mms.allen_cahn_mms import ForcedAllenCahn
    from tests.test_v0_5_s26_assembly import problem

    assert Path(pssolver.__file__).resolve().is_relative_to(root)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    results = {}
    for dtype in (torch.float64, torch.float32):
        for model in ("AllenCahn", "MMS", "S26Scalar"):
            for inplace in (False, True):
                if model == "S26Scalar":
                    engine = assemble(problem()).engine(device="cpu", dtype=dtype)
                else:
                    options = {"model": ForcedAllenCahn()} if model == "MMS" else {}
                    engine = build(dtype=dtype, **options)
                gen = torch.Generator(device="cpu").manual_seed(26)
                state = engine.initial_state(
                    {
                        name: 0.03 * torch.randn(engine.shape, dtype=dtype, generator=gen)
                        for name in engine.components[: engine.evolved_count]
                    }
                )
                rows = []
                for _ in range(20):
                    if inplace:
                        engine.step_(state)
                    else:
                        state = engine.step(state)
                    rows.append(
                        {
                            "spectral": _hash(state.spectral),
                            "spatial": _hash(state.spatial),
                            "step": state.completed_steps,
                            "refresh": state.refresh_phase,
                        }
                    )
                results[f"{dtype}/{model}/{inplace}"] = rows
    assert not torch.cuda.is_initialized()
    return results


def public_surface(root):
    sys.path.insert(0, str(root))
    import pssolver

    assert Path(pssolver.__file__).resolve().is_relative_to(root)
    assert not ({"torch", "numpy"} & sys.modules.keys())
    catalog = pssolver.capability_catalog().to_metadata()
    assert not ({"torch", "numpy"} & sys.modules.keys())
    prefix = (
        "import sys;sys.path.insert(0,sys.argv.pop(1));"
        "from pssolver.cli.plane import main;raise SystemExit(main())"
    )
    cli = {}
    for arguments in (
        ("--help",),
        ("--runtime-path", "compiled_v2"),
        ("--linear-tier", "modal_block_test"),
        ("--linear-tier", "matrix_free_test"),
        ("--integrator", "sbdf2_test"),
    ):
        for dry in (False, True):
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    prefix,
                    str(root),
                    *arguments,
                    *(("--dry-run",) if dry else ()),
                ],
                cwd=root,
                env={**os.environ, "LC_ALL": "C"},
                capture_output=True,
                check=False,
            )
            cli[" ".join(arguments) + f" dry={dry}"] = {
                "returncode": result.returncode,
                "stdout": result.stdout.decode(),
                "stderr": result.stderr.decode(),
            }
    return {"all": pssolver.__all__, "catalog": catalog, "cli": cli}


def plane_provenance(root, work):
    sys.path.insert(0, str(root))
    import pssolver
    from pssolver.runtime.simulation_run import build_run_metadata
    from qualification.identity.scenarios import scenarios
    from qualification.identity.scenes_v05 import simulation_v05

    assert Path(pssolver.__file__).resolve().is_relative_to(root)
    scene = next(
        row for row in scenarios() if row["id"] == "plane_legacy_production_float64_default"
    )
    metadata = build_run_metadata(pssolver.compile_simulation(simulation_v05(scene, work)))
    return metadata["implementation_provenance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generic", "public", "plane_provenance"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--work", type=Path, default=Path("/tmp/s29_comparison"))
    args = parser.parse_args()
    if args.mode == "plane_provenance":
        value = plane_provenance(args.root, args.work)
    else:
        value = (generic if args.mode == "generic" else public_surface)(args.root)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
