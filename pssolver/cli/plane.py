"""Historical Plane CLI through the common Engine/Runner entrance."""

from collections.abc import Sequence

from tqdm import trange

from pssolver.configuration import PlaneBerisEdwardsRunSpec, parse_plane_beris_edwards_run_spec
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.runtime.legacy_entry import run_v04_plane_translation
from pssolver.runtime.runner import RunResult


def parse_args(argv: Sequence[str] | None = None) -> PlaneBerisEdwardsRunSpec:
    """Delegate the production CLI to its immutable run specification."""

    return parse_plane_beris_edwards_run_spec(argv)


def _print_workflow_summary(workflow_result: RunResult) -> None:
    print(f"Elapsed time: {workflow_result.elapsed_seconds:.6f} seconds")
    if workflow_result.diagnostics:
        final_diagnostic = workflow_result.diagnostics[-1]
        print(
            "Final div(u) diagnostic: "
            f"max={final_diagnostic.div_max:.6e}, "
            f"rms={final_diagnostic.div_rms:.6e}, "
            f"relative={final_diagnostic.div_rel:.6e}"
        )
        print(
            "Final Schur saddle solve diagnostic: "
            f"iterations={int(final_diagnostic.schur_iterations)}, "
            f"abs_residual={final_diagnostic.schur_abs_residual:.6e}, "
            f"rel_residual={final_diagnostic.schur_rel_residual:.6e}"
        )
        print(
            "Final wall normal momentum diagnostic: "
            f"max={final_diagnostic.wall_normal_momentum_max:.6e}, "
            f"rms={final_diagnostic.wall_normal_momentum_rms:.6e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical production CLI without import-time side effects."""

    run_spec = parse_args(argv)
    components = decompose_plane_beris_edwards_run_spec(run_spec)
    progress = None if components.invocation.dry_run else trange(components.workflow.steps)
    result = run_v04_plane_translation(
        run_spec,
        progress=progress,
        emit_metadata=True,
    )
    if result is not None:
        _print_workflow_summary(result)
    return 0
