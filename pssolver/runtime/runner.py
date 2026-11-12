"""Scheduling and output ownership for explicit engine states."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pssolver.io import engine_state
from pssolver.io.diagnostic_io import diagnostics_metadata_envelope, write_structured_diagnostics
from pssolver.runtime.state_validation import require_finite_tensors


@dataclass(frozen=True, slots=True)
class RunSchedule:
    steps: int
    save_interval: int
    save_start_step: int
    diagnostics: bool
    diagnostic_interval: int
    checkpoint_interval: int | None


@dataclass(frozen=True, slots=True)
class RunResult:
    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: object
    diagnostics: tuple[object, ...]


def _save_observation(directory, observation, *, hydrodynamics):
    values = {"Q": observation.q}
    if hydrodynamics:
        values.update(u=observation.velocity, p=observation.pressure)
    for prefix, value in values.items():
        path = directory / f"{prefix}_{observation.step}.npy"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)


class Runner:
    def __init__(self, engine, *, schedule, output_directory, capture, metadata):
        self.engine = engine
        self.schedule = schedule
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.capture = capture
        self.metadata = dict(metadata)
        self.saved = []
        self.checkpoints = []
        self.diagnostics = []
        self.constraint_diagnostics = {}

    def _prepare(self, state):
        diagnostics = {}
        self.engine.prepare_constraints_(state, diagnostics=diagnostics)
        if diagnostics:
            self.constraint_diagnostics = diagnostics
        assert state.constrained_current

    def _diagnostic(self, state, step):
        assert state.constrained_current
        extra = {}
        if getattr(self.capture, "wants_constraint_diagnostics", False):
            extra["constraint_diagnostics"] = self.constraint_diagnostics
        return self.capture.diagnostic(self.engine.field_view(state), step=step, **extra)

    def run(self, state, *, start_step, progress=None):
        if not hasattr(self.capture, "begin_run"):
            return self._run(state, start_step=start_step, progress=progress)
        self.capture.begin_run(self.output_directory, self.metadata)
        started = time.time()
        try:
            return self._run(state, start_step=start_step, progress=progress)
        except BaseException as error:
            self.capture.record_failure(
                self.output_directory, self.metadata, error, state=state,
                saved=self.saved, checkpoints=self.checkpoints, elapsed=time.time() - started,
            )
            raise

    def _run(self, state, *, start_step, progress=None):
        start = start_step
        steps = self.schedule.steps
        if self.schedule.save_start_step > start + steps:
            raise ValueError("save_start_step must not exceed the absolute final step")
        # A rejected restart must not claim its output path.  In particular,
        # checkpoint integrity and compatibility are established above before
        # the first filesystem mutation owned by this run.
        self.output_directory.mkdir(parents=True, exist_ok=True)
        iterator = range(steps) if progress is None else progress
        started = time.time()
        executed = 0
        for local_step in iterator:
            if local_step != executed:
                raise ValueError(getattr(
                    self.capture, "progress_order_error",
                    "progress must yield consecutive local steps starting at zero",
                ))
            absolute = start + local_step
            diagnostic_due = (
                self.schedule.diagnostics and absolute % self.schedule.diagnostic_interval == 0
            )
            save_due = (
                absolute >= self.schedule.save_start_step
                and absolute % self.schedule.save_interval == 0
            )
            if diagnostic_due or save_due:
                self._prepare(state)
            if diagnostic_due:
                value = self._diagnostic(state, absolute)
                self.diagnostics.append(value)
                if hasattr(self.capture, "progress_diagnostic"):
                    self.capture.progress_diagnostic(iterator, value)
            if save_due:
                observation = self.capture.observation(self.engine.field_view(state), step=absolute)
                _save_observation(
                    self.output_directory,
                    observation,
                    hydrodynamics=self.capture.hydrodynamics,
                )
                self.saved.append(absolute)
            diagnostics = {}
            self.engine.step_(state, diagnostics=diagnostics)
            if diagnostics:
                self.constraint_diagnostics = diagnostics
            executed += 1
            interval = self.schedule.checkpoint_interval
            if interval is not None and state.completed_steps % interval == 0:
                fields = self.engine.field_view(state)
                names = self.engine.components[: self.engine.evolved_count]
                require_finite_tensors(
                    (
                        *((name, fields[name]) for name in names),
                        *((f"{name}.hat", fields[f"{name}.hat"]) for name in names),
                    ),
                    application_family=self.capture.application_family,
                    stage="checkpoint",
                    step=state.completed_steps,
                )
                self._prepare(state)
                engine_state.write(
                    self.output_directory / f"checkpoint_{state.completed_steps}",
                    state,
                    plan_identity_sha256=self.engine.plan.identity_sha256,
                    plan_metadata=self.engine.plan.to_metadata(),
                    refresh_interval=self.engine.plan.integrator.refresh_interval,
                    run_provenance=(
                        self.capture.checkpoint_provenance(self.metadata)
                        if hasattr(self.capture, "checkpoint_provenance")
                        else {
                            "legacy_configuration_sha256": self.metadata["configuration_sha256"],
                            "legacy_runtime_identity_sha256": self.metadata["runtime_identity_sha256"],
                        }
                    ),
                )
                self.checkpoints.append(state.completed_steps)
        if executed != steps:
            raise ValueError(getattr(
                self.capture, "progress_length_error",
                f"progress yielded {executed} steps; expected exactly {steps}",
            ))
        final_step = start + steps
        self._prepare(state)
        final = self.capture.observation(self.engine.field_view(state), step=final_step)
        if final_step not in self.saved:
            _save_observation(
                self.output_directory,
                final,
                hydrodynamics=self.capture.hydrodynamics,
            )
            self.saved.append(final_step)
        if self.schedule.diagnostics:
            self.diagnostics.append(
                self._diagnostic(state, final_step)
            )
            write_structured_diagnostics(
                self.output_directory,
                [value.as_tuple() for value in self.diagnostics],
                dtype=self.capture.diagnostic_dtype,
                header=self.capture.diagnostic_header,
            )
        elapsed = time.time() - started
        finalize = getattr(self.capture, "finalize_metadata", None)
        completion = dict(status="complete", completed_steps=final_step, elapsed_seconds=elapsed)
        if finalize is None:
            completion.update(saved_steps=self.saved, checkpoint_steps=self.checkpoints)
        self.metadata.update(
            **completion,
            diagnostics=diagnostics_metadata_envelope(
                application_family=self.capture.application_family,
                runtime_path="engine",
                enabled=self.schedule.diagnostics,
                steps=[value.step for value in self.diagnostics],
                dtype=self.capture.diagnostic_dtype,
            ),
        )
        if finalize is not None:
            finalize(self.metadata, state=state, saved=self.saved, checkpoints=self.checkpoints)
        (self.output_directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
        return RunResult(
            start,
            final_step,
            elapsed,
            tuple(self.saved),
            tuple(self.checkpoints),
            final,
            tuple(self.diagnostics),
        )
