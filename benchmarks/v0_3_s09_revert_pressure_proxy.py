#!/usr/bin/env python3
"""Exact diagnostic proxy for the pre-2573dec Channel pressure solve.

The installed candidate wheel is not edited.  Instead, this support-only
module reconstructs the ``f94b9be`` implementation of ``_solve_pressure``
from the installed ``85ba3ae`` source, verifies the exact method SHA-256, and
compiles it in the installed module namespace for the duration of one child
process.  Every replacement is fail-closed and must match exactly once.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import inspect
import textwrap
from typing import Iterator


REVERT_COMMIT = "f94b9be22fd767fee432caf30f0e5c47aad8e9c3"
REVERT_SOURCE_PATH = "pssolver/linear_solvers/stokes/channel_no_slip.py"
REVERT_SOURCE_FILE_SHA256 = (
    "c539275fa2a4992d67aeadd0ddfac3bebd53d73bf6af478a3f93d963d54d4bf4"
)
REVERT_METHOD_SHA256 = (
    "2ffb5673ebdb92525b8959ee014aa3178b91cdf37789bdbd03d3cf2a5654e8f3"
)
CANDIDATE_METHOD_SHA256 = (
    "43316e0c3919868deb2895f075be65da79d4b0e114e3ff7a45c1c2b3ab4c4116"
)

_CACHE_FIELDS = (
    "_pressure_guess_operator_action",
    "_pressure_guess_operator_action_guess",
    "_pressure_guess_operator_action_guess_version",
    "_pressure_guess_operator_action_working_exponent",
)

# Exact, ordered reverse patch from 85ba3ae to f94b9be.  Keeping the patch
# small makes it auditable while the final SHA proves that it reconstructs the
# complete 691-line historical method, not an approximation of its behavior.
_REPLACEMENTS = (
    (
        "        self._store_pressure_guess(pressure_hat)\n",
        "        self.pressure_guess = pressure_hat.detach()\n",
        1,
    ),
    (
        "    pressure_guess = self.pressure_guess\n"
        "    if pressure_guess is None or pressure_guess.shape != rhs_hat.shape:\n"
        "        self._clear_pressure_guess_operator_action()\n",
        "    if self.pressure_guess is None or self.pressure_guess.shape != rhs_hat.shape:\n",
        1,
    ),
    (
        "        pressure_operator_action = self._pressure_operator(pressure_work)\n",
        "",
        1,
    ),
    (
        "                pressure_guess.to(\n",
        "                self.pressure_guess.to(\n",
        1,
    ),
    (
        "        pressure_operator_action = self._cached_pressure_operator_action(\n"
        "            pressure_guess,\n"
        "            pressure_work,\n"
        "            int(working_exponent),\n"
        "        )\n"
        "        if pressure_operator_action is None:\n"
        "            pressure_operator_action = self._pressure_operator(\n"
        "                pressure_work\n"
        "            )\n",
        "",
        1,
    ),
    (
        "    residual = working_rhs - pressure_operator_action\n",
        "    residual = working_rhs - self._pressure_operator(pressure_work)\n",
        1,
    ),
    (
        "        self._store_pressure_guess(\n"
        "            pressure_hat,\n"
        "            operator_action=pressure_operator_action,\n"
        "            working_exponent=working_exponent,\n"
        "        )\n",
        "        self.pressure_guess = pressure_hat.detach()\n",
        2,
    ),
    (
        "    acceptable = False\n"
        "    accepted_pressure_operator_action = None\n"
        "    fixed_work_floor_pressure = None\n",
        "    acceptable = False\n"
        "    fixed_work_floor_pressure = None\n",
        1,
    ),
    (
        "        accepted_pressure_operator_action = None\n"
        "        residual = self._project_pressure_gauge(\n",
        "        residual = self._project_pressure_gauge(\n",
        1,
    ),
    (
        "            accepted_pressure_operator_action = self._pressure_operator(\n"
        "                pressure_work\n"
        "            )\n",
        "",
        1,
    ),
    (
        "                working_rhs - accepted_pressure_operator_action\n",
        "                working_rhs - self._pressure_operator(pressure_work)\n",
        1,
    ),
    (
        "            rz_old = rz_new\n"
        "            accepted_pressure_operator_action = None\n"
        "            continue\n",
        "            rz_old = rz_new\n"
        "            continue\n",
        1,
    ),
    (
        "    self._store_pressure_guess(\n"
        "        pressure_hat,\n"
        "        operator_action=accepted_pressure_operator_action,\n"
        "        working_exponent=working_exponent,\n"
        "    )\n",
        "    self.pressure_guess = pressure_hat.detach()\n",
        1,
    ),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reconstructed_revert_source() -> str:
    """Return the exact historical method or fail before executing it."""

    module = importlib.import_module(
        "pssolver.linear_solvers.stokes.channel_no_slip"
    )
    cls = module.ChannelNoSlipModalStokesSolver
    source = textwrap.dedent(inspect.getsource(cls._solve_pressure))
    actual = _sha256_text(source)
    if actual != CANDIDATE_METHOD_SHA256:
        raise RuntimeError(
            "candidate _solve_pressure source differs before revert proxy: "
            f"{actual} != {CANDIDATE_METHOD_SHA256}"
        )
    for current, historical, expected_count in _REPLACEMENTS:
        count = source.count(current)
        if count != expected_count:
            raise RuntimeError(
                "revert proxy source anchor count differs: "
                f"expected {expected_count}, observed {count}: {current[:80]!r}"
            )
        source = source.replace(current, historical, expected_count)
    actual = _sha256_text(source)
    if actual != REVERT_METHOD_SHA256:
        raise RuntimeError(
            "reconstructed f94b9be _solve_pressure SHA differs: "
            f"{actual} != {REVERT_METHOD_SHA256}"
        )
    return source


def _empty_proof(role: str) -> dict[str, object]:
    return {
        "diagnostic_role": role,
        "patch_applied": False,
        "method_calls": 0,
        "cache_fields_nonempty_after_call": 0,
        "cache_fields_empty_after_every_call": False,
        "revert_commit": REVERT_COMMIT,
        "revert_source_path": REVERT_SOURCE_PATH,
        "revert_source_file_sha256": REVERT_SOURCE_FILE_SHA256,
        "candidate_method_sha256": CANDIDATE_METHOD_SHA256,
        "revert_method_sha256": REVERT_METHOD_SHA256,
        "compiled_in_candidate_module_namespace": False,
        "proved_revert_proxy": False,
    }


@contextmanager
def use_revert_pressure_proxy(role: str) -> Iterator[dict[str, object]]:
    """Temporarily install the exact f94b9be pressure method for one child."""

    proof = _empty_proof(role)
    if role != "candidate_revert_proxy":
        yield proof
        return

    module = importlib.import_module(
        "pssolver.linear_solvers.stokes.channel_no_slip"
    )
    cls = module.ChannelNoSlipModalStokesSolver
    original = cls._solve_pressure
    source = reconstructed_revert_source()
    namespace: dict[str, object] = {}
    exec(
        compile(
            source,
            f"<{REVERT_COMMIT}:{REVERT_SOURCE_PATH}:_solve_pressure>",
            "exec",
        ),
        module.__dict__,
        namespace,
    )
    historical = namespace.get("_solve_pressure")
    if not callable(historical):
        raise RuntimeError("revert proxy compilation did not define _solve_pressure")

    def audited_historical(self, *args, **kwargs):
        result = historical(self, *args, **kwargs)
        proof["method_calls"] = int(proof["method_calls"]) + 1
        if any(getattr(self, name, None) is not None for name in _CACHE_FIELDS):
            proof["cache_fields_nonempty_after_call"] = (
                int(proof["cache_fields_nonempty_after_call"]) + 1
            )
        return result

    cls._solve_pressure = audited_historical
    proof["patch_applied"] = True
    proof["compiled_in_candidate_module_namespace"] = True
    try:
        yield proof
    finally:
        cls._solve_pressure = original
        proof["cache_fields_empty_after_every_call"] = bool(
            int(proof["method_calls"]) > 0
            and int(proof["cache_fields_nonempty_after_call"]) == 0
        )
        proof["proved_revert_proxy"] = bool(
            proof["patch_applied"]
            and proof["compiled_in_candidate_module_namespace"]
            and proof["cache_fields_empty_after_every_call"]
        )


__all__ = [
    "CANDIDATE_METHOD_SHA256",
    "REVERT_COMMIT",
    "REVERT_METHOD_SHA256",
    "REVERT_SOURCE_FILE_SHA256",
    "REVERT_SOURCE_PATH",
    "reconstructed_revert_source",
    "use_revert_pressure_proxy",
]
