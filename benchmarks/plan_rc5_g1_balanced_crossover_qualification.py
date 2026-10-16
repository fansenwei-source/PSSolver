#!/usr/bin/env python3
"""Build the formally adopted rc5 G-1 balanced-crossover contract.

Formal adoption freezes estimator semantics.  It does not itself authorize a
push or an H100 submission.  Execution additionally requires a regular-file,
source-bound authorization record for exactly one non-retrying H100 job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_balanced_crossover_contract import (
    build_plan as build_proposal,
)


SCHEMA = "pssolver.rc5.g1.balanced_crossover_qualification_contract.v1"
CLASSIFICATION = "READY_FOR_G1_BALANCED_CROSSOVER_H100_AUTHORIZATION"


def build_plan() -> dict[str, object]:
    proposal = build_proposal()
    result = dict(proposal)
    result.update(
        {
            "schema": SCHEMA,
            "classification": CLASSIFICATION,
            "purpose": (
                "formally adopt the reviewed balanced-crossover estimator "
                "while keeping push and H100 execution fail-closed"
            ),
            "adoption": {
                "proposal_schema": proposal["schema"],
                "proposal_classification": proposal["classification"],
                "proposal_record": (
                    "notes/PSSolver_v0_2_0rc5_g1_"
                    "balanced_crossover_contract_v14.json"
                ),
                "proposal_hpcc_archive": (
                    "/home/fansenwei/pssolver_rc5_g1_balanced_crossover_"
                    "replay_dabeb9a_20261016_v15"
                ),
                "proposal_archive_manifest_sha256": (
                    "bb139129d5980d9d0fae95ae00275464d7cb0c9551ec49937c2a288a98a9c073"
                ),
                "v10_v13_frozen_replay_passed": True,
                "formal_v10_classification_preserved": True,
                "formal_v13_classification_preserved": True,
                "formal_v13_pass_granted": False,
                "new_formal_h100_evidence_required": True,
            },
            "authorization": {
                "cpu_only_contract_adoption_authorized": True,
                "local_qualification_support_implementation_authorized": True,
                "formal_contract_adoption_authorized": True,
                "source_bound_execution_authorization_required": True,
                "push_authorized": False,
                "h100_submission_authorized": False,
                "maximum_h100_submissions_without_new_authorization": 0,
                "automatic_retry_authorized": False,
            },
        }
    )
    return result


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _atomic_json(args.output, build_plan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
