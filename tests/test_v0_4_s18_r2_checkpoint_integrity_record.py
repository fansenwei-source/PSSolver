from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s18_r2_checkpoint_integrity.json"


def test_r2_candidate_binds_the_confirmed_contract_and_source() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["plan_step"] == "S-18"
    assert record["batch"] == "R2"
    assert record["governance"] == {
        "decision": "D-0018",
        "design_review": "R-0054",
        "implementation_commit": "f83539e76cf027c37010205f03b7b4e4f599002e",
        "implementation_parent": "eb6a191f83ec7bc59ed41dde0b6d67dab57acc10",
        "comparison_baseline": "v0.3.0",
        "nonzero_tolerance_authorized": False,
    }
    assert record["publication_trigger"] == {
        "triggered": False,
        "reason": (
            "No target creation, staging placement, placeholder ownership, "
            "fsync, rename/renameat2, cleanup, symlink/device check, "
            "checkpoint_directory_transaction implementation, or publication "
            "call graph changed."
        ),
        "real_nfs_required_for_this_batch": False,
    }
    assert all(record["required_evidence"].values())
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_implementation": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }


def test_r2_private_owner_is_bound_and_not_publicly_exported() -> None:
    source = ROOT / "pssolver/io/_checkpoint_artifacts.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "15c65d1d77321a7f78505f945319a2575ae4f396ab80df35e422de8ffbb94d21"
    )
    public = (ROOT / "pssolver/io/__init__.py").read_text(encoding="utf-8")
    assert "_checkpoint_artifacts" not in public
