"""Forward correction for neutral R4 equivalence profile provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_99_s18_r4_equivalence_provenance.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_99_binds_neutral_profile_provenance_forward() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 99
    assert record["git_identity"] == {
        "qualification_helper_fix": "5a474652b2cb577a2cf97dc3642b4d872bd6dabf",
        "previous_helper_fix": "aebab348f8a570a5bbf1314efc1867e399047c4b",
        "runtime_implementation": "9cf54553b9f1bb2b94c2807c2e47adedeb415846",
        "record_commit_is_qualified_source": False,
    }
    relative, binding = next(iter(record["historical_successor_sources"].items()))
    assert binding["historical_sha256"] == (
        "3570adfea175a5485ef66dced5cc3f48565906b84cc55ea8cf37463f3cbc258f"
    )
    assert binding["current_sha256"] == (
        "01b934fa7042090f83f4f359ff74b8a84ae8a46f1d280949808bae1c746dd663"
    )
    assert _sha256(relative) == resolve_s03_successor(
        relative,
        binding["historical_sha256"],
    )


def test_successor_99_is_qualification_only_and_has_no_remote_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_qualification_helper_fix": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
