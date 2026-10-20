import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts_plane.create_v3_mother_manifest import create_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path):
    checkpoint = tmp_path / "Q2D_100.npy"
    np.save(checkpoint, np.zeros((8, 7, 5), dtype=np.float64))
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps({
            "candidate_2d_statistical_steady_state": True,
            "parameters": {"frank_k": 0.02, "zeta": 0.03},
            "recommended_checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
            },
        })
    )
    return checkpoint, report


def test_create_v3_manifest_binds_qualified_checkpoint_read_only(tmp_path):
    checkpoint, report = _inputs(tmp_path)
    before = (_sha256(checkpoint), _sha256(report))

    manifest = create_manifest(
        checkpoint_path=checkpoint,
        qualification_report_path=report,
        frank_k=0.02,
        zeta=0.03,
    )

    assert manifest["protocol"] == "V3"
    assert manifest["qualified"] is True
    assert manifest["checkpoint"]["sha256"] == before[0]
    assert manifest["qualification_report"]["sha256"] == before[1]
    assert manifest["parameters"]["zeta_over_k"] == pytest.approx(1.5)
    assert (_sha256(checkpoint), _sha256(report)) == before


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value.update(candidate_2d_statistical_steady_state=False), "true"),
        (lambda value: value["parameters"].update(zeta=0.04), "does not match"),
        (lambda value: value["recommended_checkpoint"].update(sha256="0" * 64), "SHA-256"),
    ],
)
def test_create_v3_manifest_rejects_unqualified_or_mismatched_input(
    tmp_path,
    mutation,
    match,
):
    checkpoint, report = _inputs(tmp_path)
    value = json.loads(report.read_text())
    mutation(value)
    report.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=match):
        create_manifest(
            checkpoint_path=checkpoint,
            qualification_report_path=report,
            frank_k=0.02,
            zeta=0.03,
        )
