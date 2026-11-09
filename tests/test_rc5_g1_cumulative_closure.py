"""Fail-closed tests for the rc5 G-1 CPU-only cumulative closure."""

from __future__ import annotations
import hashlib
from pathlib import Path

import pytest
from benchmarks.verify_rc5_g1_cumulative_archive import main as verify_main
from benchmarks.rc5_g1_cumulative_evidence import verify_archive


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_archive_verifier_is_fail_closed(tmp_path: Path):
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{digest}  payload.json\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = verify_archive(
        tmp_path,
        expected_entries=1,
        expected_manifest_sha256=manifest_digest,
    )
    assert report["passed"] is True
    assert report["verified"] == 1

    payload.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(
            tmp_path,
            expected_entries=1,
            expected_manifest_sha256=manifest_digest,
        )


def test_versioned_archive_verifier_rejects_unknown_names(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown archive name"):
        verify_main(
            [
                "--archive",
                "unknown",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
