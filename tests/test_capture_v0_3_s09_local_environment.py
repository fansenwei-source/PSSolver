"""Pure helper checks for the S-09 D-0008 environment recorder."""

from pathlib import Path

import pytest

from benchmarks.capture_v0_3_s09_local_environment import _atomic_json, _sha256


def test_sha256_and_atomic_json_are_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"s09")
    assert _sha256(source) == "69c2b041c8e57b463435a0927016fd5f1d2459b336174e189be90d548cb46670"
    output = tmp_path / "environment.json"
    _atomic_json(output, {"finite": True})
    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, {})
