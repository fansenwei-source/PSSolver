from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver.io._checkpoint_artifacts import (
    canonical_indented_json_bytes,
    indented_identity_sha256,
    raw_file_sha256,
    read_reference_tensor_record,
    sha256_bytes,
    write_reference_tensor_record,
    write_sized_tensor_record,
    write_standard_tensor_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _direct_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_reference_identity_bytes_remain_exact() -> None:
    value = {"z": [1, 2.5], "a": {"finite": True}}
    expected = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    assert canonical_indented_json_bytes(value) == expected
    assert sha256_bytes(expected) == hashlib.sha256(expected).hexdigest()
    assert indented_identity_sha256(value) == hashlib.sha256(expected).hexdigest()


def test_standard_tensor_record_preserves_bytes_and_fields(tmp_path: Path) -> None:
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)[..., ::2]
    actual_path = tmp_path / "actual.npy"
    expected_path = tmp_path / "expected.npy"

    record = write_standard_tensor_record(
        actual_path,
        value,
        hash_file=_direct_sha256,
    )
    expected_array = value.detach().to(device="cpu").contiguous().numpy()
    with expected_path.open("xb") as handle:
        np.save(handle, expected_array, allow_pickle=False)

    assert actual_path.read_bytes() == expected_path.read_bytes()
    assert record == {
        "file": actual_path.name,
        "shape": list(expected_array.shape),
        "dtype": str(expected_array.dtype),
        "sha256": _direct_sha256(actual_path),
    }


def test_reference_tensor_record_and_reader_preserve_contract(
    tmp_path: Path,
) -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    path = tmp_path / "state.npy"
    record = write_reference_tensor_record(path, value)

    assert record == {
        "filename": "state.npy",
        "shape": [3, 4],
        "numpy_dtype": "float32",
        "torch_dtype": "torch.float32",
        "finite": True,
        "sha256": raw_file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    loaded = read_reference_tensor_record(
        tmp_path,
        "state",
        record,
        filename="state.npy",
    )
    assert torch.equal(loaded, value)

    wrong = dict(record, sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint tensor 'state' SHA-256 mismatch"):
        read_reference_tensor_record(
            tmp_path,
            "state",
            wrong,
            filename="state.npy",
        )


def test_reference_writer_keeps_cpu_only_error() -> None:
    value = torch.empty((), device="meta")
    with pytest.raises(
        ValueError,
        match="reference checkpoint tensors must be on CPU",
    ):
        write_reference_tensor_record(Path("unused.npy"), value)


def test_sized_tensor_record_preserves_finite_q_layout(tmp_path: Path) -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "finite_q.npy"
    record = write_sized_tensor_record(
        path,
        value,
        hash_file=_direct_sha256,
    )
    assert record == {
        "path": "finite_q.npy",
        "size_bytes": path.stat().st_size,
        "sha256": _direct_sha256(path),
        "shape": [2, 3],
        "dtype": "float32",
    }


def test_r2_writers_delegate_to_one_private_owner() -> None:
    expected = {
        "pssolver/experimental/checkpointing.py": "write_standard_tensor_record",
        "pssolver/experimental/modal_block_checkpoint.py": (
            "write_reference_tensor_record"
        ),
        "pssolver/functional/channel_checkpoint.py": (
            "write_standard_tensor_record"
        ),
        "pssolver/integrators/reference_checkpoint.py": (
            "write_reference_tensor_record"
        ),
        "pssolver/workflows/channel_checkpoint.py": (
            "write_standard_tensor_record"
        ),
        "pssolver/workflows/finite_q_anchoring.py": (
            "write_sized_tensor_record"
        ),
        "pssolver/workflows/periodic_checkpoint.py": (
            "write_standard_tensor_record"
        ),
        "pssolver/workflows/plane_checkpoint.py": (
            "write_standard_tensor_record"
        ),
    }
    for relative, helper in expected.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "pssolver.io._checkpoint_artifacts"
            for alias in node.names
        }
        assert helper in imports or any(
            alias.asname == helper
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "pssolver.io._checkpoint_artifacts"
            for alias in node.names
        )


def test_r2_does_not_move_directory_publication_into_shared_artifacts() -> None:
    source = (
        ROOT / "pssolver/io/_checkpoint_artifacts.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "checkpoint_directory_transaction",
        "renameat2",
        "RENAME_NOREPLACE",
        "os.replace",
        "fsync",
        "symlink",
    )
    assert all(token not in source for token in forbidden)
