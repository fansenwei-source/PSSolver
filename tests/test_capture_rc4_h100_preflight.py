from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import capture_rc4_h100_preflight as preflight


class _FakeCuda:
    def __init__(self, *, available: bool, count: int, name: str):
        self._available = available
        self._count = count
        self._name = name

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_name(self, index):
        if not self._available or index >= self._count:
            raise RuntimeError("CUDA device is unavailable")
        return self._name


class _FakeTorch:
    __version__ = "2.5.1+cu121"
    version = SimpleNamespace(cuda="12.1")

    def __init__(
        self,
        *,
        available: bool = True,
        count: int = 1,
        name: str = "NVIDIA H100 PCIe",
    ):
        self.cuda = _FakeCuda(available=available, count=count, name=name)
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=True),
            ),
            cudnn=SimpleNamespace(allow_tf32=True),
        )

    def empty(self, shape, *, device):
        assert shape == ()
        assert device == "cuda"
        if not self.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        return SimpleNamespace(device="cuda:0")


def _modules(q_root: Path, torch_module=None):
    return {
        "torch_module": _FakeTorch() if torch_module is None else torch_module,
        "pssolver_module": SimpleNamespace(
            __file__=str(q_root / "pssolver" / "__init__.py")
        ),
        "diagnostic_module": SimpleNamespace(
            __file__=(
                str(q_root / "benchmarks" / "diagnose_plane_nyquist_storage.py")
            )
        ),
    }


def test_preflight_records_raw_policy_then_enforces_all_predicates(tmp_path):
    q_root = tmp_path / "q"
    q_root.mkdir()
    output = tmp_path / "preflight.json"
    fake_torch = _FakeTorch()

    report = preflight.capture_preflight(
        q_root=q_root,
        output=output,
        **_modules(q_root, fake_torch),
    )

    assert report["classification"] == preflight.PASS_CLASSIFICATION
    assert report["preflight_complete"] is True
    assert report["tf32_policy"]["before"] == {
        "cuda_matmul_allow_tf32": {"value": True, "exception": None},
        "cudnn_allow_tf32": {"value": True, "exception": None},
    }
    assert report["tf32_policy"]["after"] == {
        "cuda_matmul_allow_tf32": {"value": False, "exception": None},
        "cudnn_allow_tf32": {"value": False, "exception": None},
    }
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    assert [item["id"] for item in report["predicates"]] == [
        f"P{index:02d}" for index in range(1, 9)
    ]
    assert {item["status"] for item in report["predicates"]} == {"PASS"}
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(tmp_path.glob(".*.tmp")) == []


def test_each_predicate_is_atomically_checkpointed_before_enforcement(
    tmp_path, monkeypatch
):
    q_root = tmp_path / "q"
    q_root.mkdir()
    output = tmp_path / "preflight.json"
    snapshots = []
    original = preflight.write_json_atomic

    def recording_writer(path, payload):
        snapshots.append(deepcopy(payload))
        original(path, payload)

    monkeypatch.setattr(preflight, "write_json_atomic", recording_writer)
    preflight.capture_preflight(
        q_root=q_root,
        output=output,
        **_modules(q_root),
    )

    predicate_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot["progress"]["completed_predicate_ids"]
    ]
    first_by_count = {}
    for snapshot in predicate_snapshots:
        count = len(snapshot["progress"]["completed_predicate_ids"])
        first_by_count.setdefault(count, snapshot)
    assert sorted(first_by_count) == list(range(1, 9))
    for count, snapshot in first_by_count.items():
        assert snapshot["progress"]["completed_predicate_ids"] == [
            f"P{index:02d}" for index in range(1, count + 1)
        ]
        assert len(snapshot["predicates"]) == count


def test_failure_report_persists_all_observations_before_raise(tmp_path):
    q_root = tmp_path / "q"
    q_root.mkdir()
    outside = tmp_path / "outside"
    output = tmp_path / "preflight.json"
    modules = {
        "torch_module": _FakeTorch(available=False, count=0, name="CPU"),
        "pssolver_module": SimpleNamespace(
            __file__=str(outside / "pssolver" / "__init__.py")
        ),
        "diagnostic_module": SimpleNamespace(
            __file__=str(outside / "benchmarks" / "diagnostic.py")
        ),
    }

    with pytest.raises(preflight.PreflightFailure, match="P01.*P06"):
        preflight.capture_preflight(
            q_root=q_root,
            output=output,
            enforce=True,
            **modules,
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["classification"] == preflight.FAIL_CLASSIFICATION
    assert report["preflight_complete"] is True
    assert len(report["predicates"]) == 8
    assert report["progress"]["completed_predicate_ids"] == [
        f"P{index:02d}" for index in range(1, 9)
    ]
    assert report["enforcement"]["failed_predicate_ids"] == [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
    ]
    by_id = {item["id"]: item for item in report["predicates"]}
    assert by_id["P05"]["exception"]["type"] == "RuntimeError"
    assert by_id["P06"]["exception"]["type"] == "RuntimeError"
    assert by_id["P07"]["status"] == "PASS"
    assert by_id["P08"]["status"] == "PASS"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_no_enforce_returns_complete_failure_record(tmp_path):
    q_root = tmp_path / "q"
    q_root.mkdir()
    report = preflight.capture_preflight(
        q_root=q_root,
        output=tmp_path / "preflight.json",
        enforce=False,
        torch_module=_FakeTorch(name="NVIDIA A100"),
        pssolver_module=SimpleNamespace(
            __file__=str(q_root / "pssolver" / "__init__.py")
        ),
        diagnostic_module=SimpleNamespace(
            __file__=str(q_root / "benchmarks" / "diagnostic.py")
        ),
    )

    assert report["classification"] == preflight.FAIL_CLASSIFICATION
    assert report["enforcement_requested"] is False
    assert report["enforcement"]["failed_predicate_ids"] == ["P06"]


def test_preflight_refuses_existing_output_or_missing_q_root(tmp_path):
    q_root = tmp_path / "q"
    q_root.mkdir()
    output = tmp_path / "preflight.json"
    output.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        preflight.capture_preflight(
            q_root=q_root,
            output=output,
            **_modules(q_root),
        )
    assert output.read_text(encoding="utf-8") == "do not overwrite\n"

    with pytest.raises(NotADirectoryError):
        preflight.capture_preflight(
            q_root=tmp_path / "missing",
            output=tmp_path / "unused.json",
            **_modules(q_root),
        )
