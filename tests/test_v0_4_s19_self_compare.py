"""Checks on the sequential installed-wheel A/A/B comparison harness."""

import ast
import inspect
import json

import numpy as np
import pytest
import torch

from benchmarks import run_v0_4_s19_self_compare as harness
from benchmarks._s19_compare import HardQualificationError


def test_payload_is_saved_outside_the_reused_work_and_hash_checked(tmp_path):
    path = tmp_path / "retained" / "state.npy"
    tensor = torch.tensor([0.0, -0.0, 1.25], dtype=torch.float64)
    record = harness._persist_tensor(path, tensor)
    actual = harness._read_array(record)
    assert actual.tobytes() == tensor.numpy().tobytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(HardQualificationError, match="hash mismatch"):
        harness._read_array(record)


def test_payload_shape_and_dtype_and_finiteness_fail_closed(tmp_path):
    record = harness._persist_tensor(tmp_path / "valid.npy", torch.ones(2))
    for field, value in (("dtype", "float64"), ("shape", [3])):
        with pytest.raises(HardQualificationError, match="schema"):
            harness._read_array(dict(record, **{field: value}))
    with pytest.raises(HardQualificationError, match="non-finite"):
        harness._persist_tensor(tmp_path / "bad.npy", torch.tensor([float("nan")]))
    assert not (tmp_path / "bad.npy").exists()


def test_structured_public_diagnostics_preserve_fields_and_raw_bytes(tmp_path):
    value = np.array([(1, 0.0), (2, -0.0)], dtype=[('step', '<i8'), ('residual', '<f8')])
    rows = harness._persist_output_array(tmp_path / 'diagnostics.npy', 'diagnostics.npy', value)
    assert len(rows) == 3
    raw = next(row for name, row in rows.items() if ':raw:' in name)
    assert harness._read_array(raw).tobytes() == value.tobytes()
    value['residual'][1] = np.nan
    with pytest.raises(HardQualificationError, match='non-finite'):
        harness._persist_output_array(tmp_path / 'bad.npy', 'bad.npy', value)


@pytest.mark.parametrize(
    'runtime', ("legacy_production", "compiled_v2", 'periodic_spectral', "channel_complete_stress")
)
def test_public_execution_rebind_changes_only_device_for_all_six_paths(tmp_path, runtime):
    from tests import test_phase7_p7711_public_entry_equivalence as legacy
    from tests import test_phase8_p82_periodic_application as periodic
    from tests import test_phase8_p83_channel_complete_stress_application as channel

    if runtime in harness.PUBLIC_PATHS[:2]:
        simulation = legacy._simulation('plane' if runtime in harness.PUBLIC_PATHS[:2] else 'channel',
                                        tmp_path / runtime, runtime, steps=4)
    else:
        module = periodic if runtime == 'periodic_spectral' else channel
        simulation = module._simulation(tmp_path, output_name=runtime, steps=4)
    original = simulation.execution.to_metadata()
    result = harness._public_execution_on_device(simulation.execution, 'cuda')
    expected = dict(original, options=dict(original['options'], device='cuda'))
    assert result.to_metadata() == expected
    assert simulation.execution.to_metadata() == original
    assert harness._public_execution_on_device(simulation.execution, 'cpu').to_metadata() == original


def test_pcg_exact_arrays_and_diagnostics_both_required(tmp_path):
    record = harness._persist_tensor(tmp_path / "left.npy", torch.ones(2))
    data = {"kind": "pcg", "cases": [{"case": {"name": "fixed"}, "path": "production",
            "fields": {"p": record}, "record": {"acceptable": True, "iterations": 5}}]}
    identical = json.loads(json.dumps(data))
    assert harness._compare(data, identical)["classification"] == "PASS"
    identical["cases"][0]["record"]["iterations"] = 6
    report = harness._compare(data, identical)
    assert report["classification"] == "N"
    assert report["continue_control_and_profiles"] is True
    identical["cases"][0]["path"] = "oracle"
    with pytest.raises(HardQualificationError, match="fixture identity"):
        harness._compare(data, identical)


def test_child_command_is_isolated_and_no_bytecode_and_work_path_not_role_specific():
    source = inspect.getsource(harness.execute)
    assert '"-I", "-B"' in source
    assert '"--work", str(work)' in source
    assert 'f"{number:02d}_{role}"' in source
    assert "capture_output=True" in source
    assert "ThreadPool" not in source and "Popen(" not in source
    assert 'marker.read_text() != owner' in source
    assert source.index('atomic_write_json(output / "launches"') < source.index("shutil.rmtree(work)")


def test_pcg_device_adapter_does_not_modify_frozen_module_or_runtime():
    source = inspect.getsource(harness._pcg)
    tree = ast.parse(source)
    assigned_attributes = [node for node in ast.walk(tree)
                           if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)]
    assert not assigned_attributes
    assert "types.FunctionType" in source
    assert "namespace = dict(frozen._run_path.__globals__)" in source
    assert "for path in frozen.PATHS" in source
    assert ".to(case_device)" in source
    assert 'case_device = "cpu" if path == "oracle" else args.device' in source


def test_pcg_transpose_import_uses_the_unchanged_definition_not_stable_api():
    import subprocess

    source = inspect.getsource(harness._pcg)
    assert "from pssolver.functional.channel_pressure import ChannelPressureTransposeOperator" in source
    target = next(node.module for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.ImportFrom)
                  and any(name.name == "ChannelPressureTransposeOperator" for name in node.names))
    filename = target.replace(".", "/") + ".py"
    historical = subprocess.check_output(["git", "show", "4b72093:" + filename], text=True)
    assert any(isinstance(node, ast.ClassDef) and node.name == "ChannelPressureTransposeOperator"
               for node in ast.parse(historical).body)
