"""Tests for the RC6 recovery script/import-root separation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

from benchmarks import prepare_rc6_recovery_bootstrap as prepare
from benchmarks import validate_rc6_recovery_bootstrap as validate


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs/rc6_consumer_rebind_recovery_v6.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _script_root(tmp_path: Path) -> Path:
    root = tmp_path / "scripts"
    scripts = root / "benchmarks"
    scripts.mkdir(parents=True)
    for name in prepare.REQUIRED_SCRIPTS:
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
    return root


def _stage_plan(script_root: Path, output_root: Path, path: Path) -> Path:
    scripts = sorted((script_root / "benchmarks").glob("*.py"))
    stages = []
    for index in range(validate.STAGE_COUNT):
        script = scripts[index % len(scripts)]
        stages.append(
            {
                "name": f"stage_{index + 1}",
                "command_file_bindings": [
                    {
                        "kind": "python_script",
                        "path": str(script),
                        "sha256": _sha256(script),
                    }
                ],
                "artifact": str(output_root / f"stage_{index + 1}.json"),
            }
        )
    path.write_text(
        json.dumps({"schema": validate.PLAN_SCHEMA, "stages": stages}),
        encoding="utf-8",
    )
    return path


def test_prepare_builds_a_distinct_source_free_benchmarks_shim(tmp_path: Path):
    script_root = _script_root(tmp_path)
    import_root = tmp_path / "imports"

    result = prepare.prepare(script_root, import_root)

    assert result["passed"] is True
    assert result["script_root_contains_pssolver"] is False
    assert result["import_root_contains_pssolver"] is False
    assert sorted(path.name for path in import_root.iterdir()) == ["benchmarks"]
    initializer = import_root / "benchmarks/__init__.py"
    assert initializer.read_text(encoding="utf-8") == prepare.initializer_text(
        script_root.resolve()
    )
    assert result["initializer_sha256"] == _sha256(initializer)


def test_prepare_rejects_source_code_symlinks_and_existing_targets(tmp_path: Path):
    script_root = _script_root(tmp_path)
    (script_root / "pssolver").mkdir()
    with pytest.raises(ValueError, match="must not expose source pssolver"):
        prepare.prepare(script_root, tmp_path / "imports")

    (script_root / "pssolver").rmdir()
    linked = tmp_path / "linked_scripts"
    linked.symlink_to(script_root, target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="ordinary script root"):
        prepare.prepare(linked, tmp_path / "imports")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare.prepare(script_root, existing)


def test_validator_accepts_only_installed_pssolver_and_hash_bound_stage_scripts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    script_root = _script_root(tmp_path)
    import_root = tmp_path / "imports"
    prepare.prepare(script_root, import_root)
    output_root = tmp_path / "formal"
    control_root = tmp_path / "control"
    (control_root / "logs").mkdir(parents=True)
    plan = _stage_plan(script_root, output_root, tmp_path / "plan.json")
    purelib = tmp_path / "venv/lib/python3.10/site-packages"
    pssolver_file = purelib / "pssolver/__init__.py"
    pssolver_file.parent.mkdir(parents=True)
    pssolver_file.write_text("", encoding="utf-8")
    initializer = import_root / "benchmarks/__init__.py"
    modules = {
        "pssolver": types.SimpleNamespace(__file__=str(pssolver_file)),
        "benchmarks": types.SimpleNamespace(
            __file__=str(initializer),
            __path__=[str(script_root / "benchmarks")],
        ),
    }
    driver = Path(validate.sys.executable).absolute()
    monkeypatch.setattr(validate.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(validate.sysconfig, "get_path", lambda _name: str(purelib))
    monkeypatch.setenv("PYTHONPATH", str(import_root.resolve()))
    args = argparse.Namespace(
        driver_python=driver,
        script_root=script_root,
        import_root=import_root,
        stage_plan=plan,
        output_root=output_root,
        control_root=control_root,
    )

    result = validate.observe(args)

    assert result["passed"] is True
    assert result["failures"] == []
    assert len(result["predicates"]) == 18
    assert len(result["stage_script_bindings"]) == validate.STAGE_COUNT


def test_validator_persists_failure_before_returning_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    output = tmp_path / "bootstrap.json"
    failure = {
        "schema": validate.SCHEMA,
        "passed": False,
        "failures": ["P06"],
        "predicates": [],
    }
    monkeypatch.setattr(validate, "observe", lambda _args: failure)
    arguments = [
        "--driver-python", str(Path(validate.sys.executable).absolute()),
        "--script-root", str(tmp_path / "scripts"),
        "--import-root", str(tmp_path / "imports"),
        "--stage-plan", str(tmp_path / "plan.json"),
        "--output-root", str(tmp_path / "formal"),
        "--control-root", str(tmp_path / "control"),
        "--output", str(output),
    ]

    assert validate.main(arguments) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == failure
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate.main(arguments)


def test_recovery_job_separates_script_and_import_roots_and_is_shell_valid():
    script = JOB.read_text(encoding="utf-8")
    assert "RC6_SUPPORT_ROOT" not in script
    assert ': "${RC6_SCRIPT_ROOT:' in script
    assert ': "${RC6_IMPORT_ROOT:' in script
    assert 'export PYTHONPATH="${RC6_IMPORT_ROOT}"' in script
    assert "validate_rc6_recovery_bootstrap.py" in script
    assert script.index("validate_rc6_recovery_bootstrap.py") < script.index(
        "module load cuda/12.1"
    )
    assert 'node_root="/tmp/pssolver_rc6_${SLURM_JOB_ID}"' in script
    assert 'export TMPDIR="${pytest_tmp}"' in script
    executable_lines = [
        line for line in script.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any("sbatch" in line for line in executable_lines)
    result = subprocess.run(
        ["bash", "-n", str(JOB)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
