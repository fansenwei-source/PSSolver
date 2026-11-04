"""Repository-only V-65 contracts; real full deployment is archived separately."""
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'jobs/v0_4_s19_single_h100.sbatch'


def test_nvidia_smi_is_only_required_after_login_exit_before_node_directory():
    job = JOB.read_text()
    boundary = job.index('if [[ "${1:-}" == --preflight-only ]]; then exit 0; fi')
    check = job.index('command -v nvidia-smi')
    assert boundary < check < job.index('node_root=')
    assert 'nvidia-smi' not in job[:boundary]
    assert "MISSING_REQUIRED_COMMAND: nvidia-smi\\n" in job[check:]
    assert 'exit 2;' in job[check:job.index('node_root=')]


def test_shared_command_checks_stay_present():
    job = JOB.read_text()
    before = job.split('if [[ "${1:-}" == --preflight-only ]]; then exit 0; fi')[0]
    for text in ('command -v "$executable"', 'command -v module',
                 'command -v "$assembler"', '--mode login', '--mode job',
                 'module load cuda/12.1', 'for program in as ld cc1 cc1plus collect2'):
        assert text in before


@pytest.mark.parametrize('status,output,expected', [(0, '', 0), (0, '123 job', 1), (17, '', 17)])
def test_queue_assignment_fails_closed_on_command_failure(tmp_path, status, output, expected):
    queue = tmp_path / 'squeue'
    queue.write_text(f'#!/bin/sh\nprintf "%s" "{output}"\nexit {status}\n')
    queue.chmod(0o755)
    script = 'set -eu; queue="$(squeue -h -u "$USER")"; test -z "$queue"; printf SUBMIT_ALLOWED'
    result = subprocess.run(['/bin/bash', '-c', script], env={'PATH': str(tmp_path), 'USER': 'test'},
                            text=True, capture_output=True, check=False)
    assert result.returncode == expected
    assert ('SUBMIT_ALLOWED' in result.stdout) == (expected == 0)


def test_command_table_separates_login_and_compute_node_and_soft_calls():
    table = json.loads((ROOT / 'notes/PSSolver_v0_4_s19_retry_node_commands.json').read_text())
    for row in table['required_commands']:
        assert row['login_node'] and row['compute_node']
    soft = {row['command']: row for row in table['soft_commands']}
    assert set(soft) == {'uname -p', 'fc-list'}
    assert all(row['required'] is False for row in soft.values())


def test_r0063_import_preserves_review_status_snapshot():
    record = json.loads((ROOT / 'notes/PSSolver_v0_3_0_s03_successor_125_r0063_import.json').read_text())
    assert record['review']['classification'] == 'REWORK_REQUIRED'
    assert record['new_findings'][0]['proposed_id'] == 'V-65'
    assert {key: row['status'] for key, row in record['registry_snapshot'].items()} == {
        'V-62': 'verified', 'V-63': 'open', 'V-64': 'claimed_fixed', 'V-65': 'open'}
