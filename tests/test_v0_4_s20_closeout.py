"""S-20 closeout snapshots and forward bindings, with no external evidence IO."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'notes/PSSolver_v0_3_0_s03_successor_129_s20_closeout.json'
TRIAGE = ROOT / 'notes/PSSolver_v0_4_0_s20_closeout_triage.json'


def read(path):
    return json.loads(path.read_text())


def test_r0065_import_is_final_s19_not_final_s20_acceptance():
    r = read(RECORD)
    assert r['review']['request'] == 'R-0065'
    assert r['review']['classification'] == 'PASS'
    assert r['review']['result_sha256'] == '4eb35951d0c7f905d6c6c8c2d6f85d14685e148a4abf9b5b443e085b8ac7a3d6'
    assert r['git_identity']['implementation_parent'] == '7513787811f6bb82791972e5ea2ddaece67f2bea'
    assert r['plan_step'] == 'S-20'
    assert 'AWAITING_INTERIM_REVIEW' in r['classification']
    assert not any(r['scope'].values())


def test_d0024_snapshot_advances_only_the_approved_entries():
    r = read(RECORD)
    before, after = r['registry_before'], r['registry_snapshot']
    assert set(before) == set(after)
    assert {k for k in before if before[k] != after[k]} == {'V-63', 'R1-R9', 'V-64', 'V-65'}
    assert {k for k in before if before[k]['status'] != after[k]['status']} == {'V-63'}
    assert after['V-63']['status'] == 'accepted'
    assert after['V-63']['decision'] == 'D-0024'
    assert after['V-63']['target_release'] == '0.4 之后，随删除 functional 兼容层一并处理'
    assert after['R1-R9']['status'] == 'deferred'
    assert after['R1-R9']['decision'] == 'D-0024'
    assert after['R1-R9']['target_release'] == '0.4 之后的开发线'
    for issue in ('V-64', 'V-65'):
        assert {k for k in before[issue] if before[issue][k] != after[issue][k]} == {'notes'}
        assert '已在真实节点上证实（R-0065）' in after[issue]['notes']
    assert len(r['unchanged_issue_hashes']) == 66
    assert {k: v for k, v in r['status_counts'].items() if v} == {'verified': 65, 'accepted': 3, 'deferred': 2}
    assert r['status_counts']['open'] == 0


def test_r1_r9_remaining_items_are_decision_scoped_not_claimed_complete():
    t = read(TRIAGE)
    assert set(t['remaining_r1_r9']) == set('abcdefg')
    assert all(t['remaining_r1_r9'][key]['decision'] == 'D-0024' for key in 'defg')
    notes = read(RECORD)['registry_snapshot']['R1-R9']['notes']
    assert all(review in notes for review in ('R-0047', 'R-0051', 'R-0059'))
    assert len(t['r4_retained_files']) == 5
    assert {i['path'] for i in t['r4_retained_files']} == {
        'pssolver/integrator.py', 'pssolver/integrators/scalar_reference.py',
        'pssolver/workflows/channel_beris_edwards.py', 'pssolver/workflows/periodic_checkpoint.py',
        'pssolver/workflows/plane_checkpoint.py'}
    assert all(i['reason'] and i['decision'] == 'D-0024' for i in t['r4_retained_files'])


def test_triage_covers_all_frozen_source_labels_and_additional_action_notes():
    t = read(TRIAGE)
    assert {s['source'] for s in t['sources']} == {f'R-{i:04d}' for i in range(42, 66)}
    rows = {r['id']: r for r in t['items']}
    assert len(rows) == len(t['items'])
    for s in t['sources']:
        assert all(s['source'] + '/' + label in rows for label in s['observed_labels'])
    assert {'A-0075', 'A-0079'} <= {r['source'] for r in rows.values()}
    for row in rows.values():
        assert row['disposition'] in {'已完成', '本批完成', '延后'}
        assert row['content'] and row['reason']
        assert type(row['touches_pssolver']) is bool
    assert rows['R-0059/I-1']['disposition'] == '延后'
    assert rows['R-0060/I-4']['disposition'] == '本批完成'


def test_known_limitations_do_not_extend_the_v030_memory_exception():
    limits = {x['id']: x for x in read(TRIAGE)['known_limitations']}
    assert set(limits) == {'V-63', 'V-41', 'A2-11', 'B8/S2-07'}
    assert 'does NOT extend to 0.4' in limits['V-41']['text']
    assert 'not fixed' in limits['V-63']['text']


def test_system_cleanup_and_array_count_corrections_preserve_historical_evidence():
    c = read(RECORD)['corrections']
    assert c['cache'] + c['restart'] + c['trajectories'] == c['system_deleted'] == 1274
    assert c['restart'] + c['trajectories'] == c['data_files'] == 42
    assert (c['cache'], c['restart'], c['trajectories']) == (1232, 28, 14)
    assert c['all_npy_format_groups'] * 3 == c['all_npy_format_files'] == 6402
    assert c['npy_suffix_groups'] * 3 == c['npy_suffix_files'] == 5979
    assert not c['old_archives_modified']


def test_successor_129_binds_every_change_without_pinning_future_registry_state():
    r = read(RECORD)
    bindings = {p: v['current_sha256'] for p, v in r['historical_successor_sources'].items()}
    bindings.update(r['introduced_sources'])
    for path, digest in bindings.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, digest)
