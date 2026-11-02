"""Forward bindings for V-61; no access to external evidence during pytest."""
import hashlib
import json
from pathlib import Path
from tests.v0_3_s03_successor import resolve_s03_successor

ROOT=Path(__file__).resolve().parents[1]

def test_r0057_import_and_rework_preserve_history_and_bind_sources():
    for n, suffix in [(107,'r0057_import'),(108,'r0057_v61_rework')]:
        record=json.loads((ROOT/f'notes/PSSolver_v0_3_0_s03_successor_{n}_{suffix}.json').read_text())
        for path, b in record['historical_successor_sources'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==resolve_s03_successor(path,b['historical_sha256'])
        for path, h in record['introduced_sources'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==resolve_s03_successor(path,h)
    imported=json.loads((ROOT/'notes/PSSolver_v0_3_0_s03_successor_107_r0057_import.json').read_text())
    assert imported['registry_updates']['V-61']['status']=='open'
    assert imported['review']['result_sha256']=='948ba059a0b6cddfd659726db88c243b0e68df2c7faf71bbdad342d94a1029db'
    rework=json.loads((ROOT/'notes/PSSolver_v0_3_0_s03_successor_108_r0057_v61_rework.json').read_text())
    assert rework['registry_updates']['V-61']['status']=='claimed_fixed'
    assert rework['preserved_old_behavior']==['fixed_work_completed error label','implicit RuntimeError']
