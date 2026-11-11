"""M1 roles specialize to the baseline statement sequence, with one PCG loop."""

import ast
import copy
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = '4b720938896af6cae0719569e95be471b33e0935'
FIRST_STAGE = '80ef5e658db86436fdcc8dbbb95db6a150195397'
NUM = 'pssolver/linear_solvers/stokes/channel_no_slip.py'
RETIREMENTS = json.loads((ROOT / 'qualification/s28a_tables.json').read_text())['retired_modules']


def _retired_path(stem):
    module = next(value for value in RETIREMENTS if value.rsplit('.', 1)[-1] == stem)
    return module.replace('.', '/') + '.py'


TRANS = _retired_path('channel_pressure')
IMPL = _retired_path('channel_pressure_adjoint')


def _source(path, commit=BASE):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, text=True)


def _method(text, cls, name):
    owner = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _specialize(nodes, choices):
    result = []
    for node in nodes:
        node = copy.deepcopy(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) in choices:
            result.extend(_specialize(node.body if choices[ast.unparse(node.test)] else node.orelse, choices))
        else:
            if isinstance(node, (ast.If, ast.While)):
                node.body = _specialize(node.body, choices)
                node.orelse = _specialize(node.orelse, choices)
            if not isinstance(node, ast.Pass):
                result.append(node)
    return result


@pytest.mark.parametrize('role', ['production', 'standalone_transpose', 'implicit'])
def test_role_specialization_preserves_every_baseline_statement(role):
    common = [('self._solver', 'self'), ('self._project_gauge', 'self._project_pressure_gauge'),
              ('channel_pressure_solve_diagnostics(', 'make_diagnostics('),
              ('FunctionalConvergenceError(', 'error_type('), ('pressure_hat', 'pressure_work')]
    if role == 'production':
        path, cls, name, changes = NUM, 'ChannelNoSlipModalStokesSolver', '_solve_pressure', []
        choices = dict(production=True)
    elif role == 'standalone_transpose':
        path, cls, name = TRANS, 'ChannelPressureTransposeOperator', 'solve_pressure_transpose'
        choices = dict(production=False, standalone=True)
        changes = common + [
            ('self._transpose_solve_sequence_id', 'context._transpose_solve_sequence_id'),
            ('self._record_transpose_diagnostics', 'context._record_transpose_diagnostics'),
            ('self._fail_transpose_solve', 'context._fail_transpose_solve'),
            ('self.apply_pressure_operator_transpose(', 'action(')]
    else:
        path, cls, name = IMPL, 'ChannelImplicitPressureAdjoint', '_solve_no_grad'
        choices = dict(production=False, standalone=False)
        changes = [('self._operators._project_gauge', 'self._project_pressure_gauge'),
                   ('self._persist_solve_diagnostics', 'context._persist_solve_diagnostics'), *common]
    source = ast.unparse(_method(_source(path), cls, name))
    for before, after in changes:
        source = source.replace(before, after)
    expected = ast.parse(source).body[0].body
    expected = [n for n in expected if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    combined = _method(_source(NUM, FIRST_STAGE), 'ChannelNoSlipModalStokesSolver', '_solve_pcg')
    # The first three statements only select Python role/context values; no tensors.
    assert ast.unparse(combined.body[0]) == 'production = role is PCGRole.PRODUCTION'
    assert ast.unparse(combined.body[1]) == 'standalone = False'
    assert isinstance(combined.body[2], ast.If) and ast.unparse(combined.body[2].test) == 'not production'
    actual = _specialize(combined.body[3:], choices)
    assert [ast.dump(n) for n in actual] == [ast.dump(n) for n in expected]


def test_m1_has_one_pcg_loop_and_no_old_loop_copies():
    merged = _method((ROOT / NUM).read_text(), 'ChannelNoSlipModalStokesSolver', '_solve_pcg')
    assert sum(isinstance(n, ast.While) for n in ast.walk(merged)) == 1
    assert not (ROOT / TRANS).exists()
    assert not (ROOT / IMPL).exists()
    with pytest.raises(StopIteration):
        _method((ROOT / NUM).read_text(), 'ChannelNoSlipModalStokesSolver', '_solve_pressure')


def test_m1_oracle_stays_independent_and_ast_unchanged():
    old = _method_or_function(_source(IMPL), 'unrolled_channel_pressure_solve_oracle')
    new = _method_or_function(_source(IMPL, FIRST_STAGE), 'unrolled_channel_pressure_solve_oracle')
    assert ast.dump(new) == ast.dump(old)
    relocated = _method_or_function(
        (ROOT / 'tests/oracles/channel_pressure_unrolled.py').read_text(),
        'unrolled_channel_pressure_solve_oracle',
    )
    assert sum(isinstance(node, ast.For) for node in ast.walk(old)) == sum(
        isinstance(node, ast.For) for node in ast.walk(relocated)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {'solve', 'solve_transpose', '_solve_pcg'}
        for node in ast.walk(relocated)
    )


def _method_or_function(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize('index', range(2, 9))
def test_seven_adjoint_methods_obey_the_authority_substitutions(index):
    row = json.loads((ROOT / 'qualification/s28a_tables.json').read_text())['mechanical_moves'][index]
    path, symbol = row['source'].split(':')
    cls, name = symbol.split('.')
    original = ast.unparse(_method(_source(path), cls, name))
    for old, new in row['substitutions']:
        assert old in original
        original = original.replace(old, new)
    target_cls, target_name = row['unit'].split('.')
    moved = _method((ROOT / row['target']).read_text(), target_cls, target_name)
    assert ast.dump(moved) == ast.dump(ast.parse(original).body[0])
