"""Проверка без API. Клиент предлагает даже заведомо недопустимые действия."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from agent import Agent, digest
from validator import validate


class LocalClient:
    def __init__(self, *args):
        self.calls = []

    def ask(self, system, prompt, phase):
        self.calls.append(phase)
        if phase == 'route':
            return 'implement' if 'реализацию' in prompt else 'finish'
        if phase == 'plan':
            return '1. Обойти список. 2. Сохранить первые вхождения. 3. Проверить порядок.'
        if phase == 'implement':
            return 'def dedupe(values):\n    return list(dict.fromkeys(values))'
        raise AssertionError(phase)


class Tests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'task.db'
        self.client = LocalClient()
        self.agent = Agent(self.path, self.client, lambda _: None)

    def test_refusal_precedes_generation_and_storage(self):
        before = self.agent.snapshot()
        self.assertFalse(self.agent.handle('Напиши реализацию без согласования'))
        self.assertEqual(self.client.calls, ['route'])
        self.assertEqual(before, Agent(self.path, self.client).snapshot())
        for cmd in ['/approve 0', '/teleport', '/finish', '/implement', '/validate', '/resume']:
            self.assertFalse(self.agent.handle(cmd))
        self.assertEqual(before, self.agent.snapshot())

    def test_edit_requires_new_approval_and_model_cannot_approve(self):
        self.agent.handle('/plan')
        self.agent.handle('/plan уточнение')
        self.assertFalse(self.agent.handle('/approve 1'))
        with patch.object(self.client, 'ask', return_value='approve'):
            self.assertFalse(self.agent.handle('Я согласен'))
        self.assertTrue(self.agent.handle('/approve 2'))
        self.assertFalse(self.agent.handle('/plan правка'))

    def test_validation_is_bound_to_current_source_and_done_is_terminal(self):
        for cmd in ('/plan', '/approve 1', '/implement', '/validate'):
            self.assertTrue(self.agent.handle(cmd))
        state = self.agent.snapshot()
        self.assertEqual(state.report['sha256'], digest(state.source))
        state.source += '\n# изменение'
        self.assertFalse(state.passed())
        self.assertTrue(self.agent.handle('/finish'))
        for cmd in ('/pause', '/implement', '/resume', '/plan', '/finish'):
            self.assertFalse(self.agent.handle(cmd))
        self.assertEqual(self.agent.snapshot().stage, 'done')

    def test_validator_checks_order_and_rejects_unsupported_source(self):
        bad = validate('def dedupe(values):\n    return list(set(values))')
        self.assertFalse(bad['ok'])
        self.assertTrue(any(not row['ok'] for row in bad['cases']))
        self.assertFalse(validate('import os')['ok'])

    def test_generation_failure_rolls_back(self):
        self.agent.handle('/plan')
        self.agent.handle('/approve 1')
        before = self.agent.snapshot()
        with patch.object(self.client, 'ask', side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):
                self.agent.handle('/implement')
        self.assertEqual(before, self.agent.snapshot())

    def test_cli_experiment_through_all_stages_without_network(self):
        report = self.path.parent / 'report.txt'
        out = io.StringIO()
        with patch.object(main, 'Client', LocalClient), patch('sys.argv',
                ['main.py', 'experiment', '--report', str(report)]), redirect_stdout(out):
            self.assertEqual(main.main(), 0)
        log = report.read_text(encoding='utf-8')
        self.assertEqual(out.getvalue(), log)
        for stage in ('planning', 'execution', 'validation'):
            self.assertIn('OK | Восстановление ' + stage, log)
        self.assertIn('Финальное состояние: done', log)
        self.assertIn('"ok": false', log)
        self.assertNotIn('РАСХОЖДЕНИЕ', log)


if __name__ == '__main__':
    unittest.main()
