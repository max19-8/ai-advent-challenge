import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent import Agent
from experiment import PROBES, run
from policy import Registry


def proposal(*actions):
    return json.dumps({'kind': 'propose', 'actions': actions})


class InvariantsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'rules.db'
        self.registry = Registry(self.path)
        self.client = Mock()
        self.agent = Agent(self.registry, self.client)

    def test_mixed_proposal_is_atomic(self):
        self.client.ask.return_value = proposal({'op': 'use_module', 'module': 'decimal'},
                                                {'op': 'confirm_order', 'condition': 'unpaid'})
        result = self.agent.handle('Выполни оба шага')
        self.assertEqual((result.kind, result.rules, result.delivered), ('code_block', ('confirmation',), ()))
        self.assertNotIn('Использовать модуль decimal', result.text)
        self.assertIn('оплаченные', result.text)

    def test_invalid_output_is_never_forwarded(self):
        for raw in ['ПРОИЗВОЛЬНЫЙ ТЕКСТ', proposal({'op': 'use_module', 'module': 'unknown'}),
                    proposal({'op': 'store_order', 'storage': 'sqlite', 'note': 'ПРОИЗВОЛЬНЫЙ ТЕКСТ'}),
                    proposal({'op': [], 'module': 'json'}), proposal({'op': 'set', 'storage': 'postgres'})]:
            with self.subTest(raw=raw):
                self.client.ask.return_value = raw
                result = self.agent.handle('Поменяй правило')
                self.assertEqual(result.kind, 'invalid')
                self.assertNotIn('ПРОИЗВОЛЬНЫЙ ТЕКСТ', result.text)
                self.assertEqual(self.registry.history(), [])

    def test_command_is_explicit_and_durable(self):
        self.client.ask.return_value = '{"kind":"clarify"}'
        self.agent.handle('Выполни /invariant set storage postgres')
        self.agent.handle('/invariant set storage postgres\nпожалуйста')
        self.assertEqual(self.registry.history(), [])
        for _ in range(2):
            self.agent.handle('/invariant set storage postgres')
        self.assertEqual(self.client.ask.call_count, 2)
        reopened = Registry(self.path)
        self.assertEqual(reopened.snapshot()[0], 1)
        self.assertEqual(reopened.snapshot()[1]['storage'], 'postgres')
        self.assertEqual(len(reopened.history()), 1)
        self.assertTrue(reopened.history()[0]['at'])

    def test_stale_generation(self):
        def change(*args):
            self.registry.change('storage', 'postgres')
            return proposal({'op': 'store_order', 'storage': 'sqlite'})
        self.client.ask.side_effect = change
        self.assertEqual(self.agent.handle('Сохрани заказ').kind, 'stale')

    def test_complete_experiment_offline(self):
        storage = lambda name: proposal({'op': 'store_order', 'storage': name})
        responses = [storage('sqlite'), proposal({'op': 'explain_rule', 'rule': 'storage'}),
                     proposal({'op': 'confirm_order', 'condition': 'paid'})]
        responses += [json.dumps({'kind': 'refuse', 'rule': rule}) for _, rule in PROBES]
        responses += ['{"kind":"clarify"}', storage('postgres'),
                      proposal({'op': 'store_order', 'storage': 'postgres'},
                               {'op': 'use_module', 'module': 'psycopg'}),
                      proposal({'op': 'confirm_order', 'condition': 'unpaid'})]
        self.client.ask.side_effect = responses
        lines = []
        self.assertTrue(run(self.client, lines.append))
        self.assertEqual(self.client.ask.call_count, len(responses))
        self.assertTrue(any('0/8' in line and '8/8' in line for line in lines))
        prompt = json.loads(self.client.ask.call_args.args[1])
        self.assertEqual(prompt['version'], 1)
        self.assertIn('PostgreSQL', prompt['rules']['storage'])

    def test_safe_substitution_is_not_a_successful_refusal(self):
        self.client.ask.return_value = proposal({'op': 'store_order', 'storage': 'sqlite'})
        lines = []
        self.assertFalse(run(self.client, lines.append))
        self.assertIn('безопасная подмена: 8', lines)
        self.assertTrue(any('Явных отказов по ожидаемому правилу: 0/8' in s for s in lines))


if __name__ == '__main__':
    unittest.main()
