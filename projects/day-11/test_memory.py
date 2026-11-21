"""Локальные проверки приложения. Ответы клиента заданы только в тестах."""
import json
import unittest
from unittest.mock import Mock

from main import TokenCounter
from memory import Memory


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        question = 'Обычно я выбираю янтарь. Для этого эскиза выберу кобальт.'
        delta = {'facts': [{'key': 'цвет', 'value': 'янтарь', 'source': 'Обычно я выбираю янтарь'}],
                 'working': [{'key': 'цвет', 'value': 'кобальт', 'source': 'Для этого эскиза выберу кобальт'}]}
        self.client = Mock()
        self.client.ask.side_effect = [json.dumps(delta, ensure_ascii=False), '[true,true]', 'Принято.']
        self.memory.ask(question, self.client, lambda _: None)
        self.memory.clear('short')
        self.client.ask.side_effect = ['{"facts":[],"working":[]}', 'Капля-39']
        self.memory.ask('Предложи короткий маркер для этой реплики.', self.client, lambda _: None)

    def test_evidence_after_each_removal(self):
        counter = TokenCounter()
        question = 'Что известно?'
        included = []
        previous = counter.messages(self.memory.messages(question, included), generation=True)
        print(f'Измерение на тестовом состоянии: база {previous} ток.')
        for layer in ('facts', 'working', 'short'):
            included.append(layer)
            current = counter.messages(self.memory.messages(question, included), generation=True)
            print(f'{layer}: +{current - previous}, всего {current} ток.')
            self.assertGreater(current, previous)
            previous = current

        values = {'facts': 'янтарь', 'working': 'кобальт', 'short': 'Капля-39'}
        for removed in values:
            copy = Memory.load(self.memory.dump())
            copy.clear(removed)
            prompt = json.dumps(copy.messages(question), ensure_ascii=False)
            for layer, value in values.items():
                self.assertEqual(value in prompt, layer != removed)
            print(f'После очистки {removed}: из запроса удалено «{values[removed]}», два других значения доступны.')

    def test_source_with_changed_capitalization(self):
        question = 'Сегодня готовим макет афиши.'
        delta = {'facts': [], 'working': [{'key': 'задача', 'value': 'Макет афиши',
                                           'source': 'сегодня готовим макет афиши'}]}
        self.client.ask.side_effect = [json.dumps(delta, ensure_ascii=False), '[true]', 'Принято.']
        self.memory.ask(question, self.client, lambda _: None)
        self.assertEqual(self.memory.layers()['working']['задача'], 'Макет афиши')

    def test_verifier_rejects_unconfirmed_decision_in_mixed_turn(self):
        question = 'Я предпочитаю чай. Встречу перенесли?'
        delta = {'facts': [{'key': 'напиток', 'value': 'чай', 'source': 'Я предпочитаю чай'}],
                 'working': [{'key': 'встреча', 'value': 'перенесена', 'source': 'Встречу перенесли?'}]}
        self.client.ask.side_effect = [json.dumps(delta), '[true,false]', 'О переносе данных нет.']
        self.memory.ask(question, self.client, lambda _: None)
        self.assertNotIn('встреча', self.memory.layers()['working'])
        self.memory.switch('Новая задача')
        messages = self.memory.messages('Какой напиток я предпочитаю?')
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertEqual(json.loads(messages[-1]['content'])['Профиль пользователя']['напиток'], 'чай')

    def test_task_and_restart(self):
        original = self.memory.layers()
        self.memory.switch('Другая задача')
        self.assertEqual(self.memory.layers(), dict(facts=original['facts'], working={}, short=[]))
        restored = Memory.load(json.loads(json.dumps(self.memory.dump())))
        restored.switch('Основная')
        self.assertEqual(restored.layers(), dict(original, short=[]))

    def test_bad_source_skips_only_unsupported_change(self):
        question = 'Спасибо, пока всё. Обычно мне удобнее утром.'
        delta = {'facts': [{'key': 'время', 'value': 'утро', 'source': 'мне удобнее утром'}],
                 'working': [{'key': 'бумага', 'value': 'не выбрана',
                              'source': 'Радан, тип бумаги для печати пока не выбран.'}]}
        before = self.memory.layers()
        self.client.ask.side_effect = [json.dumps(delta), '[true]', 'До встречи.']
        log = []
        self.assertEqual(self.memory.ask(question, self.client, log.append), 'До встречи.')
        after = self.memory.layers()
        self.assertEqual(after['facts']['время'], 'утро')
        self.assertEqual(after['working'], before['working'])
        self.assertEqual(after['short'][-1]['content'], 'До встречи.')
        self.assertTrue(any('бумага' in line and 'Пропущена' in line for line in log))

    def test_invalid_json_keeps_records_and_allows_reply(self):
        before = self.memory.layers()
        self.client.ask.side_effect = ['{', 'Пожалуйста.']
        self.assertEqual(self.memory.ask('Спасибо!', self.client, lambda _: None), 'Пожалуйста.')
        self.assertEqual(self.memory.layers()['working'], before['working'])
        self.assertEqual(self.memory.layers()['facts'], before['facts'])

    def test_reply_failure_preserves_all_layers(self):
        before = self.memory.dump()
        self.client.ask.side_effect = ['{"facts":[],"working":[]}', TimeoutError()]
        with self.assertRaises(TimeoutError):
            self.memory.ask('Новая реплика', self.client, lambda _: None)
        self.assertEqual(before, self.memory.dump())


if __name__ == '__main__':
    unittest.main()
