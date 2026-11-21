"""Офлайн-проверки: ни один тест не обращается к сервису."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import main
from experiment import COMMON, SERIES, SETTINGS, assess
from persona import Profiles


class ScriptedClient:
    """Возвращает контролируемые ответы; записывает фактические запросы на генерацию."""
    def __init__(self):
        self.replies = []

    def ask(self, messages, phase):
        data = json.loads(messages[-1]['content'])
        if phase == 'extract':
            question = data['новая реплика']
            entries = []
            if question in SETTINGS.values():
                value = 'на ты' if question == SETTINGS['А'] else 'на вы'
                values = {'tone': value, 'code': 'без кода' if question == SETTINGS['А'] else 'Python'}
                values.update({'length': '70 слов'} if question == SETTINGS['А'] else {'detail': 'подробно'})
                entries = [dict(key='pref.' + k, value=v, source=question) for k, v in values.items()]
            elif question.startswith('С этого момента'):
                entries = [dict(key='pref.code', value='без кода', source=question)]
            return json.dumps(dict(facts=entries, working=[]))
        if phase == 'verify':
            return '```json\n' + json.dumps([True] * len(data['кандидаты'])) + '\n```'
        if phase == 'reply':
            self.replies.append((messages, data))
            return 'Ты можешь использовать множество.' if data['Предпочтения'].get('tone') == 'на ты' else 'Вы можете использовать множество.'
        if phase == 'judge':
            return json.dumps({k: dict(status='да', reason='Тестовая оценка', quote='')
                               for k in data['критерии']})
        raise AssertionError(phase)


class Tests(unittest.TestCase):
    def test_cli_experiment_runs_offline_and_file_matches_console(self):
        client = ScriptedClient()
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'report.txt'
            out = io.StringIO()
            with patch.object(main, 'Client', return_value=client), patch.dict(main.os.environ, POLZA_API_KEY='test'), \
                    patch.object(sys, 'argv', ['main.py', 'experiment', '--report', str(report)]), redirect_stdout(out):
                self.assertEqual(main.main(), 0)
            self.assertEqual(report.read_text(), out.getvalue())
        questions = [d['Вопрос пользователя'] for _, d in client.replies]
        common = [(m, d) for m, d in client.replies if d['Вопрос пользователя'] == COMMON]
        self.assertEqual([d['Предпочтения']['tone'] for _, d in common], ['на ты', 'на вы'])
        self.assertTrue(all(len(m) == 2 for m, _ in common))
        start = questions.index(SERIES[0])
        self.assertEqual(tuple(questions[start:start + 10]), SERIES)
        self.assertIn('Первое наблюдаемое нарушение:', out.getvalue())
        self.assertTrue(out.getvalue().rstrip().endswith('Эксперимент завершён.'))

    def test_chat_reload_preserves_user_and_one_turn_does_not_replace_profile(self):
        client = ScriptedClient()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state' / 'profiles.json'
            main.chat(client, lambda _: None, path, ['/user Первый', SETTINGS['А'], '/task Доклад',
                      '/user Второй', SETTINGS['Б'], 'Только сейчас ответьте на ты.'])
            again = Profiles.load(json.loads(path.read_text()))
            self.assertEqual(again.active, 'Второй')
            self.assertEqual(again.memory.profile()['tone'], 'на вы')
            self.assertEqual(again.memory.task, 'Основная')
            again.switch('Первый')
            self.assertEqual(again.memory.task, 'Доклад')
            self.assertEqual(again.memory.profile()['tone'], 'на ты')
            self.assertNotIn('Только сейчас', json.dumps(again.memory.dump(), ensure_ascii=False))
            again.ask('С этого момента примеры без кода.', client, lambda _: None)
            self.assertEqual(again.memory.profile()['code'], 'без кода')

    def test_uncertain_score_and_word_limit(self):
        client = Mock()
        client.ask.return_value = '{}'
        self.assertEqual(assess(client, 'А', COMMON, 'Короткий ответ.', lambda _: None), 'не подтверждено')
        self.assertEqual(assess(client, 'А', COMMON, 'слово ' * 71, lambda _: None), 'нарушено')


if __name__ == '__main__':
    unittest.main()
