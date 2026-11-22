import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from assistant import Assistant, parse_changes
from experiment import run
from memory import Memory


def item(kind, key, value):
    return dict(kind=kind, key=key, value=value)


class MemoryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'memory.db'
        self.m = Memory(self.path)
        self.task = 'Первая'

    def test_delayed_correction_and_cancellation(self):
        older, newer = [self.m.arrive(self.task) for _ in range(2)]
        self.m.apply(self.task, [item('current', 'requirement', None)], newer)
        self.assertEqual(self.m.apply(self.task, [item('current', 'requirement', 'Бумага')], older), ['IGNORE_STALE'])
        self.assertIsNone(self.m.snapshot(self.task)['facts'][0]['value'])
        self.assertEqual(self.m.history(self.task)[-1]['event'], 'IGNORE_STALE')

    def test_timestamp_before_sequence_and_ties(self):
        with patch('memory.now', side_effect=['2026-11-22T12:00:00+00:00','2026-11-22T11:00:00+00:00',
                                              '2026-11-22T12:00:00+00:00']):
            for value, expected in [('Первый', 'CREATE'), ('Поздняя доставка', 'IGNORE_STALE'), ('Новый', 'REPLACE')]:
                self.assertEqual(self.m.apply(self.task, [item('current', 'status', value)],
                                              self.m.arrive(self.task)), [expected])
        event = self.m.history(self.task)[-1]
        self.assertEqual((event['old'], event['new']), ('Первый', 'Новый'))

    def test_lists_restart_and_task_isolation(self):
        updates = [item('decision','a','Один'), item('decision','b','Два'), item('working','pass','Липа')]
        for task in (self.task, 'Другая'):
            self.m.apply(task, updates, self.m.arrive(task))
        self.assertEqual(self.m.finish(self.task), 1)
        reopened = Memory(self.path)
        self.assertEqual([r['value'] for r in reopened.snapshot(self.task)['facts']], ['Один', 'Два'])
        self.assertEqual(len(reopened.snapshot('Другая')['facts']), 3)
        self.assertEqual(reopened.finish(self.task), 0)
        with self.assertRaises(ValueError):
            reopened.apply(self.task, updates, reopened.arrive(self.task))

    def test_malformed_batch_reports_failure_without_partial_write(self):
        app = Assistant(self.path, Mock(), lambda _: None)
        app.client.complete.return_value = 'Обновление отклонено.'
        good = item('current', 'version', 'v2')
        for bad in [dict(good, kind='other'), good, dict(good, value=[])]:
            app.client.ask.return_value = json.dumps({'changes': [good, bad]})
            result = app.send(self.task, 'Версия v2')
            self.assertTrue(result.error)
            self.assertEqual(self.m.snapshot(self.task)['facts'], [])
        app.client.ask.return_value = json.dumps({'changes': [good]})
        self.assertFalse(app.send(self.task, 'Теперь версия v2').error)

    def test_origin_is_the_actual_message_for_any_subject(self):
        app = Assistant(self.path, Mock(), lambda _: None)
        app.client.complete.return_value = 'Учтено.'
        for index, (msg, value) in enumerate([('Теперь встречаемся по средам.', 'среда'),
                                              ('Площадку сменили на «Маяк».', 'Маяк')]):
            change = item('current', 'field' + str(index), value)
            if index:
                change['source'] = 'ненадёжный текст модели'
            app.client.ask.return_value = json.dumps({'changes': [change]})
            self.assertFalse(app.send(self.task, msg).error)
            self.assertEqual(self.m.history(self.task)[-1]['source'], msg)

    def test_updates_and_question_never_replay_old_value(self):
        app = Assistant(self.path, Mock(), lambda _: None)
        app.client.complete.return_value = 'v2'
        for version in ['v1','v2']:
            app.client.ask.return_value = json.dumps({'changes': [
                dict(item('current','version',version), source=version)]})
            app.send(self.task, 'версия '+version)
        app.client.ask.return_value = '{"changes":[]}'
        result = app.send(self.task, 'Всё ещё v1?')
        self.assertNotIn('v1', str(result.messages))
        self.assertIs(result.messages, app.client.complete.call_args.args[0])
        self.assertEqual(result.answer, 'v2')

    def test_scenario_uses_model_first_answer(self):
        client = Mock()
        def extract(_, payload, phase):
            msg = payload['message']
            rules = [('Срок сдачи —', [('current','deadline','12 декабря'), ('current','status','Черновик'),
                                      ('current','requirement','Бумажный дубль'), ('current','approach','Каскад')]),
                     ('Приняли отдельное', [('decision','протокол','вести протокол встреч')]),
                     ('Приняли ещё', [('decision','отчёт','публиковать отчёт по пятницам')]),
                     ('Временная', [('working','пропуск','Липа-73')]),
                     ('Дату сдачи', [('current','deadline','19 декабря')]),
                     ('Статус проекта теперь', [('current','status','Согласован')]),
                     ('Требование «', [('current','requirement',None)]),
                     ('Вместо подхода', [('current','approach','Пилот')])]
            changes = [dict(item(*args), source=msg) for start, entries in rules if msg.startswith(start)
                       for args in entries]
            return json.dumps({'changes':changes})
        def reply(messages, phase):
            payload = json.loads(messages[-1]['content'])
            values = {r['key']:r['value'] for r in payload['memory']['facts']}
            q = payload['question']
            for word, key in [('срок','deadline'), ('статус','status'), ('требование','requirement'),
                              ('подход','approach'), ('пропуск','пропуск')]:
                if word in q:
                    return values.get(key) or ('отменено' if key == 'requirement' else 'нет данных')
            return 'Записано.'
        client.ask.side_effect, client.complete.side_effect = extract, reply
        report = []
        self.assertTrue(run(client, report.append), report)
        client.complete.side_effect = lambda messages, phase: '12 декабря'
        self.assertFalse(run(client, report.append))
        self.assertIn('устаревших 4/4', report[-1])


if __name__ == '__main__':
    unittest.main()
