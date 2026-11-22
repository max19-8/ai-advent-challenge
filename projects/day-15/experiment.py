"""Фиксированные реплики и независимые ожидаемые ответы для воспроизводимого прогона."""
import json
import re
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

from assistant import Assistant, Turn, dump


CHANGES = [
    ('Дату сдачи перенесли с 12 декабря на 19 декабря.', 'deadline', '12 декабря', '19 декабря',
     'Какой сейчас срок сдачи? Ответь только датой.'),
    ('Статус проекта теперь «Согласован».', 'status', 'Черновик', 'Согласован',
     'Какой сейчас статус проекта? Напиши только статус.'),
    ('Требование «Бумажный дубль» отменили.', 'requirement', 'Бумажный дубль', None,
     'Есть ли ещё действующее требование? Если оно отменено, ответь только «отменено».'),
    ('Вместо подхода «Каскад» выбираем «Пилот».', 'approach', 'Каскад', 'Пилот',
     'Какой подход мы сейчас выбрали? Напиши только название.'),
]


def normalize(text):
    return ' '.join(re.findall(r'[\w-]+', text.casefold()))


def completed_turns(report, model):
    headers = list(re.finditer(r'^День 15 \| experiment \| ([^|]+) \| .*', report, re.M))
    if not headers or headers[-1].group(1).strip() != model:
        raise ValueError('Нет протокола эксперимента с текущей моделью.')
    body = report[headers[-1].end():]
    if 'Прогон прерван:' not in body or '\nПервые ответы:' in body:
        raise ValueError('Продолжить можно только прерванный прогон.')
    turns = deque()
    for block in body.split('\nВы: ')[1:]:
        message, _, rest = block.partition('\n')
        lines, marker, answer = rest.partition('\nАссистент: ')
        if not marker:
            for line in reversed(rest.splitlines()):
                if line.startswith('Используется ранее полученное извлечение: '):
                    turns.append((message, json.loads(line.partition(': ')[2])))
                    break
                if line.startswith('Извлечение отклонено:') and ' Ответ модели: ' in line:
                    raw = json.loads(line.partition(' Ответ модели: ')[2])
                    if isinstance(raw, str):
                        turns.append((message, raw))
                    break
            break
        fields = dict(line.split(': ', 1) for line in lines.splitlines() if ': ' in line)
        answer = re.split(r'\n(?:OK —|НЕ ПРОЙДЕНО —|Первый ответ:|Перед завершением:|Журнал:|Прогон прерван:)', answer)[0].strip()
        if not answer or 'Журнал хода' not in fields or 'Промпт ответа' not in fields:
            raise ValueError('Неполная запись завершённой реплики в протоколе.')
        turns.append((message, json.loads(fields['Журнал хода']), json.loads(fields['Промпт ответа']), answer, fields.get('Ошибка обновления', '')))
    return turns


def restore_turn(app, task, message, saved, emit):
    if message != saved[0]:
        raise ValueError('Сценарий изменился относительно сохранённого прогона.')
    if len(saved) == 2:
        emit('Используется ранее полученное извлечение: ' + dump(saved[1]))
        return app.send(task, message, recorded_extract=saved[1])
    original, journal, messages, answer, error = saved
    stamp = app.memory.arrive(task, message)
    if journal:
        stamp = (journal[0]['at'], stamp[1])
    changes = [dict(kind=r['kind'], key=r['key'], value=r['new']) for r in journal]
    events = app.memory.apply(task, changes, stamp)
    if (events != [r['event'] for r in journal]
            or app.memory.snapshot(task) != json.loads(messages[-1]['content'])['memory']):
        raise ValueError('Восстановленная память расходится с сохранённым промптом.')
    emit('Восстановлено из протокола без обращения к модели.')
    emit('Журнал хода: ' + dump(journal))
    emit('Промпт ответа: ' + dump(messages))
    return Turn(answer, messages, events, error)


def run(client, emit, saved=None):
    checks = []
    counts = dict(correct=0, stale=0, other=0, leaks=0)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'memory.db'
        app = Assistant(path, client, emit)
        task = 'Подготовка релиза'

        def say(message):
            emit('\nВы: ' + message)
            result = restore_turn(app, task, message, saved.popleft(), emit) if saved else app.send(task, message)
            if result.error:
                emit('Ошибка обновления: ' + result.error)
                checks.append(False)
            emit('Ассистент: ' + result.answer)
            return result

        def check(label, ok):
            checks.append(ok)
            emit(('OK — ' if ok else 'НЕ ПРОЙДЕНО — ') + label)

        say('Срок сдачи — 12 декабря. Статус проекта — Черновик. Требование — Бумажный дубль. '
            'Выбранный подход — Каскад.')
        say('Приняли отдельное решение: вести протокол встреч.')
        say('Приняли ещё одно независимое решение: публиковать отчёт по пятницам.')
        say('Временная рабочая заметка до завершения: пропуск Липа-73.')
        old_values = []
        for correction, key, old, new, probe in CHANGES:
            before = {r['key']: r['value'] for r in app.memory.snapshot(task)['facts'] if r['kind'] == 'current'}
            check(f'Исходное значение {key}', normalize(before.get(key) or '') == normalize(old))
            say(correction)
            after = {r['key']: r['value'] for r in app.memory.snapshot(task)['facts'] if r['kind'] == 'current'}
            actual = after.get(key)
            check(f'Текущая запись {key}', key in after and (actual == new if new is None else
                                                           normalize(actual or '') == normalize(new)))
            old_values.append(old)
            result = say(probe)
            answer = normalize(result.answer)
            stale = any(normalize(x) in answer for x in old_values)
            expected = normalize(new or 'отменено')
            category = 'stale' if stale else 'correct' if answer == expected else 'other'
            counts[category] += 1
            serialized = dump(result.messages).casefold()
            found = [x for x in old_values if x.casefold() in serialized]
            counts['leaks'] += bool(found)
            emit(f'Первый ответ: {category}. Ожидалось: {expected}. Старые значения в промпте: {found}.')
            check('Вопрос не изменил память', not result.events)

        before = app.memory.snapshot(task)['facts']
        durable = [r for r in before if r['kind'] != 'working']
        decisions = [r['value'] or '' for r in durable if r['kind'] == 'decision']
        check('Сохранены оба самостоятельных решения', any('протокол' in x.lower() for x in decisions)
              and any('пятниц' in x.lower() for x in decisions) and len(decisions) >= 2)
        check('Есть рабочая заметка с маркером', any(r['kind'] == 'working' and 'Липа-73' in (r['value'] or '')
                                                    for r in before))
        emit('Перед завершением: ' + dump(before))
        cleared = app.memory.finish(task)
        app = Assistant(path, client, emit)
        restored = app.memory.snapshot(task)
        emit('После завершения и перезапуска: ' + dump(restored))
        check('Рабочие записи очищены, долговременные сохранены', cleared > 0 and restored['closed']
              and restored['facts'] == durable)
        probe = say('Какой пропуск указан во временной заметке? Если сведений нет, ответь только «нет данных».')
        check('Заметка отсутствует в запросе и ответе', 'липа-73' not in dump(probe.messages).casefold()
              and normalize(probe.answer) == 'нет данных')
        check('Текущий подход доступен после перезапуска',
              normalize(say('Какой подход был выбран? Ответь только названием.').answer) == 'пилот')
        emit('Журнал: ' + dump(app.memory.history(task)))
    emit(f"Первые ответы: верных {counts['correct']}/4, устаревших {counts['stale']}/4, "
         f"прочих {counts['other']}/4. Промптов со старым значением: {counts['leaks']}/4.")
    return all(checks) and counts['correct'] == 4 and counts['leaks'] == 0
