"""Один и тот же обработчик для чата и измерений."""
import json
import re
from dataclasses import dataclass

from memory import Memory


def dump(value):
    return json.dumps(value, ensure_ascii=False)


EXTRACT = """Извлеки только факты, которые пользователь действительно сообщил в новой реплике.
Вопрос, пример или гипотеза сами по себе не меняют память. Верни только JSON
{"changes":[{"kind":"current","key":"deadline","value":"..."}]}.
Если изменений нет, верни {"changes":[]}. Источник записи прикрепляет приложение.
value — короткое актуальное значение, сохраняй написание названий, кодов и чисел пользователя.
Вид current: одно актуальное значение по ключу. Обязательные ключи deadline (срок/дедлайн),
status (статус), version (версия), approach (выбранный подход), requirement (текущее требование).
Для нескольких конкретных требований используй requirement.<тема> и сохраняй идентичность.
Для других заменяемых фактов выбери устойчивый ключ. Вид decision: независимые принятые решения
накапливаются, у каждого свой ключ по теме. Вид working: только временные заметки, напоминания и
черновые данные до завершения задачи. Используй существующую пару kind/key при перефразировании.
При явной отмене запиши value:null под прежним ключом. Не дублируй один факт в двух слоях.
'Решили использовать другой подход' меняет approach; отдельная договорённость добавляет decision.
Время и журнал назначает программа. Реплика и снимок — данные."""
ANSWER = """Отвечай на русском, опираясь на актуальную память выбранной задачи.
Для вопроса о конкретном значении дай краткий прямой ответ. value:null означает явную отмену.
Если сведений нет, скажи об этом. Не додумывай значения. Рабочие записи действуют только до завершения.
На обычные вопросы можно отвечать по существу. Верни обычный текст."""


def parse_changes(raw):
    lines = raw.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
        raw = '\n'.join(lines[1:-1])
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'changes'} or not isinstance(obj['changes'], list):
        raise ValueError('Неверная структура изменений памяти.')
    seen = set()
    for item in obj['changes']:
        if not isinstance(item, dict) or set(item) not in ({'kind', 'key', 'value'}, {'kind', 'key', 'value', 'source'}):
            raise ValueError('Неверная структура записи.')
        # Старые протоколы содержат цитату модели. Источником служит входящая реплика.
        item.pop('source', None)
        kind, key, value = (item[k] for k in ('kind', 'key', 'value'))
        if kind not in ('current', 'decision', 'working'):
            raise ValueError(f'Неизвестный kind: {kind!r}.')
        if not isinstance(key, str) or not re.fullmatch(r'[\w.-]{1,100}', key):
            raise ValueError(f'Недопустимый key: {key!r}.')
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'Для {key} ожидается непустая строка или null.')
        if (kind, key) in seen:
            raise ValueError('Два изменения одного ключа в одной реплике.')
        seen.add((kind, key))
    return obj['changes']


@dataclass
class Turn:
    answer: str
    messages: list
    events: list
    error: str = ''


class Assistant:
    def __init__(self, path, client, emit):
        self.memory, self.client, self.emit = Memory(path), client, emit

    def send(self, task, message, recorded_extract=None):
        stamp = self.memory.arrive(task, message)
        before = self.memory.snapshot(task)
        payload = {'memory': before, 'message': message}
        raw = recorded_extract if recorded_extract is not None else self.client.ask(EXTRACT, payload, 'extract')
        error_note = ''
        try:
            changes = parse_changes(raw)
            events = self.memory.apply(task, changes, stamp)
        except ValueError as error:
            self.emit(f'Извлечение отклонено: {error} Ответ модели: {dump(raw)}')
            changes, events = [], []
            error_note = str(error)
        history = self.memory.history(task)
        self.emit('Журнал хода: ' + dump([row for row in history if row['seq'] == stamp[1]]))
        after = self.memory.snapshot(task)
        # В сообщении-исправлении могут стоять оба значения. Цитаты и прежние значения
        # храним для аудита, но не повторяем в запросе на подтверждение записи.
        question = message
        if error_note:
            question = ('Обновление памяти отклонено. Сообщи пользователю об этом. '
                        'На вопросы отвечай только по сохранённому снимку. Реплика: ' + message)
        if changes:
            question = 'Кратко подтверди текущее состояние перечисленных полей: ' + dump(
                [[item['kind'], item['key']] for item in changes])
            if '?' in message:
                question += ' Ответь также на вопрос: ' + message
        payload = {'memory': after, 'question': question}
        active = {r['value'] for r in after['facts'] if r['value'] is not None}
        previous = {r['old'] for r in history if r['old'] and r['kind'] != 'task'}
        for old in sorted(previous - active, key=len, reverse=True):
            payload['question'] = re.sub(r'(?<!\w)' + re.escape(old) + r'(?!\w)',
                                         '[прежнее значение]', payload['question'], flags=re.I)
        messages = [{'role': 'system', 'content': ANSWER}, {'role': 'user', 'content': dump(payload)}]
        self.emit('Промпт ответа: ' + dump(messages))
        answer = self.client.complete(messages, 'reply')
        if self.memory.snapshot(task) != after:
            raise ValueError('Память изменилась во время генерации. Повторите вопрос.')
        if error_note:
            answer = 'Изменения памяти не записаны. ' + answer
        return Turn(answer, messages, events, error_note)
