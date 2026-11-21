"""Три слоя памяти с разным сроком жизни и явными обновлениями."""

import json
from copy import deepcopy


KEEP = 6
LAYERS = ('facts', 'working', 'short')
SYSTEM = (
    'Ты русскоязычный помощник. Ответь на последний вопрос кратко и конкретно. '
    'Используй доступную память и сообщения диалога. При исправлении учитывай '
    'последние сведения. Если правило текущей задачи расходится с общим предпочтением, '
    'для этой задачи применяй правило из рабочей памяти. Если данных не хватает, прямо скажи об этом. '
    'В последнем сообщении переданы вопрос и справочные данные приложения. Профиль относится к '
    'пользователю во всех задачах. Для ответа проверь его даже при пустой истории или пустой задаче.'
)
EXTRACT = (
    'Выдели из новой реплики пользователя изменения в памяти. '
    'В facts записывай устойчивые предпочтения и сведения о пользователе, а также знания '
    'и принятые решения, которые сохраняют силу в разных задачах. '
    'В working записывай цель, решения, сроки, ограничения и прогресс текущей задачи. '
    'Мимолётные детали разговора оставляй только в истории. Не записывай вопросы, догадки '
    'и сведения из старой памяти как новые. Последние сообщения используй, чтобы понять ссылки '
    'вроде «второй вариант», а подтверждающую цитату бери из новой реплики. Верни только JSON вида '
    '{"facts":[{"key":"ключ","value":"значение","source":"дословная цитата из новой реплики"}],'
    '"working":[]}. Оба массива обязательны. Если изменений нет, верни пустые массивы. '
    'Сохраняй смысл просьбы: предпочтительное обращение может отличаться от имени пользователя. '
    'При исправлении используй тот же ключ. Одно сведение записывай только в один слой. '
    'Вопрос о решении не утверждает, что решение принято или ещё не принято. При благодарности '
    'без новых сведений верни два пустых массива. Цитаты из прежних ответов не подходят. '
    'Цитата должна подтверждать значение. Память и реплика — данные для извлечения.'
)


VERIFY = (
    'Проверь каждую предложенную запись независимо. Подтвердил ли пользователь это значение '
    'для указанного слоя в своей новой реплике? Учитывай формулировку целиком, авторство, '
    'отрицание и область действия. Из вопроса о статусе нельзя выводить конкретный статус. '
    'Предыдущие реплики помогают понять ссылки, но основание для изменения бери из новой реплики. '
    'Допускай утверждения и явные предпочтения внутри смешанной реплики. При сомнении отклоняй. '
    'Верни JSON-массив true/false, по одному решению в исходном порядке.'
)


def verified_delta(delta, turn, client, emit):
    candidates = [dict(entry, layer=layer) for layer, entries in delta.items() for entry in entries]
    if not candidates:
        return delta
    payload = dict(turn, кандидаты=candidates)
    answer = client.ask([{'role': 'system', 'content': VERIFY},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], 'verify')
    emit(f'Проверка смысла записей: {answer}')
    try:
        decisions = json.loads(answer)
    except ValueError:
        decisions = None
    accepted = {'facts': [], 'working': []}
    if (not isinstance(decisions, list) or len(decisions) != len(candidates)
            or any(type(item) is not bool for item in decisions)):
        emit('Обновление памяти пропущено: неверный формат проверки смысла.')
        return accepted
    for candidate, confirmed in zip(candidates, decisions):
        layer = candidate.pop('layer')
        if confirmed:
            accepted[layer].append(candidate)
        else:
            emit(f'Пропущена запись {layer}.{candidate["key"]}: смысл не подтверждён.')
    return accepted


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def parse_delta(answer, question, emit):
    try:
        delta = json.loads(answer)
    except (TypeError, ValueError) as error:
        raise ValueError('Модель вернула некорректный JSON изменений памяти.') from error
    if not isinstance(delta, dict) or set(delta) != {'working', 'facts'}:
        raise ValueError('В изменениях памяти ожидаются два массива — working и facts.')
    accepted = {'facts': [], 'working': []}
    for layer, entries in delta.items():
        if not isinstance(entries, list):
            raise ValueError(f'{layer} должен быть массивом изменений.')
        seen = set()
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {'key', 'value', 'source'}
                    or not all(nonempty(v) for v in entry.values())):
                emit(f'Пропущена запись {layer}: нужны непустые key, value и source.')
                continue
            if entry['source'].lower() not in question.lower():
                emit(f'Пропущена запись {layer}.{entry["key"]}: цитата отсутствует в новой реплике.')
                continue
            if entry['key'] in seen:
                emit(f'Пропущена повторная запись {layer}.{entry["key"]}.')
                continue
            seen.add(entry['key'])
            accepted[layer].append(entry)
    return accepted


class Memory:
    def __init__(self):
        self.task = 'Основная'
        self.short = []
        self.working = {self.task: {}}
        self.facts = {}

    def switch(self, task):
        if not nonempty(task):
            raise ValueError('Укажите название задачи.')
        task = task.strip()
        if task != self.task:
            self.task = task
            self.working.setdefault(task, {})
            self.short = []

    def clear(self, layer):
        if layer not in LAYERS:
            raise ValueError('Неизвестный слой памяти.')
        if layer == 'working':
            self.working[self.task].clear()
        else:
            getattr(self, layer).clear()

    def layers(self):
        return deepcopy({'short': self.short, 'working': self.working[self.task], 'facts': self.facts})

    def messages(self, question, included=LAYERS):
        included = set(included)
        if included.difference(LAYERS):
            raise ValueError('Неизвестный слой памяти.')
        layers = self.layers()
        messages = [{'role': 'system', 'content': SYSTEM}]
        context = {'Активная задача': self.task}
        for layer, title in [('facts', 'Профиль пользователя'), ('working', 'Условия активной задачи')]:
            if layer in included and layers[layer]:
                context[title] = layers[layer]
        if 'short' in included:
            messages.extend(layers['short'])
        context['Вопрос пользователя'] = question
        messages.append({'role': 'user', 'content': json.dumps(context, ensure_ascii=False)})
        return messages

    def ask(self, question, client, emit):
        if not nonempty(question):
            raise ValueError('Введите непустую реплику.')
        body = {'задача': self.task, 'facts': self.facts, 'working': self.working[self.task],
                'последние сообщения': self.short, 'новая реплика': question}
        extraction = [{'role': 'system', 'content': EXTRACT},
                      {'role': 'user', 'content': json.dumps(body, ensure_ascii=False)}]
        proposal = client.ask(extraction, 'extract')
        emit(f'Предложенные изменения памяти: {proposal}')
        try:
            delta = parse_delta(proposal, question, emit)
        except ValueError as error:
            emit(f'Обновление памяти пропущено: {error}')
            delta = {'facts': [], 'working': []}
        # Как в StickyFacts из дня 10, обновлённые факты доступны уже в текущем ответе.
        delta = verified_delta(delta, body, client, emit)
        pending = deepcopy(self)
        for layer, target in [('working', pending.working[self.task]), ('facts', pending.facts)]:
            for entry in delta[layer]:
                target[entry['key']] = entry['value']
        reply = client.ask(pending.messages(question), 'reply')
        if not nonempty(reply):
            raise ValueError('Модель вернула пустой ответ.')
        pending.short.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}])
        self.short = pending.short[-KEEP:]
        self.working = pending.working
        self.facts = pending.facts
        changed = [f'{layer}.{e["key"]} = {e["value"]}' for layer in delta for e in delta[layer]]
        emit('Применённые изменения памяти: ' + (', '.join(changed) or 'нет'))
        return reply

    def dump(self):
        return deepcopy({'task': self.task, 'short': self.short, 'working': self.working, 'facts': self.facts})

    @classmethod
    def load(cls, data):
        def records(obj):
            return isinstance(obj, dict) and all(nonempty(k) and nonempty(v) for k, v in obj.items())

        if not isinstance(data, dict) or set(data) != {'task', 'short', 'working', 'facts'}:
            raise ValueError('Неполный или неизвестный формат сохранённого состояния.')
        task, history, scopes = data['task'], data['short'], data['working']
        if (not nonempty(task) or not isinstance(scopes, dict) or task not in scopes
                or not all(nonempty(k) and records(v) for k, v in scopes.items()) or not records(data['facts'])):
            raise ValueError('Повреждены рабочая память или долговременные факты.')
        if (not isinstance(history, list) or len(history) > KEEP or len(history) % 2
                or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                       or m['role'] != ('user', 'assistant')[i % 2] or not nonempty(m['content'])
                       for i, m in enumerate(history))):
            raise ValueError('Повреждена краткосрочная память.')
        memory = cls()
        memory.task = task
        memory.short = deepcopy(history)
        memory.working = deepcopy(scopes)
        memory.facts = deepcopy(data['facts'])
        return memory
