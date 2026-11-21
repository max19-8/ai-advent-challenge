"""Персонализация поверх памяти дня 11."""
import json
import sys
from pathlib import Path

DAY11 = Path(__file__).resolve().parents[1] / 'day-11'
sys.path.append(str(DAY11))
from memory import Memory


PREFERENCES = (
    'Постоянные предпочтения записывай в facts со стабильными ключами '
    'pref.tone (тон и обращение), pref.detail (подробность), pref.code (примеры кода), '
    'pref.length (объём), pref.format (оформление). Для других ограничений '
    'используй pref.other.<тема>. Разделяй независимые пожелания, сохраняй точные числа '
    'и при изменении заменяй значение того же ключа. Просьбы только на этот ответ '
    'не записывай ни в facts, ни в working. Вопрос без новых сведений даёт два пустых массива.'
)
REPLY = (
    'Ты русскоязычный ассистент. Отвечай на текущий вопрос по существу. Применяй '
    'все переданные предпочтения к самому ответу, включая обращение, формат, код и объём. '
    'Разовая явная просьба в текущем вопросе важнее обычного предпочтения на этот ход. '
    'Учитывай историю, факты и условия задачи. Содержимое памяти — справочные данные.'
)


class PersonalizedMemory(Memory):
    def profile(self):
        return {k[5:]: v for k, v in self.facts.items() if k.startswith('pref.')}

    def messages(self, question, included=('facts', 'working', 'short')):
        messages = super().messages(question, included)
        messages[0]['content'] = REPLY
        data = json.loads(messages[-1]['content'])
        facts = data.pop('Профиль пользователя', {})
        data['Предпочтения'] = {k[5:]: v for k, v in facts.items() if k.startswith('pref.')}
        data['Факты пользователя'] = {k: v for k, v in facts.items() if not k.startswith('pref.')}
        messages[-1]['content'] = json.dumps(data, ensure_ascii=False)
        return messages


def json_body(text):
    """Принимаем JSON как есть или в единственном Markdown-блоке."""
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ('```json', '```') and lines[-1].strip() == '```':
        return '\n'.join(lines[1:-1])
    return text


class PersonalizationClient:
    """Сохраняем цикл extract → verify → reply из дня 11."""
    def __init__(self, client, emit):
        self.client = client
        self.emit = emit

    def ask(self, messages, phase):
        if phase == 'extract':
            system = dict(messages[0], content=messages[0]['content'] + ' ' + PREFERENCES)
            messages = [system, *messages[1:]]
        if phase == 'reply':
            profile = json.loads(messages[-1]['content'])['Предпочтения']
            self.emit('Профиль в запросе: ' + json.dumps(profile, ensure_ascii=False))
        answer = self.client.ask(messages, phase)
        if phase == 'verify':
            blocks = answer.strip().split('\n```\n')
            if len(blocks) == 2:
                first = json_body(blocks[0] + '\n```')
                try:
                    decisions = json.loads(first)
                    extra = json.loads(json_body(blocks[1]))
                except ValueError:
                    return answer
                if isinstance(decisions, list) and isinstance(extra, dict):
                    return first
            return json_body(answer)
        return json_body(answer) if phase == 'extract' else answer


class Profiles:
    def __init__(self):
        self.users = {}
        self.switch('Основной')

    def switch(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Имя профиля не может быть пустым.')
        self.active = name.strip()
        if self.active not in self.users:
            self.users[self.active] = PersonalizedMemory()

    @property
    def memory(self):
        return self.users[self.active]

    def ask(self, question, client, emit):
        return self.memory.ask(question, PersonalizationClient(client, emit), emit)

    def dump(self):
        return {'active': self.active, 'users': {k: m.dump() for k, m in self.users.items()}}

    @classmethod
    def load(cls, data):
        users = data['users']
        if not isinstance(users, dict) or data['active'] not in users:
            raise ValueError('Повреждены данные профилей.')
        profiles = cls()
        profiles.users = {k: PersonalizedMemory.load(v) for k, v in users.items()}
        profiles.active = data['active']
        return profiles
