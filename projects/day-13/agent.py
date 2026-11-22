"""Автомат одной учебной задачи. Все изменения проходят через handle()."""
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from validator import RULES, validate


TASK = 'Удалить дубликаты из списка целых чисел, сохранив порядок первых вхождений.'
# События, заканчивающие этап, и события внутри него.
ALLOWED = {'planning': ('plan', 'approve'), 'execution': ('implement',),
           'validation': ('validate', 'finish'), 'done': ()}
ACTIONS = ('plan', 'approve', 'implement', 'validate', 'finish', 'pause', 'resume', 'status')
ROUTE = (
    'Определи действие, которое пользователь ПРОСИТ, даже если оно запрещено. '
    'Верни только одно слово: plan (составить/изменить план), approve (согласие), '
    'implement (написать код), validate (проверить реализацию), finish (объявить готовым), '
    'pause, resume или status (объяснить состояние). Если просит сразу несколько этапов, '
    'назови самый дальний запрошенный этап. Не выполняй задачу.'
)


class Denied(ValueError):
    pass


@dataclass
class State:
    task: str = TASK
    stage: str = 'planning'
    paused: bool = False
    plan: str = ''
    revision: int = 0
    approved: int = 0
    source: str = ''
    report: dict = None

    def expected(self):
        if self.paused:
            return 'resume'
        if self.stage == 'planning':
            return 'approve' if self.plan else 'plan'
        if self.stage == 'execution':
            return 'implement'
        if self.stage == 'validation':
            return 'finish' if self.passed() else 'validate'
        return None

    def passed(self):
        return (self.report and self.report.get('ok') is True
                and self.report['sha256'] == digest(self.source))

    def step(self):
        if self.stage == 'planning':
            return f'Согласовать план v{self.revision}' if self.plan else 'Составить план'
        if self.stage == 'execution':
            return 'Исправить реализацию по отчёту' if self.report else 'Подготовить реализацию'
        if self.stage == 'validation':
            return 'Зафиксировать итог' if self.passed() else 'Проверить сохранённую реализацию'
        return 'Задача завершена'

    def summary(self):
        return (f'{self.stage} | шаг: {self.step()} | ожидается: {self.expected() or "—"}'
                + (' | пауза' if self.paused else ''))

    def guard(self, action, command, arg):
        if action not in ACTIONS:
            raise Denied('Неизвестное действие.')
        if action == 'status':
            return
        if self.paused:
            if action != 'resume':
                raise Denied('Задача на паузе: сначала /resume.')
            return
        if action == 'resume':
            raise Denied('Задача не на паузе.')
        if action == 'pause' and self.stage != 'done':
            return
        if action not in ALLOWED[self.stage]:
            raise Denied(f'Действие {action} недоступно на этапе {self.stage}.')
        if action == 'approve':
            if not self.plan:
                raise Denied('Сначала нужен сохранённый план.')
            if not command or arg != str(self.revision):
                raise Denied(f'Утвердить текущую версию может пользователь: /approve {self.revision}.')
        if action == 'implement' and (not self.plan or self.approved != self.revision):
            raise Denied('Нет утверждённой версии плана.')
        if action == 'finish' and not self.passed():
            raise Denied('Нужна успешная проверка текущей реализации.')


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def unfence(text):
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ('```python', '```') and lines[-1].strip() == '```':
        return '\n'.join(lines[1:-1])
    return text.strip()


class Agent:
    def __init__(self, path, client, emit=print):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.emit = emit
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS task (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO task VALUES (1, ?)', (json.dumps(asdict(State())),))

    @staticmethod
    def read(db):
        return State(**json.loads(db.execute('SELECT data FROM task WHERE id=1').fetchone()[0]))

    def snapshot(self):
        with closing(sqlite3.connect(self.path)) as db:
            return self.read(db)

    def handle(self, message, submitted=None):
        self.emit('Пользователь: ' + message)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            state = self.read(db)
            self.emit('До: ' + state.summary())
            command = message.startswith('/')
            if command:
                action, _, arg = message[1:].partition(' ')
                arg = arg.strip()
            else:
                action = self.client.ask(ROUTE, message, 'route').strip().lower()
                arg = ''
            self.emit('Действие: ' + action)
            try:
                state.guard(action, command, arg)
            except Denied as error:
                self.emit(f'Отклонено: {error} {state.summary()}')
                return False
            if action == 'status':
                self.emit(json.dumps(asdict(state), ensure_ascii=False, indent=2))
                return True
            if action == 'pause':
                state.paused = True
            elif action == 'resume':
                state.paused = False
            elif action == 'plan':
                state.plan = self.client.ask('Составь короткий пошаговый план на русском. '
                                             'Только план, без реализации или кода.',
                                             json.dumps([state.task, state.plan, message], ensure_ascii=False),
                                             'plan').strip()
                if not state.plan:
                    raise ValueError('Получен пустой план.')
                state.revision += 1
                state.approved = 0
                self.emit(f'План v{state.revision}: {state.plan}')
            elif action == 'approve':
                state.approved = state.revision
                state.stage = 'execution'
            elif action == 'implement':
                context = {'задача': state.task, 'утверждённый план': state.plan,
                           'прежняя реализация': state.source, 'результат проверки': state.report}
                source = submitted if submitted is not None else self.client.ask(
                    RULES + ' Верни только исходный код.', json.dumps(context, ensure_ascii=False), 'implement')
                state.source = unfence(source)
                if not state.source:
                    raise ValueError('Получена пустая реализация.')
                state.report = None
                state.stage = 'validation'
                self.emit('Реализация:\n' + state.source)
            elif action == 'validate':
                state.report = dict(validate(state.source), sha256=digest(state.source))
                self.emit('Проверка: ' + json.dumps(state.report, ensure_ascii=False))
                if not state.passed():
                    state.stage = 'execution'
            elif action == 'finish':
                state.stage = 'done'
            db.execute('UPDATE task SET data=? WHERE id=1', (json.dumps(asdict(state), ensure_ascii=False),))
        self.emit('После: ' + state.summary())
        return True
