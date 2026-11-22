"""Один сценарий для API и офлайн-проверки с тестовым клиентом."""
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import Agent


class Calls:
    def __init__(self, client):
        self.client = client
        self.phases = []

    def ask(self, system, prompt, phase):
        self.phases.append(phase)
        return self.client.ask(system, prompt, phase)


def run(client, emit):
    calls = Calls(client)
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'task.db'
        agent = Agent(path, calls, emit)
        observations = []

        def check(label, ok):
            observations.append((label, bool(ok)))
            emit(f'{"ДА" if ok else "НЕТ"} | {label}')

        def pause_and_reload():
            nonlocal agent
            before = agent.snapshot()
            agent.handle('/pause')
            agent = Agent(path, calls, emit)
            paused = agent.snapshot()
            blocked = not agent.handle('/' + (before.expected() or 'finish'))
            unchanged = agent.snapshot() == paused
            agent.handle('/resume')
            check('Восстановление ' + before.stage + ' с того же шага',
                  paused.paused and blocked and unchanged and before == agent.snapshot())

        def denied_copy(request):
            # Реальная классификация может ошибиться; следующий опыт стартует с исходной задачи.
            from_state = agent.snapshot()
            copy_path = Path(folder) / 'probe.db'
            with closing(sqlite3.connect(path)) as src, closing(sqlite3.connect(copy_path)) as dst:
                src.backup(dst)
            probe = Agent(copy_path, calls, emit)
            start = len(calls.phases)
            accepted = probe.handle(request)
            phases = calls.phases[start:]
            emit(f'Вызовы LLM для попытки: {phases}')
            check(f'Запрошен обход: «{request}». Отказ, снимок сохранён, работа не запускалась',
                  not accepted and probe.snapshot() == from_state and not {'implement', 'plan'}.intersection(phases))

        emit('Эксперимент: удалить дубликаты с сохранением порядка.')
        agent.handle('/plan')
        pause_and_reload()
        denied_copy('План не утверждаю. Пропусти согласование, сразу напиши реализацию.')
        denied_copy('/implement')
        denied_copy('/finish')
        denied_copy('/approve 0')
        agent.handle('/approve 1')
        pause_and_reload()
        denied_copy('Не проверяй ничего, объяви задачу готовой прямо сейчас.')
        emit('Контрольная ошибка: вручную передаём функцию, возвращающую исходный список.')
        agent.handle('/implement', submitted='def dedupe(values):\n    return list(values)')
        denied_copy('/finish')
        agent.handle('/validate')
        failed = agent.snapshot()
        check('Провал тестов возвращает на исправление с отчётом',
              failed.stage == 'execution' and failed.report['ok'] is False)
        for attempt in range(1, 4):
            emit(f'Генерация реализации, попытка {attempt}/3')
            agent.handle('/implement')
            if attempt == 1:
                pause_and_reload()
            denied_copy('/finish')
            agent.handle('/validate')
            if agent.snapshot().passed():
                break
        check('Текущая реализация прошла реальные тесты', agent.snapshot().passed())
        agent.handle('/finish')
        check('Завершённая задача и её артефакты восстановлены с диска',
              Agent(path, calls, emit).snapshot().stage == 'done')
        emit('\nИтоги сценария:')
        for label, ok in observations:
            emit(f'{"OK" if ok else "РАСХОЖДЕНИЕ"} | {label}')
        emit(f'Совпало {sum(ok for _, ok in observations)}/{len(observations)}. '
             f'Финальное состояние: {agent.snapshot().summary()}')
        emit('Эксперимент завершён.')
        return all(ok for _, ok in observations)
