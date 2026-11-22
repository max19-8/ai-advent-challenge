"""День 15. Устаревание и конфликты памяти."""
import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from assistant import Assistant, dump
from experiment import completed_turns, run


HERE = Path(__file__).resolve().parent
MODEL = 'qwen/qwen-2.5-72b-instruct'


class Client:
    def __init__(self, emit):
        self.emit = emit

    def ask(self, instruction, payload, phase):
        return self.complete([{'role': 'system', 'content': instruction},
                              {'role': 'user', 'content': dump(payload)}], phase)

    def complete(self, messages, phase):
        key = os.getenv('POLZA_API_KEY', '').strip()
        if not key:
            raise ValueError('Укажите POLZA_API_KEY в корневом .env.')
        response = requests.post('https://polza.ai/api/v1/chat/completions',
                                 headers={'Authorization': 'Bearer ' + key},
                                 json={'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 1000},
                                 timeout=(10, 120))
        response.raise_for_status()
        data = response.json()
        choice = data['choices'][0]
        self.emit('API ' + phase + ': ' + dump(data.get('usage', {})))
        content = choice['message']['content']
        if choice.get('finish_reason') != 'stop' or not isinstance(content, str) or not content.strip():
            raise ValueError('Пустой или незавершённый ответ на этапе ' + phase)
        return content.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['chat', 'experiment'])
    parser.add_argument('--task', default='Основная')
    parser.add_argument('--state', type=Path, default=HERE / 'state/memory.db')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--resume', action='store_true', help='Продолжить прерванный эксперимент по отчёту.')
    args = parser.parse_args()
    load_dotenv(HERE.parents[1] / '.env')
    report = args.report or (HERE / 'results.txt' if args.mode == 'experiment' else HERE / 'state/chat.txt')
    saved = None
    if args.resume:
        if args.mode != 'experiment' or not report.is_file():
            parser.error('Для --resume нужен существующий отчёт эксперимента.')
        try:
            saved = completed_turns(report.read_text(encoding='utf-8'), MODEL)
        except ValueError as error:
            parser.error(str(error))
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as stream:
        def emit(text):
            print(text, flush=True)
            print(text, file=stream, flush=True)

        emit(f'\nДень 15 | {args.mode} | {MODEL} | {datetime.now().astimezone().isoformat()}')
        client = Client(emit)
        try:
            if args.mode == 'experiment':
                return 0 if run(client, emit, saved) else 2
            app = Assistant(args.state, client, emit)
            task = args.task
            emit(f'Задача: {task}. Команды: /task ИМЯ, /memory, /log, /finish, /exit.')
            for line in sys.stdin:
                query = line.strip()
                if not query:
                    continue
                emit('Вы: ' + query)
                if query == '/exit':
                    break
                if query.startswith('/task ') and query[6:].strip():
                    task = query[6:].strip()
                    emit('Задача: ' + task)
                elif query == '/finish':
                    emit('Задача завершена, очищено рабочих записей: ' + str(app.memory.finish(task)))
                elif query == '/memory':
                    emit(dump(app.memory.snapshot(task)))
                elif query == '/log':
                    emit(dump(app.memory.history(task)))
                elif query.startswith('/'):
                    emit('Неизвестная команда.')
                else:
                    try:
                        emit('Ассистент: ' + app.send(task, query).answer)
                    except (ValueError, requests.RequestException) as exc:
                        emit('Ход прерван: ' + str(exc))
        except (OSError, ValueError, TypeError, KeyError, IndexError, requests.RequestException, sqlite3.Error) as exc:
            emit('Прогон прерван: ' + str(exc))
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
