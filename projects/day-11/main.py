"""День 11. Интерактивный чат и эксперимент с независимыми слоями памяти."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from memory import Memory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'day-10'))
from tokens import TokenCounter, save_json

MODEL = 'qwen/qwen-2.5-72b-instruct'
URL = 'https://polza.ai/api/v1/chat/completions'


class Client:
    def __init__(self, key, emit):
        self.key = key
        self.emit = emit
        self.counter = TokenCounter()

    def ask(self, messages, phase):
        local = self.counter.messages(messages, generation=True)
        self.emit(f'Запрос {phase}, вход локально: {local} ток.')
        response = requests.post(URL, headers={'Authorization': f'Bearer {self.key}'},
                                 json={'model': MODEL, 'messages': messages, 'temperature': 0,
                                       'max_tokens': 1000 if phase == 'extract' else 350},
                                 timeout=(10, 120))
        response.raise_for_status()
        data = response.json()
        usage = data.get('usage') or {}
        self.emit(f"API {phase}: вход {usage.get('prompt_tokens', 'н/д')}, "
                  f"выход {usage.get('completion_tokens', 'н/д')} ток.")
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError(f"Незавершённый ответ API: {choice.get('finish_reason')}")
        answer = choice['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Пустой ответ модели.')
        return answer


def chat(client, emit, path):
    memory = Memory.load(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else Memory()
    emit('Команды: /task ИМЯ, /new (очистить диалог), /clear short|working|facts, /memory, /exit')
    emit(f'Текущая задача: {memory.task}')
    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        emit('Вы: ' + question)
        if question == '/exit':
            break
        if question == '/memory':
            emit(json.dumps(memory.layers(), indent=2, ensure_ascii=False))
            continue
        if question.startswith('/task ') and question[6:].strip():
            memory.switch(question[6:].strip())
            emit(f'Текущая задача: {memory.task}')
        elif question == '/new':
            memory.clear('short')
            emit('Начат новый диалог.')
        elif question in ('/clear short', '/clear working', '/clear facts'):
            layer = question.split()[1]
            memory.clear(layer)
            emit('Очищен слой ' + layer)
        elif question.startswith('/'):
            emit('Неизвестная команда.')
            continue
        else:
            answer = memory.ask(question, client, emit)
            emit('Ассистент: ' + answer)
        save_json(path, memory.dump())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['chat', 'experiment'])
    parser.add_argument('--report', type=Path, help='Файл журнала (дополняется при запуске)')
    args = parser.parse_args()
    load_dotenv(HERE.parents[1] / '.env')
    state = HERE / 'state'
    report = args.report or (state / 'chat.txt' if args.mode == 'chat' else HERE / 'results.txt')
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as stream:
        def emit(text):
            print(text, flush=True)
            print(text, file=stream, flush=True)

        emit(f'\nДень 11 | {args.mode} | {MODEL} | {datetime.now().astimezone().isoformat(timespec="seconds")}')
        try:
            key = os.getenv('POLZA_API_KEY', '').strip()
            if not key:
                raise ValueError('Заполните POLZA_API_KEY в корневом .env.')
            client = Client(key, emit)
            if args.mode == 'experiment':
                from experiment import run
                run(client, emit)
            else:
                chat(client, emit, state / 'memory.json')
        except (requests.RequestException, ValueError, OSError, KeyError, IndexError, TypeError) as exc:
            emit(f'Выполнение прервано: {exc}')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
