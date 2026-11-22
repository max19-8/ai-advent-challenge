"""День 14. Инварианты сервиса заказов."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import sys

import requests
from dotenv import load_dotenv

from agent import Agent
from experiment import run
from policy import Registry


HERE = Path(__file__).resolve().parent
MODEL = 'qwen/qwen-2.5-72b-instruct'


class Client:
    def __init__(self, key, emit):
        self.key, self.emit = key, emit

    def ask(self, system, prompt, phase):
        if not self.key:
            raise ValueError('Заполните POLZA_API_KEY в корневом .env.')
        self.emit('Запрос LLM: ' + phase)
        response = requests.post('https://polza.ai/api/v1/chat/completions',
                                 headers={'Authorization': 'Bearer ' + self.key},
                                 json={'model': MODEL, 'temperature': 0, 'max_tokens': 900,
                                       'messages': [{'role': 'system', 'content': system},
                                                    {'role': 'user', 'content': prompt}]}, timeout=(10, 120))
        response.raise_for_status()
        data = response.json()
        usage = data.get('usage') or {}
        self.emit(f"API {phase}: вход {usage.get('prompt_tokens', 'н/д')}, "
                  f"выход {usage.get('completion_tokens', 'н/д')} ток.")
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop' or not choice['message']['content']:
            raise ValueError('Пустой или незавершённый ответ модели.')
        return choice['message']['content']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['chat', 'experiment'])
    parser.add_argument('--report', type=Path)
    parser.add_argument('--state', type=Path, default=HERE / 'state/rules.db', help='База правил для чата.')
    args = parser.parse_args()
    load_dotenv(HERE.parents[1] / '.env')
    report = args.report or (HERE / 'results.txt' if args.mode == 'experiment' else HERE / 'state/chat.txt')
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as stream:
        def emit(line):
            print(line, flush=True)
            print(line, file=stream, flush=True)

        emit(f'\nДень 14 | {args.mode} | {MODEL} | {datetime.now().astimezone().isoformat(timespec="seconds")}')
        client = Client(os.getenv('POLZA_API_KEY', '').strip(), emit)
        try:
            if args.mode == 'experiment':
                return 0 if run(client, emit) else 2
            agent = Agent(Registry(args.state), client)
            emit(agent.command(['/invariant', 'show']))
            emit('Команды: /invariant show, /invariant history, /invariant set ПРАВИЛО ЗНАЧЕНИЕ, /exit.')
            for line in sys.stdin:
                query = line.strip()
                if query == '/exit':
                    break
                if query:
                    emit('Пользователь: ' + query)
                    result = agent.handle(query)
                    emit('Ассистент: ' + result.text)
        except (OSError, ValueError, KeyError, IndexError, TypeError, requests.RequestException, sqlite3.Error) as exc:
            emit('Прогон прерван: ' + str(exc))
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
