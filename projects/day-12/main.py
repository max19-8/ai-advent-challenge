"""День 12. Персонализированный чат и эксперимент."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from experiment import run
from persona import Profiles, DAY11
import sys

sys.path.append(str(DAY11.parent / 'day-10'))
from tokens import save_json

HERE = Path(__file__).resolve().parent
MODEL = 'qwen/qwen-2.5-72b-instruct'


class Client:
    def __init__(self, key, emit):
        self.key = key
        self.emit = emit

    def ask(self, messages, phase):
        response = requests.post('https://polza.ai/api/v1/chat/completions',
                                 headers={'Authorization': f'Bearer {self.key}'},
                                 json={'model': MODEL, 'messages': messages, 'temperature': 0,
                                       'max_tokens': 1400}, timeout=(10, 120))
        response.raise_for_status()
        data = response.json()
        usage = data.get('usage') or {}
        self.emit(f"API {phase}: вход {usage.get('prompt_tokens', 'н/д')}, "
                  f"выход {usage.get('completion_tokens', 'н/д')} ток.")
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError(f"Незавершённый ответ {phase}: {choice.get('finish_reason')}")
        answer = choice['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'Пустой ответ {phase}.')
        return answer


def chat(client, emit, path, lines):
    profiles = Profiles.load(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else Profiles()
    emit('Команды: /user ИМЯ, /profile, /task ИМЯ, /new, /exit.')
    emit('Активный пользователь: ' + profiles.active)
    for line in lines:
        query = line.strip()
        if not query:
            continue
        emit('Вы: ' + query)
        if query == '/exit':
            break
        if query.startswith('/user '):
            profiles.switch(query[6:])
            emit('Активный пользователь: ' + profiles.active)
        elif query.startswith('/task '):
            profiles.memory.switch(query[6:])
        elif query == '/new':
            profiles.memory.clear('short')
        elif query == '/profile':
            emit(json.dumps(profiles.memory.profile(), indent=2, ensure_ascii=False))
            continue
        elif query.startswith('/'):
            emit('Неизвестная команда.')
            continue
        else:
            emit('Ассистент: ' + profiles.ask(query, client, emit))
        save_json(path, profiles.dump())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('chat', 'experiment'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    load_dotenv(HERE.parents[1] / '.env')
    state = HERE / 'state'
    report = args.report or (HERE / 'results.txt' if args.mode == 'experiment' else state / 'chat.txt')
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as output:
        def emit(text):
            print(text, flush=True)
            print(text, file=output, flush=True)

        emit(f'\nДень 12 | {args.mode} | {MODEL} | {datetime.now().astimezone().isoformat(timespec="seconds")}')
        try:
            key = os.getenv('POLZA_API_KEY', '').strip()
            if not key:
                raise ValueError('Заполните POLZA_API_KEY в корневом .env.')
            client = Client(key, emit)
            if args.mode == 'experiment':
                run(client, emit)
            else:
                chat(client, emit, state / 'profiles.json', sys.stdin)
        except (ValueError, requests.RequestException, KeyError, TypeError, IndexError, OSError) as error:
            emit(f'Прогон прерван: {error}')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
