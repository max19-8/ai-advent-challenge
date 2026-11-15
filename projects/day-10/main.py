"""Сравнение трёх стратегий на реальных диалогах и ветвление от checkpoint."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent import (Agent, AgentError, Branches, STRATEGIES, SYSTEM, SUMMARY_SYSTEM, FACTS_SYSTEM,
                   KEEP, EVERY, REPLY_OUTPUT, SUMMARY_OUTPUT, FACTS_OUTPUT)
from tokens import (Metrics, TokenCounter, TOKENIZER_REPO, TOKENIZER_REVISION,
                    render_reports, save_json)


DIRECTORY = Path(__file__).resolve().parent
MODEL = 'qwen/qwen-2.5-72b-instruct'
API_URL = 'https://polza.ai/api/v1/chat/completions'
STAGES = ('dialogue', 'probe', 'branches')
DIALOGUE = [
    'Готовим учебный вебинар. Ответственная за подготовку — Алина Ветрова. Подтверди.',
    'Срок готовности материалов — 20 декабря 2026 года. Зафиксируй этот дедлайн.',
    'Первоначальный бюджет — 30 000 рублей. Пока это черновая сумма, её могут изменить.',
    'Важное ограничение: запись вебинара запрещена. Зафиксируй его.',
    'Принято решение проводить вебинар только онлайн. Очного участия не будет.',
    'На вебинаре будет 18 участников. Запомни количество.',
    'Начало вебинара — в 19:30 по московскому времени. Зафиксируй время и часовой пояс.',
    'Исправление бюджета: утверждены 42 000 рублей. Черновые 30 000 рублей больше не актуальны.',
    'Продолжительность вебинара — 90 минут. Запомни её.',
    'Выбранная платформа для вебинара — Контур.Толк. Зафиксируй название.',
    'Материалы для участников передаём одним PDF-файлом, без видео. Запомни формат.',
    'Вводные закончены. Подтверди готовность продолжить подготовку, не перечисляя параметры заново.',
]
FACTS = [
    ('responsible', 'Кто отвечает за подготовку вебинара? Назови имя и фамилию.', 'Алина Ветрова', 1),
    ('deadline', 'К какому сроку должны быть готовы материалы? Укажи полную дату.', '20 декабря 2026 года', 2),
    ('budget', 'Какой бюджет в итоге утверждён? Назови действующую сумму.', '42 000 рублей', 8),
    ('restriction', 'Какое ограничение установлено на запись вебинара?', 'Запись вебинара запрещена', 4),
    ('decision', 'Какой формат проведения вебинара мы выбрали?', 'Только онлайн, без очного участия', 5),
    ('participants', 'Сколько участников будет на вебинаре?', '18 участников', 6),
    ('start', 'Во сколько начинается вебинар и по какому времени?', '19:30 по московскому времени', 7),
    ('duration', 'Какова продолжительность вебинара?', '90 минут', 9),
    ('platform', 'Какая платформа выбрана для вебинара?', 'Контур.Толк', 10),
    ('materials', 'В каком формате передаём материалы участникам?', 'Один PDF-файл, без видео', 11),
]

BRANCHES = {
    'A': ['В этой ветке утверждённый бюджет меняется на 50 000 рублей. Прежняя сумма больше не действует.',
          'В этой ветке срок готовности материалов меняется на 22 декабря 2026 года. Прежний срок отменён.'],
    'B': ['В этой ветке утверждённый бюджет меняется на 35 000 рублей. Прежняя сумма больше не действует.',
          'В этой ветке срок готовности материалов меняется на 18 декабря 2026 года. Прежний срок отменён.'],
}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def signature():
    return digest([MODEL, SYSTEM, SUMMARY_SYSTEM, FACTS_SYSTEM, KEEP, EVERY,
                   REPLY_OUTPUT, SUMMARY_OUTPUT, FACTS_OUTPUT, DIALOGUE, FACTS, BRANCHES])


def metadata(api_key):
    response = requests.get('https://polza.ai/api/v1/models',
                            headers={'Authorization': f'Bearer {api_key}'}, timeout=(10, 60))
    response.raise_for_status()
    model = next((item for item in response.json()['data'] if item.get('id') == MODEL), None)
    if model is None or type(model.get('context_length')) is not int:
        raise AgentError(f'Модель {MODEL} или её context_length не найдены в каталоге.')
    return {'model': MODEL, 'context_length': model['context_length'],
            'tokenizer': TOKENIZER_REPO, 'tokenizer_revision': TOKENIZER_REVISION,
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'temperature': 0}


class Client:
    def __init__(self, api_key, counter, metrics, scope='comparison'):
        self.api_key, self.counter, self.metrics, self.scope = api_key, counter, metrics, scope

    def ask(self, messages, phase, item, max_tokens=REPLY_OUTPUT):
        record = {'scope': self.scope, 'phase': phase, 'item': item, 'state': 'sent',
                  'local': self.counter.messages(messages, generation=True)}
        self.metrics.records.append(record)
        self.metrics.save()
        try:
            response = requests.post(API_URL, headers={'Authorization': f'Bearer {self.api_key}'},
                                     json={'model': MODEL, 'messages': messages, 'temperature': 0,
                                           'max_tokens': max_tokens}, timeout=(10, 180))
            record['http'] = response.status_code
            data = response.json()
            if isinstance(data, dict):
                self.metrics.observe(record, data)
            if response.status_code >= 400 or (isinstance(data, dict) and data.get('error')):
                raise AgentError(f'API вернул HTTP {response.status_code}; подробности в metrics.txt.')
            choice = data['choices'][0]
            record['finish'] = choice.get('finish_reason')
            message = choice['message']
            answer = message.get('content')
            if isinstance(answer, str):
                record['answer'] = answer.replace(self.api_key, '[HIDDEN]')
                readable = answer
                if phase == 'facts':
                    try:
                        facts = json.loads(answer)
                        if isinstance(facts, dict):
                            readable = '\n'.join(f'{key}: {value}' for key, value in facts.items()) or '(нет фактов)'
                    except ValueError:
                        pass
                self.metrics.result(f'{self.scope} / {phase} / {item}', messages[-1]['content'], readable)
            if record['finish'] != 'stop':
                raise AgentError(f'Ответ {item} не завершён: finish_reason={record["finish"]}.')
            if (message.get('role') != 'assistant' or message.get('tool_calls')
                    or not isinstance(answer, str) or not answer.strip()):
                raise AgentError('Ожидался непустой текст assistant без вызовов инструментов.')
            record['state'] = 'received'
            return answer
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AgentError) as error:
            record['state'] = 'error'
            record['error'] = (response.text if 'response' in locals() else str(error)).replace(self.api_key, '[HIDDEN]')
            raise AgentError(str(error).replace(self.api_key, '[HIDDEN]')) from None
        finally:
            self.metrics.save()


def start_run(strategy, new, api_key):
    state = DIRECTORY / 'state' / strategy
    if state.exists() and not new:
        raise AgentError(f'Прогон {strategy} уже существует. Для нового укажи --stage dialogue --new.')
    info = metadata(api_key)
    if state.exists():
        archive = (DIRECTORY.parents[1].parent / 'ai-advent-challenge-archive/day-10'
                   / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / strategy)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(state), str(archive))
    run = {'metadata': info, 'strategy': strategy, 'signature': signature(),
           'stages': dict.fromkeys(STAGES, 'pending')}
    save_json(state / 'run.json', run)
    render_reports(DIRECTORY, api_key)
    return run


def checkpoint(client, state, run):
    snapshot = load_json(state / 'checkpoint.json')
    if digest(snapshot) != run['checkpoint_sha']:
        raise AgentError('Checkpoint был изменён после завершения диалога.')
    if snapshot['strategy'] != run['strategy']:
        raise AgentError('Стратегия checkpoint не соответствует прогону.')
    return Agent.restore(client, snapshot)


def dialogue_stage(client, state, run):
    agent = Agent(client, run['strategy'])
    transcript = []
    for number, question in enumerate(DIALOGUE, 1):
        answer = agent.ask(question, f'turn-{number:02d}')
        transcript.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
        save_json(state / 'dialogue.json', transcript)
        save_json(state / 'current.json', agent.snapshot())
    agent.memory.finalize(client)
    snapshot = agent.snapshot()
    save_json(state / 'checkpoint.json', snapshot)
    run['checkpoint_sha'] = digest(snapshot)
    run['context'] = {'messages_in_dialogue': len(transcript),
                      'recent_messages': len(agent.memory.history),
                      'local_tokens': client.counter.messages(agent.context(), generation=True)}
    if agent.strategy == 'summary':
        run['context'].update(covered_messages=agent.memory.covered, summary_updates=agent.memory.updates)
    if agent.strategy == 'facts':
        run['context']['fact_keys'] = len(agent.memory.facts)


def probe_stage(client, state, run):
    agent = checkpoint(client, state, run)
    initial = agent.snapshot()
    checks = []
    for key, question, expected, _ in FACTS:
        # probe() формирует новый запрос и не изменяет состояние агента.
        answer = agent.probe(question, key)
        checks.append({'id': key, 'question': question, 'expected': expected, 'answer': answer})
        save_json(state / 'checks.json', checks)
    if agent.snapshot() != initial:
        raise AgentError('Проверочные вопросы изменили память.')
    table = ['Деталь | Эталон | Ответ']
    table += [' | '.join(check[key].replace('\n', ' ').replace('|', '/')
                         for key in ('id', 'expected', 'answer')) for check in checks]
    client.metrics.result('Проверка деталей: ручная оценка смысла ответов',
                          'Сопоставить ответы с эталонами', '\n'.join(table))


def branches_stage(client, state, run):
    original = checkpoint(client, state, run).snapshot()
    store = Branches(state / 'branches', client)
    for name in BRANCHES:
        store.fork(name, original)
    for name, questions in BRANCHES.items():
        store.switch(name)
        for number, question in enumerate(questions, 1):
            store.ask(question, f'{name}-turn-{number}')
    saved = {name: load_json(state / 'branches' / (name + '.json')) for name in BRANCHES}
    probes = [fact for fact in FACTS if fact[0] in ('budget', 'deadline')]
    expected = {'A': {'budget': '50 000 рублей', 'deadline': '22 декабря 2026 года'},
                'B': {'budget': '35 000 рублей', 'deadline': '18 декабря 2026 года'}}
    checks = []
    # Действительно перечитываем ветки с диска: A → B → A после их продолжения.
    for number, name in enumerate(('A', 'B', 'A'), 1):
        agent = store.switch(name)
        for key, question, _, _ in probes:
            answer = agent.probe(question, f'switch-{number}-{name}-{key}')
            checks.append({'branch': name, 'switch': number, 'id': key,
                           'expected': expected[name][key], 'answer': answer})
            save_json(state / 'branch-checks.json', checks)
    base = checkpoint(client, state, run)
    for key, question, expected_value, _ in probes:
        answer = base.probe(question, 'checkpoint-' + key)
        checks.append({'branch': 'checkpoint', 'id': key, 'expected': expected_value, 'answer': answer})
        save_json(state / 'branch-checks.json', checks)
    if load_json(state / 'checkpoint.json') != original or any(
            load_json(state / 'branches' / (name + '.json')) != snapshot for name, snapshot in saved.items()):
        raise AgentError('Переключение или проверка изменили сохранённую ветку/checkpoint.')
    table = ['Ветка | Деталь | Эталон | Ответ']
    table += [' | '.join(str(check[key]).replace('\n', ' ').replace('|', '/')
                         for key in ('branch', 'id', 'expected', 'answer')) for check in checks]
    client.metrics.result('Ветвление: ручная оценка ответов',
                          'Сравнить A, B и исходный checkpoint', '\n'.join(table))


def execute(strategy, args, api_key, counter):
    state = DIRECTORY / 'state' / strategy
    run = (start_run(strategy, args.new, api_key) if args.stage == 'dialogue'
           else load_json(state / 'run.json'))
    if run['signature'] != signature():
        raise AgentError('Настройки изменились после начала прогона. Нужен новый диалог.')
    for required in STAGES[:STAGES.index(args.stage)]:
        if run['stages'][required] != 'complete':
            raise AgentError(f'Сначала заверши {strategy} --stage {required}.')
    if run['stages'][args.stage] != 'pending':
        raise AgentError(f'{strategy}: этап {args.stage} уже запускался. Автоповтор отключён.')
    metrics = Metrics(DIRECTORY, strategy, api_key)
    client = Client(api_key, counter, metrics, 'branches' if args.stage == 'branches' else 'comparison')
    run['stages'][args.stage] = 'running'
    save_json(state / 'run.json', run)
    try:
        metrics.save()
        {'dialogue': dialogue_stage, 'probe': probe_stage, 'branches': branches_stage}[args.stage](client, state, run)
        run['stages'][args.stage] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        run['stages'][args.stage] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        run['error'] = str(error).replace(api_key, '[HIDDEN]')
        raise
    finally:
        save_json(state / 'run.json', run)
        metrics.save()


def main():
    parser = argparse.ArgumentParser(description='День 10. Три стратегии памяти и независимые ветки.')
    parser.add_argument('--strategy', choices=('all', *STRATEGIES), default='all')
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--new', action='store_true', help='Архивировать прежний прогон выбранной стратегии.')
    parser.add_argument('--prepare', action='store_true', help='Проверить каталог модели и токенизатор без генерации.')
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.stage):
        parser.error('Укажи либо --prepare, либо --stage dialogue/probe/branches.')
    if args.new and args.stage != 'dialogue':
        parser.error('--new используется только с --stage dialogue.')
    load_dotenv(DIRECTORY.parents[1] / '.env')
    api_key = os.getenv('POLZA_API_KEY', '').strip()
    if not api_key:
        raise SystemExit('Заполни POLZA_API_KEY в корневом .env.')
    try:
        counter = TokenCounter()
        if args.prepare:
            metadata(api_key)
            return 0
        strategies = tuple(STRATEGIES) if args.strategy == 'all' else (args.strategy,)
        for strategy in strategies:
            execute(strategy, args, api_key, counter)
        return 0
    except KeyboardInterrupt:
        return 130
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, OSError, AgentError) as error:
        raise SystemExit(str(error).replace(api_key, '[HIDDEN]')) from None


if __name__ == '__main__':
    raise SystemExit(main())
