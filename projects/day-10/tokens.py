"""Локальный токенизатор Qwen и текстовый отчёт фактических метрик API."""

import json
import ast
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests


TOKENIZER_REPO = "Qwen/Qwen2.5-72B-Instruct"
TOKENIZER_REVISION = "495f393"


class TokenCounter:
    def __init__(self):
        try:
            from tokenizers import Tokenizer
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ValueError("Установи зависимости: python -m pip install -r projects/day-10/requirements.txt") from None
        cache = Path.home() / ".cache/ai-advent-challenge" / TOKENIZER_REPO / TOKENIZER_REVISION
        cache.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            path = cache / name
            if not path.exists():
                # Только словарь и шаблон токенизатора, веса модели не скачиваются.
                response = requests.get(
                    f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/{name}",
                    timeout=(10, 120),
                )
                response.raise_for_status()
                json.loads(response.content)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
        self.tokenizer = Tokenizer.from_file(str(cache / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        config = json.loads((cache / "tokenizer_config.json").read_text())
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        self.template = environment.from_string(config["chat_template"])

    def text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def messages(self, messages, generation=False):
        prompt = self.template.render(messages=messages, tools=[],
                                      add_generation_prompt=generation)
        return self.text(prompt)


def integer(value):
    return value if type(value) is int and value >= 0 else None


def cost(usage):
    value = usage.get("cost_rub")
    if value is None:
        value = usage.get("cost")
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_json(path, data):
    save_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def total(records, key):
    values = [r.get(key) for r in records]
    if any(value is None for value in values):
        return None
    return sum((Decimal(value) if key == "cost" else value for value in values),
               Decimal(0) if key == "cost" else 0)


def display(value):
    return "н/д" if value is None else str(value)


def complexity(directory):
    """Непустые строки классов памяти, без строк с одними комментариями."""
    source = (directory / 'agent.py').read_text(encoding='utf-8')
    lines = source.splitlines()
    counts = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            counts[node.name] = sum(bool(line.strip()) and not line.lstrip().startswith('#')
                                    for line in lines[node.lineno - 1:node.end_lineno])
    return {'window': counts['Window'], 'facts': counts['Window'] + counts['StickyFacts'],
            'summary': counts['Window'] + counts['Summary']}


def render_reports(directory, api_key):
    """Единые читаемые отчёты; JSON и исходные ответы API остаются в локальном state/."""
    results = ['День 10. Сравнение стратегий управления контекстом',
               'Правильность ответов оцениваем вручную по эталонам, не по совпадению строк.']
    lines = ['День 10. Фактические метрики API',
             'Вход локально — токенизатор, вход/выход API и цена — usage сервиса.',
             'н/д означает отсутствие данных. Итоги незавершённых этапов частичные.', '']
    code = complexity(directory)
    all_records = []
    for strategy in ('window', 'facts', 'summary'):
        state = directory / 'state' / strategy
        if not (state / 'run.json').exists():
            continue
        run = json.loads((state / 'run.json').read_text())
        records = json.loads((state / 'usage.json').read_text()) if (state / 'usage.json').exists() else []
        all_records.extend(records)
        results += ['', '=== ' + strategy + ' ===', 'Этапы: ' + str(run['stages'])]
        if (state / 'results.txt').exists():
            results.append((state / 'results.txt').read_text())
        lines += ['=== ' + strategy + ' ===', 'Модель: ' + run['metadata']['model'],
                  'Начало UTC: ' + run['metadata']['started_at_utc'], 'Этапы: ' + str(run['stages']),
                  '№ | Раздел | Назначение | Обращение | Вход лок. | Вход API | Выход API | Кеш API | ₽ | HTTP | Завершение']
        for number, record in enumerate(records, 1):
            values = [number, record['scope'], record['phase'], record['item'], record['local'],
                      record.get('input'), record.get('output'), record.get('cached'), record.get('cost'),
                      record.get('http'), record.get('finish', record['state'])]
            lines.append(' | '.join(display(value) for value in values))
        lines += ['', 'Раздел | Назначение | Вызовы | Вход API | Выход API | Всего токенов | ₽']
        for scope in ('comparison', 'branches'):
            for phase in ('dialogue', 'facts', 'summary', 'probe', 'ВСЕГО'):
                group = [record for record in records if record['scope'] == scope
                         and (phase == 'ВСЕГО' or record['phase'] == phase)]
                a, b = total(group, 'input'), total(group, 'output')
                lines.append(' | '.join(map(display, [scope, phase, len(group), a, b,
                                                     None if a is None or b is None else a + b,
                                                     total(group, 'cost')])))
        main = [record for record in records if record['scope'] == 'comparison']
        support = [record for record in main if record['phase'] in ('facts', 'summary')]
        lines += [f'Поддержка памяти в основном сценарии: {len(support)} вызовов.',
                  f'Строк классов памяти, включая базовый Window: {code[strategy]}. '
                  'Общий Agent, Branches, клиент, промпты и отчёт сюда не включены.']
        if run.get('context'):
            lines.append('Локальный контекст перед проверкой: ' + str(run['context']))
        for number, record in enumerate(records, 1):
            if record.get('model'):
                lines.append(f'Обращение {number}: модель={record["model"]}; '
                             f'провайдер={record.get("provider") or "н/д"}')
            if record.get('error'):
                lines += [f'Ошибка обращения {number}:', record['error']]
    lines += ['', 'Весь фактический эксперимент (все стратегии и ветвление):',
              'Вызовы: ' + str(len(all_records))]
    for key in ('input', 'output', 'cost'):
        lines.append(key + ': ' + display(total(all_records, key)))
    save_text(directory / 'results.txt', ('\n'.join(results).rstrip() + '\n').replace(api_key, '[HIDDEN]'))
    save_text(directory / 'metrics.txt', ('\n'.join(lines) + '\n').replace(api_key, '[HIDDEN]'))


class Metrics:
    def __init__(self, directory, strategy, api_key):
        self.directory, self.strategy, self.api_key = directory, strategy, api_key
        self.path = directory / 'state' / strategy / 'usage.json'
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self):
        save_json(self.path, self.records)
        render_reports(self.directory, self.api_key)

    def result(self, title, question, answer):
        path = self.path.parent / 'results.txt'
        with path.open('a', encoding='utf-8') as output:
            output.write(f'\n{title}\nВопрос:\n{question}\n\nОтвет:\n{answer}\n'.replace(self.api_key, '[HIDDEN]'))
        render_reports(self.directory, self.api_key)

    def observe(self, record, data):
        usage = data.get('usage') or {}
        if not isinstance(usage, dict):
            usage = {}
        value = cost(usage)
        details = usage.get('prompt_tokens_details') or {}
        record.update(input=integer(usage.get('prompt_tokens')),
                      output=integer(usage.get('completion_tokens')),
                      cached=integer(details.get('cached_tokens')) if isinstance(details, dict) else None,
                      cost=str(value) if value is not None else None,
                      model=data.get('model'), provider=data.get('provider'))
