"""Локальный токенизатор Qwen и текстовый отчёт фактических метрик API."""

import json
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
            raise ValueError("Установи зависимости: python -m pip install -r projects/day-09/requirements.txt") from None
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


class Metrics:
    """Все этапы одного эксперимента: запросы summary не теряются между запусками."""

    def __init__(self, directory, run, api_key):
        self.directory, self.run, self.api_key = directory, run, api_key
        self.path = directory / "state/usage.json"
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self):
        save_json(self.path, self.records)
        lines = ["День 9. Сжатие истории", f"Модель: {self.run['metadata']['model']}",
                 f"Начало UTC: {self.run['metadata']['started_at_utc']}",
                 f"Этапы: {self.run['stages']}",
                 "Локальный вход — оценка по официальному токенизатору; API — фактический usage.",
                 "Стоимость берётся из cost_rub или алиаса cost один раз; н/д не равно нулю.", "",
                 "№ | Этап | Обращение | Вход лок. | Вход API | Выход API | Кеш API | ₽ | HTTP | Завершение"]
        for number, record in enumerate(self.records, 1):
            values = [number, record['phase'], record['item'], record['local'], record.get('input'),
                      record.get('output'), record.get('cached'), record.get('cost'),
                      record.get('http'), record.get('finish', record['state'])]
            lines.append(" | ".join(display(value) for value in values))
        lines += ["", "Сводка записанных обращений (до завершения этапов это частичные данные):",
                  "Этап | Попытки | Вход API | Выход API | ₽"]
        groups = {}
        for phase in ('dialogue', 'summary', 'probe-full', 'probe-compressed'):
            group = [r for r in self.records if r['phase'] == phase]
            groups[phase] = {key: total(group, key) for key in ('input', 'output', 'cost')}
            lines.append(" | ".join([phase, str(len(group)), *[display(groups[phase][key])
                                                          for key in ('input', 'output', 'cost')]]))
        lines.append("Все фактические попытки: " + "; ".join(
            f"{key}={display(total(self.records, key))}" for key in ('input', 'output', 'cost')))
        if all(state == 'complete' for state in self.run['stages'].values()):
            full = [r for r in self.records if r['phase'] == 'probe-full']
            compressed = [r for r in self.records if r['phase'] in ('summary', 'probe-compressed')]
            lines += ["", "Сравнение проверки фактов: full против summary + compressed.",
                      "Общая подготовка dialogue вынесена отдельно, а стоимость всех summary включена."]
            for key in ('input', 'output', 'cost'):
                a, b = total(full, key), total(compressed, key)
                difference = None if a is None or b is None else a - b
                lines.append(f"{key}: full={display(a)}; summary+compressed={display(b)}; "
                             f"разность full−compressed={display(difference)}")
            a = total(full, 'input'); b = total(full, 'output')
            c = total(compressed, 'input'); d = total(compressed, 'output')
            if None not in (a, b, c, d) and a + b:
                percent = Decimal(a + b - c - d) / Decimal(a + b) * 100
                lines.append(f"Разница токенов проверки с учётом summary, без общей подготовки: "
                             f"{percent:.2f}% (минус означает рост расхода).")
        if self.run.get('context'):
            lines += ["", "Размер сохранённого контекста локально:",
                      *[f"{key}: {value}" for key, value in self.run['context'].items()]]
        for number, record in enumerate(self.records, 1):
            if record.get('model') or record.get('provider'):
                lines.append(f"Обращение {number}: модель={record.get('model') or 'н/д'}; "
                             f"провайдер={record.get('provider') or 'н/д'}")
            if record.get('error_body'):
                lines += [f"Ошибка обращения {number}:", record['error_body']]
        save_text(self.directory / 'metrics.txt', ('\n'.join(lines) + '\n').replace(self.api_key, '[HIDDEN]'))

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
