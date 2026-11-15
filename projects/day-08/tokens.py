"""Локальный токенизатор Qwen и текстовый отчёт фактических метрик API."""

import copy
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
            raise ValueError("Установи зависимости: python -m pip install -r projects/day-08/requirements.txt") from None
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

    def messages(self, messages, tools=None, generation=False):
        normalized = copy.deepcopy(messages)
        for message in normalized:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
        prompt = self.template.render(messages=normalized, tools=tools or [],
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


class Metrics:
    def __init__(self, path, metadata, api_key):
        self.path, self.metadata, self.api_key = path, metadata, api_key
        self.previous = path.read_text(encoding="utf-8") if path.exists() else ""
        if self.previous:
            self.previous = self.previous.rstrip() + "\n\n" + "=" * 60 + "\n"
        self.records = []

    def save(self):
        lines = ["День 8. Токены и стоимость",
                 *[f"{key}: {value}" for key, value in self.metadata.items()], "",
                 "Вопрос/история/вход локально — подсчёт официальным токенизатором; API — фактический usage.",
                 "История — до нового вопроса, без схем tools; вход локально — все текущие messages + tools + префикс ответа.",
                 "Выход API включает вызовы функций; текст ответа локально их не включает.",
                 "н/д означает отсутствие метрики, а не ноль. Стоимость — cost_rub или его алиас cost, один раз.", "",
                 "№ | Вопрос | Шаг | Вопрос лок. | История лок. | Вход лок. | Вход API | Выход API | Всего API | Текст ответа лок. | Кеш API | ₽ | Σ вход | Σ выход | Σ ₽ | HTTP | Завершение"]
        totals = {"input": 0, "output": 0, "cost": Decimal(0)}
        def display(value):
            return "н/д" if value is None else str(value)
        for index, record in enumerate(self.records, 1):
            for key in totals:
                value = record.get(key)
                totals[key] = None if totals[key] is None or value is None else totals[key] + value
            values = [index, record["turn"], record["step"], record["question_tokens"],
                      record["history_tokens"], record["input_local"], record.get("input"),
                      record.get("output"), record.get("total"), record.get("answer_local"), record.get("cached"),
                      record.get("cost"), totals["input"], totals["output"], totals["cost"],
                      record.get("http"), record.get("finish", record["state"])]
            lines.append(" | ".join(display(v) for v in values))
        for index, record in enumerate(self.records, 1):
            if record.get("actual_model") or record.get("provider"):
                lines += ["", f"Обращение №{index}: модель {record.get('actual_model') or 'н/д'}, "
                          f"провайдер {record.get('provider') or 'н/д'}"]
            if record.get("error_body"):
                lines += ["", f"Ответ API / ошибка обращения №{index}:", record["error_body"]]
        text = "\n".join(lines).replace(self.api_key, "[HIDDEN]") + "\n"
        temporary = self.path.with_suffix(".txt.tmp")
        temporary.write_text(self.previous + text, encoding="utf-8")
        temporary.replace(self.path)

    def observe(self, record, data):
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("prompt_tokens_details") or {}
        record.update(input=integer(usage.get("prompt_tokens")),
                      output=integer(usage.get("completion_tokens")), total=integer(usage.get("total_tokens")), cost=cost(usage),
                      cached=integer(details.get("cached_tokens")) if isinstance(details, dict) else None,
                      actual_model=data.get("model"), provider=data.get("provider"))
