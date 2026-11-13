"""Первый агент: модель выбирает инструменты, Python выполняет их в цикле."""

import json
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator, ValidationError


DIRECTORY = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash"
API_URL = "https://polza.ai/api/v1/chat/completions"
QUESTIONS = (
    "Объясни простыми словами, что такое API.",
    "Какой учебный курс USD к RUB задан в программе?",
    "Переведи 250 долларов в рубли по учебному курсу.",
)
SYSTEM_PROMPT = (
    "Ты помощник с доступом к инструментам. Для получения учебного курса валют "
    "используй get_exchange_rate, для арифметических вычислений — calculate. "
    "Сам выбирай необходимые инструменты и их аргументы. Не придумывай результаты "
    "вызовов. Курсы фиксированные учебные, а не актуальные рыночные. "
    "Получив необходимые результаты, ответь пользователю по-русски. "
    "Если инструменты не нужны, отвечай сразу. Если инструмент сообщил об ошибке, "
    "исправь аргументы, если возможно, или объясни ограничение пользователю."
)


def get_exchange_rate(from_currency, to_currency):
    """Стоимость одной единицы исходной валюты в целевой; курсы учебные."""
    rub_per_unit = {"RUB": 1, "USD": 90, "EUR": 100}
    return {"from_currency": from_currency, "to_currency": to_currency,
            "rate": rub_per_unit[from_currency] / rub_per_unit[to_currency],
            "kind": "fixed_educational"}


def calculate(operation, a, b):
    """Одна операция над двумя числами, без eval и выполнения чужого кода."""
    operations = {"add": lambda: a + b, "subtract": lambda: a - b,
                  "multiply": lambda: a * b, "divide": lambda: a / b}
    result = operations[operation]()
    if not math.isfinite(result):
        raise ValueError("Результат должен быть конечным числом.")
    return {"operation": operation, "a": a, "b": b, "result": result}


TOOLS = [
    {"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": (
            "Получить фиксированный учебный курс: сколько единиц to_currency "
            "стоит одна единица from_currency. Это не текущий рыночный курс. "
            "Поддерживаются RUB, USD, EUR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "enum": ["RUB", "USD", "EUR"]},
                "to_currency": {"type": "string", "enum": ["RUB", "USD", "EUR"]},
            },
            "required": ["from_currency", "to_currency"], "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": (
            "Выполнить одну арифметическую операцию над числами a и b: "
            "add (a+b), subtract (a-b), multiply (a*b), divide (a/b). "
            "При конвертации валют умножить сумму на полученный курс."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"}, "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"], "additionalProperties": False,
        },
    }},
]
FUNCTIONS = {"get_exchange_rate": get_exchange_rate, "calculate": calculate}
SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS}


class AgentError(Exception):
    """Запрос агента не завершился нормальным финальным ответом."""


def reject_constant(value):
    raise ValueError(f"Недопустимое JSON-число: {value}")


class Agent:
    def __init__(self, api_key, max_steps=8):
        self.api_key = api_key
        self.max_steps = max_steps

    def request(self, messages):
        payload = {"model": MODEL, "messages": list(messages), "tools": TOOLS,
                   "tool_choice": "auto", "reasoning": {"enabled": False}}
        try:
            response = requests.post(
                API_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=(10, 180),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status is not None else "сеть или таймаут"
            raise AgentError(f"Ошибка обращения к API: {detail}. Повторных попыток нет.") from None
        try:
            data = json.loads(response.text, parse_constant=reject_constant)
            if not isinstance(data, dict) or data.get("error"):
                raise ValueError
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
            finish = choice.get("finish_reason")
            if finish == "length":
                raise AgentError("Ответ обрезан по лимиту генерации; запрос не завершён.")
            if finish not in ("stop", "tool_calls"):
                raise AgentError(f"Неожиданная причина завершения: {finish!r}.")
            return message, finish
        except (ValueError, KeyError, IndexError, TypeError):
            raise AgentError("API вернул неожиданный формат ответа.") from None

    def execute_tool(self, name, raw_arguments):
        try:
            if name not in FUNCTIONS:
                raise ValueError(f"Неизвестный инструмент: {name}")
            arguments = json.loads(raw_arguments, parse_constant=reject_constant)
            Draft202012Validator(SCHEMAS[name]).validate(arguments)
            # JSON допускает очень большие числа; Python также читает 1e999 как inf.
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in arguments.values()):
                raise ValueError("Аргументы должны содержать только конечные числа.")
            return FUNCTIONS[name](**arguments)
        except (ValueError, TypeError, ValidationError, ZeroDivisionError, OverflowError) as error:
            detail = error.message if isinstance(error, ValidationError) else str(error)
            return {"error": detail}

    def run(self, question):
        if not isinstance(question, str) or not question.strip():
            raise AgentError("Вопрос не должен быть пустым.")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question.strip()}]
        for _ in range(self.max_steps):
            message, finish = self.request(messages)
            calls = message.get("tool_calls")
            if calls is None or calls == []:
                answer = message.get("content")
                if finish != "stop" or not isinstance(answer, str) or not answer.strip():
                    raise AgentError("Нет ни вызовов инструментов, ни завершённого текстового ответа.")
                return answer
            # Проверяем весь набор до выполнения: для каждого результата нужен корректный ID.
            if not isinstance(calls, list) or len(calls) > 16:
                raise AgentError("Некорректный список вызовов или более 16 вызовов за шаг.")
            ids = set()
            for call in calls:
                if not isinstance(call, dict):
                    raise AgentError("Некорректный вызов инструмента.")
                call_id, function = call.get("id"), call.get("function")
                if (not isinstance(call_id, str) or not call_id or call_id in ids
                        or call.get("type") != "function" or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise AgentError("У вызова отсутствуют корректные ID, имя или JSON-аргументы.")
                ids.add(call_id)
            messages.append(message)
            for call in calls:
                function = call["function"]
                result = self.execute_tool(function["name"], function["arguments"])
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
        raise AgentError(f"Достигнут предел {self.max_steps} обращений к модели. Финальный ответ не получен.")


def main():
    load_dotenv(DIRECTORY.parents[1] / ".env")
    api_key = os.getenv("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")
    agent = Agent(api_key)
    failures = 0
    with (DIRECTORY / "results.txt").open("a", encoding="utf-8") as output:
        for question in QUESTIONS:
            output.write(f"\nВопрос:\n{question}\n\nОтвет:\n")
            output.flush()
            try:
                answer = agent.run(question)
            except AgentError as error:
                failures += 1
                answer = f"Ошибка: {error}"
            except KeyboardInterrupt:
                output.write("Запрос прерван пользователем.\n")
                return 130
            output.write(answer.strip().replace(api_key, "[HIDDEN]") + "\n")
            output.flush()
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
