"""Один вопрос → запрос к модели → ответ в терминале."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Общий .env лежит в корне репозитория, на два уровня выше папки дня.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
api_key = os.getenv("POLZA_API_KEY", "").strip()
if not api_key:
    raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")

question = input("Твой вопрос: ").strip()
if not question:
    raise SystemExit("Вопрос не должен быть пустым.")

# json= превращает словарь Python в JSON и задаёт Content-Type.
payload = {
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": question}],
    "reasoning": {"enabled": False},
}

print("Ждём ответ модели…", flush=True)
try:
    response = requests.post(
        "https://polza.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=(10, 120),
    )
    response.raise_for_status()
except requests.HTTPError as error:
    raise SystemExit(f"Ошибка API: HTTP {error.response.status_code}.") from None
except requests.RequestException:
    raise SystemExit("Не удалось получить ответ: проверь соединение и доступность API.") from None

# API возвращает JSON; первый ответ находится в choices[0].
try:
    choice = response.json()["choices"][0]
    answer = choice["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError
except (ValueError, KeyError, IndexError, TypeError):
    raise SystemExit("API вернул ответ без ожидаемого текста.") from None

print("\nОтвет модели:")
print(answer)
if choice.get("finish_reason") == "length":
    print("\nОтвет обрезан: достигнут лимит генерации сервиса.")
