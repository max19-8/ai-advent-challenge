# День 1. Первый запрос к LLM через API

Вводим вопрос в терминале, отправляем его в DeepSeek V4 Flash через Polza.ai
и выводим полученный текст. Каждый запуск — один независимый вопрос.

## Запуск

Из корня репозитория при первой настройке:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

В корневом `.env` должна быть строка `POLZA_API_KEY=твой_ключ`.
Для новой установки можно скопировать `.env.example` в `.env`.
Ключ не публикуется в Git.

```bash
source .venv/bin/activate
python projects/day-01/main.py
```

Введи вопрос, например: «Объясни простыми словами, что такое API».

[Модель](https://polza.ai/models/deepseek/deepseek-v4-flash)
· [API](https://github.com/polza-ai/docs/blob/main/gaidy/chat-completions.mdx)
