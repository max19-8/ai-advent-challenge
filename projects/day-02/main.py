"""Пять способов управления ответом × три повтора одного вопроса."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator


DIRECTORY = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash"
PROVIDER = "streamlake/fp8"
REPEATS = 3
QUESTION = (
    "Расскажи о планете Марс: укажи её порядковый номер от Солнца, "
    "назови естественные спутники и кратко опиши поверхность."
)
FORMAT = (
    "\nВерни только JSON-объект без Markdown и пояснений вне JSON. "
    "Поля: planet — название планеты (строка), position — порядковый номер "
    "от Солнца (целое число), satellites — названия спутников (массив строк), "
    "surface — описание поверхности (строка). Все четыре поля обязательны, других полей нет."
)
STOP = "<<<END>>>"
STOP_INSTRUCTION = (
    f"\nСначала укажи название планеты и её порядковый номер от Солнца. "
    f"Затем на отдельной строке напиши маркер {STOP}. "
    "После маркера назови естественные спутники и кратко опиши поверхность."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "planet": {"type": "string", "description": "Название планеты"},
        "position": {"type": "integer", "description": "Порядковый номер от Солнца"},
        "satellites": {"type": "array", "items": {"type": "string"},
                       "description": "Названия естественных спутников"},
        "surface": {"type": "string", "description": "Краткое описание поверхности"},
    },
    "required": ["planet", "position", "satellites", "surface"],
    "additionalProperties": False,
}
# Сначала проверяем схему реальным запросом, который входит в общий прогон.
VARIANTS = {
    "schema": "Строгая JSON Schema",
    "baseline": "Обычный ответ",
    "formatted": "Формат словами",
    "limited": "Только max_tokens=50",
    "stop": "Маркер в середине, со stop",
}


def build_payload(variant):
    prompt = QUESTION
    if variant == "formatted":
        prompt += FORMAT
    if variant == "stop":
        prompt += STOP_INSTRUCTION
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": False},
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
    }
    if variant == "limited":
        payload["max_tokens"] = 50
    if variant == "stop":
        payload["stop"] = [STOP]
    if variant == "schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "planet", "strict": True, "schema": SCHEMA},
        }
    return payload


def check_answer(answer):
    # Проверяем весь текст: не убираем Markdown и не исправляем JSON модели.
    try:
        parsed = json.loads(answer)
    except ValueError as error:
        return {"json_ok": False, "schema_ok": False, "errors": [str(error)]}
    errors = [error.message for error in Draft202012Validator(SCHEMA).iter_errors(parsed)]
    return {"json_ok": True, "schema_ok": not errors, "errors": errors}


def save_results(run):
    path = DIRECTORY / "results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    load_dotenv(DIRECTORY.parents[1] / ".env")
    api_key = os.getenv("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")
    Draft202012Validator.check_schema(SCHEMA)
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "provider": PROVIDER, "question": QUESTION,
        "schema": SCHEMA, "repeats": REPEATS, "state": "running", "records": [],
        "protocol": "stop-middle-v1",
    }
    total = len(VARIANTS) * REPEATS
    print(f"{MODEL}, провайдер {PROVIDER}: {total} запросов, без повторных попыток.")
    print("Результаты каждого запуска заменяют projects/day-02/results.json.", flush=True)
    record = None
    try:
        save_results(run)
        for repeat in range(1, REPEATS + 1):
            for variant, title in VARIANTS.items():
                record = {
                    "variant": variant, "repeat": repeat, "state": "pending",
                    "request": build_payload(variant),
                }
                run["records"].append(record)
                save_results(run)
                print(f"\n[{len(run['records'])}/{total}] {title}, повтор {repeat}", flush=True)
                try:
                    response = requests.post(
                        "https://polza.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=record["request"],
                    )
                    record["http_status"] = response.status_code
                    record["raw_response"] = response.text.replace(api_key, "[HIDDEN]")
                    response.raise_for_status()
                    data = json.loads(record["raw_response"])
                    if not isinstance(data, dict) or data.get("error"):
                        raise ValueError("API сообщил об ошибке.")
                    choice = data["choices"][0]
                    answer = choice["message"]["content"]
                    if not isinstance(answer, str):
                        raise ValueError("API вернул content не строкой.")
                    # Лимит может целиком уйти на рассуждения: пустой итог при length допустим.
                    if not answer.strip() and choice.get("finish_reason") != "length":
                        raise ValueError("В завершённом ответе API нет текста.")
                    if choice.get("finish_reason") not in ("stop", "length"):
                        raise ValueError("API не подтвердил завершение текстового ответа.")
                    record.update(
                        state="received", answer=answer, provider=data.get("provider"),
                        finish_reason=choice.get("finish_reason"), usage=data.get("usage"),
                        checks=check_answer(answer) if variant in ("formatted", "schema") else None,
                    )
                except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AttributeError):
                    record.update(state="api_error", error="Ошибка сети, HTTP или структуры ответа API; см. raw_response, если он получен.")
                    run["state"] = "api_error"
                    print("Прогон остановлен: " + record["error"])
                    return 1
                finally:
                    save_results(run)
                print(record["answer"] if record["answer"].strip() else "Итоговый текст отсутствует.")
                tokens = (record["usage"] or {}).get("completion_tokens", "н/д")
                print(f"Завершение: {record['finish_reason']}; выходных токенов: {tokens}")
                if record["checks"] is not None:
                    print(f"JSON: {record['checks']['json_ok']}; схема: {record['checks']['schema_ok']}")
                if variant == "stop":
                    print(f"Маркер {STOP} в тексте: {STOP in record['answer']}")
        run["state"] = "completed"
        return 0
    except KeyboardInterrupt:
        run["state"] = "interrupted"
        if record is not None and record["state"] == "pending":
            record["state"] = "interrupted"
        print("\nПрервано. Полученные ответы сохранены.")
        return 130
    finally:
        save_results(run)
        print("\nВариант | Ответов API | Обрезаны | Выходные токены | JSON / схема | С маркером")
        for variant in ("baseline", "formatted", "limited", "stop", "schema"):
            title = VARIANTS[variant]
            records = [r for r in run["records"] if r["variant"] == variant and r["state"] == "received"]
            truncated = sum(r["finish_reason"] == "length" for r in records)
            tokens = ", ".join(str((r["usage"] or {}).get("completion_tokens", "н/д")) for r in records) or "—"
            parsing = "—"
            if variant in ("formatted", "schema"):
                json_count = sum(r["checks"]["json_ok"] for r in records)
                schema_count = sum(r["checks"]["schema_ok"] for r in records)
                parsing = f"{json_count}/{len(records)} / {schema_count}/{len(records)}" if records else "нет данных"
            markers = str(sum(STOP in r["answer"] for r in records)) if variant == "stop" else "—"
            print(f"{title} | {len(records)}/{REPEATS} | {truncated} | {tokens} | {parsing} | {markers}")
        print("Состояние: " + run["state"])
        print("Результаты: " + str(DIRECTORY / "results.json"))


if __name__ == "__main__":
    raise SystemExit(main())
