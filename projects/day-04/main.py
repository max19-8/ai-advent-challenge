"""Три задачи × три температуры × пять повторов: 45 независимых запросов."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


DIRECTORY = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash"
PROVIDER = "streamlake/fp8"
TEMPERATURES = (0, 0.7, 1.2)
REPEATS = 5
TASKS = [
    {
        "id": "logic", "title": "Задача с одним правильным ответом",
        "question": (
            "Каждый из четверых — либо рыцарь, который всегда говорит правду, либо лжец, "
            "который всегда лжёт. Анна говорит: «Борис — лжец». "
            "Борис говорит: «Анна и Вера одного типа». Вера говорит: «Глеб — рыцарь». "
            "Глеб говорит: «Анна и Вера разных типов». Кто из них рыцарь, а кто лжец?"
        ),
        "expected": "Борис — рыцарь; Анна, Вера и Глеб — лжецы",
    },
    {
        "id": "creative", "title": "Свободная генерация названий",
        "question": "Придумай названия приложения для формирования полезных привычек.",
    },
    {
        "id": "format", "title": "Соблюдение текстового формата",
        "question": (
            "Придумай три названия приложения для формирования полезных привычек. "
            "Ответь ровно тремя строками по шаблону:\n"
            "1. <название>\n2. <название>\n3. <название>\n"
            "Без текста до и после списка."
        ),
    },
]


def build_payload(task, temperature):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": task["question"]}],
        "temperature": temperature,
        "reasoning": {"enabled": False},
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
    }


def format_errors(answer):
    # Краевые пробелы/переносы не учитываем; внутри нужны три пронумерованные строки.
    lines = answer.strip().splitlines()
    if len(lines) != 3:
        return [f"Ожидалось 3 строки, получено {len(lines)}."]
    return [f"Строка {i} должна начинаться с «{i}. » и содержать название."
            for i, line in enumerate(lines, 1) if not re.fullmatch(rf"{i}\.\s+\S.*", line)]


def summary_text(run):
    lines = ["Сводка", "Задача | Температура | Получено | Обрезаны | Уникальных полных текстов | Нарушений формата"]
    for task in run["tasks"]:
        for temperature in run["temperatures"]:
            records = [r for r in run["records"] if r["task_id"] == task["id"]
                       and r["temperature"] == temperature and r["state"] == "received"]
            complete = [r for r in records if r["finish_reason"] == "stop" and r["answer"].strip()]
            unique = len({r["answer"] for r in complete})
            truncated = sum(r["finish_reason"] == "length" for r in records)
            errors = str(sum(bool(r["format_errors"]) for r in records)) if task["id"] == "format" else "—"
            lines.append(f"{task['title']} | {temperature} | {len(records)}/{run['repeats']} | {truncated} | {unique}/{len(complete)} | {errors}")
    lines += ["Правильность логических решений и полезность названий оцениваем вручную.",
              "Уникальность текста сравнивается посимвольно, включая пробелы и оформление.",
              "Состояние: " + run["state"], "Результаты: " + str(DIRECTORY / "results.txt")]
    return "\n".join(lines)


def save_results(run):
    lines = ["День 4. Температура", f"Начало UTC: {run['started_at']}",
             f"Модель: {run['model']}", f"Запрошенный провайдер: {run['provider']}",
             "Рассуждения: отключены. HTTP-таймаут не задан. Повторных попыток нет.",
             f"Состояние: {run['state']}", ""]
    for task in run["tasks"]:
        lines += ["=" * 60, task["title"], "Вопрос:", task["question"]]
        if "expected" in task:
            lines += ["Локальный эталон (не передаётся модели): " + task["expected"]]
        for temperature in run["temperatures"]:
            lines += ["", f"Температура: {temperature}"]
            records = [r for r in run["records"] if r["task_id"] == task["id"] and r["temperature"] == temperature]
            if not records:
                lines.append("Запросы ещё не выполнены.")
            for record in records:
                lines += ["", f"Повтор {record['repeat']}. Состояние: {record['state']}",
                          f"HTTP: {record.get('http_status', 'ответ не получен')}"]
                if record["state"] == "received":
                    usage = record.get("usage") or {}
                    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "н/д")
                    lines += [f"Завершение: {record['finish_reason']}",
                              f"Токены: вход {usage.get('prompt_tokens', 'н/д')}, выход {usage.get('completion_tokens', 'н/д')}, reasoning {reasoning}",
                              "", "Ответ модели:", record["answer"], ""]
                    if task["id"] == "format":
                        lines.append("Нарушения формата: " + ("; ".join(record["format_errors"]) or "нет"))
                else:
                    lines.append(record.get("error", "Ответ ещё не получен."))
                    if record.get("raw_response"):
                        lines += ["Ответ API с ошибкой:", record["raw_response"]]
    lines += ["", summary_text(run)]
    path = DIRECTORY / "results.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    load_dotenv(DIRECTORY.parents[1] / ".env")
    api_key = os.getenv("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")
    run = {"started_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
           "provider": PROVIDER, "tasks": TASKS, "temperatures": list(TEMPERATURES),
           "repeats": REPEATS, "state": "running", "records": []}
    total = len(TASKS) * len(TEMPERATURES) * REPEATS
    print(f"{MODEL}, провайдер {PROVIDER}: {total} запросов. Результаты заменят results.txt.", flush=True)
    record = None
    try:
        save_results(run)
        for repeat in range(1, REPEATS + 1):
            for task in TASKS:
                for temperature in TEMPERATURES:
                    record = {"task_id": task["id"], "temperature": temperature, "repeat": repeat,
                              "state": "pending", "request": build_payload(task, temperature)}
                    run["records"].append(record)
                    save_results(run)
                    print(f"\n[{len(run['records'])}/{total}] {task['title']}, t={temperature}, повтор {repeat}", flush=True)
                    try:
                        response = requests.post(
                            "https://polza.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"}, json=record["request"],
                        )
                        record["http_status"] = response.status_code
                        record["raw_response"] = response.text.replace(api_key, "[HIDDEN]")
                        response.raise_for_status()
                        data = json.loads(record["raw_response"])
                        if not isinstance(data, dict) or data.get("error"):
                            raise ValueError("API сообщил об ошибке")
                        choice = data["choices"][0]
                        answer, finish = choice["message"]["content"], choice.get("finish_reason")
                        if not isinstance(answer, str) or finish not in ("stop", "length"):
                            raise ValueError("Неожиданный текст или причина завершения")
                        if not answer.strip() and finish != "length":
                            raise ValueError("Пустой завершённый ответ")
                        if data.get("usage") is not None and not isinstance(data["usage"], dict):
                            raise ValueError("Неожиданный формат usage")
                        record.update(state="received", answer=answer, finish_reason=finish,
                                      usage=data.get("usage"), format_errors=format_errors(answer) if task["id"] == "format" else None)
                    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AttributeError):
                        record.update(state="api_error", error="Ошибка сети, HTTP или структуры ответа; подробности в results.txt, если ответ получен.")
                        run["state"] = "api_error"
                        print(record["error"])
                        return 1
                    finally:
                        save_results(run)
                    print(record["answer"] or "Итоговый текст отсутствует.")
                    print("Завершение: " + record["finish_reason"], flush=True)
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
        print("\n" + summary_text(run))


if __name__ == "__main__":
    raise SystemExit(main())
