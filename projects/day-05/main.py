"""Пять задач × три модели с настройками по умолчанию: 15 запросов."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter

import requests
from dotenv import load_dotenv


DIRECTORY = Path(__file__).resolve().parent
MODELS = {
    "qwen/qwen-2.5-7b-instruct": "Маленькая: Qwen2.5 7B Instruct",
    "meta-llama/llama-3.3-70b-instruct": "Средняя: Llama 3.3 70B Instruct",
    "anthropic/claude-opus-5.5": "Флагманская: Opus 5.5",
}
TASKS = [
    {
        "id": "knights", "title": "Рыцари и лжецы",
        "question": (
            "Каждый из троих — либо рыцарь, который всегда говорит правду, либо лжец, "
            "который всегда лжёт. Анна говорит: «Борис — лжец». "
            "Борис говорит: «Анна и Вера одного типа». Вера говорит: «Анна — рыцарь». "
            "Кто из них рыцарь, а кто лжец?"
        ),
        "expected": "Анна и Вера — лжецы, Борис — рыцарь",
        "reference": "Единственный согласованный вариант из восьми назначений типов.",
    },
    {
        "id": "bridge", "title": "Мост и фонарь",
        "question": (
            "Пять человек должны перейти мост ночью. Их времена перехода — 1, 2, 7, 10 "
            "и 12 минут. На мосту одновременно могут находиться один или два человека. "
            "Для каждого перехода нужен единственный фонарь, который переносят люди; "
            "бросать его нельзя. Двое идут со скоростью более медленного. Все начинают "
            "на одном берегу вместе с фонарём. Каково минимальное время, за которое "
            "все окажутся на другом берегу?"
        ),
        "expected": "27 минут",
        "reference": "1,2 →; 1 ←; 10,12 →; 2 ←; 1,2 →; 1 ←; 1,7 →. Сумма 27; минимум проверен поиском по состояниям.",
    },
    {
        "id": "knapsack", "title": "Рюкзак",
        "question": (
            "В рюкзак помещается вес не более 17 кг. Есть шесть неделимых предметов, "
            "каждый можно взять не более одного раза. Вес и ценность: A — 3 кг и 8; "
            "B — 4 кг и 10; C — 5 кг и 14; D — 6 кг и 16; E — 7 кг и 19; F — 9 кг и 24. "
            "Какова максимальная суммарная ценность выбранных предметов?"
        ),
        "expected": "46",
        "reference": "A+C+F: вес 17, ценность 46. Это единственный максимум среди 64 подмножеств.",
    },
    {
        "id": "route", "title": "Маршрут с купоном",
        "question": (
            "Есть неориентированный граф с вершинами A, B, C, D, E. Стоимость прохода "
            "по рёбрам: A-B=4, A-C=7, A-D=11, B-C=2, B-D=6, B-E=13, C-D=3, C-E=9, D-E=5. "
            "Других рёбер нет. Нужно пройти из A в E, обязательно посетив C. "
            "Можно повторно посещать вершины и проходить рёбра. Есть один купон: "
            "его нужно ровно один раз применить к одному прохождению ребра, "
            "уменьшив стоимость этого прохождения на 50%. Какова минимальная "
            "суммарная стоимость маршрута?"
        ),
        "expected": "10,5",
        "reference": "A-B-C-E, скидка на C-E: 4+2+9/2=10,5. Минимум проверен поиском с состоянием купона и посещения C.",
    },
    {
        "id": "probability", "title": "Условная вероятность",
        "question": (
            "В урне 5 красных, 4 синих и 3 зелёных шара. Случайно выбирают три шара "
            "без возвращения; все наборы из трёх физических шаров равновероятны. "
            "Известно, что среди выбранных есть хотя бы один зелёный. Какова условная "
            "вероятность того, что среди выбранных ровно два красных шара?"
        ),
        "expected": "15/68 ≈ 0,220588",
        "reference": "C(5,2)·C(3,1) / (C(12,3)−C(9,3)) = 30/136 = 15/68.",
    },
]


def build_payload(model, task):
    # Только исходное условие; эталон и дополнительные параметры не передаём.
    return {"model": model, "messages": [{"role": "user", "content": task["question"]}]}


def cost_rub(usage):
    value = usage.get("cost_rub")
    if value is None:
        value = usage.get("cost")  # Документированный алиас; тоже рубли.
    try:
        result = Decimal(str(value))
        return str(result) if result.is_finite() and result >= 0 else "н/д"
    except InvalidOperation:
        return "н/д"


def read_stream(response, record, save):
    """Собираем итоговый текст; usage и стоимость приходят в конце SSE."""
    data = {"choices": [{"message": {"content": ""}, "finish_reason": None}]}
    parts = []
    event_lines = []
    count = 0
    last_notice = perf_counter()
    for line in response.iter_lines(chunk_size=1):
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("data:"):
            event_lines.append(line[5:].lstrip(" "))
        if line or not event_lines:
            continue
        event = "\n".join(event_lines)
        event_lines = []
        if event == "[DONE]":
            if data["choices"][0]["finish_reason"] not in ("stop", "length"):
                raise ValueError("SSE завершился без причины завершения ответа")
            data["choices"][0]["message"]["content"] = "".join(parts)
            return data
        chunk = json.loads(event)
        if chunk.get("error"):
            raise ValueError(str(chunk["error"]))
        for key, field in (("id", "generation"), ("model", "actual_model"), ("provider", "provider")):
            if chunk.get(key):
                data[key] = record[field] = chunk[key]
        if chunk.get("usage") is not None:
            data["usage"] = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            content = (choice.get("delta") or {}).get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Неожиданный текст в SSE")
                parts.append(content)
            if choice.get("finish_reason"):
                data["choices"][0]["finish_reason"] = choice["finish_reason"]
        count += 1
        if count == 1 or perf_counter() - last_notice >= 15:
            print(f"SSE: событий {count}, символов итогового ответа {sum(map(len, parts))}", flush=True)
            save()
            last_notice = perf_counter()
    raise ValueError("Соединение закрыто без SSE [DONE]")


def request_once(record, task, api_key, save):
    started = perf_counter()
    try:
        print("Ожидание HTTP-заголовков…", flush=True)
        payload = build_payload(record["model"], task)
        if record.get("stream"):
            payload.update(stream=True, stream_options={"include_usage": True})
        with requests.post(
            "https://polza.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload, stream=True,
        ) as response:
            record["http"] = str(response.status_code)
            save()
            print(f"HTTP {response.status_code}", flush=True)
            if record.get("stream") and "text/event-stream" in response.headers.get("Content-Type", ""):
                response.raise_for_status()
                data = read_stream(response, record, save)
                data["choices"][0]["message"]["content"] = data["choices"][0]["message"]["content"].replace(api_key, "[HIDDEN]")
            else:
                print("Ожидаем полное тело ответа…", flush=True)
                raw = response.text.replace(api_key, "[HIDDEN]")
                response.raise_for_status()
                data = json.loads(raw)
        if data.get("error"):
            raise ValueError(str(data["error"]))
        usage = data.get("usage") or {}
        record.update(generation=data.get("id", "н/д"),
                      actual_model=data.get("model", "н/д"), provider=data.get("provider", "н/д"),
                      input=str(usage.get("prompt_tokens") if usage.get("prompt_tokens") is not None else "н/д"),
                      output=str(usage.get("completion_tokens") if usage.get("completion_tokens") is not None else "н/д"),
                      reasoning=str((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "н/д")),
                      cost=cost_rub(usage))
        choice = data["choices"][0]
        answer = choice["message"]["content"]
        if not isinstance(answer, str):
            raise ValueError("API не вернул текст ответа")
        record.update(answer=answer, finish=str(choice.get("finish_reason", "н/д")))
        if data.get("model") and data["model"] != record["model"]:
            raise ValueError("API указал другую модель; проверь поле «Модель в ответе»")
        record["state"] = "completed" if record["finish"] == "stop" and answer.strip() else "incomplete"
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        record.update(state="api_error", error=f"{type(error).__name__}: {error}".replace(api_key, "[HIDDEN]"))
    except KeyboardInterrupt:
        record["state"] = "interrupted"
        raise
    finally:
        record["seconds"] = str(round(perf_counter() - started, 6))
        save()


def sum_field(records, field):
    if not records or any(r.get(field, "н/д") == "н/д" for r in records):
        return "н/д"
    try:
        values = [Decimal(r[field]) for r in records]
        if not all(v.is_finite() and v >= 0 for v in values):
            return "н/д"
        return str(sum(values))
    except (InvalidOperation, KeyError):
        return "н/д"


def summary(records):
    lines = ["Сводка", "Модель | Полных ответов | Верно | Время, с | Вход / выход / reasoning | Стоимость, ₽"]
    for model, title in MODELS.items():
        attempts = [r for r in records if r["model"] == model]
        complete = len({r["task"] for r in attempts if r["state"] == "completed"})
        tokens = " / ".join(sum_field(attempts, f) for f in ("input", "output", "reasoning"))
        lines.append(f"{title} | {complete}/{len(TASKS)} | н/д | {sum_field(attempts, 'seconds')} | {tokens} | {sum_field(attempts, 'cost')}")
    small = [r for r in records if r["model"] == next(iter(MODELS))]
    large = [r for r in records if r["model"] == list(MODELS)[-1]]
    if all(len({r["task"] for r in group if r["state"] == "completed"}) == len(TASKS) for group in (small, large)):
        a, b = sum_field(small, "cost"), sum_field(large, "cost")
        if a != "н/д" and b != "н/д" and Decimal(a) > 0:
            lines.append(f"Opus / Qwen по фактической стоимости: {Decimal(b) / Decimal(a):.2f} раза.")
    lines += ["Верно — ручная оценка итоговых ответов, пока н/д.",
              "Токены, время и стоимость включают все попытки, в том числе неполные.",
              "н/д означает отсутствие данных хотя бы одной попытки, а не нулевой расход.",
              "Reasoning уже входит в выходные токены и повторно не прибавляется."]
    return "\n".join(lines)


# Простые текстовые поля нужны также для продолжения прерванного прогона.
FIELDS = {
    "model": "Модель", "task": "ID задачи", "state": "Состояние",
    "generation": "ID генерации", "actual_model": "Модель в ответе", "provider": "Провайдер",
    "http": "HTTP", "seconds": "Время, с", "finish": "Завершение",
    "input": "Входные токены", "output": "Выходные токены",
    "reasoning": "Reasoning-токены", "cost": "Стоимость API, ₽",
}


def save_results(records, started_at):
    lines = ["День 5. Версии моделей", "Начало UTC: " + started_at,
             "Условия: только model и messages; прочие параметры — по умолчанию.",
             "Правильность оцениваем вручную по локальным эталонам."]
    for record in records:
        task = next(t for t in TASKS if t["id"] == record["task"])
        lines += ["", "=== ЗАПРОС ==="]
        lines += [f"{title}: {record.get(key, 'н/д')}" for key, title in FIELDS.items()]
        if record.get("stream"):
            lines.append("Запрошена потоковая доставка: SSE")
        lines += ["", "Задача:", task["question"], "", "Локальный эталон (не передаётся модели):",
                  task["expected"], task["reference"], "", "Ответ модели:", record.get("answer", ""),
                  "", "Ошибка обращения:", record.get("error", ""), "=== КОНЕЦ ==="]
    lines += ["", summary(records)]
    path = DIRECTORY / "results.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_results():
    text = (DIRECTORY / "results.txt").read_text(encoding="utf-8")
    if not text.startswith("День 5. Версии моделей\nНачало UTC: "):
        raise ValueError("Неизвестный формат отчёта")
    started_at = text.splitlines()[1].removeprefix("Начало UTC: ")
    records = []
    for block in text.split("\n=== ЗАПРОС ===\n")[1:]:
        metadata, body = block.split("\n\nЗадача:\n", 1)
        record = {}
        for (key, title), line in zip(FIELDS.items(), metadata.splitlines()):
            if not line.startswith(title + ": "):
                raise ValueError("Не удалось прочитать поля отчёта")
            record[key] = line[len(title) + 2:]
        if len(record) != len(FIELDS) or record["model"] not in MODELS:
            raise ValueError("Набор моделей или формат отчёта изменился")
        if "Запрошена потоковая доставка: SSE" in metadata.splitlines():
            record["stream"] = True
        task = next(t for t in TASKS if t["id"] == record["task"])
        question, body = body.split("\n\nЛокальный эталон (не передаётся модели):\n", 1)
        if question != task["question"]:
            raise ValueError("Условие задачи изменилось")
        _, answer = body.split("\n\nОтвет модели:\n", 1)
        record["answer"], rest = answer.split("\n\nОшибка обращения:\n", 1)
        record["error"], _ = rest.split("\n=== КОНЕЦ ===", 1)
        if record["state"] == "pending":
            record["state"] = "interrupted"
            record["error"] = "Предыдущая попытка не завершена; статус и расход на сервере неизвестны."
        records.append(record)
    if not records:
        raise ValueError("В отчёте нет обращений")
    return records, started_at


def main(resume=False, streaming=False):
    load_dotenv(DIRECTORY.parents[1] / ".env")
    api_key = os.getenv("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")
    records, started_at = [], datetime.now(timezone.utc).isoformat()
    path = DIRECTORY / "results.txt"
    if resume:
        try:
            records, started_at = load_results()
        except (OSError, ValueError, KeyError, StopIteration) as error:
            raise SystemExit(f"Не удалось продолжить отчёт: {error}")
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = DIRECTORY.parents[2] / "ai-advent-challenge-archive" / "day-05" / f"results-{stamp}.txt"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        print(f"Копия предыдущего отчёта: {backup}")
    save = lambda: save_results(records, started_at)
    try:
        for task in TASKS:
            for model, title in MODELS.items():
                if any(r["model"] == model and r["task"] == task["id"] and r["state"] == "completed" for r in records):
                    continue
                record = dict(model=model, task=task["id"], state="pending")
                if streaming:
                    record["stream"] = True
                records.append(record)
                save()
                print(f"\n{task['title']} · {title} · попытка №{len(records)}", flush=True)
                request_once(record, task, api_key, save)
                print(record.get("answer") or "Итоговый текст отсутствует.", flush=True)
                if record["state"] == "api_error":
                    print(record["error"])
                    return 1
    except KeyboardInterrupt:
        print("\nПрервано. Полученные ответы сохранены; продолжение — с --resume.")
        return 130
    finally:
        save()
        print(summary(records))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="День 5: Qwen 7B / Llama 70B / Opus, 15 запросов.")
    parser.add_argument("--stream", action="store_true", help="Получать новые ответы потоком SSE вместе с usage и стоимостью.")
    parser.add_argument("--resume", action="store_true", help="Пропустить готовые ответы в results.txt.")
    args = parser.parse_args()
    raise SystemExit(main(resume=args.resume, streaming=args.stream))
