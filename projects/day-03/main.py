"""Пять задач × четыре способа × два режима: 40 комбинаций, 50 API-вызовов."""

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import requests
from dotenv import load_dotenv


DIRECTORY = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash"
PROVIDER = "streamlake/fp8"
EFFORTS = ("none", "high")
METHODS = {
    "direct": "Прямой запрос",
    "step": "Решай пошагово",
    "prompt": "Составление промпта",
    "experts": "Группа экспертов",
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


def build_prompt(task, method):
    question = task["question"]
    if method == "step":
        return question + "\nРешай пошагово."
    if method == "prompt":
        return (
            "Подготовь готовый промпт для другого решателя, которому предстоит решить "
            "задачу ниже. Промпт должен просить найти ответ и объяснить решение. "
            "На этом этапе нужен текст инструкции будущему решателю, а не решение задачи. "
            "Это ограничение относится только к автору промпта: будущий решатель должен решить задачу."
            "\n\nЗадача:\n" + question
        )
    if method == "experts":
        return (
            question + "\nРассмотри задачу как группу из аналитика, инженера и критика. "
            "Аналитик предлагает решение, инженер проверяет его применимость и расчёты, "
            "критик ищет ошибки и противоречия. Затем эксперты обсуждают разногласия, "
            "сопоставляют аргументы и выбирают одно общее решение. "
            "Покажи только согласованное решение и один итоговый ответ от всей группы."
        )
    return question


def build_payload(prompt, effort):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": effort},
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
    }


def read_stream(response, call, progress):
    """Собираем текст из SSE; usage приходит отдельным событием в конце."""
    answer = []
    usage = None
    finish = None
    event_lines = []
    last_notice = perf_counter()
    events = 0
    for line in response.iter_lines(chunk_size=1):
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("data:"):
            event_lines.append(line[5:].lstrip(" "))
        if line or not event_lines:
            continue
        event = "\n".join(event_lines)
        event_lines = []
        if event == "[DONE]":
            if finish not in ("stop", "length"):
                raise ValueError("Поток завершён без причины завершения ответа")
            return {"choices": [{"message": {"content": "".join(answer)}, "finish_reason": finish}],
                    "usage": usage, "provider": call.get("provider")}
        data = json.loads(event)
        if data.get("error"):
            raise ValueError(str(data["error"]))
        events += 1
        if data.get("id"):
            call["generation_id"] = data["id"]
        if data.get("provider"):
            call["provider"] = data["provider"]
        if data.get("usage") is not None:
            usage = data["usage"]
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            content = (choice.get("delta") or {}).get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Неожиданный текст в SSE")
                answer.append(content)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        if events == 1 or perf_counter() - last_notice >= 15:
            progress(f"SSE: событий {events}, символов итогового ответа {sum(map(len, answer))}")
            last_notice = perf_counter()
    raise ValueError("Соединение закрыто без завершающего события SSE [DONE]")


def request_once(call, api_key, on_progress=None):
    started = perf_counter()
    phase = "ожидание HTTP-заголовков"
    received = 0

    def progress(detail):
        amount = "" if call["request"].get("stream") else f"; получено {received} байт"
        call["diagnostic"] = f"{detail}; прошло {perf_counter() - started:.1f} с" + amount
        call["elapsed_seconds"] = round(perf_counter() - started, 6)
        print("  HTTP: " + call["diagnostic"], flush=True)
        if on_progress is not None:
            on_progress()

    try:
        progress(phase)
        # stream=True здесь управляет чтением HTTP; SSE включает только флаг --stream.
        with requests.post(
            "https://polza.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"}, json=call["request"],
            stream=True,
        ) as response:
            call["http_status"] = response.status_code
            phase = "чтение HTTP-тела"
            content_type = response.headers.get("Content-Type", "не указан")
            progress(f"заголовки получены: статус {response.status_code}, Content-Type {content_type}")
            if call["request"].get("stream") and response.ok:
                phase = "чтение событий SSE"
                data = read_stream(response, call, progress)
                # Не выводим внутренние рассуждения; сохраняем итог и usage.
                data["choices"][0]["message"]["content"] = data["choices"][0]["message"]["content"].replace(api_key, "[HIDDEN]")
            else:
                chunks = []
                last_notice = perf_counter()
                for chunk in response.iter_content(chunk_size=1024):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    received += len(chunk)
                    if len(chunks) == 1 or perf_counter() - last_notice >= 15:
                        progress(phase)
                        last_notice = perf_counter()
                call["raw_response"] = b"".join(chunks).decode("utf-8-sig", errors="replace").replace(api_key, "[HIDDEN]")
                progress("HTTP-тело получено полностью")
                response.raise_for_status()
                data = json.loads(call["raw_response"])
        phase = "проверка ответа"
        progress(phase)
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
        call.update(state="received", answer=answer, finish_reason=finish,
                    usage=data.get("usage"), provider=data.get("provider"))
        progress("ответ разобран")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        detail = f"{type(error).__name__}: {error}".replace(api_key, "[HIDDEN]")
        call.update(state="api_error", error=f"Ошибка на этапе «{phase}»: {detail}")
        progress("ошибка: " + phase)
    except KeyboardInterrupt:
        call["state"] = "interrupted"
        progress("прервано: " + phase)
        raise
    finally:
        call["elapsed_seconds"] = round(perf_counter() - started, 6)


def save_results(run):
    lines = ["День 3. Разные способы рассуждения", f"Начало UTC: {run['started_at']}",
             f"Модель: {run['model']}", f"Запрошенный провайдер: {run['provider']}",
             f"Состояние: {run['state']}", "Режимы: " + ", ".join(run["efforts"]),
             "", "Задачи и локальные эталоны (не передаются модели):"]
    tasks = {task["id"]: task for task in run["tasks"]}
    for task in run["tasks"]:
        lines += ["", task["title"], task["question"], "Эталон: " + task["expected"], task["reference"]]
    for index, record in enumerate(run["records"], 1):
        title = tasks[record["task_id"]]["title"]
        lines += ["\n" + "=" * 60,
                  f"{index}. {title} | {run['methods'][record['method']]} | effort={record['effort']}",
                  "Состояние решения: " + record["state"], "=" * 60]
        for call in record["calls"]:
            stage = "Составление промпта" if call["stage"] == "prompt_author" else "Решение задачи"
            lines += ["", "Этап: " + stage, "Состояние обращения: " + call["state"],
                      "Отправленный текст:", call["request"]["messages"][0]["content"], "",
                      f"HTTP: {call.get('http_status', 'ответ не получен')}",
                      f"Время: {call.get('elapsed_seconds', 'н/д')} с"]
            if call.get("diagnostic"):
                lines.append("Диагностика HTTP: " + call["diagnostic"])
            if call["request"].get("stream"):
                lines.append("Передача ответа: SSE")
            if call.get("generation_id"):
                lines.append("ID генерации: " + call["generation_id"])
            if call["state"] == "received":
                usage = call.get("usage") or {}
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "н/д")
                lines += [f"Завершение: {call['finish_reason']}",
                          f"Токены: вход {usage.get('prompt_tokens', 'н/д')}, выход {usage.get('completion_tokens', 'н/д')}, reasoning {reasoning}",
                          "", "Ответ модели:", call["answer"], ""]
            elif call.get("error"):
                lines.append(call["error"])
                if call.get("raw_response"):
                    lines += ["Ответ API с ошибкой:", call["raw_response"]]
    lines += ["", summary_text(run)]
    path = DIRECTORY / "results.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_results():
    """Восстанавливаем этапы из собственного текстового отчёта без отдельного файла состояния."""
    text = (DIRECTORY / "results.txt").read_text(encoding="utf-8")
    for line in (f"Модель: {MODEL}", f"Запрошенный провайдер: {PROVIDER}"):
        if line not in text.splitlines():
            raise ValueError("Модель или провайдер отчёта отличаются от текущих настроек.")
    for task in TASKS:
        if task["question"] not in text:
            raise ValueError("Условия задач изменились; продолжать этот отчёт нельзя.")
    run = dict(started_at=re.search(r"^Начало UTC: (.+)$", text, re.M)[1],
               model=MODEL, provider=PROVIDER, tasks=TASKS, methods=METHODS,
               efforts=[], state="running", records=[])
    body = text.split("\nСводка\n", 1)[0]
    divider = "=" * 60
    blocks = re.split(r"\n" + divider + r"\n\d+\. ", body)[1:]
    for block in blocks:
        heading, state, rest = block.split("\n", 2)
        title, method_title, effort = heading.split(" | ")
        task = next(t for t in TASKS if t["title"] == title)
        method = next(k for k, v in METHODS.items() if v == method_title)
        effort = effort.removeprefix("effort=")
        if effort not in ("none", "high", "xhigh"):
            raise ValueError("Неизвестный effort в отчёте.")
        if effort not in run["efforts"]:
            run["efforts"].append(effort)
        record = dict(task_id=task["id"], method=method, effort=effort,
                      state=state.removeprefix("Состояние решения: "), calls=[])
        if any((r["task_id"], r["method"], r["effort"]) ==
               (task["id"], method, effort) for r in run["records"]):
            raise ValueError("Повтор комбинации в отчёте.")
        for part in rest.split("\nЭтап: ")[1:]:
            stage_title, status, part = part.split("\n", 2)
            stage = {"Составление промпта": "prompt_author", "Решение задачи": "solution"}[stage_title]
            sent, details = part.removeprefix("Отправленный текст:\n").split("\n\nHTTP: ", 1)
            call = dict(stage=stage, state=status.removeprefix("Состояние обращения: "),
                        request=build_payload(sent, effort))
            http, timing, details = details.split("\n", 2)
            if http != "ответ не получен":
                call["http_status"] = int(http)
            seconds = timing.removeprefix("Время: ").removesuffix(" с")
            if seconds != "н/д":
                call["elapsed_seconds"] = float(seconds)
            if details.startswith("Диагностика HTTP: "):
                diagnostic, _, details = details.partition("\n")
                call["diagnostic"] = diagnostic.removeprefix("Диагностика HTTP: ")
            if details.startswith("Передача ответа: SSE\n"):
                call["request"].update(stream=True, stream_options={"include_usage": True})
                details = details.removeprefix("Передача ответа: SSE\n")
            if details.startswith("ID генерации: "):
                identifier, _, details = details.partition("\n")
                call["generation_id"] = identifier.removeprefix("ID генерации: ")
            if call["state"] == "received":
                finish, tokens, answer = details.split("\n", 2)
                values = re.fullmatch(r"Токены: вход (\d+|н/д), выход (\d+|н/д), reasoning (\d+|н/д)", tokens)
                if values is None or not answer.startswith("\nОтвет модели:\n"):
                    raise ValueError("Не удалось прочитать ответ или токены.")
                numbers = [int(v) if v != "н/д" else None for v in values.groups()]
                call.update(finish_reason=finish.removeprefix("Завершение: "),
                            answer=answer.removeprefix("\nОтвет модели:\n").rstrip("\n"),
                            usage=dict(prompt_tokens=numbers[0], completion_tokens=numbers[1],
                                       completion_tokens_details={"reasoning_tokens": numbers[2]}))
            else:
                call["error"] = details.rstrip("\n")
            expected = build_prompt(task, method)
            if method == "prompt" and stage == "solution":
                author = next(c for c in reversed(record["calls"]) if c["stage"] == "prompt_author" and c["state"] == "received")
                expected = "Реши исходную задачу по следующей инструкции.\n\nЗадача:\n" + task["question"] + "\n\nИнструкция решателю:\n" + author["answer"]
            if sent.strip() != expected.strip():
                raise ValueError("Промпты изменились; продолжать этот отчёт нельзя.")
            record["calls"].append(call)
        if not record["calls"]:
            raise ValueError("В отчёте отсутствуют этапы решения.")
        run["records"].append(record)
    if not run["records"]:
        raise ValueError("В отчёте не найдены обращения; файл оставлен без изменений.")
    planned = re.search(r"^Режимы: (.+)$", text, re.M)
    if planned:
        efforts = planned[1].split(", ")
        if not set(run["efforts"]).issubset(efforts) or any(e not in ("none", "high", "xhigh") for e in efforts):
            raise ValueError("Неизвестные или несовпадающие режимы отчёта.")
        run["efforts"] = efforts
    return run


def token_sum(calls, key, reasoning=False):
    values = []
    for call in calls:
        usage = call.get("usage") or {}
        if reasoning:
            usage = usage.get("completion_tokens_details") or {}
        values.append(usage.get(key))
    return sum(values) if values and all(isinstance(v, int) for v in values) else "н/д"


def summary_text(run):
    planned = len(run["tasks"])
    lines = ["Сводка", "Способ | Effort | Верно | Полных решений | Токены: вход / выход / reasoning | Время, с"]
    for method, title in run["methods"].items():
        for effort in run["efforts"]:
            records = [r for r in run["records"] if r["method"] == method and r["effort"] == effort]
            complete = sum(r["state"] == "completed" for r in records)
            attempts = [c for r in records for c in r["calls"]]
            calls = [c for c in attempts if c["state"] == "received"]
            tokens = f"{token_sum(calls, 'prompt_tokens')} / {token_sum(calls, 'completion_tokens')} / {token_sum(calls, 'reasoning_tokens', True)}"
            elapsed = f"{sum(c.get('elapsed_seconds', 0) for c in calls):.3f}" if calls else "—"
            lines.append(f"{title} | {effort} | н/д | {complete}/{planned} | {tokens} | {elapsed}")
    lines += ["Правильность оцениваем вручную; итоговую таблицу с оценками запишем в README.",
              "Reasoning входит в выходные токены; повторно не прибавляется.",
              "В таблице учтены полученные ответы, включая неполные; неудачные обращения указаны отдельно.",
              "Неудачных/прерванных обращений: " + str(sum(c["state"] != "received" for r in run["records"] for c in r["calls"])),
              "Время неудачных обращений, с: " + str(round(sum(c.get("elapsed_seconds", 0) for r in run["records"] for c in r["calls"] if c["state"] != "received"), 3)),
              "Расход токенов неудачных обращений неизвестен, если API не вернул usage.",
              "Состояние: " + run["state"], "Результаты: " + str(DIRECTORY / "results.txt")]
    return "\n".join(lines)


def main(efforts=EFFORTS, resume=False, streaming=False):
    load_dotenv(DIRECTORY.parents[1] / ".env")
    api_key = os.getenv("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Заполни POLZA_API_KEY в .env в корне репозитория.")
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "provider": PROVIDER, "tasks": TASKS, "efforts": list(efforts),
        "methods": METHODS, "state": "running", "records": [],
    }
    if resume:
        try:
            run = load_results()
        except (ValueError, KeyError, IndexError, StopIteration, OSError, AttributeError) as error:
            raise SystemExit(f"Не удалось продолжить отчёт: {error}")
        if set(efforts) != set(run["efforts"]):
            raise SystemExit("Укажи те же режимы --effort, что в сохранённом отчёте.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = DIRECTORY.parents[2] / "ai-advent-challenge-archive" / "day-03" / f"before-resume-{stamp}.txt"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DIRECTORY / "results.txt", backup)
        print(f"Копия прежнего отчёта: {backup}")
        for saved_record in run["records"]:
            for saved_call in saved_record["calls"]:
                if saved_call["state"] == "pending":
                    saved_call.update(state="interrupted", error="В предыдущем запуске итог обращения не был сохранён; статус на сервере неизвестен.")
    total = len(TASKS) * len(efforts) * (len(METHODS) + 1)
    combinations = len(TASKS) * len(efforts) * len(METHODS)

    def call_stage(record, stage, prompt):
        for saved in reversed(record["calls"]):
            if saved["stage"] == stage and saved["state"] == "received" and saved["finish_reason"] == "stop" and saved["answer"].strip():
                return saved
        call = {"stage": stage, "state": "pending", "request": build_payload(prompt, record["effort"])}
        if streaming:
            call["request"].update(stream=True, stream_options={"include_usage": True})
        record["calls"].append(call)
        save_results(run)
        number = sum(len(r["calls"]) for r in run["records"])
        print(f"  Обращение №{number} (включая прежние попытки): {stage}", flush=True)
        try:
            request_once(call, api_key, on_progress=lambda: save_results(run))
        finally:
            save_results(run)
        return call

    print(f"{MODEL}, провайдер {PROVIDER}. {combinations} комбинаций, план — {total} API-вызовов.")
    print("Продолжаем сохранённый прогон." if resume else "Новый запуск заменяет results.txt.", flush=True)
    record = None
    try:
        save_results(run)
        for effort in efforts:
            for task in TASKS:
                for method, title in METHODS.items():
                    record = next((r for r in run["records"] if
                                   (r["task_id"], r["method"], r["effort"]) ==
                                   (task["id"], method, effort)), None)
                    if record is not None and record["state"] == "completed":
                        continue
                    if record is None:
                        record = {"task_id": task["id"], "method": method, "effort": effort,
                                  "state": "pending", "calls": []}
                        run["records"].append(record)
                    record["state"] = "pending"
                    print(f"\n{task['title']} · {title} · {effort}", flush=True)
                    prompt = build_prompt(task, method)
                    if method == "prompt":
                        author = call_stage(record, "prompt_author", prompt)
                        if author["state"] == "api_error":
                            record["state"] = run["state"] = "api_error"
                            print(author["error"])
                            return 1
                        print("Составленный промпт:\n" + author["answer"])
                        if author["finish_reason"] != "stop" or not author["answer"].strip():
                            record["state"] = "incomplete"
                            save_results(run)
                            print("Промпт не завершён; второй этап этой комбинации не выполняется.")
                            continue
                        prompt = "Реши исходную задачу по следующей инструкции.\n\nЗадача:\n" + task["question"] + "\n\nИнструкция решателю:\n" + author["answer"]
                    solution = call_stage(record, "solution", prompt)
                    if solution["state"] == "api_error":
                        record["state"] = run["state"] = "api_error"
                        print(solution["error"])
                        return 1
                    record["answer"] = solution["answer"]
                    record["state"] = "completed" if solution["finish_reason"] == "stop" and solution["answer"].strip() else "incomplete"
                    save_results(run)
                    print(solution["answer"] or "Итоговый текст отсутствует.", flush=True)
        run["state"] = "completed"
        return 0
    except KeyboardInterrupt:
        run["state"] = "interrupted"
        if record is not None and record["state"] == "pending":
            record["state"] = "interrupted"
            if record["calls"] and record["calls"][-1]["state"] == "pending":
                record["calls"][-1]["state"] = "interrupted"
        print("\nПрервано. Полученные данные сохранены.")
        return 130
    finally:
        save_results(run)
        print(summary_text(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="День 03: четыре способа решения, reasoning none/high.")
    parser.add_argument("--effort", choices=("none", "high", "xhigh"), help="Выполнить только выбранный режим: 25 API-вызовов.")
    parser.add_argument("--stream", action="store_true", help="Получать новые ответы по SSE, сохраняя настройки модели.")
    parser.add_argument("--resume", action="store_true", help="Продолжить results.txt, пропустив готовые ответы.")
    parser.add_argument("--report", action="store_true", help="Показать сохранённый текстовый отчёт без запросов к API.")
    args = parser.parse_args()
    if args.report:
        print((DIRECTORY / "results.txt").read_text(encoding="utf-8"))
    else:
        raise SystemExit(main((args.effort,) if args.effort else EFFORTS, resume=args.resume, streaming=args.stream))
