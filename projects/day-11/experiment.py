"""Воспроизводимый диалог и проверки трёх слоёв памяти."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from memory import Memory


LAYERS = ("facts", "working", "short")
LABELS = {"facts": "факты о пользователе", "working": "память задачи", "short": "недавние реплики"}
TASK = "Подготовка макета"
UNKNOWN = "Нет данных"


def show_state(memory, emit):
    layers = memory.layers()
    for layer in LAYERS:
        value = json.dumps(layers[layer], ensure_ascii=False)
        emit(f"  {LABELS[layer]}: {value}")


def copy_memory(memory):
    return Memory.load(json.loads(json.dumps(memory.dump(), ensure_ascii=False)))


def normalized(text):
    return text.strip().strip(' .«»"“”`').casefold()


def probe(memory, client, emit, question, expected):
    before = json.dumps(memory.dump(), sort_keys=True)
    messages = memory.messages(question)
    tokens = client.counter.messages(messages, generation=True)
    answer = client.ask(messages, "probe")
    verdict = "совпало" if normalized(answer) == normalized(expected) else "расхождение"
    emit(f"Вопрос: {question}\nОжидание: {expected}\nОтвет: {answer}\nИтог: {verdict}. Вход: {tokens} токенов.")
    if json.dumps(memory.dump(), sort_keys=True) != before:
        raise RuntimeError("Проверочный вопрос изменил состояние памяти")
    return tokens, answer


def check_baseline(memory, questions, marker, emit):
    """Проверяем буквальное присутствие эталонов в активных слоях."""
    state = {
        layer: json.dumps(value, ensure_ascii=False).casefold()
        for layer, value in memory.layers().items()
    }
    issues = {layer: [] for layer in LAYERS}
    for layer, _, target in questions:
        found = [name for name, content in state.items() if target.casefold() in content]
        if layer not in found:
            issues[layer].append("эталон отсутствует в своём слое")
        for name in found:
            if name != layer:
                issues[layer].append(f"эталон также найден в слое «{LABELS[name]}»")
    for layer, content in state.items():
        if "тополь-47" in content:
            issues["working"].append(f"прежний шифр сохранился в слое «{LABELS[layer]}»")
    if marker != "Звон731":
        issues["short"].append("последний ответ отличается от заданного маркера")
    for layer, reasons in issues.items():
        emit(f"Проверка исходного слоя «{LABELS[layer]}»: " + (", ".join(reasons) or "эталон изолирован"))
    return issues


def emit_comparison(observed, questions, issues, emit):
    emit("\nРезультаты удаления слоёв. В ячейке приведены ответ и его оценка относительно полного контекста.")
    emit("Условия | Обращение | Шифр макета | Последний ответ")
    emit("--- | --- | --- | ---")
    full = observed[0][1]
    for title, answers in observed:
        cells = [title]
        for layer, _, target in questions:
            answer = answers[layer]
            if issues[layer]:
                status = "неопределённо: исходный слой"
            elif normalized(full[layer]) != normalized(target):
                status = "неопределённо: ошибка при всех слоях"
            elif answers is full:
                status = "эталон получен"
            elif normalized(answer) == normalized(target):
                status = "сохранён"
            else:
                status = "утрачен"
            display = " ".join(answer.splitlines()).replace("|", "│")
            cells.append(f"{display} ({status})")
        emit(" | ".join(cells))


def run(client, emit):
    memory = Memory()
    memory.switch(TASK)
    emit("Сценарий. Пользователь готовит макет и исправляет шифр задачи.")
    def turn(message):
        emit(f"Пользователь: {message}")
        reply = memory.ask(message, client, emit)
        emit(f"Ассистент: {reply}")
        return reply

    for message in (
        "Когда мы общаемся, обращайся ко мне Радан. Сегодня готовим макет афиши, "
        "его рабочий шифр — Тополь-47. Коротко подтверди.",
        "Уточнение по макету: рабочий шифр теперь Липа-83, старый шифр больше не используем.",
    ):
        turn(message)

    for message in (
        "Мы уже выбрали тип бумаги для печати? Если данных нет, так и скажи.",
        "Спасибо, пока всё.",
    ):
        before = copy_memory(memory).layers()
        turn(message)
        for layer in ("facts", "working"):
            verdict = "без изменений" if memory.layers()[layer] == before[layer] else "изменился"
            emit(f"После реплики слой «{LABELS[layer]}»: {verdict} (ожидание: без изменений).")

    # Эти нейтральные ходы вытесняют исходные реплики с проверяемыми значениями.
    for _ in range(12):
        recent = json.dumps(memory.layers()["short"], ensure_ascii=False).casefold()
        if not any(value in recent for value in ("радан", "тополь", "липа")):
            break
        turn("Ответь только словом «хорошо».")
    else:
        emit("Достигнут предел нейтральных реплик. Проверяем фактическое содержимое слоёв.")

    marker = turn(
        "Для этой реплики склей два фрагмента «Звон» и «731» без пробела. "
        "Ответь только получившейся строкой.",
    ).strip()
    emit(f"Ожидаемый маркер: Звон731. Фактический последний ответ: {marker}")
    emit("Общее исходное состояние для всех проверок")
    show_state(memory, emit)

    suffix = f" Если ответа в контексте нет, напиши только «{UNKNOWN}»."
    questions = (
        ("facts", "Как я прошу ко мне обращаться? Ответь только именем." + suffix, "Радан"),
        ("working", "Какой рабочий шифр у нашего макета? Ответь только шифром." + suffix, "Липа-83"),
        ("short", "Повтори дословно свой последний ответ в этом диалоге." + suffix, marker),
    )
    issues = check_baseline(memory, questions, marker, emit)
    emit("\nВходные токены при последовательном добавлении слоёв для одного и того же вопроса")
    measured_question = questions[0][1]
    emit(f"Вопрос: {measured_question}")
    previous = client.counter.messages(memory.messages(measured_question, included=()), generation=True)
    emit(f"Без памяти: {previous} ток.")
    for end, layer in enumerate(LAYERS, start=1):
        messages = memory.messages(measured_question, included=LAYERS[:end])
        total = client.counter.messages(messages, generation=True)
        emit(f"Добавлен слой «{LABELS[layer]}»: {total} ток. Изменение: {total - previous:+d} ток.")
        previous = total

    variants = [("Все слои", ())]
    variants.extend((f"Исключён слой «{LABELS[layer]}»", (layer,)) for layer in LAYERS)
    variants.append(("Все слои очищены", LAYERS))
    snapshot = json.dumps(memory.dump(), sort_keys=True)
    full_tokens = {}
    observed = []
    for title, removed in variants:
        emit(f"\n{title}")
        variant = copy_memory(memory)
        for layer in removed:
            variant.clear(layer)
        show_state(variant, emit)
        answers = {}
        for layer, question, target in questions:
            expected = UNKNOWN if layer in removed else target
            count, answers[layer] = probe(copy_memory(variant), client, emit, question, expected)
            if not removed:
                full_tokens[layer] = count
            else:
                emit(f"Разница с полным запросом: {full_tokens[layer] - count} токенов убрано.")

        observed.append((title, answers))

    emit("\nСмена задачи и возврат")
    switched = copy_memory(memory)
    switched.switch("Подготовка экскурсии")
    show_state(switched, emit)
    probe(switched, client, emit, questions[1][1], UNKNOWN)
    probe(switched, client, emit, questions[0][1], questions[0][2])
    switched.switch(TASK)
    show_state(switched, emit)
    probe(switched, client, emit, questions[1][1], questions[1][2])

    emit("\nНовый диалог после загрузки состояния из JSON")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        path.write_text(json.dumps(memory.dump(), ensure_ascii=False), encoding="utf-8")
        restored = Memory.load(json.loads(path.read_text(encoding="utf-8")))
    if json.dumps(restored.dump(), sort_keys=True) != snapshot:
        raise RuntimeError("Загрузка из JSON изменила состояние памяти")
    restored.clear("short")
    show_state(restored, emit)
    for layer, question, target in questions:
        probe(restored, client, emit, question, UNKNOWN if layer == "short" else target)
    if json.dumps(memory.dump(), sort_keys=True) != snapshot:
        raise RuntimeError("Эксперимент изменил исходный снимок памяти")
    emit_comparison(observed, questions, issues, emit)
    emit("Исходный снимок сохранился после всех проверок.")
