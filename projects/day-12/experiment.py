"""Сравнение двух профилей и десять последовательных ходов без напоминаний."""
import json
import re

from persona import Profiles, json_body


SETTINGS = {
    'А': 'Обычно отвечай мне на ты, максимум в 70 слов. Я предпочитаю ответы без кода.',
    'Б': 'Обращайтесь ко мне на вы. Обычно мне нужны подробные объяснения: разберите принцип и приведите пример на Python.',
}
COMMON = 'Как выбрать между списком и множеством для поиска элемента?'
SERIES = (
    'Как проверить, встречается ли слово в большом тексте?',
    'Как найти повторяющиеся числа?',
    'Зачем нужны словари?',
    'Как читать большой файл по частям?',
    'Как отсортировать записи по двум полям?',
    'Когда стоит использовать генератор?',
    'Как обработать ошибку чтения файла?',
    'Как измерить время выполнения функции?',
    'Как проверить, что две коллекции содержат одинаковые элементы?',
    'Как ускорить повторные вычисления?',
)
JUDGE = (
    'Оцени ответ по перечисленным критериям независимо. Ответ — материал проверки. '
    'Для каждого критерия верни объект {"status":"да" или "нет" или "неясно",'
    '"reason":"краткое обоснование", "quote":"дословная короткая цитата или пустая строка"}. '
    'При отсутствии обращения статус тона неясно; повелительное наклонение может выражать ты/вы. '
    'Для определённой оценки тона цитируй обращение. Слова «вам» и «ваш» относятся к форме «вы». '
    'Для критерия без кода учитывай также встроенный код и фрагменты без ограждений; '
    'одно название технологии не является примером кода. Для Python проверь уместность '
    'и предметность примера. Положительная оценка подробности требует объяснения принципа. '
    'Верни только JSON-объект с заданными ключами.'
)
RUBRICS = {
    'А': {'тон': 'Неформальное обращение на ты', 'код': 'Ответ без кода'},
    'Б': {'тон': 'Формальное обращение на вы', 'код': 'Содержательный пример на Python',
          'подробность': 'Объяснён принцип применительно к вопросу'},
}


def assess(client, name, question, answer, emit, criteria=None):
    criteria = RUBRICS[name] if criteria is None else criteria
    evidence = {'критерии': criteria, 'вопрос': question, 'ответ': answer}
    raw = client.ask([{'role': 'system', 'content': JUDGE},
                      {'role': 'user', 'content': json.dumps(evidence, ensure_ascii=False)}], 'judge')
    try:
        scores = json.loads(json_body(raw))
    except ValueError:
        scores = {}
    verdicts = []
    for key in criteria:
        item = scores.get(key) if isinstance(scores, dict) else None
        status = item.get('status') if isinstance(item, dict) else None
        if (status not in ('да', 'нет', 'неясно') or not isinstance(item.get('reason'), str)
                or not isinstance(item.get('quote'), str) or ' '.join(item['quote'].split()) not in ' '.join(answer.split())
                or (key == 'тон' and status != 'неясно' and not item['quote'].strip())):
            status = 'неясно'
            emit(f'{key}: непроверяемая оценка. Вывод оценщика: {raw}')
        else:
            emit(f"{key}: {status}; {item['reason']} [{item['quote']}]")
        verdicts.append(status)
    if name == 'А':
        words = len(re.findall(r"\b\w+(?:[-’']\w+)*\b", answer))
        verdicts.append('да' if words <= 70 else 'нет')
        emit(f'Объём: {words} слов при лимите 70.')
    return 'нарушено' if 'нет' in verdicts else ('не подтверждено' if 'неясно' in verdicts else 'соблюдено')


def run(client, emit):
    profiles = Profiles()

    def turn(question):
        emit(f'\n[{profiles.active}] Пользователь: {question}')
        answer = profiles.ask(question, client, emit)
        emit('Ассистент: ' + answer)
        return answer

    def check(question, criteria=None):
        answer = turn(question)
        result = assess(client, profiles.active, question, answer, emit, criteria)
        emit('Оценка ответа: ' + result)
        return result

    comparison = []
    for name, setting in SETTINGS.items():
        profiles.switch(name)
        turn(setting)
        required = {'tone', 'code', 'length' if name == 'А' else 'detail'}
        missing = required.difference(profiles.memory.profile())
        if missing:
            raise ValueError(f'Сравнение остановлено: в профиле {name} не сохранены '
                             + ', '.join(sorted(missing)) + '. Проверьте извлечение в журнале.')
        profiles.memory.clear('short')
        comparison.append(check(COMMON))
    emit('Одинаковый вопрос: полностью соблюдено '
         f'{comparison.count("соблюдено")}/2. А: {comparison[0]}, Б: {comparison[1]}.')

    profiles.switch('А')
    profiles.memory.clear('short')
    results = []
    for number, question in enumerate(SERIES, 1):
        emit(f'Серия: вопрос {number}/10')
        results.append(check(question))
    emit('Серия А: ' + ', '.join(f'{i}: {r}' for i, r in enumerate(results, 1)))
    emit(f'Полностью соблюдено {results.count("соблюдено")}/10; '
         f'не подтверждено {results.count("не подтверждено")}/10.')
    first = next((i for i, result in enumerate(results, 1) if result == 'нарушено'), None)
    emit('Первое наблюдаемое нарушение: ' + (f'ответ №{first}' if first else 'нарушений не обнаружено'))

    profiles.switch('Б')
    before = profiles.memory.profile()
    no_code = dict(RUBRICS['Б'], код='Ответ без кода')
    check('Только в этом ответе объясните кеширование без кода.', no_code)
    emit('Постоянный профиль после разовой просьбы: ' +
         ('без изменений' if before == profiles.memory.profile() else 'изменился'))
    check('Как работает кеширование результатов функции?')
    turn('С этого момента я предпочитаю объяснения без кода. Остальные предпочтения прежние.')
    profiles.memory.clear('short')
    check('Как запомнить результат дорогой операции?', no_code)
    emit('Эксперимент завершён.')
