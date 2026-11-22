"""Реестр инвариантов сервиса заказов и проверка предложений модели."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


DEFAULTS = {'storage': 'sqlite', 'stack': 'stdlib', 'confirmation': 'paid'}
OPTIONS = {'storage': ('sqlite', 'postgres', 'json_file'), 'stack': ('stdlib', 'catalog'),
           'confirmation': ('paid', 'any')}
STORES = {'sqlite': 'SQLite', 'postgres': 'PostgreSQL', 'json_file': 'JSON-файл',
          'redis': 'Redis', 'memory': 'оперативная память'}
# Закрытый каталог: всё, что не описано, требует уточнения.
MODULES = {'sqlite3': True, 'json': True, 'decimal': True, 'datetime': True,
           'requests': False, 'sqlalchemy': False, 'psycopg': False, 'redis': False}
CONDITIONS = {'paid': 'оплата подтверждена', 'unpaid': 'заказ не оплачен',
              'any': 'статус оплаты произвольный'}
OPERATIONS = {'store_order': ('storage', STORES), 'use_module': ('module', MODULES),
              'confirm_order': ('condition', CONDITIONS), 'explain_rule': ('rule', OPTIONS)}


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS rules (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS changes (version INTEGER PRIMARY KEY, '
                       'at TEXT, name TEXT, old TEXT, new TEXT)')
            db.executemany('INSERT OR IGNORE INTO rules VALUES (?,?)', DEFAULTS.items())

    def snapshot(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('BEGIN')
            version = db.execute('SELECT COALESCE(MAX(version), 0) FROM changes').fetchone()[0]
            return version, dict(db.execute('SELECT name,value FROM rules'))

    def history(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM changes ORDER BY version')]

    def change(self, name, new):
        if name not in OPTIONS or new not in OPTIONS[name]:
            raise ValueError('Неизвестное правило или значение. Допустимы: ' + json.dumps(OPTIONS))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT value FROM rules WHERE name=?', (name,)).fetchone()[0]
            if old == new:
                return 'Значение уже установлено; изменений нет.'
            db.execute('UPDATE rules SET value=? WHERE name=?', (new, name))
            row = db.execute('INSERT INTO changes(at,name,old,new) VALUES (?,?,?,?)',
                             (datetime.now(timezone.utc).isoformat(), name, old, new))
            return f'Правила v{row.lastrowid}: {name}: {old} → {new}.'


def describe(rule, values):
    if rule == 'storage':
        return 'Постоянное хранилище заказов — только ' + STORES[values[rule]] + '.'
    if rule == 'stack':
        return ('Разрешены только модули стандартной библиотеки из каталога.'
                if values[rule] == 'stdlib' else 'Разрешены все модули учебного каталога.')
    return ('Подтверждать можно только оплаченные заказы.' if values[rule] == 'paid'
            else 'Подтверждение разрешено при любом статусе оплаты.')


def conflicts(actions, rules):
    rejected = []
    for action in actions:
        op = action['op']
        if op == 'store_order' and action['storage'] != rules['storage']:
            rejected.append(('storage', 'Запрошено хранилище ' + STORES[action['storage']] + '.'))
        if op == 'use_module' and rules['stack'] == 'stdlib' and not MODULES[action['module']]:
            rejected.append(('stack', action['module'] + ' — сторонняя зависимость.'))
        if op == 'confirm_order' and rules['confirmation'] == 'paid' and action['condition'] != 'paid':
            rejected.append(('confirmation', 'Предложено подтверждение без обязательной оплаты.'))
    return rejected


def parse(raw):
    lines = raw.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ('```json', '```') and lines[-1].strip() == '```':
        raw = '\n'.join(lines[1:-1])
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError('Нужен объект.')
    if obj.get('kind') == 'refuse' and set(obj) == {'kind', 'rule'} and obj['rule'] in OPTIONS:
        return obj
    if obj == {'kind': 'clarify'}:
        return obj
    if set(obj) != {'kind', 'actions'} or obj['kind'] != 'propose':
        raise ValueError('Неизвестный формат.')
    actions = obj['actions']
    if not isinstance(actions, list) or not 1 <= len(actions) <= 12:
        raise ValueError('Нужны действия.')
    for action in actions:
        if not isinstance(action, dict) or action.get('op') not in OPERATIONS:
            raise ValueError('Неизвестная операция.')
        field, choices = OPERATIONS[action['op']]
        if set(action) != {'op', field} or action[field] not in choices:
            raise ValueError('Неизвестные параметры.')
    return obj


def render(action, rules):
    op = action['op']
    if op == 'store_order':
        return 'Сохранять заказы в ' + STORES[action['storage']] + '.'
    if op == 'use_module':
        return 'Использовать модуль ' + action['module'] + '.'
    if op == 'confirm_order':
        return 'Переводить заказ в confirmed при условии: ' + CONDITIONS[action['condition']] + '.'
    return describe(action['rule'], rules)
