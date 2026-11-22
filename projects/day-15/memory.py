"""Текущий снимок и отдельный журнал изменений для каждой задачи."""
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


class Memory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks(name TEXT PRIMARY KEY, closed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS arrivals(id INTEGER PRIMARY KEY, at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources(seq INTEGER PRIMARY KEY, message TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS facts(
                    task TEXT, kind TEXT, key TEXT, value TEXT, at TEXT NOT NULL, seq INTEGER NOT NULL,
                    PRIMARY KEY(task,kind,key));
                CREATE TABLE IF NOT EXISTS log(
                    id INTEGER PRIMARY KEY, task TEXT, kind TEXT, key TEXT, event TEXT,
                    old TEXT, new TEXT, at TEXT, seq INTEGER);
            """)

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN IMMEDIATE')
            yield db

    def arrive(self, task, message=None):
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO tasks(name) VALUES (?)', (task,))
            at = now()
            seq = db.execute('INSERT INTO arrivals(at) VALUES (?)', (at,)).lastrowid
            if message is not None:
                db.execute('INSERT INTO sources VALUES (?,?)', (seq, message))
            return at, seq

    def snapshot(self, task):
        with self.transaction() as db:
            rows = db.execute('SELECT kind,key,value,at FROM facts WHERE task=? ORDER BY kind,key', (task,))
            closed = db.execute('SELECT closed FROM tasks WHERE name=?', (task,)).fetchone()
            return {'task': task, 'closed': bool(closed and closed[0]), 'facts': [dict(r) for r in rows]}

    def history(self, task):
        with self.transaction() as db:
            return [dict(r) for r in db.execute(
                'SELECT log.*,sources.message AS source FROM log LEFT JOIN sources USING(seq) '
                'WHERE task=? ORDER BY log.id', (task,))]

    @staticmethod
    def record(db, task, kind, key, event, old, new, stamp):
        db.execute('INSERT INTO log(task,kind,key,event,old,new,at,seq) VALUES (?,?,?,?,?,?,?,?)',
                   (task, kind, key, event, old, new, *stamp))

    def apply(self, task, changes, stamp):
        with self.transaction() as db:
            if changes and db.execute('SELECT closed FROM tasks WHERE name=?', (task,)).fetchone()[0]:
                raise ValueError('Задача завершена. Для новых записей откройте другую задачу.')
            events = []
            for item in changes:
                kind, key, value = item['kind'], item['key'], item['value']
                identity = (task, kind, key)
                row = db.execute('SELECT value,at,seq FROM facts WHERE task=? AND kind=? AND key=?',
                                 identity).fetchone()
                old = row['value'] if row else None
                if row and stamp <= (row['at'], row['seq']):
                    event = 'IGNORE_STALE'
                else:
                    if value is None:
                        event = 'REVOKE'
                    elif not row or old is None:
                        event = 'APPEND' if kind == 'decision' else 'CREATE'
                    else:
                        event = 'CONFIRM' if old == value else 'REPLACE'
                    db.execute('INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?)', (*identity, value, *stamp))
                self.record(db, *identity, event, old, value, stamp)
                events.append(event)
            return events

    def finish(self, task):
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO tasks(name) VALUES (?)', (task,))
            if not db.execute('UPDATE tasks SET closed=1 WHERE name=? AND closed=0', (task,)).rowcount:
                return 0
            at = now()
            seq = db.execute('INSERT INTO arrivals(at) VALUES (?)', (at,)).lastrowid
            rows = db.execute("SELECT key,value FROM facts WHERE task=? AND kind='working'", (task,)).fetchall()
            for row in rows:
                self.record(db, task, 'working', row['key'], 'CLEAR', row['value'], None, (at, seq))
            db.execute("DELETE FROM facts WHERE task=? AND kind='working'", (task,))
            self.record(db, task, 'task', 'closed', 'FINISH', 'open', 'closed', (at, seq))
            return len(rows)
