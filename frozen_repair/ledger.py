import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from .common import Blocked, BudgetExceeded, Invalid, atomic_json, file_hash, canonical


@contextlib.contextmanager
def run_lock(directory):
    """OS lock is released on process exit. Never infer liveness from lock-file age."""
    path = Path(directory) / "run.lock"
    f = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise Blocked("Run is already locked") from exc
        else:
            import fcntl
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise Blocked("Run is already locked") from exc
        yield
    finally:
        f.close()


class Ledger:
    def __init__(self, directory, limits, existing=False, expected_tasks=None, readonly=False):
        self.directory, self.limits = Path(directory), limits
        path = self.directory / "ledger.sqlite"
        self.expected_tasks = None if expected_tasks is None else set(expected_tasks)
        self.db = None
        try:
            if existing:
                if self.expected_tasks is None:
                    raise Invalid("Recovery requires manifest task IDs")
                self.db = sqlite3.connect(path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True)
                self.validate()
            else:
                if path.exists() or readonly:
                    raise Invalid("Refuse initializing an existing ledger")
                self.db = sqlite3.connect(path)
                self.db.execute("PRAGMA foreign_keys=ON")
                with self.db:
                    self.db.execute("PRAGMA user_version=2")
                    self.db.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, hash TEXT, error TEXT)")
                    self.db.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY, kind TEXT, task TEXT, started REAL, detail TEXT)")
                    self.db.execute("CREATE TABLE cache (id TEXT PRIMARY KEY, hash TEXT, call_id INTEGER NOT NULL UNIQUE REFERENCES calls(id))")
                    self.db.execute("CREATE TABLE proofs (name TEXT PRIMARY KEY, hash TEXT NOT NULL)")
                    if self.expected_tasks is not None:
                        self.db.executemany("INSERT INTO tasks VALUES (?, 'pending', NULL, NULL)",
                                            [(t,) for t in sorted(self.expected_tasks)])
            if not readonly:
                self.db.execute("PRAGMA foreign_keys=ON")
                self.db.execute("PRAGMA synchronous=FULL")
        except Exception as exc:
            if self.db is not None:
                self.db.close()
            if isinstance(exc, Invalid):
                raise
            raise Invalid(f"Ledger open/validation failed: {type(exc).__name__}: {exc}") from exc

    def validate(self):
        from .common import digest
        if self.db.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise Invalid("Unsupported ledger schema; no automatic migration")
        if self.db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise Invalid("SQLite integrity check failed")
        schemas = dict(tasks=["id", "status", "hash", "error"], calls=["id", "kind", "task", "started", "detail"],
                       cache=["id", "hash", "call_id"], proofs=["name", "hash"])
        for table, fields in schemas.items():
            columns = self.db.execute(f"PRAGMA table_info({table})").fetchall()
            types = {"id": "INTEGER" if table == "calls" else "TEXT", "started": "REAL", "call_id": "INTEGER"}
            if ([c[1] for c in columns] != fields or not columns or columns[0][5] != 1 or
                    any(c[2].upper() != types.get(c[1], "TEXT") for c in columns)):
                raise Invalid(f"Invalid/missing ledger table structure: {table}")
            if table == "cache" and (columns[2][3] != 1 or
                    [(r[2], r[3], r[4]) for r in self.db.execute("PRAGMA foreign_key_list(cache)")] != [("calls", "call_id", "id")]):
                raise Invalid("Invalid cache foreign key schema")
        if self.db.execute("PRAGMA foreign_key_check").fetchall():
            raise Invalid("Ledger foreign key failure")
        tasks = self.db.execute("SELECT id,status,hash,error FROM tasks").fetchall()
        for task, state, checksum, error in tasks:
            if self.expected_tasks is not None and task not in self.expected_tasks:
                raise Invalid("Unexpected task ID in ledger")
            if state not in ("pending", "running", "completed", "failed"):
                raise Invalid("Unknown task state")
            if state == "completed" and (not checksum or error is not None):
                raise Invalid("Invalid completed task record")
        calls = {r[0]: r[1:] for r in self.db.execute("SELECT id,kind,task,detail FROM calls")}
        allowed_tasks = self.expected_tasks if self.expected_tasks is not None else {r[0] for r in tasks}
        for kind, task, detail in calls.values():
            semantic = kind in ("semantic_full", "semantic_partial")
            if kind not in self.limits or (semantic and task != "verify") or (not semantic and task not in allowed_tasks):
                raise Invalid("Unknown call kind/task")
            try:
                json.loads(detail)
            except (TypeError, ValueError) as exc:
                raise Invalid("Invalid call detail") from exc
        if any(n > self.limits[k] for k, n in self.counts().items()):
            raise Invalid("Recorded calls exceed fixed budget")
        used = set()
        for ident, checksum, call_id in self.db.execute("SELECT id,hash,call_id FROM cache"):
            if call_id not in calls or call_id in used:
                raise Invalid("Cache has no unique charged call")
            used.add(call_id)
            kind, task, detail = calls[call_id]
            if ident != digest([kind, task, json.loads(detail)]):
                raise Invalid("Cache/charged-call identity mismatch")
            path = self.directory / "raw" / "calls" / (ident + ".json")
            if not path.is_file() or file_hash(path) != checksum:
                raise Invalid("Cached call checksum mismatch")
        for task, state, _, _ in tasks:
            if state == "completed":
                self.completed(task)

    def commit_checks(self):
        if self.db.execute("SELECT 1 FROM proofs WHERE name='checks'").fetchone():
            raise Invalid("Checks proof already committed")
        self.db.execute("INSERT INTO proofs VALUES ('checks', ?)", (file_hash(self.directory / "checks.json"),))
        self.db.commit()

    def add(self, task):
        self.db.execute("INSERT OR IGNORE INTO tasks VALUES (?, 'pending', NULL, NULL)", (task,))
        self.db.commit()

    def begin(self, task):
        self.db.execute("UPDATE tasks SET status='running', error=NULL WHERE id=?", (task,))
        self.db.commit()

    def charge(self, kind, task, detail=None):
        with self.db:
            count = self.db.execute("SELECT count(*) FROM calls WHERE kind=?", (kind,)).fetchone()[0]
            if count >= self.limits[kind]:
                raise BudgetExceeded(f"Budget exhausted: {kind}; interrupted calls remain charged")
            cursor = self.db.execute("INSERT INTO calls(kind,task,started,detail) VALUES (?,?,?,?)",
                            (kind, task, time.time(), canonical(detail)))
            return cursor.lastrowid

    def complete(self, task, result):
        path = self.directory / "raw" / (task + ".json")
        atomic_json(path, result)
        h = file_hash(path)
        self.db.execute("UPDATE tasks SET status='completed', hash=?, error=NULL WHERE id=?", (h, task))
        self.db.commit()

    def cached(self, kind, task, key, execute):
        from .common import digest, read_json
        ident = digest([kind, task, key])
        path = self.directory / "raw" / "calls" / (ident + ".json")
        row = self.db.execute("SELECT hash FROM cache WHERE id=?", (ident,)).fetchone()
        if row:
            if not path.is_file() or file_hash(path) != row[0]:
                raise Invalid("Cached call checksum mismatch")
            return read_json(path)
        call_id = self.charge(kind, task, key)
        result = execute()
        atomic_json(path, result)
        self.db.execute("INSERT INTO cache VALUES (?,?,?)", (ident, file_hash(path), call_id))
        self.db.commit()
        return result

    def fail(self, task, error):
        self.db.execute("UPDATE tasks SET status='failed', error=? WHERE id=?", (str(error), task))
        self.db.commit()

    def completed(self, task):
        row = self.db.execute("SELECT status,hash FROM tasks WHERE id=?", (task,)).fetchone()
        if row and row[0] == "completed":
            path = self.directory / "raw" / (task + ".json")
            if not path.is_file() or file_hash(path) != row[1]:
                raise Invalid(f"Completed task checksum failure: {task}")
            return True
        return False

    def counts(self):
        return dict(self.db.execute("SELECT kind,count(*) FROM calls GROUP BY kind"))

    def close(self):
        self.db.close()


def status(directory):
    path = Path(directory).resolve() / "ledger.sqlite"
    if not path.is_file():
        raise Blocked("No ledger")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        return dict(tasks=dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status")),
                    calls=dict(db.execute("SELECT kind,count(*) FROM calls GROUP BY kind")))
    finally:
        db.close()
