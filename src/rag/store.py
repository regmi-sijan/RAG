"""SQLite stores source versions, passages, and vectors in the same transaction."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import RagError, unit_vector


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / "rag.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, locator TEXT NOT NULL,
                    content_hash TEXT, fingerprint TEXT, model TEXT, updated TEXT,
                    active INTEGER NOT NULL DEFAULT 1, error TEXT, warnings TEXT DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY, source_id TEXT NOT NULL,
                    text TEXT NOT NULL, location TEXT NOT NULL, vector TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id));
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, finished TEXT, results TEXT);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def source(self, source_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            return dict(row) if row else None

    def replace(self, source, content_hash, fingerprint, model, passages, vectors, warnings):
        if not passages or len(passages) != len(vectors):
            raise ValueError("Cannot index empty or misaligned passages and vectors.")
        with self.connection() as db:
            db.execute("""INSERT INTO sources(id,kind,locator,content_hash,fingerprint,model,updated,active,error,warnings)
                VALUES(?,?,?,?,?,?,?,1,NULL,?) ON CONFLICT(id) DO UPDATE SET
                content_hash=excluded.content_hash,fingerprint=excluded.fingerprint,model=excluded.model,
                updated=excluded.updated,active=1,error=NULL,warnings=excluded.warnings""",
                       (source["id"], source["kind"], source["locator"], content_hash, fingerprint, model, now(), json.dumps(warnings)))
            db.execute("DELETE FROM chunks WHERE source_id=?", (source["id"],))
            db.executemany("INSERT INTO chunks(source_id,text,location,vector) VALUES(?,?,?,?)", [
                (source["id"], p["text"], json.dumps({k: v for k, v in p.items() if k != "text"}), json.dumps(v))
                for p, v in zip(passages, vectors)])

    def failure(self, source, error):
        with self.connection() as db:
            db.execute("""INSERT INTO sources(id,kind,locator,error) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET error=excluded.error,active=1""",
                       (source["id"], source["kind"], source["locator"], str(error)[:2000]))

    def synchronize(self, ids):
        with self.connection() as db:
            db.execute("UPDATE sources SET active=0")
            db.executemany("UPDATE sources SET active=1 WHERE id=?", [(i,) for i in ids])

    def summary(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("""SELECT s.*, COUNT(c.id) AS chunks FROM sources s
                LEFT JOIN chunks c ON c.source_id=s.id GROUP BY s.id ORDER BY s.locator""")]

    def passages(self, allowed, model):
        if not allowed:
            return []
        # Bind one source at a time to avoid SQLite's parameter limit for larger libraries.
        rows = []
        with self.connection() as db:
            for source_id in allowed:
                rows.extend(dict(row) for row in db.execute("""SELECT c.*, s.kind, s.locator, s.updated, s.error FROM chunks c
                    JOIN sources s ON s.id=c.source_id WHERE s.active=1 AND s.id=? AND s.model=?""", (source_id, model)))
        result = []
        try:
            for row in rows:
                location = json.loads(row["location"])
                if not isinstance(location, dict):
                    raise ValueError("Invalid location metadata.")
                result.append({**row, "location": location, "vector": unit_vector(json.loads(row["vector"]))})
        except (ValueError, TypeError, RagError) as error:
            raise RagError("Stored index data is invalid. Rebuild the index before asking questions.") from error
        return result

    def log_run(self, results):
        with self.connection() as db:
            db.execute("INSERT INTO runs(finished,results) VALUES(?,?)", (now(), json.dumps(results)))

    def recent_runs(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 5")]
