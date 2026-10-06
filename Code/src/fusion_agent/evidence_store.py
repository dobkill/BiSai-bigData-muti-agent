"""Local structured evidence retrieval and versioned assignment snapshots."""
import json
from pathlib import Path
import sqlite3

from .models import DayResult, Document, Observation
from .utils import stable_id


class EvidenceStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY, day TEXT NOT NULL, source TEXT NOT NULL,
                path TEXT NOT NULL, regions TEXT NOT NULL, directions TEXT NOT NULL,
                observation TEXT NOT NULL, original_text TEXT
            );
            CREATE INDEX IF NOT EXISTS evidence_day_source ON evidence(day, source);
            CREATE TABLE IF NOT EXISTS assignment_runs (
                id TEXT PRIMARY KEY, day TEXT NOT NULL, previous_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, result TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS latest_assignment (day TEXT PRIMARY KEY, run_id TEXT NOT NULL);
        """)

    def close(self):
        self.connection.close()

    def replace_day(self, day: str, observations: list[Observation], docs: list[Document]):
        texts = {d.id: d.text for d in docs}
        with self.connection:
            self.connection.execute("DELETE FROM evidence WHERE day = ?", (day,))
            self.connection.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                (o.id, day, o.source, o.path, json.dumps(o.facts.get("region", []), ensure_ascii=False),
                 json.dumps(o.facts.get("direction", []), ensure_ascii=False), o.model_dump_json(), texts.get(o.id))
                for o in observations])

    def observations(self, day: str) -> list[Observation]:
        return [Observation.model_validate_json(row[0]) for row in self.connection.execute(
            "SELECT observation FROM evidence WHERE day = ? ORDER BY source, path, id", (day,))]

    def documents(self, day: str) -> list[Document]:
        return [Document(id=row[0], date=day, source=row[1], path=row[2], text=row[3])
                for row in self.connection.execute(
                    "SELECT id, source, path, original_text FROM evidence WHERE day = ? AND original_text IS NOT NULL",
                    (day,))]

    def latest(self, day: str) -> DayResult | None:
        row = self.connection.execute("SELECT result FROM assignment_runs JOIN latest_assignment "
                                      "ON assignment_runs.id = latest_assignment.run_id WHERE latest_assignment.day = ?",
                                      (day,)).fetchone()
        return DayResult.model_validate_json(row[0]) if row else None

    def record(self, result: DayResult):
        previous = self.connection.execute("SELECT assignment_runs.id, result FROM assignment_runs JOIN latest_assignment "
                                           "ON assignment_runs.id = latest_assignment.run_id WHERE latest_assignment.day = ?",
                                           (result.date,)).fetchone()
        payload = result.model_dump_json()
        if previous and previous[1] == payload:
            return
        # Returning to an earlier assignment is a new revision, not a pointer
        # back to an old snapshot that would hide the intervening decisions.
        run_id = stable_id((previous[0] if previous else "") + "|" + payload)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO assignment_runs (id, day, previous_id, result) VALUES (?, ?, ?, ?)",
                                    (run_id, result.date, previous[0] if previous else None, payload))
            self.connection.execute("INSERT OR REPLACE INTO latest_assignment VALUES (?, ?)", (result.date, run_id))
