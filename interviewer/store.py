"""SQLite persistence for interviews, transcripts, scorecards, reviews and
integrity signals — plus the retention/erasure machinery the DPDP Act implies.

Pure standard library (``sqlite3``). One connection guarded by a lock is plenty
for a prototype; swap for Postgres when you outgrow it.

Compliance helpers
------------------
* ``retention_until`` is stamped on every interview; ``purge_expired()`` deletes
  the ones past their window.
* ``export_candidate(email)`` returns everything held about one person (a data
  access request).
* ``delete_interview`` / ``delete_candidate`` honour erasure requests.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id TEXT PRIMARY KEY,
    created_at REAL, updated_at REAL,
    status TEXT,
    mode TEXT, interview_type TEXT, role TEXT, level TEXT,
    candidate_name TEXT, candidate_email TEXT,
    company TEXT, language TEXT,
    consent INTEGER DEFAULT 0,
    consent_at REAL, consent_version TEXT,
    retention_until REAL,
    llm TEXT,
    weighted_total REAL,
    recommendation TEXT,
    summary TEXT,
    integrity_score REAL, flags TEXT, proctor_events INTEGER DEFAULT 0,
    review_status TEXT DEFAULT 'unreviewed',
    reviewer TEXT, reviewer_notes TEXT,
    human_score REAL, human_recommendation TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT, idx INTEGER,
    role TEXT, text TEXT, stage TEXT, question_id TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS competencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT, comp_id TEXT, name TEXT, weight REAL,
    score REAL, evidence TEXT, comment TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_interview ON turns(interview_id);
CREATE INDEX IF NOT EXISTS idx_comp_interview ON competencies(interview_id);
CREATE INDEX IF NOT EXISTS idx_email ON interviews(candidate_email);
"""

# columns added after the first release — migrated in place on open
_NEW_COLUMNS = {
    "consent_at": "REAL", "consent_version": "TEXT", "retention_until": "REAL",
    "integrity_score": "REAL", "flags": "TEXT", "proctor_events": "INTEGER DEFAULT 0",
}


class Store:
    def __init__(self, path="interviews.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(interviews)")}
        for name, typ in _NEW_COLUMNS.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE interviews ADD COLUMN {name} {typ}")
        self._conn.commit()

    # -- writes -----------------------------------------------------------
    def create_interview(self, config: dict, llm: str = "") -> str:
        iid = uuid.uuid4().hex[:12]
        now = time.time()
        consent = int(bool(config.get("consent")))
        days = int(config.get("retention_days") or 90)
        with self._lock:
            self._conn.execute(
                """INSERT INTO interviews
                   (id, created_at, updated_at, status, mode, interview_type, role,
                    level, candidate_name, candidate_email, company, language,
                    consent, consent_at, consent_version, retention_until, llm,
                    review_status, proctor_events)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (iid, now, now, "active", config.get("mode"), config.get("interview_type"),
                 config.get("role"), config.get("level"), config.get("candidate_name"),
                 config.get("candidate_email"), config.get("company"),
                 config.get("language"), consent, now if consent else None,
                 config.get("consent_version"), now + days * 86400, llm,
                 "unreviewed", 0),
            )
            self._conn.commit()
        return iid

    def add_turn(self, interview_id, idx, turn: dict):
        with self._lock:
            self._conn.execute(
                """INSERT INTO turns (interview_id, idx, role, text, stage, question_id, kind)
                   VALUES (?,?,?,?,?,?,?)""",
                (interview_id, idx, turn.get("role"), turn.get("text"), turn.get("stage"),
                 turn.get("question_id"), turn.get("kind")),
            )
            self._conn.execute("UPDATE interviews SET updated_at=? WHERE id=?",
                               (time.time(), interview_id))
            self._conn.commit()

    def save_scorecard(self, interview_id, scorecard: dict):
        with self._lock:
            self._conn.execute("DELETE FROM competencies WHERE interview_id=?", (interview_id,))
            for c in scorecard.get("competencies", []):
                self._conn.execute(
                    """INSERT INTO competencies
                       (interview_id, comp_id, name, weight, score, evidence, comment)
                       VALUES (?,?,?,?,?,?,?)""",
                    (interview_id, c.get("id"), c.get("name"), c.get("weight"),
                     c.get("score"), c.get("evidence"), c.get("comment")),
                )
            self._conn.execute(
                """UPDATE interviews SET status='completed', updated_at=?,
                   weighted_total=?, recommendation=?, summary=? WHERE id=?""",
                (time.time(), scorecard.get("weighted_total"),
                 scorecard.get("recommendation"), scorecard.get("summary"), interview_id),
            )
            self._conn.commit()

    def save_integrity(self, interview_id, summary: dict):
        with self._lock:
            self._conn.execute(
                """UPDATE interviews SET integrity_score=?, flags=?, proctor_events=?,
                   updated_at=? WHERE id=?""",
                (summary.get("score"), json.dumps(summary.get("flags", [])),
                 summary.get("events", 0), time.time(), interview_id),
            )
            self._conn.commit()

    def save_review(self, interview_id, reviewer, notes, human_score, human_recommendation):
        with self._lock:
            self._conn.execute(
                """UPDATE interviews SET review_status='reviewed', reviewer=?,
                   reviewer_notes=?, human_score=?, human_recommendation=?,
                   updated_at=? WHERE id=?""",
                (reviewer, notes, human_score, human_recommendation, time.time(), interview_id),
            )
            self._conn.commit()

    def delete_interview(self, interview_id):
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM competencies WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM interviews WHERE id=?", (interview_id,))
            self._conn.commit()

    def delete_candidate(self, email):
        """Erasure request: remove everything held about one person."""
        ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM interviews WHERE candidate_email=?", (email,)).fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

    def purge_expired(self, now=None) -> int:
        """Delete interviews past their retention window. Returns the count removed."""
        now = now or time.time()
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM interviews WHERE retention_until IS NOT NULL AND retention_until < ?",
                (now,)).fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

    # -- reads ------------------------------------------------------------
    def list_interviews(self, status=None, sort="score") -> list:
        order = {
            "score": "COALESCE(weighted_total, -1) DESC, created_at DESC",
            "recent": "created_at DESC",
            "name": "candidate_name ASC",
            "integrity": "COALESCE(integrity_score, -1) ASC, created_at DESC",
        }.get(sort, "created_at DESC")
        q = "SELECT * FROM interviews"
        args = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += f" ORDER BY {order}"
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    def get_interview(self, interview_id):
        with self._lock:
            row = self._conn.execute("SELECT * FROM interviews WHERE id=?",
                                     (interview_id,)).fetchone()
            if not row:
                return None
            turns = [dict(r) for r in self._conn.execute(
                "SELECT * FROM turns WHERE interview_id=? ORDER BY idx", (interview_id,))]
            comps = [dict(r) for r in self._conn.execute(
                "SELECT * FROM competencies WHERE interview_id=? ORDER BY id", (interview_id,))]
        out = dict(row)
        out["turns"] = turns
        out["competencies"] = comps
        try:
            out["flags"] = json.loads(out["flags"]) if out["flags"] else []
        except (TypeError, ValueError):
            out["flags"] = []
        return out

    def export_candidate(self, email) -> list:
        rows = self._conn.execute(
            "SELECT id FROM interviews WHERE candidate_email=?", (email,)).fetchall()
        return [self.get_interview(r["id"]) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) c FROM interviews").fetchone()["c"]
            done = self._conn.execute(
                "SELECT COUNT(*) c FROM interviews WHERE status='completed'").fetchone()["c"]
            reviewed = self._conn.execute(
                "SELECT COUNT(*) c FROM interviews WHERE review_status='reviewed'").fetchone()["c"]
            flagged = self._conn.execute(
                "SELECT COUNT(*) c FROM interviews WHERE integrity_score IS NOT NULL "
                "AND integrity_score < 90").fetchone()["c"]
            avg = self._conn.execute(
                "SELECT AVG(weighted_total) a FROM interviews WHERE status='completed'"
            ).fetchone()["a"]
        return {
            "total": total,
            "completed": done,
            "reviewed": reviewed,
            "flagged": flagged,
            "avg_score": round(avg, 1) if avg is not None else None,
        }
