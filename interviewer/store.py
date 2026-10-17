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
    session_id TEXT,
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
    human_score REAL, human_recommendation TEXT,
    resume_text TEXT, job_description TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT, idx INTEGER,
    role TEXT, text TEXT, stage TEXT, question_id TEXT, kind TEXT,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS competencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT, comp_id TEXT, name TEXT, weight REAL,
    score REAL, evidence TEXT, comment TEXT
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    section TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_interview ON turns(interview_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx);
CREATE INDEX IF NOT EXISTS idx_comp_interview ON competencies(interview_id);
CREATE INDEX IF NOT EXISTS idx_chunks_interview ON document_chunks(interview_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(interview_id, doc_type);
CREATE INDEX IF NOT EXISTS idx_email ON interviews(candidate_email);
"""

# columns added after the first release — migrated in place on open
_NEW_COLUMNS = {
    "session_id": "TEXT",
    "consent_at": "REAL", "consent_version": "TEXT", "retention_until": "REAL",
    "integrity_score": "REAL", "flags": "TEXT", "proctor_events": "INTEGER DEFAULT 0",
    "resume_text": "TEXT", "job_description": "TEXT",
    "evaluation_report": "TEXT",
}


class Store:
    def __init__(self, path="interviews.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=10000;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(interviews)")}
        for name, typ in _NEW_COLUMNS.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE interviews ADD COLUMN {name} {typ}")

        turn_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(turns)")}
        if "evidence" not in turn_cols:
            self._conn.execute("ALTER TABLE turns ADD COLUMN evidence TEXT")

        # Ensure index creation after all columns are confirmed
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id)")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx)")

        self._conn.commit()


    # -- writes -----------------------------------------------------------
    def create_interview(self, config: dict, llm: str = "", session_id: str = "") -> str:
        iid = uuid.uuid4().hex[:12]
        now = time.time()
        consent = int(bool(config.get("consent")))
        days = int(config.get("retention_days") or 90)
        with self._lock:
            self._conn.execute(
                """INSERT INTO interviews
                   (id, session_id, created_at, updated_at, status, mode, interview_type, role,
                    level, candidate_name, candidate_email, company, language,
                    consent, consent_at, consent_version, retention_until, llm,
                    review_status, proctor_events, resume_text, job_description)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (iid, session_id or None, now, now, "active", config.get("mode"), config.get("interview_type"),
                 config.get("role"), config.get("level"), config.get("candidate_name"),
                 config.get("candidate_email"), config.get("company"),
                 config.get("language"), consent, now if consent else None,
                 config.get("consent_version"), now + days * 86400, llm,
                 "unreviewed", 0, str(config.get("resume_text") or ""),
                 str(config.get("job_description") or "")),
            )
            self._conn.commit()
        return iid

    def add_turn(self, interview_id, idx, turn: dict):
        ev = turn.get("evidence")
        ev_str = json.dumps(ev) if ev is not None and not isinstance(ev, str) else ev
        with self._lock:
            self._conn.execute(
                """INSERT INTO turns (interview_id, idx, role, text, stage, question_id, kind, evidence)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(interview_id, idx) DO UPDATE SET
                       role=excluded.role, text=excluded.text, stage=excluded.stage,
                       question_id=excluded.question_id, kind=excluded.kind, evidence=excluded.evidence""",
                (interview_id, idx, turn.get("role"), turn.get("text"), turn.get("stage"),
                 turn.get("question_id"), turn.get("kind"), ev_str),
            )
            self._conn.execute("UPDATE interviews SET updated_at=? WHERE id=?",
                               (time.time(), interview_id))
            self._conn.commit()


    def save_scorecard(self, interview_id, scorecard: dict):
        report_json = json.dumps(scorecard)
        with self._lock:
            self._conn.execute("DELETE FROM competencies WHERE interview_id=?", (interview_id,))
            for c in scorecard.get("competencies", []):
                cmt = str(c.get("explanation") or c.get("comment") or "")
                self._conn.execute(
                    """INSERT INTO competencies
                       (interview_id, comp_id, name, weight, score, evidence, comment)
                       VALUES (?,?,?,?,?,?,?)""",
                    (interview_id, c.get("id"), c.get("name"), c.get("weight"),
                     c.get("score"), c.get("evidence"), cmt),
                )
            self._conn.execute(
                """UPDATE interviews SET status='completed', updated_at=?,
                   weighted_total=?, recommendation=?, summary=?, evaluation_report=? WHERE id=?""",
                (time.time(), scorecard.get("weighted_total"),
                 scorecard.get("recommendation"), scorecard.get("summary"),
                 report_json, interview_id),
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

    # -- RAG Document Chunks ----------------------------------------------
    def save_document_chunks(self, interview_id: str, chunks: list, doc_type: str | None = None):
        """Persist embedded chunks for an interview. Idempotently cleans up previous chunks."""
        with self._lock:
            if doc_type:
                self._conn.execute(
                    "DELETE FROM document_chunks WHERE interview_id=? AND doc_type=?",
                    (interview_id, doc_type),
                )
            else:
                self._conn.execute(
                    "DELETE FROM document_chunks WHERE interview_id=?",
                    (interview_id,),
                )
            for c in chunks:
                c_dict = c if isinstance(c, dict) else vars(c)
                emb = c_dict.get("embedding")
                emb_json = json.dumps(emb) if emb is not None else "[]"
                self._conn.execute(
                    """INSERT INTO document_chunks
                       (interview_id, doc_type, section, chunk_index, text, embedding, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (interview_id, c_dict.get("doc_type"), c_dict.get("section"),
                     int(c_dict.get("chunk_index", 0)), str(c_dict.get("text", "")),
                     emb_json, float(c_dict.get("created_at") or time.time())),
                )
            self._conn.commit()

    def get_document_chunks(self, interview_id: str, doc_type: str | None = None) -> list:
        """Fetch chunks for an interview session. Strictly filters by interview_id for isolation."""
        with self._lock:
            q = "SELECT * FROM document_chunks WHERE interview_id=?"
            args = [interview_id]
            if doc_type:
                q += " AND doc_type=?"
                args.append(doc_type)
            q += " ORDER BY chunk_index ASC"
            rows = self._conn.execute(q, args).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["embedding"] = json.loads(d["embedding"]) if d.get("embedding") else []
            except Exception:
                d["embedding"] = []
            result.append(d)
        return result

    def delete_document_chunks(self, interview_id: str, doc_type: str | None = None):
        with self._lock:
            if doc_type:
                self._conn.execute(
                    "DELETE FROM document_chunks WHERE interview_id=? AND doc_type=?",
                    (interview_id, doc_type),
                )
            else:
                self._conn.execute(
                    "DELETE FROM document_chunks WHERE interview_id=?",
                    (interview_id,),
                )
            self._conn.commit()

    def update_interview_documents(
        self,
        interview_id: str,
        resume_text: str | None = None,
        job_description: str | None = None,
    ) -> bool:
        """Update resume_text and/or job_description in interviews table and refresh timestamps."""
        updates = []
        params = []
        if resume_text is not None:
            updates.append("resume_text=?")
            params.append(resume_text)
        if job_description is not None:
            updates.append("job_description=?")
            params.append(job_description)

        if not updates:
            return False

        updates.append("updated_at=?")
        params.append(time.time())
        params.append(interview_id)

        sql = f"UPDATE interviews SET {', '.join(updates)} WHERE id=?"
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.rowcount > 0

    def update_interview_status(self, interview_id: str, status: str) -> bool:
        """Update session status (e.g., active, completed, interrupted)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE interviews SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), interview_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_interview_by_session_id(self, session_id: str) -> dict | None:
        """Fetch full interview details by session_id."""
        if not session_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM interviews WHERE session_id=? OR id=?",
                (session_id, session_id),
            ).fetchone()
        if not row:
            return None
        return self.get_interview(row["id"])


    def delete_interview(self, interview_id):
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM competencies WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM document_chunks WHERE interview_id=?", (interview_id,))
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
            # Chunk statistics
            chunk_stats_rows = self._conn.execute(
                "SELECT doc_type, COUNT(*) as c FROM document_chunks WHERE interview_id=? GROUP BY doc_type",
                (interview_id,),
            ).fetchall()
        out = dict(row)
        for t in turns:
            if t.get("evidence"):
                try:
                    t["evidence"] = json.loads(t["evidence"])
                except Exception:
                    pass
        out["turns"] = turns
        out["competencies"] = comps
        chunk_counts = {r["doc_type"]: r["c"] for r in chunk_stats_rows}
        out["chunk_stats"] = {
            "resume_chunks": chunk_counts.get("resume", 0),
            "jd_chunks": chunk_counts.get("job_description", 0),
            "total": sum(chunk_counts.values()),
        }
        try:
            out["flags"] = json.loads(out["flags"]) if out["flags"] else []
        except (TypeError, ValueError):
            out["flags"] = []
        if out.get("evaluation_report"):
            try:
                out["scorecard"] = json.loads(out["evaluation_report"])
            except Exception:
                pass
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
