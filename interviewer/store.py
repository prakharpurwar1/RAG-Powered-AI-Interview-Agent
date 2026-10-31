"""Storage and persistence layer for interviews, transcripts, scorecards, reviews,
integrity signals, and RAG document chunks.

Supports both:
1. Local SQLite (zero-config file or :memory:, ideal for offline development and testing)
2. Managed PostgreSQL (Neon, Supabase, RDS) via connection pooling, suitable for
   stateless serverless execution where local disk is ephemeral.

Compliance helpers (DPDP Act):
* ``retention_until`` is stamped on every interview; ``purge_expired()`` deletes
  the ones past their window.
* ``export_candidate(email)`` returns everything held about one person (data access request).
* ``delete_interview`` / ``delete_candidate`` honour erasure requests.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from .llm import sanitize_secret_text

logger = logging.getLogger("interviewer.store")


class StoreError(RuntimeError):
    """Exception raised for database connectivity, pooling, or operational errors."""
    pass


try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None  # type: ignore


# ---------------------------------------------------------------------------
# SQLite Schema & Migration
# ---------------------------------------------------------------------------
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

# columns added after the first release — migrated in place on open for SQLite
_NEW_COLUMNS = {
    "session_id": "TEXT",
    "consent_at": "REAL", "consent_version": "TEXT", "retention_until": "REAL",
    "integrity_score": "REAL", "flags": "TEXT", "proctor_events": "INTEGER DEFAULT 0",
    "resume_text": "TEXT", "job_description": "TEXT",
    "evaluation_report": "TEXT",
}


# ---------------------------------------------------------------------------
# PostgreSQL Schema
# ---------------------------------------------------------------------------
POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(128),
    created_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION,
    status VARCHAR(64),
    mode VARCHAR(64),
    interview_type VARCHAR(64),
    role VARCHAR(255),
    level VARCHAR(64),
    candidate_name VARCHAR(255),
    candidate_email VARCHAR(255),
    company VARCHAR(255),
    language VARCHAR(64),
    consent INTEGER DEFAULT 0,
    consent_at DOUBLE PRECISION,
    consent_version VARCHAR(64),
    retention_until DOUBLE PRECISION,
    llm VARCHAR(64),
    weighted_total DOUBLE PRECISION,
    recommendation VARCHAR(128),
    summary TEXT,
    integrity_score DOUBLE PRECISION,
    flags TEXT,
    proctor_events INTEGER DEFAULT 0,
    review_status VARCHAR(64) DEFAULT 'unreviewed',
    reviewer VARCHAR(255),
    reviewer_notes TEXT,
    human_score DOUBLE PRECISION,
    human_recommendation VARCHAR(128),
    resume_text TEXT,
    job_description TEXT,
    evaluation_report TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id SERIAL PRIMARY KEY,
    interview_id VARCHAR(64) NOT NULL,
    idx INTEGER NOT NULL,
    role VARCHAR(64),
    text TEXT,
    stage VARCHAR(64),
    question_id VARCHAR(64),
    kind VARCHAR(64),
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS competencies (
    id SERIAL PRIMARY KEY,
    interview_id VARCHAR(64) NOT NULL,
    comp_id VARCHAR(64),
    name VARCHAR(255),
    weight DOUBLE PRECISION,
    score DOUBLE PRECISION,
    evidence TEXT,
    comment TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    interview_id VARCHAR(64) NOT NULL,
    doc_type VARCHAR(64) NOT NULL,
    section VARCHAR(128) NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id);
CREATE INDEX IF NOT EXISTS idx_email ON interviews(candidate_email);
CREATE INDEX IF NOT EXISTS idx_turns_interview ON turns(interview_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx);
CREATE INDEX IF NOT EXISTS idx_comp_interview ON competencies(interview_id);
CREATE INDEX IF NOT EXISTS idx_chunks_interview ON document_chunks(interview_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(interview_id, doc_type);
"""


# ===========================================================================
# SQLite Storage Backend
# ===========================================================================
class SQLiteStoreBackend:
    """Thread-safe SQLite storage implementation."""

    def __init__(self, path: str = "interviews.db"):
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

        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id)")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx)")
        self._conn.commit()

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

    def add_turn(self, interview_id: str, idx: int, turn: dict):
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

    def save_scorecard(self, interview_id: str, scorecard: dict):
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

    def save_integrity(self, interview_id: str, summary: dict):
        with self._lock:
            self._conn.execute(
                """UPDATE interviews SET integrity_score=?, flags=?, proctor_events=?,
                   updated_at=? WHERE id=?""",
                (summary.get("score"), json.dumps(summary.get("flags", [])),
                 summary.get("events", 0), time.time(), interview_id),
            )
            self._conn.commit()

    def save_review(self, interview_id: str, reviewer, notes, human_score, human_recommendation):
        with self._lock:
            self._conn.execute(
                """UPDATE interviews SET review_status='reviewed', reviewer=?,
                   reviewer_notes=?, human_score=?, human_recommendation=?,
                   updated_at=? WHERE id=?""",
                (reviewer, notes, human_score, human_recommendation, time.time(), interview_id),
            )
            self._conn.commit()

    def save_document_chunks(self, interview_id: str, chunks: list, doc_type: str | None = None):
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
        with self._lock:
            cur = self._conn.execute(
                "UPDATE interviews SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), interview_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_interview_by_session_id(self, session_id: str) -> dict | None:
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

    def delete_interview(self, interview_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM competencies WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM document_chunks WHERE interview_id=?", (interview_id,))
            self._conn.execute("DELETE FROM interviews WHERE id=?", (interview_id,))
            self._conn.commit()

    def delete_candidate(self, email: str) -> int:
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM interviews WHERE candidate_email=?", (email,)).fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

    def purge_expired(self, now=None) -> int:
        now = now or time.time()
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM interviews WHERE retention_until IS NOT NULL AND retention_until < ?",
                (now,)).fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

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

    def get_interview(self, interview_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM interviews WHERE id=?",
                                     (interview_id,)).fetchone()
            if not row:
                return None
            turns = [dict(r) for r in self._conn.execute(
                "SELECT * FROM turns WHERE interview_id=? ORDER BY idx", (interview_id,))]
            comps = [dict(r) for r in self._conn.execute(
                "SELECT * FROM competencies WHERE interview_id=? ORDER BY id", (interview_id,))]
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

    def export_candidate(self, email: str) -> list:
        with self._lock:
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

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ===========================================================================
# PostgreSQL Storage Backend
# ===========================================================================
class PostgresStoreBackend:
    """PostgreSQL storage implementation using connection pooling for serverless."""

    def __init__(self, database_url: str, min_conn: int = 1, max_conn: int = 5):
        if not HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Please install it with: pip install psycopg2-binary"
            )
        self.raw_url = database_url
        self.min_conn = min_conn
        self.max_conn = max_conn
        try:
            self.url = self._normalize_url(database_url)
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.min_conn,
                maxconn=self.max_conn,
                dsn=self.url,
            )
            self._init_schema()
        except Exception as exc:
            raw_err = str(exc)
            m = re.match(r"^postgres(?:ql)?://[^/@]+:([^@]+)@", self.raw_url)
            if m:
                raw_pw = urllib.parse.unquote(m.group(1))
                if raw_pw and len(raw_pw) >= 3:
                    raw_err = raw_err.replace(raw_pw, "[REDACTED]")
            sanitized_err = sanitize_secret_text(raw_err)
            masked_url = sanitize_secret_text(getattr(self, "url", self.raw_url))
            logger.error("Failed to connect to PostgreSQL database (%s): %s", masked_url, sanitized_err)
            raise StoreError(f"Database unavailable: {sanitized_err}") from exc

    def _normalize_url(self, url: str) -> str:
        """Normalize PostgreSQL URI for Supabase compatibility.

        - Converts 'postgres://' to 'postgresql://'
        - Safely partitions URI components without relying on urllib.parse.urlsplit
          (which raises ValueError: Invalid IPv6 URL on bracketed pooler host templates or passwords)
        - Strips template/placeholder brackets from hostname (e.g. aws-0-[region].pooler.supabase.com)
          while preserving valid RFC 3986 bracketed IPv6 literals (e.g. [::1])
        - Percent-encodes special characters in database passwords (e.g. #, %, @, [, ])
        - Enforces 'sslmode=require' for remote cloud instances (Supabase) if absent
        - Configures TCP keepalives to prevent pooler idle timeouts
        """
        url = url.strip().strip("'\"")
        if url.startswith("postgres://"):
            scheme = "postgresql://"
            rest = url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            scheme = "postgresql://"
            rest = url[len("postgresql://"):]
        else:
            return url

        # Separate query string
        base, _, query = rest.partition("?")

        # Separate path (database name)
        auth_part, slash, path = base.partition("/")

        # Separate userinfo from hostport (use rpartition to safely support passwords containing '@')
        if "@" in auth_part:
            userinfo, _, hostport = auth_part.rpartition("@")
            if ":" in userinfo:
                user, _, password = userinfo.partition(":")
            else:
                user, password = userinfo, ""
        else:
            user, password = "", ""
            hostport = auth_part

        # Clean user placeholder brackets if present e.g. postgres.[project-ref] -> postgres.project-ref
        if user:
            if "[" in user or "]" in user:
                user = user.replace("[", "").replace("]", "")
            user = urllib.parse.quote(urllib.parse.unquote(user), safe=".-_~")

        # Safely quote password
        if password:
            unquoted_pw = urllib.parse.unquote(password)
            password = urllib.parse.quote(unquoted_pw, safe="")

        # Parse and clean host and port
        if hostport.startswith("["):
            idx = hostport.rfind("]")
            if idx != -1:
                raw_host = hostport[:idx + 1]
                raw_port = hostport[idx + 1:]
                port = raw_port[1:] if raw_port.startswith(":") else ""
            else:
                raw_host = hostport
                port = ""
        elif ":" in hostport:
            raw_host, _, port = hostport.rpartition(":")
            if not port.isdigit():
                raw_host = hostport
                port = ""
        else:
            raw_host = hostport
            port = ""

        # Clean host: preserve valid IPv6 literals (e.g. [::1]), otherwise strip placeholder brackets
        clean_host = raw_host
        if raw_host.startswith("[") and raw_host.endswith("]"):
            inner = raw_host[1:-1]
            try:
                ipaddress.IPv6Address(inner)
                clean_host = raw_host  # valid IPv6 literal
            except ValueError:
                clean_host = inner.replace("[", "").replace("]", "")
        else:
            clean_host = raw_host.replace("[", "").replace("]", "")

        # Reconstruct authority
        user_prefix = ""
        if user or password:
            user_prefix = f"{user}:{password}@" if password else f"{user}@"
        port_suffix = f":{port}" if port else ""
        authority = f"{user_prefix}{clean_host}{port_suffix}"

        # Parse query parameters safely without urlsplit
        q = urllib.parse.parse_qs(query) if query else {}
        params_to_add = []

        # Enforce sslmode=require and keepalives for remote databases
        is_local = clean_host.lower() in ("localhost", "127.0.0.1", "::1", "[::1]", "")
        if not is_local:
            if "sslmode" not in q:
                params_to_add.append("sslmode=require")
            if "supabase" in clean_host.lower() and "keepalives" not in q:
                params_to_add.append("keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5")

        if params_to_add:
            delim = "&" if query else ""
            new_query = f"{query}{delim}{'&'.join(params_to_add)}"
        else:
            new_query = query

        query_suffix = f"?{new_query}" if new_query else ""
        path_part = f"/{path}" if slash else ""

        return f"{scheme}{authority}{path_part}{query_suffix}"

    @contextmanager
    def _cursor(self, commit: bool = False):
        conn = self._pool.getconn()
        try:
            # Replace broken or closed connection before cursor checkout (0 = open, 1 = closed, 2 = broken)
            if getattr(conn, "closed", 0) in (1, 2):
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self._pool.getconn()

            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                yield cur
            if commit:
                conn.commit()
            else:
                conn.rollback()  # Cleanly conclude read transaction before returning to pool
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._pool.putconn(conn)

    def _init_schema(self):
        with self._cursor(commit=True) as cur:
            cur.execute(POSTGRES_SCHEMA)
        self._migrate()

    def _migrate(self):
        """Idempotently ensure all required columns and indexes exist on pre-existing tables without data loss."""
        with self._cursor(commit=True) as cur:
            cur.execute("""
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS session_id VARCHAR(128);
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS consent_at DOUBLE PRECISION;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS consent_version VARCHAR(64);
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS retention_until DOUBLE PRECISION;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS integrity_score DOUBLE PRECISION;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS flags TEXT;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS proctor_events INTEGER DEFAULT 0;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS resume_text TEXT;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS job_description TEXT;
                ALTER TABLE interviews ADD COLUMN IF NOT EXISTS evaluation_report TEXT;
                ALTER TABLE turns ADD COLUMN IF NOT EXISTS evidence TEXT;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id);
                CREATE INDEX IF NOT EXISTS idx_email ON interviews(candidate_email);
                CREATE INDEX IF NOT EXISTS idx_turns_interview ON turns(interview_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx);
                CREATE INDEX IF NOT EXISTS idx_comp_interview ON competencies(interview_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_interview ON document_chunks(interview_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(interview_id, doc_type);
            """)

    def create_interview(self, config: dict, llm: str = "", session_id: str = "") -> str:
        iid = uuid.uuid4().hex[:12]
        now = time.time()
        consent = int(bool(config.get("consent")))
        days = int(config.get("retention_days") or 90)
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO interviews
                   (id, session_id, created_at, updated_at, status, mode, interview_type, role,
                    level, candidate_name, candidate_email, company, language,
                    consent, consent_at, consent_version, retention_until, llm,
                    review_status, proctor_events, resume_text, job_description)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (iid, session_id or None, now, now, "active", config.get("mode"), config.get("interview_type"),
                 config.get("role"), config.get("level"), config.get("candidate_name"),
                 config.get("candidate_email"), config.get("company"),
                 config.get("language"), consent, now if consent else None,
                 config.get("consent_version"), now + days * 86400, llm,
                 "unreviewed", 0, str(config.get("resume_text") or ""),
                 str(config.get("job_description") or "")),
            )
        return iid

    def add_turn(self, interview_id: str, idx: int, turn: dict):
        ev = turn.get("evidence")
        ev_str = json.dumps(ev) if ev is not None and not isinstance(ev, str) else ev
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO turns (interview_id, idx, role, text, stage, question_id, kind, evidence)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(interview_id, idx) DO UPDATE SET
                       role=EXCLUDED.role, text=EXCLUDED.text, stage=EXCLUDED.stage,
                       question_id=EXCLUDED.question_id, kind=EXCLUDED.kind, evidence=EXCLUDED.evidence""",
                (interview_id, idx, turn.get("role"), turn.get("text"), turn.get("stage"),
                 turn.get("question_id"), turn.get("kind"), ev_str),
            )
            cur.execute("UPDATE interviews SET updated_at=%s WHERE id=%s",
                        (time.time(), interview_id))

    def save_scorecard(self, interview_id: str, scorecard: dict):
        report_json = json.dumps(scorecard)
        with self._cursor(commit=True) as cur:
            cur.execute("DELETE FROM competencies WHERE interview_id=%s", (interview_id,))
            for c in scorecard.get("competencies", []):
                cmt = str(c.get("explanation") or c.get("comment") or "")
                cur.execute(
                    """INSERT INTO competencies
                       (interview_id, comp_id, name, weight, score, evidence, comment)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (interview_id, c.get("id"), c.get("name"), c.get("weight"),
                     c.get("score"), c.get("evidence"), cmt),
                )
            cur.execute(
                """UPDATE interviews SET status='completed', updated_at=%s,
                   weighted_total=%s, recommendation=%s, summary=%s, evaluation_report=%s WHERE id=%s""",
                (time.time(), scorecard.get("weighted_total"),
                 scorecard.get("recommendation"), scorecard.get("summary"),
                 report_json, interview_id),
            )

    def save_integrity(self, interview_id: str, summary: dict):
        with self._cursor(commit=True) as cur:
            cur.execute(
                """UPDATE interviews SET integrity_score=%s, flags=%s, proctor_events=%s,
                   updated_at=%s WHERE id=%s""",
                (summary.get("score"), json.dumps(summary.get("flags", [])),
                 summary.get("events", 0), time.time(), interview_id),
            )

    def save_review(self, interview_id: str, reviewer, notes, human_score, human_recommendation):
        with self._cursor(commit=True) as cur:
            cur.execute(
                """UPDATE interviews SET review_status='reviewed', reviewer=%s,
                   reviewer_notes=%s, human_score=%s, human_recommendation=%s,
                   updated_at=%s WHERE id=%s""",
                (reviewer, notes, human_score, human_recommendation, time.time(), interview_id),
            )

    def save_document_chunks(self, interview_id: str, chunks: list, doc_type: str | None = None):
        with self._cursor(commit=True) as cur:
            if doc_type:
                cur.execute(
                    "DELETE FROM document_chunks WHERE interview_id=%s AND doc_type=%s",
                    (interview_id, doc_type),
                )
            else:
                cur.execute(
                    "DELETE FROM document_chunks WHERE interview_id=%s",
                    (interview_id,),
                )
            for c in chunks:
                c_dict = c if isinstance(c, dict) else vars(c)
                emb = c_dict.get("embedding")
                emb_json = json.dumps(emb) if emb is not None else "[]"
                cur.execute(
                    """INSERT INTO document_chunks
                       (interview_id, doc_type, section, chunk_index, text, embedding, created_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (interview_id, c_dict.get("doc_type"), c_dict.get("section"),
                     int(c_dict.get("chunk_index", 0)), str(c_dict.get("text", "")),
                     emb_json, float(c_dict.get("created_at") or time.time())),
                )

    def get_document_chunks(self, interview_id: str, doc_type: str | None = None) -> list:
        with self._cursor() as cur:
            q = "SELECT * FROM document_chunks WHERE interview_id=%s"
            args = [interview_id]
            if doc_type:
                q += " AND doc_type=%s"
                args.append(doc_type)
            q += " ORDER BY chunk_index ASC"
            cur.execute(q, tuple(args))
            rows = cur.fetchall()
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
        with self._cursor(commit=True) as cur:
            if doc_type:
                cur.execute(
                    "DELETE FROM document_chunks WHERE interview_id=%s AND doc_type=%s",
                    (interview_id, doc_type),
                )
            else:
                cur.execute(
                    "DELETE FROM document_chunks WHERE interview_id=%s",
                    (interview_id,),
                )

    def update_interview_documents(
        self,
        interview_id: str,
        resume_text: str | None = None,
        job_description: str | None = None,
    ) -> bool:
        updates = []
        params = []
        if resume_text is not None:
            updates.append("resume_text=%s")
            params.append(resume_text)
        if job_description is not None:
            updates.append("job_description=%s")
            params.append(job_description)

        if not updates:
            return False

        updates.append("updated_at=%s")
        params.append(time.time())
        params.append(interview_id)

        sql = f"UPDATE interviews SET {', '.join(updates)} WHERE id=%s"
        with self._cursor(commit=True) as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount > 0

    def update_interview_status(self, interview_id: str, status: str) -> bool:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE interviews SET status=%s, updated_at=%s WHERE id=%s",
                (status, time.time(), interview_id),
            )
            return cur.rowcount > 0

    def get_interview_by_session_id(self, session_id: str) -> dict | None:
        if not session_id:
            return None
        with self._cursor() as cur:
            cur.execute(
                "SELECT id FROM interviews WHERE session_id=%s OR id=%s LIMIT 1",
                (session_id, session_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return self.get_interview(row["id"])

    def delete_interview(self, interview_id: str):
        with self._cursor(commit=True) as cur:
            cur.execute("DELETE FROM turns WHERE interview_id=%s", (interview_id,))
            cur.execute("DELETE FROM competencies WHERE interview_id=%s", (interview_id,))
            cur.execute("DELETE FROM document_chunks WHERE interview_id=%s", (interview_id,))
            cur.execute("DELETE FROM interviews WHERE id=%s", (interview_id,))

    def delete_candidate(self, email: str) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT id FROM interviews WHERE candidate_email=%s", (email,))
            ids = [r["id"] for r in cur.fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

    def purge_expired(self, now=None) -> int:
        now = now or time.time()
        with self._cursor() as cur:
            cur.execute(
                "SELECT id FROM interviews WHERE retention_until IS NOT NULL AND retention_until < %s",
                (now,),
            )
            ids = [r["id"] for r in cur.fetchall()]
        for iid in ids:
            self.delete_interview(iid)
        return len(ids)

    def list_interviews(self, status=None, sort="score") -> list:
        order = {
            "score": "COALESCE(weighted_total, -1) DESC, created_at DESC",
            "recent": "created_at DESC",
            "name": "candidate_name ASC",
            "integrity": "COALESCE(integrity_score, -1) ASC, created_at DESC",
        }.get(sort, "created_at DESC")
        q = "SELECT * FROM interviews"
        args = []
        if status:
            q += " WHERE status=%s"
            args.append(status)
        q += f" ORDER BY {order}"
        with self._cursor() as cur:
            cur.execute(q, tuple(args))
            return [dict(r) for r in cur.fetchall()]

    def get_interview(self, interview_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM interviews WHERE id=%s", (interview_id,))
            row = cur.fetchone()
            if not row:
                return None
            cur.execute("SELECT * FROM turns WHERE interview_id=%s ORDER BY idx", (interview_id,))
            turns = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT * FROM competencies WHERE interview_id=%s ORDER BY id", (interview_id,))
            comps = [dict(r) for r in cur.fetchall()]
            cur.execute(
                "SELECT doc_type, COUNT(*) as c FROM document_chunks WHERE interview_id=%s GROUP BY doc_type",
                (interview_id,),
            )
            chunk_stats_rows = cur.fetchall()

        out = dict(row)
        for t in turns:
            if t.get("evidence"):
                try:
                    t["evidence"] = json.loads(t["evidence"])
                except Exception:
                    pass
        out["turns"] = turns
        out["competencies"] = comps
        chunk_counts = {r["doc_type"]: int(r["c"]) for r in chunk_stats_rows}
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

    def export_candidate(self, email: str) -> list:
        with self._cursor() as cur:
            cur.execute("SELECT id FROM interviews WHERE candidate_email=%s", (email,))
            rows = cur.fetchall()
        return [self.get_interview(r["id"]) for r in rows]

    def stats(self) -> dict:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM interviews")
            total = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) as c FROM interviews WHERE status='completed'")
            done = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) as c FROM interviews WHERE review_status='reviewed'")
            reviewed = int(cur.fetchone()["c"])
            cur.execute(
                "SELECT COUNT(*) as c FROM interviews WHERE integrity_score IS NOT NULL "
                "AND integrity_score < 90"
            )
            flagged = int(cur.fetchone()["c"])
            cur.execute("SELECT AVG(weighted_total) as a FROM interviews WHERE status='completed'")
            avg_row = cur.fetchone()["a"]
            avg = float(avg_row) if avg_row is not None else None
        return {
            "total": total,
            "completed": done,
            "reviewed": reviewed,
            "flagged": flagged,
            "avg_score": round(avg, 1) if avg is not None else None,
        }

    def close(self):
        try:
            if self._pool and not self._pool.closed:
                self._pool.closeall()
        except Exception:
            pass


# ===========================================================================
# Unified Store Interface / Adapter
# ===========================================================================
class Store:
    """Unified persistence store supporting SQLite and PostgreSQL backends."""

    def __init__(
        self,
        path: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        self.path = path
        self.database_url = database_url

        db_env = os.environ.get("DATABASE_URL", "").strip()
        has_pg_env = bool(db_env and (db_env.startswith("postgres://") or db_env.startswith("postgresql://")))

        # Consistent Selection Rule:
        # 1. Explicit database_url argument takes highest precedence
        if database_url:
            self._backend = PostgresStoreBackend(database_url)
            self._backend_type = "postgres"
            self.path = self.path or "postgresql"
        # 2. Path provided as a PostgreSQL connection URI
        elif path and (path.startswith("postgres://") or path.startswith("postgresql://")):
            self._backend = PostgresStoreBackend(path)
            self._backend_type = "postgres"
            self.path = "postgresql"
        # 3. Explicit SQLite test/scratch file (e.g. ends with .db/.sqlite or is :memory:)
        #    Preserves test isolation even if DATABASE_URL happens to be in ambient environment
        elif path and (path.endswith(".db") or path.endswith(".sqlite") or path == ":memory:") and path not in ("interviews.db", "/tmp/interviews.db"):
            self._backend = SQLiteStoreBackend(path)
            self._backend_type = "sqlite"
        # 4. DATABASE_URL configured in environment (production / serverless)
        elif has_pg_env:
            self._backend = PostgresStoreBackend(db_env)
            self._backend_type = "postgres"
            self.path = self.path or "postgresql"
        # 5. Default: SQLite backend (path or INTERVIEW_DB or interviews.db)
        else:
            sqlite_path = path or os.environ.get("INTERVIEW_DB", "interviews.db")
            self._backend = SQLiteStoreBackend(sqlite_path)
            self._backend_type = "sqlite"
            self.path = sqlite_path

    @property
    def backend_type(self) -> str:
        return self._backend_type

    @property
    def _conn(self):
        """Expose raw sqlite connection when using SQLite backend for backward compatibility."""
        if hasattr(self._backend, "_conn"):
            return self._backend._conn
        raise AttributeError("PostgreSQL backend does not expose a raw _conn property.")

    # Public method delegates:
    def create_interview(self, config: dict, llm: str = "", session_id: str = "") -> str:
        return self._backend.create_interview(config, llm=llm, session_id=session_id)

    def add_turn(self, interview_id, idx, turn: dict):
        return self._backend.add_turn(interview_id, idx, turn)

    def save_scorecard(self, interview_id, scorecard: dict):
        return self._backend.save_scorecard(interview_id, scorecard)

    def save_integrity(self, interview_id, summary: dict):
        return self._backend.save_integrity(interview_id, summary)

    def save_review(self, interview_id, reviewer, notes, human_score, human_recommendation):
        return self._backend.save_review(interview_id, reviewer, notes, human_score, human_recommendation)

    def save_document_chunks(self, interview_id: str, chunks: list, doc_type: str | None = None):
        return self._backend.save_document_chunks(interview_id, chunks, doc_type=doc_type)

    def get_document_chunks(self, interview_id: str, doc_type: str | None = None) -> list:
        return self._backend.get_document_chunks(interview_id, doc_type=doc_type)

    def delete_document_chunks(self, interview_id: str, doc_type: str | None = None):
        return self._backend.delete_document_chunks(interview_id, doc_type=doc_type)

    def update_interview_documents(
        self,
        interview_id: str,
        resume_text: str | None = None,
        job_description: str | None = None,
    ) -> bool:
        return self._backend.update_interview_documents(interview_id, resume_text, job_description)

    def update_interview_status(self, interview_id: str, status: str) -> bool:
        return self._backend.update_interview_status(interview_id, status)

    def get_interview_by_session_id(self, session_id: str) -> dict | None:
        return self._backend.get_interview_by_session_id(session_id)

    def delete_interview(self, interview_id):
        return self._backend.delete_interview(interview_id)

    def delete_candidate(self, email):
        return self._backend.delete_candidate(email)

    def purge_expired(self, now=None) -> int:
        return self._backend.purge_expired(now=now)

    def list_interviews(self, status=None, sort="score") -> list:
        return self._backend.list_interviews(status=status, sort=sort)

    def get_interview(self, interview_id):
        return self._backend.get_interview(interview_id)

    def export_candidate(self, email) -> list:
        return self._backend.export_candidate(email)

    def stats(self) -> dict:
        return self._backend.stats()

    def close(self):
        return self._backend.close()
