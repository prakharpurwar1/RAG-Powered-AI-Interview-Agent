"""Automated Tests for Phase 3 Step 1: PostgreSQL Schema & Database Adapter.

Distinguishes between:
1. SIMULATED / MOCKED UNIT TESTS: Validates selection logic, DDL definitions,
   psycopg2 query parameterization (%s), transaction rollback, connection pool
   checkout/checkin, and data mapping without requiring a running PostgreSQL server.
2. LIVE INTEGRATION TESTS: Tested only when a real PostgreSQL database URI is
   provided via TEST_DATABASE_URL. Otherwise clearly marked as skipped.

Run: python -m pytest tests/test_postgres_adapter.py -v
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from interviewer.store import HAS_PSYCOPG2, POSTGRES_SCHEMA, PostgresStoreBackend, SQLiteStoreBackend, Store, StoreError

try:
    import psycopg2
except ImportError:
    psycopg2 = None


# ===========================================================================
# 1. Selection & Configuration Tests
# ===========================================================================

def test_store_defaults_to_sqlite_when_no_database_url(monkeypatch, tmp_path):
    """Verify Store uses SQLite backend when DATABASE_URL is not set."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    test_db = str(tmp_path / "test.db")
    st = Store(path=test_db)
    assert st.backend_type == "sqlite"
    assert isinstance(st._backend, SQLiteStoreBackend)
    assert hasattr(st, "_conn")
    st.close()


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2-binary not installed in this environment")
def test_store_selects_postgres_when_database_url_passed(monkeypatch):
    """Verify Store selects PostgreSQL backend when database_url is passed."""
    fake_url = "postgresql://user:secret@ep-test-123.us-east-2.aws.neon.tech/neondb"
    with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_pool_cls.return_value = mock_pool

        st = Store(database_url=fake_url)
        assert st.backend_type == "postgres"
        assert isinstance(st._backend, PostgresStoreBackend)
        st.close()


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2-binary not installed in this environment")
def test_store_selects_postgres_from_env_variable(monkeypatch):
    """Verify Store selects PostgreSQL when DATABASE_URL is present in environment."""
    fake_url = "postgresql://user:pass@localhost:5432/interviews"
    monkeypatch.setenv("DATABASE_URL", fake_url)

    with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_pool_cls.return_value = mock_pool

        st = Store()
        assert st.backend_type == "postgres"
        assert st._backend.url == fake_url
        st.close()


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2-binary not installed in this environment")
def test_store_normalizes_postgres_url_scheme():
    """Verify postgres:// is normalized to postgresql:// for compatibility."""
    raw_url = "postgres://user:pass@localhost:5432/interviews"
    expected = "postgresql://user:pass@localhost:5432/interviews"

    with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_pool_cls.return_value = mock_pool

        backend = PostgresStoreBackend(raw_url)
        assert backend.url == expected
        backend.close()


def test_explicit_sqlite_path_overrides_database_url_env(monkeypatch, tmp_path):
    """Verify passing an explicit file path preserves SQLite even if DATABASE_URL is in env."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    test_db = str(tmp_path / "explicit.db")
    st = Store(path=test_db)
    assert st.backend_type == "sqlite"
    st.close()


# ===========================================================================
# 2. Schema DDL Structure Tests
# ===========================================================================

def test_postgres_schema_ddl_structure():
    """Verify POSTGRES_SCHEMA contains required tables, types, and indexes."""
    schema = POSTGRES_SCHEMA

    # Tables exist
    assert "CREATE TABLE IF NOT EXISTS interviews" in schema
    assert "CREATE TABLE IF NOT EXISTS turns" in schema
    assert "CREATE TABLE IF NOT EXISTS competencies" in schema
    assert "CREATE TABLE IF NOT EXISTS document_chunks" in schema

    # Proper PostgreSQL types
    assert "DOUBLE PRECISION" in schema
    assert "SERIAL PRIMARY KEY" in schema
    assert "VARCHAR" in schema
    assert "TEXT" in schema

    # Unique index for upsert conflict handling
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique_idx ON turns(interview_id, idx)" in schema

    # Query and retrieval indexes
    assert "CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id)" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_email ON interviews(candidate_email)" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_turns_interview ON turns(interview_id)" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_chunks_interview ON document_chunks(interview_id)" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON document_chunks(interview_id, doc_type)" in schema


# ===========================================================================
# 3. Simulated / Mocked PostgreSQL Query & Transaction Tests
# ===========================================================================

@pytest.fixture
def mock_postgres_backend():
    """Fixture providing a PostgresStoreBackend with mocked connection pool."""
    if not HAS_PSYCOPG2:
        pytest.skip("psycopg2-binary not installed in this environment")
    fake_url = "postgresql://user:pass@localhost:5432/testdb"
    with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()

        # Connect mock cursor to connection
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_pool_cls.return_value = mock_pool

        backend = PostgresStoreBackend(fake_url)
        backend._mock_pool = mock_pool
        backend._mock_conn = mock_conn
        backend._mock_cur = mock_cur
        yield backend
        backend.close()


def test_simulated_postgres_cursor_lifecycle_and_rollback(mock_postgres_backend):
    """Verify connection pool checkout, rollback on error, and guaranteed checkin."""
    backend = mock_postgres_backend
    mock_pool = backend._mock_pool
    mock_conn = backend._mock_conn

    # 1. Normal execution with commit=True (writes)
    with backend._cursor(commit=True) as cur:
        cur.execute("SELECT 1")

    mock_pool.getconn.assert_called()
    mock_conn.commit.assert_called()
    mock_pool.putconn.assert_called_with(mock_conn)

    # 2. Read execution with commit=False: verifies transaction rollback for clean return
    mock_conn.commit.reset_mock()
    mock_conn.rollback.reset_mock()
    mock_pool.putconn.reset_mock()

    with backend._cursor(commit=False) as cur:
        cur.execute("SELECT 1")

    mock_conn.commit.assert_not_called()
    mock_conn.rollback.assert_called_once()  # Concludes read transaction cleanly
    mock_pool.putconn.assert_called_once_with(mock_conn)

    # 3. Error handling: verifies rollback and checkin even on exception
    mock_conn.commit.reset_mock()
    mock_conn.rollback.reset_mock()
    mock_pool.putconn.reset_mock()

    with pytest.raises(ValueError, match="Simulated DB Error"):
        with backend._cursor(commit=True) as cur:
            raise ValueError("Simulated DB Error")

    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_simulated_postgres_create_interview_parameterization(mock_postgres_backend):
    """Verify create_interview executes parameterized INSERT with %s placeholders."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    cfg = {
        "mode": "hiring",
        "interview_type": "technical",
        "role": "Lead Architect",
        "level": "staff",
        "candidate_name": "Aarav",
        "candidate_email": "aarav@example.com",
        "company": "TechCorp",
        "language": "en-IN",
        "consent": True,
        "consent_version": "1.0",
        "retention_days": 180,
        "resume_text": "Experienced Python and Cloud Architect.",
        "job_description": "Cloud migration architect.",
    }

    iid = backend.create_interview(cfg, llm="MockLLM", session_id="sess_pg_001")
    assert isinstance(iid, str)
    assert len(iid) == 12

    # Verify query and parameters
    cur.execute.assert_called()
    last_call = cur.execute.call_args_list[-1]
    query, params = last_call[0]

    assert "INSERT INTO interviews" in query
    assert "%s" in query
    assert "?" not in query  # Must use PostgreSQL parameter syntax
    assert len(params) == 22
    assert params[0] == iid
    assert params[1] == "sess_pg_001"
    assert params[7] == "Lead Architect"
    assert params[9] == "Aarav"
    assert params[13] == 1  # Consent integer


def test_simulated_postgres_add_turn_upsert(mock_postgres_backend):
    """Verify add_turn executes ON CONFLICT (interview_id, idx) upsert query."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    turn = {
        "role": "interviewer",
        "text": "What is connection pooling?",
        "stage": "core",
        "question_id": "q_pool_1",
        "kind": "message",
        "evidence": {"doc": "resume", "match": "pooling"},
    }

    backend.add_turn("test_iid", 0, turn)
    assert cur.execute.call_count >= 2

    insert_call = cur.execute.call_args_list[-2]
    query, params = insert_call[0]

    assert "INSERT INTO turns" in query
    assert "ON CONFLICT(interview_id, idx) DO UPDATE" in query
    assert "EXCLUDED.role" in query
    assert "%s" in query
    assert params[0] == "test_iid"
    assert params[1] == 0
    assert params[2] == "interviewer"
    assert json.loads(params[7]) == {"doc": "resume", "match": "pooling"}


def test_simulated_postgres_save_scorecard_transaction(mock_postgres_backend):
    """Verify save_scorecard updates competencies and interview status transactionally."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    card = {
        "competencies": [
            {"id": "c1", "name": "System Design", "weight": 0.4, "score": 4.5, "evidence": "good", "comment": "strong"}
        ],
        "weighted_total": 85.0,
        "recommendation": "Strong Hire",
        "summary": "Excellent technical competence.",
    }

    backend.save_scorecard("test_iid", card)
    calls = cur.execute.call_args_list

    queries = [c[0][0] for c in calls]
    assert any("DELETE FROM competencies WHERE interview_id=%s" in q for q in queries)
    assert any("INSERT INTO competencies" in q for q in queries)
    assert any("UPDATE interviews SET status='completed'" in q for q in queries)


def test_simulated_postgres_document_chunks_save_and_retrieve(mock_postgres_backend):
    """Verify document chunks embedding JSON serialization and retrieval."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    chunks = [
        {
            "doc_type": "resume",
            "section": "experience",
            "chunk_index": 0,
            "text": "Designed high throughput stream processor.",
            "embedding": [0.12, -0.34, 0.56],
            "created_at": 1700000000.0,
        }
    ]

    backend.save_document_chunks("test_iid", chunks, doc_type="resume")
    calls = cur.execute.call_args_list

    insert_call = calls[-1]
    query, params = insert_call[0]
    assert "INSERT INTO document_chunks" in query
    assert params[0] == "test_iid"
    assert params[1] == "resume"
    assert json.loads(params[5]) == [0.12, -0.34, 0.56]

    # Test retrieval parsing
    cur.fetchall.return_value = [
        {
            "id": 1,
            "interview_id": "test_iid",
            "doc_type": "resume",
            "section": "experience",
            "chunk_index": 0,
            "text": "Designed high throughput stream processor.",
            "embedding": json.dumps([0.12, -0.34, 0.56]),
            "created_at": 1700000000.0,
        }
    ]

    retrieved = backend.get_document_chunks("test_iid", doc_type="resume")
    assert len(retrieved) == 1
    assert retrieved[0]["embedding"] == [0.12, -0.34, 0.56]


def test_simulated_postgres_stats(mock_postgres_backend):
    """Verify stats calculation with proper type coercion."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    cur.fetchone.side_effect = [
        {"c": 10},     # total
        {"c": 7},      # completed
        {"c": 5},      # reviewed
        {"c": 1},      # flagged
        {"a": 82.46},  # avg_score
    ]

    res = backend.stats()
    assert res == {
        "total": 10,
        "completed": 7,
        "reviewed": 5,
        "flagged": 1,
        "avg_score": 82.5,
    }


# ===========================================================================
# 4. Genuine Integration Tests (Live Database Required)
# ===========================================================================

@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="Real PostgreSQL instance not configured (set TEST_DATABASE_URL to run live integration tests)",
)
def test_genuine_postgres_live_lifecycle():
    """Live integration test executed ONLY against an actual PostgreSQL instance.

    To run:
      export TEST_DATABASE_URL="postgresql://user:password@host:port/dbname"
      pytest tests/test_postgres_adapter.py -k test_genuine_postgres_live_lifecycle -v
    """
    db_url = os.environ["TEST_DATABASE_URL"]
    st = Store(database_url=db_url)
    assert st.backend_type == "postgres"

    try:
        # 1. Create interview
        iid = st.create_interview(
            {
                "mode": "hiring",
                "role": "Distributed Systems Engineer",
                "candidate_name": "Integration Test Candidate",
                "candidate_email": "live_test@example.com",
                "consent": True,
            },
            llm="MockLLM",
            session_id="sess_live_pg_001",
        )
        assert iid

        # 2. Add turns
        st.add_turn(iid, 0, {"role": "interviewer", "text": "Live Question 1", "stage": "core"})
        st.add_turn(iid, 1, {"role": "candidate", "text": "Live Answer 1", "stage": "core"})

        # 3. Save scorecard
        st.save_scorecard(
            iid,
            {
                "competencies": [
                    {"id": "comp_1", "name": "Concurrency", "weight": 0.5, "score": 4.0, "comment": "Good"}
                ],
                "weighted_total": 80.0,
                "recommendation": "Hire",
                "summary": "Passed live test.",
            },
        )

        # 4. Save & retrieve RAG document chunks
        st.save_document_chunks(
            iid,
            [
                {
                    "doc_type": "resume",
                    "section": "skills",
                    "chunk_index": 0,
                    "text": "PostgreSQL, Neon, Serverless",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
        )
        chunks = st.get_document_chunks(iid, doc_type="resume")
        assert len(chunks) == 1
        assert chunks[0]["embedding"] == [0.1, 0.2, 0.3]

        # 5. Fetch full interview
        rec = st.get_interview(iid)
        assert rec is not None
        assert rec["candidate_name"] == "Integration Test Candidate"
        assert rec["status"] == "completed"
        assert len(rec["turns"]) == 2
        assert rec["weighted_total"] == 80.0

        # 6. Delete interview and verify cleanup
        st.delete_interview(iid)
        assert st.get_interview(iid) is None
        assert len(st.get_document_chunks(iid)) == 0

    finally:
        st.close()


# ===========================================================================
# 5. Targeted Fix Validation Tests
# ===========================================================================

def test_proctoring_signal_persistence_and_rehydration(tmp_path, monkeypatch):
    """Verify that proctoring telemetry is persisted immediately on signal arrival,
    surviving an in-memory cache wipe and correctly rehydrating into IntegrityMonitor.
    """
    from starlette.testclient import TestClient
    from interviewer.app import app, _SESSIONS, get_or_rehydrate_session
    import interviewer.app as app_module

    db_path = str(tmp_path / "proctor_test.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("INTERVIEW_DB", db_path)
    app_module._STORE = None
    app_module._STORE_PATH = None

    client = TestClient(app)

    # 1. Start interview session with proctoring enabled
    start_resp = client.post(
        "/api/start",
        json={
            "role": "Backend Engineer",
            "level": "mid",
            "candidate_name": "Aarav",
            "candidate_email": "aarav@example.com",
            "consent": True,
            "proctoring": True,
            "mock": True,
        },
    )
    assert start_resp.status_code == 200
    sid = start_resp.json()["session_id"]
    iid = start_resp.json()["interview_id"]

    # 2. Candidate triggers blur signal (tab switch)
    sig_resp = client.post("/api/signal", json={"session_id": sid, "kind": "blur"})
    assert sig_resp.status_code == 200
    assert sig_resp.json()["ok"] is True
    assert sig_resp.json()["score"] == 95.0

    # 3. Verify immediately persisted in store without waiting for interview completion
    st = app_module.get_store()
    row = st.get_interview(iid)
    assert row["integrity_score"] == 95.0
    assert any(f.get("kind") == "blur" for f in row.get("flags", []))

    # 4. Candidate triggers paste signal
    sig_resp2 = client.post("/api/signal", json={"session_id": sid, "kind": "paste"})
    assert sig_resp2.status_code == 200
    assert sig_resp2.json()["score"] == 80.0  # 100 - 5(blur) - 15(paste)

    # 5. Simulate server restart / instance change by completely wiping in-memory session cache
    _SESSIONS.clear()
    assert sid not in _SESSIONS

    # 6. Rehydrate session from database
    sess = get_or_rehydrate_session(sid)
    assert sess is not None
    assert sess["iid"] == iid
    # Integrity monitor must have recovered both signals
    monitor = sess["monitor"]
    assert monitor.score() == 80.0
    flags = monitor.flags()
    kinds = {f["kind"] for f in flags}
    assert "blur" in kinds
    assert "paste" in kinds


def test_consistent_database_selection_app_and_server(monkeypatch, tmp_path):
    """Verify that interviewer.app and interviewer.server maintain consistent database selection."""
    import interviewer.app as app_module
    import interviewer.server as server_module

    fake_pg_url = "postgresql://user:pass@ep-pooler.neon.tech/neondb?sslmode=require"

    # Scenario A: DATABASE_URL is set -> both use PostgreSQL
    monkeypatch.setenv("DATABASE_URL", fake_pg_url)
    app_module._STORE = None
    app_module._STORE_PATH = None
    server_module._STORE = None
    server_module._STORE_PATH = None

    with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_pool_cls.return_value = mock_pool

        app_store = app_module.get_store()
        server_store = server_module._store()

        assert app_store.backend_type == "postgres"
        assert server_store.backend_type == "postgres"
        assert app_store._backend.url == fake_pg_url
        assert server_store._backend.url == fake_pg_url

    # Scenario B: DATABASE_URL is unset -> both use SQLite
    monkeypatch.delenv("DATABASE_URL", raising=False)
    test_db = str(tmp_path / "consistent_test.db")
    monkeypatch.setenv("INTERVIEW_DB", test_db)
    app_module._STORE = None
    app_module._STORE_PATH = None
    server_module._STORE = None
    server_module._STORE_PATH = None

    app_store_sqlite = app_module.get_store()
    server_store_sqlite = server_module._store()

    assert app_store_sqlite.backend_type == "sqlite"
    assert server_store_sqlite.backend_type == "sqlite"
    assert app_store_sqlite._backend.path == test_db
    assert server_store_sqlite._backend.path == test_db


# ===========================================================================
# 6. Supabase Connection, Normalization & Migration Tests
# ===========================================================================

def test_supabase_url_normalization_with_ssl_and_keepalives():
    """Verify Supabase Session Pooler URLs are normalized with sslmode=require,
    keepalives, and special character password encoding.
    """
    raw_supabase = "postgres://postgres.myref:p#ss%word@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
    backend = PostgresStoreBackend.__new__(PostgresStoreBackend)
    normalized = backend._normalize_url(raw_supabase)

    assert normalized.startswith("postgresql://")
    assert "sslmode=require" in normalized
    assert "keepalives=1" in normalized
    assert "p%23ss%25word" in normalized  # # and % are percent-encoded
    assert "aws-0-ap-south-1.pooler.supabase.com" in normalized


def test_supabase_url_preserves_explicit_sslmode():
    """Verify explicit sslmode is preserved and not duplicated."""
    raw = "postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=verify-full"
    backend = PostgresStoreBackend.__new__(PostgresStoreBackend)
    normalized = backend._normalize_url(raw)

    assert "sslmode=verify-full" in normalized
    assert normalized.count("sslmode") == 1


def test_postgres_schema_and_migration_idempotent(mock_postgres_backend):
    """Verify _init_schema and _migrate execute schema DDL and idempotent column additions."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    backend._init_schema()
    backend._migrate()

    calls = [c[0][0] for c in cur.execute.call_args_list]
    # DDL schema executed
    assert any("CREATE TABLE IF NOT EXISTS interviews" in sql for sql in calls)
    # Migration ADD COLUMN IF NOT EXISTS executed
    assert any("ALTER TABLE interviews ADD COLUMN IF NOT EXISTS evaluation_report" in sql for sql in calls)
    assert any("ALTER TABLE turns ADD COLUMN IF NOT EXISTS evidence" in sql for sql in calls)
    assert any("CREATE INDEX IF NOT EXISTS idx_interviews_session" in sql for sql in calls)


# ===========================================================================
# 7. Session, Turn, Chunks & Scorecard Persistence Across Restarts
# ===========================================================================

def test_postgres_session_and_turn_persistence_across_restarts(mock_postgres_backend):
    """Verify that candidate state, turns, and question counts persist and rehydrate
    from PostgreSQL backend across simulated application restarts.
    """
    from interviewer.engine import InterviewEngine
    from interviewer.llm import MockLLM

    backend = mock_postgres_backend
    cur = backend._mock_cur

    # Mock DB return for get_interview
    mock_interview_row = {
        "id": "int_pg_999",
        "session_id": "sess_pg_999",
        "created_at": 1700000000.0,
        "updated_at": 1700000100.0,
        "status": "active",
        "mode": "hiring",
        "interview_type": "technical",
        "role": "Cloud Architect",
        "level": "senior",
        "candidate_name": "Priya",
        "candidate_email": "priya@example.com",
        "company": "Enterprise Cloud",
        "language": "en-IN",
        "consent": 1,
        "consent_at": 1700000000.0,
        "consent_version": "1.0",
        "retention_until": 1707776000.0,
        "llm": "MockLLM",
        "weighted_total": None,
        "recommendation": None,
        "summary": None,
        "integrity_score": 100.0,
        "flags": "[]",
        "proctor_events": 0,
        "review_status": "unreviewed",
        "reviewer": None,
        "reviewer_notes": None,
        "human_score": None,
        "human_recommendation": None,
        "resume_text": "AWS, GCP, Supabase PostgreSQL, Kubernetes expert.",
        "job_description": "Architect cloud database solutions.",
        "evaluation_report": None,
    }
    mock_turns = [
        {"id": 1, "interview_id": "int_pg_999", "idx": 0, "role": "interviewer", "text": "Welcome Priya. Tell me about your cloud experience.", "stage": "core", "question_id": "q1", "kind": "message", "evidence": None},
        {"id": 2, "interview_id": "int_pg_999", "idx": 1, "role": "candidate", "text": "I have designed distributed architectures using Supabase PostgreSQL and microservices.", "stage": "core", "question_id": "q1", "kind": "answer", "evidence": None},
    ]

    # Setup fetchone for interview and fetchall for turns/comps/chunks
    cur.fetchone.return_value = mock_interview_row
    cur.fetchall.side_effect = [
        mock_turns,  # turns
        [],          # competencies
        [],          # document_chunks stats
    ]

    # Rehydrate
    engine = InterviewEngine.rehydrate_from_store(backend, "int_pg_999", session_id="sess_pg_999", llm=MockLLM())
    assert engine is not None
    assert engine.config.candidate_name == "Priya"
    assert engine.config.role == "Cloud Architect"
    assert len(engine.turns) == 2
    assert engine.turns[0].text == "Welcome Priya. Tell me about your cloud experience."
    assert engine.turns[1].text == "I have designed distributed architectures using Supabase PostgreSQL and microservices."
    assert engine.questions_asked == 1


def test_postgres_scorecard_persistence_and_idempotency(mock_postgres_backend):
    """Verify scorecard persistence updates status, evaluation_report, and competencies idempotently."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    card = {
        "competencies": [
            {"id": "comp_db", "name": "Database Design", "weight": 0.6, "score": 4.8, "evidence": "strong", "comment": "Excellent Supabase knowledge"}
        ],
        "weighted_total": 92.0,
        "recommendation": "Strong Hire",
        "summary": "Outstanding candidate.",
    }

    # First call
    backend.save_scorecard("int_pg_999", card)
    calls_1 = len(cur.execute.call_args_list)

    # Second call (idempotent evaluation)
    backend.save_scorecard("int_pg_999", card)
    calls_2 = len(cur.execute.call_args_list)

    assert calls_2 > calls_1
    executed_sql = [c[0][0] for c in cur.execute.call_args_list]
    assert any("DELETE FROM competencies WHERE interview_id=%s" in sql for sql in executed_sql)
    assert any("UPDATE interviews SET status='completed'" in sql for sql in executed_sql)


def test_postgres_privacy_deletion_and_isolation(mock_postgres_backend):
    """Verify privacy deletion cascades across turns, competencies, and chunks for only the target interview."""
    backend = mock_postgres_backend
    cur = backend._mock_cur

    # Delete interview
    backend.delete_interview("int_isolated_001")
    calls = [c[0] for c in cur.execute.call_args_list]

    # Check that all 4 tables had targeted DELETE queries
    delete_queries = [call[0] for call in calls if "DELETE FROM" in call[0]]
    assert len(delete_queries) == 4
    for call in calls:
        if "DELETE FROM" in call[0]:
            assert call[1] == ("int_isolated_001",)  # Only target interview deleted


# ===========================================================================
# 8. Database Unavailable Resilience Tests
# ===========================================================================

def test_database_unavailable_sanitizes_credentials_and_raises_store_error():
    """Verify that if database connection fails, credentials are not exposed in logs or exceptions."""
    secret_pw = "SuperSecretPass123"
    fake_url = f"postgresql://postgres:{secret_pw}@db.fake-project.supabase.co:5432/postgres"

    with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=psycopg2.OperationalError(f"connection to server at 'db.fake-project.supabase.co', password '{secret_pw}' failed")):
        with pytest.raises(StoreError) as exc_info:
            PostgresStoreBackend(fake_url)

        err_msg = str(exc_info.value)
        assert secret_pw not in err_msg
        assert "Database unavailable" in err_msg


def test_api_health_when_database_unavailable(monkeypatch):
    """Verify /api/health gracefully reports db_status: unavailable without throwing HTTP 500."""
    from starlette.testclient import TestClient
    from interviewer.app import app
    import interviewer.app as app_module

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:secret@db.fake.supabase.co:5432/postgres")
    app_module._STORE = None
    app_module._STORE_PATH = None

    with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=psycopg2.OperationalError("host down")):
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db_status"] == "unavailable"


def test_api_start_when_database_unavailable(monkeypatch):
    """Verify /api/start returns HTTP 503 when the database is unavailable."""
    from starlette.testclient import TestClient
    from interviewer.app import app
    import interviewer.app as app_module

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:secret@db.fake.supabase.co:5432/postgres")
    app_module._STORE = None
    app_module._STORE_PATH = None

    with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=psycopg2.OperationalError("connection refused")):
        client = TestClient(app)
        resp = client.post("/api/start", json={
            "role": "Backend Engineer",
            "level": "senior",
            "candidate_name": "Test",
            "mock": True,
        })
        assert resp.status_code == 503
        data = resp.json()
        assert "Database service unavailable" in data["detail"]


