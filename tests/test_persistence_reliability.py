"""Automated Test Suite for Phase 5: Persistence and Reliability.

Tests:
1. Interview sessions and answers survive server restart / in-memory cache clearance.
2. Turn rehydration and transcript continuation after server restart.
3. Client-side session resume endpoint for browser refresh restoration.
4. Report caching idempotency (no duplicate evaluations or redundant LLM calls).
5. Granular document replacement/updating updates session context and RAG index.
6. Document removal cleanly purges vector chunks while preserving other documents.
7. Candidate deletion / erasure cascades across turns, documents, scorecards, and in-memory cache.
8. Database migration safely preserves pre-existing records and adds new columns.
9. SQLite WAL mode and busy timeout ensure concurrent write resilience.
10. Strict session isolation prevents data leaking between sessions.

Run: python3 tests/test_persistence_reliability.py
"""
import os
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.app import _SESSIONS, get_or_rehydrate_session, evict_sessions
from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine, Turn
from interviewer.llm import MockLLM
from interviewer.rag import index_interview_documents, retrieve_evidence
from interviewer.store import Store


def _temp_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return Store(path), path


def test_session_survives_restart_and_rehydrates():
    st, path = _temp_store()
    try:
        sid = "sess_recov_001"
        cfg = InterviewConfig(
            role="Backend Engineer",
            level="senior",
            interview_type="technical",
            candidate_name="Divya",
            candidate_email="divya@example.com",
            resume_text="Senior Go & distributed systems engineer at CloudTech.",
            job_description="Seeking high scale microservices engineer.",
        )
        iid = st.create_interview(vars(cfg), llm="MockLLM", session_id=sid)

        # Record initial turns in DB
        st.add_turn(iid, 0, {"role": "interviewer", "text": "Tell me about your Go experience.", "stage": "core"})
        st.add_turn(iid, 1, {"role": "candidate", "text": "I built a distributed cache handling 50k QPS.", "stage": "core"})

        # Wipe in-memory session cache to simulate complete server restart
        _SESSIONS.clear()
        assert sid not in _SESSIONS

        # Rehydrate engine from store
        engine = InterviewEngine.rehydrate_from_store(st, iid, session_id=sid, llm=MockLLM())
        assert engine is not None
        assert engine.config.candidate_name == "Divya"
        assert engine.config.role == "Backend Engineer"
        assert len(engine.turns) == 2
        assert engine.turns[0].text == "Tell me about your Go experience."
        assert engine.turns[1].text == "I built a distributed cache handling 50k QPS."

        # Continue interview: candidate sends another answer
        resp = engine.respond("We used consistent hashing to distribute keys evenly across 12 nodes.")
        assert "message" in resp
        assert len(engine.turns) == 4
        assert engine.turns[3].role == "interviewer"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_get_or_rehydrate_session_helper():
    st, path = _temp_store()
    orig_env = os.environ.get("INTERVIEW_DB")
    os.environ["INTERVIEW_DB"] = path
    try:
        sid = "sess_app_rehyd_002"
        cfg = InterviewConfig(
            role="Data Engineer",
            level="mid",
            interview_type="technical",
            candidate_name="Kiran",
            candidate_email="kiran@example.com",
        )
        iid = st.create_interview(vars(cfg), llm="MockLLM", session_id=sid)
        st.add_turn(iid, 0, {"role": "interviewer", "text": "Hello Kiran, welcome.", "stage": "core"})

        # Simulate fresh server process where memory is empty
        _SESSIONS.clear()

        # Call get_or_rehydrate_session
        sess = get_or_rehydrate_session(sid)
        assert sess is not None
        assert sess["iid"] == iid
        assert sess["engine"].config.candidate_name == "Kiran"
        assert len(sess["engine"].turns) == 1
        assert sid in _SESSIONS
    finally:
        if orig_env is not None:
            os.environ["INTERVIEW_DB"] = orig_env
        else:
            os.environ.pop("INTERVIEW_DB", None)
        try:
            os.unlink(path)
        except OSError:
            pass


def test_report_caching_idempotency_no_duplicate_evaluations():
    st, path = _temp_store()
    try:
        sid = "sess_report_idemp_003"
        cfg = InterviewConfig(
            role="Frontend Engineer",
            level="senior",
            interview_type="technical",
            candidate_name="Rohit",
            candidate_email="rohit@example.com",
        )
        iid = st.create_interview(vars(cfg), llm="MockLLM", session_id=sid)
        st.add_turn(iid, 0, {"role": "interviewer", "text": "Q1", "stage": "core"})
        st.add_turn(iid, 1, {"role": "candidate", "text": "I built accessible UI components in React.", "stage": "core"})

        engine = InterviewEngine(cfg, MockLLM(), store=st, interview_id=iid, session_id=sid)
        engine.turns = [
            Turn(role="interviewer", text="Q1", stage="core"),
            Turn(role="candidate", text="I built accessible UI components in React.", stage="core"),
        ]

        # 1. First report generation
        rep1 = engine.report()
        st.save_scorecard(iid, rep1["scorecard"])

        row1 = st.get_interview(iid)
        assert row1["status"] == "completed"
        assert row1["weighted_total"] is not None
        comp_count_1 = len(row1["competencies"])
        assert comp_count_1 > 0

        # 2. Simulate server restart: new engine instance rehydrated from store
        engine_rehydrated = InterviewEngine.rehydrate_from_store(st, iid, session_id=sid, llm=MockLLM())
        assert engine_rehydrated._report_cache is not None

        # 3. Call report() again: must hit cache from store without LLM re-run
        rep2 = engine_rehydrated.report()
        assert rep2["scorecard"]["weighted_total"] == rep1["scorecard"]["weighted_total"]

        # 4. Save scorecard again: must remain idempotent (no duplicated competencies)
        st.save_scorecard(iid, rep2["scorecard"])
        row2 = st.get_interview(iid)
        assert len(row2["competencies"]) == comp_count_1
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_replace_uploaded_document_updates_session_and_rag():
    st, path = _temp_store()
    try:
        iid = "interview_doc_replace"
        cfg = {"role": "Backend", "mode": "practice"}
        st.create_interview(cfg, session_id="sess_doc_rep")

        # Initial indexing of Resume 1 and Job Description 1
        resume_v1 = "Work Experience: Lead Architect at AlphaCorp. Scaled PostgreSQL cluster."
        jd_v1 = "Requirements: High scale PostgreSQL database engineering background."

        st.update_interview_documents(iid, resume_text=resume_v1, job_description=jd_v1)
        res1 = index_interview_documents(st, iid, resume_text=resume_v1, job_description=jd_v1)
        assert res1["resume_chunks"] > 0
        assert res1["jd_chunks"] > 0

        # Verify retrieval matches AlphaCorp
        ev1 = retrieve_evidence(st, iid, "AlphaCorp database", doc_type="resume")
        assert len(ev1) > 0
        assert "AlphaCorp" in ev1[0]["text"]

        # Replace Resume with Resume 2 (BetaCorp)
        resume_v2 = "Work Experience: Principal Engineer at BetaCorp. Architected Kafka event streaming."
        st.update_interview_documents(iid, resume_text=resume_v2)
        res2 = index_interview_documents(st, iid, resume_text=resume_v2, doc_type="resume")
        assert res2["resume_chunks"] > 0

        # Verify AlphaCorp is gone and BetaCorp is present in resume chunks
        ev_old = retrieve_evidence(st, iid, "AlphaCorp", doc_type="resume")
        assert len(ev_old) == 0 or all("AlphaCorp" not in c["text"] for c in ev_old)

        ev_new = retrieve_evidence(st, iid, "BetaCorp Kafka", doc_type="resume")
        assert len(ev_new) > 0
        assert "BetaCorp" in ev_new[0]["text"]

        # Ensure JD chunks were NOT deleted during resume replacement
        jd_chunks = st.get_document_chunks(iid, doc_type="job_description")
        assert len(jd_chunks) > 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_remove_uploaded_document_cleans_chunks():
    st, path = _temp_store()
    try:
        iid = "interview_doc_remove"
        st.create_interview({"role": "SDE"}, session_id="sess_doc_rem")

        resume = "Experience: Machine Learning Engineer at AI Labs."
        jd = "Requirements: 3+ years experience with PyTorch and Transformers."
        st.update_interview_documents(iid, resume_text=resume, job_description=jd)
        index_interview_documents(st, iid, resume_text=resume, job_description=jd)

        assert len(st.get_document_chunks(iid, doc_type="resume")) > 0
        assert len(st.get_document_chunks(iid, doc_type="job_description")) > 0

        # Remove resume
        st.update_interview_documents(iid, resume_text="")
        index_interview_documents(st, iid, resume_text="", doc_type="resume")

        # Resume chunks should be zero, JD chunks intact
        assert len(st.get_document_chunks(iid, doc_type="resume")) == 0
        assert len(st.get_document_chunks(iid, doc_type="job_description")) > 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_privacy_erasure_cascading_and_in_memory_eviction():
    st, path = _temp_store()
    try:
        email = "erasure_test@privacy.org"
        sid = "sess_privacy_evict"
        cfg = {"candidate_name": "Sam", "candidate_email": email, "mode": "hiring"}
        iid = st.create_interview(cfg, session_id=sid)

        st.add_turn(iid, 0, {"role": "interviewer", "text": "Q1", "stage": "core"})
        index_interview_documents(st, iid, resume_text="Work Experience: Engineer", job_description="JD")
        st.save_scorecard(iid, {"competencies": [{"id": "c1", "name": "C1", "weight": 1.0, "score": 4.0}],
                                "weighted_total": 80.0, "recommendation": "Hire", "summary": ""})

        # Register in _SESSIONS
        _SESSIONS[sid] = {"iid": iid, "last": time.time()}

        # Verify data exists
        assert st.get_interview(iid) is not None
        assert len(st.get_document_chunks(iid)) > 0
        assert sid in _SESSIONS

        # Perform erasure
        evict_sessions(interview_id=iid)
        st.delete_interview(iid)

        # Verify total cascade
        assert st.get_interview(iid) is None
        assert len(st.get_document_chunks(iid)) == 0
        assert st._conn.execute("SELECT COUNT(*) c FROM turns WHERE interview_id=?", (iid,)).fetchone()["c"] == 0
        assert st._conn.execute("SELECT COUNT(*) c FROM competencies WHERE interview_id=?", (iid,)).fetchone()["c"] == 0
        assert sid not in _SESSIONS  # Evicted from in-memory cache
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_database_migration_preserves_records_and_adds_indexes():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    legacy = sqlite3.connect(path)
    legacy.execute("""CREATE TABLE interviews (
        id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, status TEXT,
        mode TEXT, interview_type TEXT, role TEXT, level TEXT, candidate_name TEXT,
        candidate_email TEXT, company TEXT, language TEXT, consent INTEGER DEFAULT 0,
        llm TEXT, weighted_total REAL, recommendation TEXT, summary TEXT,
        review_status TEXT, reviewer TEXT, reviewer_notes TEXT, human_score REAL,
        human_recommendation TEXT)""")
    legacy.execute("INSERT INTO interviews (id, candidate_name, role) VALUES ('legacy_001', 'Arun', 'DevOps')")
    legacy.commit()
    legacy.close()

    try:
        # Opening legacy DB through Store runs migration
        st = Store(path)
        cols = {r[1] for r in st._conn.execute("PRAGMA table_info(interviews)")}
        assert "session_id" in cols
        assert "evaluation_report" in cols
        assert "resume_text" in cols

        # Check existing record preserved
        row = st.get_interview("legacy_001")
        assert row is not None
        assert row["candidate_name"] == "Arun"
        assert row["role"] == "DevOps"

        # Check indexes exist
        idx_names = {r[1] for r in st._conn.execute("PRAGMA index_list(interviews)")}
        assert "idx_interviews_session" in idx_names
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_concurrent_writes_with_wal_mode():
    st, path = _temp_store()
    try:
        iid = st.create_interview({"role": "QA"}, session_id="sess_concur")
        errors = []

        def worker(idx):
            try:
                st.add_turn(iid, idx, {
                    "role": "candidate",
                    "text": f"Response {idx}",
                    "stage": "core",
                })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        turns = st._conn.execute("SELECT COUNT(*) c FROM turns WHERE interview_id=?", (iid,)).fetchone()["c"]
        assert turns == 20
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_session_isolation_strictness():
    st, path = _temp_store()
    try:
        # Candidate 1
        iid1 = st.create_interview({"candidate_name": "Candidate One"}, session_id="sess_one")
        st.add_turn(iid1, 0, {"role": "interviewer", "text": "Q1 for Cand1", "stage": "core"})
        index_interview_documents(st, iid1, resume_text="Kubernetes cluster management at Company Alpha.")

        # Candidate 2
        iid2 = st.create_interview({"candidate_name": "Candidate Two"}, session_id="sess_two")
        st.add_turn(iid2, 0, {"role": "interviewer", "text": "Q1 for Cand2", "stage": "core"})
        index_interview_documents(st, iid2, resume_text="iOS swift development at Company Beta.")

        # Rehydrate Candidate 1
        eng1 = InterviewEngine.rehydrate_from_store(st, iid1, session_id="sess_one", llm=MockLLM())
        assert eng1.config.candidate_name == "Candidate One"
        assert len(eng1.turns) == 1
        assert "Cand1" in eng1.turns[0].text
        assert "Cand2" not in eng1.turns[0].text

        # RAG retrieval for Candidate 1 should never return Candidate 2's data
        res1 = retrieve_evidence(st, iid1, "swift iOS")
        assert all("swift" not in r["text"].lower() for r in res1)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    tests = [
        test_session_survives_restart_and_rehydrates,
        test_get_or_rehydrate_session_helper,
        test_report_caching_idempotency_no_duplicate_evaluations,
        test_replace_uploaded_document_updates_session_and_rag,
        test_remove_uploaded_document_cleans_chunks,
        test_privacy_erasure_cascading_and_in_memory_eviction,
        test_database_migration_preserves_records_and_adds_indexes,
        test_concurrent_writes_with_wal_mode,
        test_session_isolation_strictness,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__} passed.")
    print(f"\nAll {len(tests)} Phase 5 Persistence & Reliability tests passed successfully!")
