"""Tests for SQLite persistence.

Run:  python tests/test_store.py
"""
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.store import Store


def _store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return Store(path), path


def test_lifecycle():
    st, path = _store()
    try:
        iid = st.create_interview({
            "mode": "hiring", "interview_type": "technical", "role": "SDE",
            "level": "mid", "candidate_name": "Asha", "candidate_email": "a@x.com",
            "company": "Acme", "language": "en-IN", "consent": True,
        }, llm="MockLLM")
        assert iid

        st.add_turn(iid, 0, {"role": "interviewer", "text": "Q1", "stage": "core",
                             "question_id": "t1", "kind": "message"})
        st.add_turn(iid, 1, {"role": "candidate", "text": "A1", "stage": "core"})
        st.save_scorecard(iid, {
            "competencies": [{"id": "fundamentals", "name": "Fundamentals",
                              "weight": 0.3, "score": 4.0, "evidence": "good",
                              "comment": "solid"}],
            "weighted_total": 72.0, "recommendation": "Hire", "summary": "ok",
        })

        row = st.get_interview(iid)
        assert row["status"] == "completed"
        assert row["weighted_total"] == 72.0
        assert len(row["turns"]) == 2
        assert row["competencies"][0]["score"] == 4.0
        assert row["consent"] == 1

        st.save_review(iid, "reviewer1", "strong hire", 4.5, "Strong hire")
        row = st.get_interview(iid)
        assert row["review_status"] == "reviewed"
        assert row["human_recommendation"] == "Strong hire"

        assert st.stats()["completed"] == 1
        assert st.stats()["reviewed"] == 1
        assert len(st.list_interviews()) == 1

        st.delete_interview(iid)
        assert st.get_interview(iid) is None
        assert st.stats()["total"] == 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_sorting_by_score():
    st, path = _store()
    try:
        a = st.create_interview({"candidate_name": "Low", "mode": "hiring", "consent": True})
        b = st.create_interview({"candidate_name": "High", "mode": "hiring", "consent": True})
        st.save_scorecard(a, {"competencies": [], "weighted_total": 40.0,
                              "recommendation": "No hire", "summary": ""})
        st.save_scorecard(b, {"competencies": [], "weighted_total": 90.0,
                              "recommendation": "Strong hire", "summary": ""})
        order = [r["candidate_name"] for r in st.list_interviews(sort="score")]
        assert order[0] == "High"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_purge_expired():
    st, path = _store()
    try:
        iid = st.create_interview({"mode": "hiring", "consent": True, "retention_days": 1})
        st._conn.execute("UPDATE interviews SET retention_until=? WHERE id=?",
                         (time.time() - 10, iid))
        st._conn.commit()
        assert st.purge_expired() == 1
        assert st.get_interview(iid) is None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_candidate_export_and_erasure():
    st, path = _store()
    try:
        st.create_interview({"candidate_email": "a@x.com", "mode": "hiring", "consent": True})
        st.create_interview({"candidate_email": "a@x.com", "mode": "hiring", "consent": True})
        st.create_interview({"candidate_email": "b@x.com", "mode": "hiring", "consent": True})
        assert len(st.export_candidate("a@x.com")) == 2
        assert st.delete_candidate("a@x.com") == 2
        assert st.export_candidate("a@x.com") == []
        assert st.stats()["total"] == 1
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_integrity_persisted():
    st, path = _store()
    try:
        iid = st.create_interview({"mode": "hiring", "consent": True})
        st.save_integrity(iid, {"score": 85.0, "events": 1,
                                "flags": [{"kind": "blur", "label": "tab", "count": 1,
                                           "penalty": 5}]})
        row = st.get_interview(iid)
        assert row["integrity_score"] == 85.0
        assert row["flags"][0]["kind"] == "blur"
        assert row["proctor_events"] == 1
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_consent_recorded_with_timestamp():
    st, path = _store()
    try:
        iid = st.create_interview({"mode": "hiring", "consent": True,
                                   "consent_version": "2.1", "retention_days": 30})
        row = st.get_interview(iid)
        assert row["consent"] == 1 and row["consent_at"]
        assert row["consent_version"] == "2.1"
        assert row["retention_until"] > time.time()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_migration_adds_new_columns():
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
    legacy.commit()
    legacy.close()
    try:
        st = Store(path)                     # opening should migrate in place
        cols = {r[1] for r in st._conn.execute("PRAGMA table_info(interviews)")}
        assert {"retention_until", "integrity_score", "flags",
                "consent_at", "consent_version"}.issubset(cols)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
