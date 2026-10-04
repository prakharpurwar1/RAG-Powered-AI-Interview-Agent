"""Tests for the integrity monitor.

Run:  python tests/test_integrity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.integrity import IntegrityMonitor


def test_clean_session_scores_100():
    m = IntegrityMonitor()
    assert m.score() == 100.0
    assert m.flags() == []


def test_paste_penalty_is_capped():
    m = IntegrityMonitor()
    for _ in range(5):
        m.record("paste")
    f = {x["kind"]: x for x in m.flags()}
    assert f["paste"]["count"] == 5
    assert f["paste"]["penalty"] == 15 * 3      # capped at 3 occurrences
    assert m.score() == 100 - 45


def test_combined_penalties_floor_at_zero():
    m = IntegrityMonitor()
    for _ in range(3):
        m.record("paste")     # 45
    for _ in range(3):
        m.record("blur")      # 15
    for _ in range(2):
        m.record("fast")      # 20
    for _ in range(2):
        m.record("copy")      # 16
    m.record("devtools")      # 10  -> total 106
    assert m.score() == 0.0


def test_disabled_monitor_ignores_everything():
    m = IntegrityMonitor(enabled=False)
    m.record("paste")
    assert m.events == []
    assert m.score() == 100.0


def test_unknown_kind_ignored():
    m = IntegrityMonitor()
    m.record("somethingelse")
    assert m.events == []


def test_summary_shape():
    m = IntegrityMonitor()
    m.record("blur")
    s = m.summary()
    assert set(["enabled", "score", "flags", "events", "detail"]).issubset(s)
    assert s["events"] == 1 and s["score"] == 95.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
