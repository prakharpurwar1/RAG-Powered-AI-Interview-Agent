"""End-to-end tests for the interview engine (offline, via MockLLM).

Run:  python -m pytest -q   (or)   python tests/test_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine
from interviewer.llm import MockLLM


def test_full_flow_reaches_done():
    cfg = InterviewConfig(mode="practice", interview_type="behavioral",
                          max_questions=3, candidate_name="Asha")
    eng = InterviewEngine(cfg, MockLLM())
    first = eng.start()
    assert "Asha" in first

    # answer everything with a substantive reply so we advance quickly
    answer = ("I led the migration of our billing service. I owned the plan, "
              "coordinated three teams, and cut errors by 40 percent over two quarters.")
    steps = 0
    out = None
    while not eng.finished and steps < 30:
        out = eng.respond(answer)
        steps += 1
    assert eng.finished, "interview should finish"
    assert out["done"] is True


def test_thin_answer_triggers_followup():
    cfg = InterviewConfig(interview_type="mixed", max_questions=2, allow_followups=True)
    eng = InterviewEngine(cfg, MockLLM())
    eng.start()
    out = eng.respond("I don't know")          # thin -> follow-up
    assert out["stage"] == eng.stage
    assert eng.turns[-1].kind == "followup"


def test_scorecard_weights_sum_and_total():
    cfg = InterviewConfig(interview_type="technical", max_questions=2)
    eng = InterviewEngine(cfg, MockLLM())
    eng.start()
    for _ in range(12):
        if eng.finished:
            break
        eng.respond("I built a service that processed a million events a day and "
                    "explained the trade-offs between consistency and availability.")
    rep = eng.report()
    sc = rep["scorecard"]
    weights = sum(c["weight"] for c in sc["competencies"])
    assert abs(weights - 1.0) < 1e-6, f"weights should sum to 1, got {weights}"
    assert 0 <= sc["weighted_total"] <= 100
    assert sc["recommendation"]


def test_mock_llm_never_needs_a_key():
    assert MockLLM().chat([{"role": "user", "content": "hi"}], task="followup")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
