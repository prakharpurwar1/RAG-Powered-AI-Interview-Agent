import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine, TurnDecision, validate_turn_decision
from interviewer.llm import MockLLM, LLMError
from interviewer.scoring import build_scorecard
from interviewer.store import Store


def test_validate_turn_decision_expanded_intents_and_actions():
    """Validates that all new Phase 2 intents and actions normalize and validate correctly."""
    # Test valid expanded intents and actions
    valid_samples = [
        {"action": "conversational", "candidate_intent": "greeting", "spoken_response": "Hello! Let's begin."},
        {"action": "clarify", "candidate_intent": "clarification", "spoken_response": "To clarify, we mean..."},
        {"action": "rephrase", "candidate_intent": "rephrase", "spoken_response": "Sure, let me rephrase that..."},
        {"action": "conversational", "candidate_intent": "meta_question", "spoken_response": "I am an AI interviewer driven by an LLM..."},
        {"action": "conversational", "candidate_intent": "off_topic", "spoken_response": "Let's focus on the interview question."},
        {"action": "follow_up", "candidate_intent": "answer", "spoken_response": "Could you provide metrics?"},
        {"action": "next_question", "candidate_intent": "answer", "spoken_response": "Moving to the next topic..."},
    ]
    for s in valid_samples:
        d = validate_turn_decision(s)
        assert isinstance(d, TurnDecision)
        assert d.action == s["action"]
        assert d.candidate_intent == s["candidate_intent"]

    # Test synonym normalization
    synonym_samples = [
        ({"action": "small_talk", "spoken_response": "Hi there!"}, "conversational", "greeting"),
        ({"action": "clarification", "spoken_response": "Explanation here"}, "clarify", "clarification"),
        ({"action": "repeat", "spoken_response": "Restating question"}, "rephrase", "rephrase"),
        ({"action": "meta", "spoken_response": "I use LLM and RAG"}, "conversational", "meta_question"),
        ({"action": "divert", "spoken_response": "Let's stick to the topic"}, "conversational", "off_topic"),
    ]
    for raw, expected_action, expected_intent in synonym_samples:
        d = validate_turn_decision(raw)
        assert d.action == expected_action
        assert d.candidate_intent == expected_intent


def test_candidate_intent_rule_based_meta_question():
    """Scenario: Candidate asks 'Are you rule-based?'.

    Interviewer must explain that it is an LLM/RAG driven interviewer and guide
    them back to the current question without advancing interview question counters.
    """
    cfg = InterviewConfig(
        role="Machine Learning Engineer",
        interview_type="technical",
        max_questions=3,
        allow_followups=True,
    )
    engine = InterviewEngine(cfg, MockLLM())
    opening = engine.start()
    assert opening

    initial_q_count = engine.questions_asked
    initial_f_count = engine.followups_used
    initial_comp = engine.current_competency

    # Candidate asks the exact question
    resp = engine.respond("Are you rule-based?")
    assert not resp["done"]

    decision = resp.get("decision")
    assert decision is not None
    assert decision["candidate_intent"] == "meta_question"
    assert decision["action"] == "conversational"

    spoken = resp["message"].lower()
    assert any(term in spoken for term in ["rule-based", "llm", "rag", "artificial intelligence", "ai"])
    assert any(guide in spoken for guide in ["back to", "return to", "question", "background", "experience"])

    # Progress counters and competency MUST NOT be changed
    assert engine.questions_asked == initial_q_count
    assert engine.followups_used == initial_f_count
    assert engine.current_competency == initial_comp


def test_candidate_intent_greeting_and_small_talk():
    """Scenario: Candidate sends greeting/small talk.

    Interviewer responds warmly, prompts candidate for the current question,
    and does not advance question or followup counters.
    """
    cfg = InterviewConfig(
        role="Backend Developer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    initial_q_count = engine.questions_asked
    initial_f_count = engine.followups_used
    initial_comp = engine.current_competency

    resp = engine.respond("Good morning! Nice to meet you today, hope you are doing well.")
    assert not resp["done"]

    decision = resp.get("decision")
    assert decision is not None
    assert decision["candidate_intent"] == "greeting"
    assert decision["action"] == "conversational"

    spoken = resp["message"].lower()
    assert any(g in spoken for g in ["hello", "welcome", "good morning", "glad to have you", "nice to meet"])

    # Progress counters MUST remain unchanged
    assert engine.questions_asked == initial_q_count
    assert engine.followups_used == initial_f_count
    assert engine.current_competency == initial_comp


def test_candidate_intent_clarification_request():
    """Scenario: Candidate asks for clarification regarding the question.

    Interviewer clarifies the concept without giving away the answer,
    invites them to answer, and does NOT increment question counters.
    """
    cfg = InterviewConfig(
        role="Systems Engineer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    initial_q_count = engine.questions_asked
    initial_f_count = engine.followups_used
    initial_comp = engine.current_competency

    resp = engine.respond("Could you clarify what you mean by system trade-offs in this context?")
    assert not resp["done"]

    decision = resp.get("decision")
    assert decision is not None
    assert decision["candidate_intent"] == "clarification"
    assert decision["action"] == "clarify"

    spoken = resp["message"].lower()
    # Verifies explanation is given and candidate is invited to answer
    assert any(term in spoken for term in ["trade-off", "clarify", "balance", "compromise", "example"])
    
    # Question counters MUST NOT advance
    assert engine.questions_asked == initial_q_count
    assert engine.followups_used == initial_f_count
    assert engine.current_competency == initial_comp


def test_candidate_intent_repeat_or_rephrase_request():
    """Scenario: Candidate asks to repeat or rephrase the question.

    Interviewer restates/rephrases the question without penalty or advancing progression.
    """
    cfg = InterviewConfig(
        role="Software Engineer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    initial_q_count = engine.questions_asked
    initial_f_count = engine.followups_used
    initial_comp = engine.current_competency

    resp = engine.respond("Could you please rephrase that question in simpler terms?")
    assert not resp["done"]

    decision = resp.get("decision")
    assert decision is not None
    assert decision["candidate_intent"] == "rephrase"
    assert decision["action"] == "rephrase"

    assert engine.questions_asked == initial_q_count
    assert engine.followups_used == initial_f_count
    assert engine.current_competency == initial_comp


def test_candidate_intent_off_topic_redirection():
    """Scenario: Candidate makes an off-topic comment.

    Interviewer politely redirects them back to the active interview question without penalty.
    """
    cfg = InterviewConfig(
        role="Frontend Engineer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    initial_q_count = engine.questions_asked
    initial_f_count = engine.followups_used

    resp = engine.respond("Did you watch the Champions League match yesterday? It was incredible.")
    assert not resp["done"]

    decision = resp.get("decision")
    assert decision is not None
    assert decision["candidate_intent"] == "off_topic"
    assert decision["action"] == "conversational"

    spoken = resp["message"].lower()
    assert any(redirect in spoken for redirect in ["focus", "back to", "return to", "interview", "question"])

    assert engine.questions_asked == initial_q_count
    assert engine.followups_used == initial_f_count


def test_multi_turn_non_answer_then_strong_answer_progression():
    """Scenario: Candidate interacts with multiple conversational turns before answering.

    Verifies counters stay stable through non-answers, and only advance once
    a substantive technical answer is provided.
    """
    cfg = InterviewConfig(
        role="DevOps Engineer",
        interview_type="technical",
        max_questions=3,
        allow_followups=True,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    # Step 1: Greeting
    resp1 = engine.respond("Hi there, good morning!")
    assert resp1["decision"]["candidate_intent"] == "greeting"
    assert engine.questions_asked == 1
    assert engine.followups_used == 0

    # Step 2: Meta question
    resp2 = engine.respond("Are you rule-based or an AI agent?")
    assert resp2["decision"]["candidate_intent"] == "meta_question"
    assert engine.questions_asked == 1
    assert engine.followups_used == 0

    # Step 3: Clarification
    resp3 = engine.respond("What specifically are you looking for in this question?")
    assert resp3["decision"]["candidate_intent"] == "clarification"
    assert engine.questions_asked == 1
    assert engine.followups_used == 0

    # Step 4: Substantive technical answer with metrics
    technical_answer = (
        "In my previous project at CloudScale, I engineered a multi-region Kubernetes cluster "
        "using Terraform and ArgoCD. We achieved zero-downtime blue/green deployments and reduced "
        "deployment rollback times by 75% while maintaining 99.99% availability."
    )
    resp4 = engine.respond(technical_answer)
    assert resp4["decision"]["candidate_intent"] == "answer"
    # A complete answer with metrics transitions to next_question
    assert resp4["decision"]["action"] == "next_question"
    assert engine.questions_asked == 2


def test_rehydration_preserves_non_answer_turns_without_counter_corruption(tmp_path):
    """Verifies that persisting and rehydrating turns with conversational/clarification kinds

    does not distort `questions_asked`, `followups_used`, or `current_competency`.
    """
    db_path = str(tmp_path / "test_store.db")
    store = Store(path=db_path)

    cfg = InterviewConfig(
        role="Backend Engineer",
        interview_type="technical",
        max_questions=3,
        allow_followups=True,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    # Create interview in store
    iid = store.create_interview(vars(cfg), session_id="sess_rehydrate_test")

    # Add a greeting turn
    engine.respond("Hello, happy to be here!")
    # Add a clarification turn
    engine.respond("Can you explain what you mean?")

    # Persist all current engine turns into store
    for idx, t in enumerate(engine.turns):
        store.add_turn(iid, idx, vars(t))

    # Rehydrate into a new engine instance
    new_engine = InterviewEngine.rehydrate_from_store(store, iid)
    assert new_engine is not None

    # Counters in rehydrated engine must match original engine
    assert new_engine.questions_asked == 1
    assert new_engine.followups_used == 0
    assert len(new_engine.turns) == len(engine.turns)

    # Candidate turns kinds must be preserved
    kinds = [t.kind for t in new_engine.turns]
    assert "greeting" in kinds
    assert "clarification" in kinds


def test_scoring_isolation_only_non_answers_yields_unassessed_scorecard():
    """Scenario: Candidate only says greetings and asks meta-questions.

    Candidate MUST NOT be assigned failing scores (e.g. 1.0) or penalized.
    The scorecard must return 'insufficient_evidence' across competencies.
    """
    cfg = InterviewConfig(
        role="Security Engineer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    engine.respond("Hi, good morning!")
    engine.respond("Are you rule-based?")
    engine.respond("Could you clarify the question?")

    scorecard = build_scorecard(MockLLM(), engine.rubric, engine.turns, cfg)

    assert scorecard.recommendation == "Insufficient Data"
    assert scorecard.weighted_total == 0.0
    for comp in scorecard.competencies:
        assert comp.status == "insufficient_evidence"
        assert comp.score is None


def test_scoring_isolation_non_answers_do_not_dilute_strong_answer():
    """Scenario: Candidate has greeting + meta-question + strong answer.

    Evaluation must evaluate the strong answer without penalty for the conversational turns.
    """
    cfg = InterviewConfig(
        role="Data Engineer",
        interview_type="technical",
        max_questions=3,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    engine.respond("Good morning!")
    engine.respond("Are you rule-based?")
    strong_answer = (
        "In my recent project, I designed a real-time data streaming pipeline using Apache Flink and Kafka. "
        "I was the lead engineer responsible for state management and fault tolerance. "
        "We reduced end-to-end data latency from 15 minutes to under 2 seconds and processed 50,000 events/sec."
    )
    engine.respond(strong_answer)

    scorecard = build_scorecard(MockLLM(), engine.rubric, engine.turns, cfg)

    # Should evaluate the competency without penalty
    evaluated = [c for c in scorecard.competencies if c.status == "evaluated"]
    assert len(evaluated) > 0
    for comp in evaluated:
        assert comp.score is not None
        assert comp.score >= 3.0


def test_api_message_intent_handling_meta_question_preserves_counters():
    """Integration test: candidate asks 'Are you rule-based?' over HTTP POST /api/message.

    Verifies response is returned, done is false, turn is persisted, and
    questions counter is preserved.
    """
    from starlette.testclient import TestClient
    from interviewer.app import app

    client = TestClient(app)
    start_payload = {
        "role": "Backend Engineer",
        "candidate_name": "Test Candidate",
        "mock": True,
        "consent": True,
    }
    start_resp = client.post("/api/start", json=start_payload)
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    session_id = start_data["session_id"]

    # Send meta question
    msg_resp = client.post("/api/message", json={
        "session_id": session_id,
        "message": "Are you rule-based or an AI agent?",
    })
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()

    assert not msg_data.get("done")
    spoken = msg_data.get("message", "").lower()
    assert any(term in spoken for term in ["rule-based", "llm", "rag", "artificial intelligence", "ai"])


def test_llm_failure_during_intent_response_returns_502():
    """Verifies that if the LLM fails with LLMError during a candidate message turn,

    the API returns HTTP 502 with a sanitized error message (retaining Phase 1 robustness).
    """
    from starlette.testclient import TestClient
    from interviewer.app import app, _SESSIONS

    client = TestClient(app)
    start_resp = client.post("/api/start", json={
        "role": "Backend Engineer",
        "candidate_name": "Test Candidate",
        "mock": True,
        "consent": True,
    })
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # Inject an LLM that fails on decide_turn
    class FailingLLM(MockLLM):
        def chat(self, messages, *, task="question", **kwargs):
            if task == "decide_turn":
                raise LLMError("Upstream server timeout while deciding turn", status_code=504)
            return super().chat(messages, task=task, **kwargs)

    sess = _SESSIONS.get(session_id)
    assert sess is not None
    sess["engine"].llm = FailingLLM()

    msg_resp = client.post("/api/message", json={
        "session_id": session_id,
        "message": "Are you rule-based?",
    })
    assert msg_resp.status_code == 502
    assert "Upstream AI service request failed (HTTP 504)" in msg_resp.json()["detail"]

