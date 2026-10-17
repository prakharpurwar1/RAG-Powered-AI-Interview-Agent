"""Automated tests for Adaptive Interview Logic and RAG Verification.

Scenarios tested:
A. The candidate mentions a specific project; the next question probes that project.
B. The candidate gives a vague answer; the interviewer asks for clarification.
C. The candidate gives a complete answer; the interviewer advances to a different competency.
D. The same question is not repeated unnecessarily.
E. The retrieved resume/JD chunks reach the model-generation function.
F. Documents belonging to another session are never retrieved.

Run: python3 tests/test_adaptive_interview.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine, validate_turn_decision, TurnDecision
from interviewer.llm import MockLLM
from interviewer.rag import index_interview_documents, retrieve_evidence
from interviewer.store import Store


def test_scenario_a_project_probing():
    """Scenario A: The candidate mentions a specific project; the next question probes that project."""
    cfg = InterviewConfig(
        role="Data Scientist",
        interview_type="technical",
        max_questions=3,
        allow_followups=True,
        candidate_name="Prakhar",
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()

    # Candidate mentions their HR attrition project with IBM dataset
    candidate_answer = (
        "In my previous project, I built an HR attrition prediction model using the IBM dataset. "
        "I trained machine-learning classification models to identify high-risk employees."
    )
    resp = engine.respond(candidate_answer)

    assert not resp["done"]
    decision = resp.get("decision")
    assert decision is not None, "Engine must return validated structured decision"

    # Verify that the decision explicitly references the project mentioned
    spoken = resp["message"].lower()
    reason = decision["reason_for_decision"].lower()
    evidence = " ".join(decision["evidence_used"]).lower()

    assert ("hr attrition" in spoken or "hr attrition" in reason or "hr attrition" in evidence or
            "ibm" in spoken or "ibm" in reason), (
        f"Interviewer question/decision must probe the candidate's stated HR attrition project. Spoken: {resp['message']}"
    )
    print("  ✓ Scenario A passed: Interviewer probed candidate's HR attrition project.")


def test_scenario_b_vague_answer_clarification():
    """Scenario B: The candidate gives a vague answer; the interviewer asks for clarification."""
    cfg = InterviewConfig(
        role="Backend Engineer",
        interview_type="mixed",
        max_questions=3,
        allow_followups=True,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()
    initial_q_count = engine.questions_asked
    assert initial_q_count == 1
    assert engine.followups_used == 0

    # Vague answer lacking technical detail or personal contribution
    vague_answer = "I learned new technologies and adapted to them."
    resp = engine.respond(vague_answer)

    assert not resp["done"]
    decision = resp["decision"]
    assert decision["action"] == "follow_up", f"Expected action='follow_up' for vague answer, got {decision['action']}"
    assert engine.followups_used == 1, "Followup counter must increment"
    assert engine.questions_asked == initial_q_count, (
        f"Main question counter must NOT advance on follow-up! Expected {initial_q_count}, got {engine.questions_asked}"
    )

    spoken = resp["message"].lower()
    assert any(term in spoken for term in ("concrete", "contribution", "measured", "hurdle", "specific")), (
        f"Clarification follow-up should probe for specifics. Got: {resp['message']}"
    )
    print("  ✓ Scenario B passed: Vague answer resulted in clarification follow-up without advancing question counter.")


def test_scenario_c_complete_answer_advances_competency():
    """Scenario C: The candidate gives a complete answer; the interviewer advances to a different competency."""
    cfg = InterviewConfig(
        role="Full Stack Engineer",
        interview_type="technical",
        max_questions=4,
        allow_followups=True,
    )
    engine = InterviewEngine(cfg, MockLLM())
    engine.start()
    initial_comp = engine.current_competency
    initial_q_count = engine.questions_asked

    # Complete answer with technical architecture and concrete measurable impact
    complete_answer = (
        "I architected a high-throughput event processing pipeline using Kafka and Go. "
        "The system handled over 45,000 requests per second across 12 worker nodes, "
        "reducing end-to-end p99 ingestion latency by 42% from 380ms to 220ms and achieving 99.99% uptime."
    )
    resp = engine.respond(complete_answer)

    assert not resp["done"]
    decision = resp["decision"]
    assert decision["action"] == "next_question", f"Expected action='next_question' for complete answer, got {decision['action']}"
    assert engine.questions_asked == initial_q_count + 1, "Questions asked counter must advance on next_question"
    assert engine.followups_used == 0, "Followups counter must reset to 0 on next main question"
    assert engine.current_competency != initial_comp, (
        f"Interviewer must advance to a different competency. Initial: {initial_comp}, New: {engine.current_competency}"
    )
    print(f"  ✓ Scenario C passed: Complete answer advanced competency from '{initial_comp}' to '{engine.current_competency}'.")


def test_scenario_d_no_question_repeated():
    """Scenario D: The same question is not repeated unnecessarily."""
    cfg = InterviewConfig(
        role="Software Engineer",
        interview_type="mixed",
        max_questions=3,
        allow_followups=True,
    )
    recorded_prompts = []

    class MockInspectLLM(MockLLM):
        def chat(self, messages, task=None, **kwargs):
            if task == "decide_turn":
                recorded_prompts.append(messages[-1]["content"])
            return super().chat(messages, task=task, **kwargs)

    engine = InterviewEngine(cfg, MockInspectLLM())
    engine.start()

    # Step through turns
    engine.respond("I designed the money-management budgeting tool using React and Node.js.")
    engine.respond("I used PostgreSQL with indexed transactions and cut query latency by 50% for 100k active users.")

    interviewer_questions = [t.text.strip() for t in engine.turns if t.role == "interviewer"]
    assert len(interviewer_questions) >= 3

    # Check uniqueness
    unique_questions = set(interviewer_questions)
    assert len(unique_questions) == len(interviewer_questions), (
        f"Detected duplicate questions in interview turns: {interviewer_questions}"
    )

    # Check that decision prompt explicitly listed PREVIOUSLY ASKED QUESTIONS to prevent repeats
    assert len(recorded_prompts) >= 2
    for p in recorded_prompts:
        assert "PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT)" in p
        assert "DO NOT repeat or rephrase any question" in p
    print("  ✓ Scenario D passed: No duplicate questions asked; prompt enforces non-repetition.")


def test_scenario_e_rag_chunks_reach_model_generation():
    """Scenario E: The retrieved resume/JD chunks reach the model-generation function."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        iid = "session_test_rag_e"
        resume_content = (
            "Experience:\n"
            "Staff AI Engineer at TechCorp (2021-2024).\n"
            "Spearheaded IBM HR attrition modeling initiative using gradient boosting and random forests. "
            "Evaluated employee retention factors and automated HR alert triggers."
        )
        jd_content = (
            "Requirements:\n"
            "Must have extensive experience with machine learning pipelines, scikit-learn, and predictive analytics."
        )

        cfg = InterviewConfig(
            role="Data Science Lead",
            interview_type="technical",
            resume_text=resume_content,
            job_description=jd_content,
            max_questions=3,
        )

        # Index documents via RAG
        index_result = index_interview_documents(store, iid, resume_text=resume_content, job_description=jd_content)
        assert index_result["total_indexed"] > 0

        intercepted_prompts = []

        class InterceptingLLM:
            name = "InterceptingLLM"

            def chat(self, messages, task=None, **kwargs):
                if task == "decide_turn":
                    intercepted_prompts.append(messages[-1]["content"])
                    return json.dumps({
                        "action": "follow_up",
                        "spoken_response": "How did you validate the IBM HR attrition dataset features against bias?",
                        "reason_for_decision": "Probing ML feature engineering on candidate's stated experience",
                        "competency": "experience",
                        "evidence_used": ["IBM HR attrition modeling initiative"]
                    })
                return "Greeting message"

        engine = InterviewEngine(cfg, InterceptingLLM(), store=store, interview_id=iid, session_id="sess_e_123")
        engine.start()

        # Candidate gives an answer that triggers RAG retrieval
        resp = engine.respond("I built the HR attrition predictive models on IBM employee records.")

        assert len(intercepted_prompts) == 1, "Model decide_turn prompt must be generated"
        prompt_sent = intercepted_prompts[0]

        # Verify that verified RAG evidence reached the prompt
        assert "--- RETRIEVED VERIFIED EVIDENCE (RAG) ---" in prompt_sent
        assert "attrition" in prompt_sent.lower() or "ibm" in prompt_sent.lower(), (
            f"Expected retrieved chunks to be present in prompt. Prompt snippet:\n{prompt_sent[:800]}"
        )

        # Verify response includes evidence metadata
        assert resp["evidence"] is not None
        assert len(resp["evidence"]) > 0
        assert any(e["doc_type"] == "resume" for e in resp["evidence"])
        print("  ✓ Scenario E passed: Retrieved RAG chunks successfully injected into model prompt.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_scenario_f_cross_session_isolation():
    """Scenario F: Documents belonging to another session are never retrieved."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        session_alice = "sess_alice_001"
        session_bob = "sess_bob_002"

        alice_resume = (
            "Work Experience:\n"
            "Lead Cryptography Engineer at SecretVault.\n"
            "Architected Project Alpha: a proprietary zero-knowledge blockchain privacy layer."
        )
        bob_resume = (
            "Work Experience:\n"
            "Senior Quantitative Developer at HighFreqFund.\n"
            "Built Project Beta: ultra low latency C++ order execution gateway processing NASDAQ order book."
        )

        # Index both candidates' documents into the shared database
        index_interview_documents(store, session_alice, resume_text=alice_resume)
        index_interview_documents(store, session_bob, resume_text=bob_resume)

        # 1. Query Alice's session looking for high frequency trading / NASDAQ
        alice_results = retrieve_evidence(store, session_alice, "ultra low latency C++ NASDAQ trading", top_k=5)
        # Even with a query matching Bob's keywords, Alice's retrieval MUST NEVER return Bob's chunks!
        for c in alice_results:
            assert "Project Beta" not in c["text"]
            assert "NASDAQ" not in c["text"]
            assert "HighFreqFund" not in c["text"]
            assert c["doc_type"] == "resume"

        # 2. Query Bob's session looking for zero-knowledge blockchain
        bob_results = retrieve_evidence(store, session_bob, "proprietary zero-knowledge blockchain crypto", top_k=5)
        # Bob's retrieval MUST NEVER return Alice's chunks!
        for c in bob_results:
            assert "Project Alpha" not in c["text"]
            assert "SecretVault" not in c["text"]
            assert "zero-knowledge" not in c["text"]
            assert c["doc_type"] == "resume"

        print("  ✓ Scenario F passed: Strict multi-tenant session isolation verified across vector retrieval.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_structured_decision_validation_and_error_handling():
    """Verify that invalid model decisions raise LLMError and are never silently swallowed."""
    # 1. Invalid JSON
    try:
        validate_turn_decision("not a dict")
        assert False, "Should raise ValueError for non-dict"
    except ValueError:
        pass

    # 2. Invalid action
    try:
        validate_turn_decision({"action": "random_action", "spoken_response": "Hello"})
        assert False, "Should raise ValueError for invalid action"
    except ValueError:
        pass

    # 3. Empty spoken_response
    try:
        validate_turn_decision({"action": "follow_up", "spoken_response": ""})
        assert False, "Should raise ValueError for empty spoken_response"
    except ValueError:
        pass

    # 4. Valid structure
    valid = validate_turn_decision({
        "action": "follow_up",
        "spoken_response": "Can you elaborate on your role?",
        "reason_for_decision": "Answer was too brief.",
        "competency": "experience",
        "evidence_used": ["Candidate mentioned project X"],
    })
    assert isinstance(valid, TurnDecision)
    assert valid.action == "follow_up"
    print("  ✓ Decision validation schema test passed.")


if __name__ == "__main__":
    tests = [
        test_scenario_a_project_probing,
        test_scenario_b_vague_answer_clarification,
        test_scenario_c_complete_answer_advances_competency,
        test_scenario_d_no_question_repeated,
        test_scenario_e_rag_chunks_reach_model_generation,
        test_scenario_f_cross_session_isolation,
        test_structured_decision_validation_and_error_handling,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} adaptive interview & RAG verification tests passed successfully!")
