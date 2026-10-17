"""Automated tests for Phase 4: Interview Evaluation Engine.

Tests:
1. Strong answers with evidence (metrics, architecture, STAR grounding).
2. Vague answers with gaps (properly identified, actionable advice, no fabrication).
3. Incomplete interviews (unprobed competencies marked 'insufficient_evidence', not penalized).
4. Anti-hallucination guardrail (no fabricated metrics or unsupported claims).
5. Scoring consistency and rubric normalization (weights sum to 1.0, normalized scores).
6. Session-specific evaluation isolation (Candidate A's results never leak to Candidate B).

Run: python3 tests/test_evaluation.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine, Turn
from interviewer.llm import MockLLM, LLMError
from interviewer.rubric import rubric_for
from interviewer.scoring import (
    build_scorecard,
    validate_and_build_scorecard,
    is_evidence_grounded,
    ScoreCard,
    CompetencyScore,
)
from interviewer.store import Store


def test_evaluation_strong_answers_with_evidence():
    """Verify that strong answers receive high scores with verbatim evidence and STAR improvements."""
    cfg = InterviewConfig(
        role="Machine Learning Engineer",
        interview_type="technical",
        max_questions=3,
        candidate_name="Prakhar",
    )
    rubric = rubric_for("technical")

    turns = [
        Turn(role="interviewer", text="Walk me through your key machine learning project.", stage="core", kind="message"),
        Turn(
            role="candidate",
            text=(
                "I built an HR attrition machine learning model using an IBM dataset. "
                "I trained Random Forest and Gradient Boosting classifiers with scikit-learn, "
                "achieving an 84% accuracy score and reducing quarterly attrition by 15%."
            ),
            stage="core",
            kind="message",
        ),
        Turn(role="interviewer", text="How did you handle model deployment and monitoring?", stage="core", kind="message"),
        Turn(
            role="candidate",
            text=(
                "I deployed the model as a FastAPI microservice in Docker containers on Kubernetes. "
                "We set up automated Prometheus alerts for feature drift, maintaining 99.9% uptime."
            ),
            stage="core",
            kind="message",
        ),
    ]

    card = build_scorecard(MockLLM(), rubric, turns, cfg)
    assert isinstance(card, ScoreCard)
    assert card.weighted_total >= 70.0, f"Expected strong total >= 70, got {card.weighted_total}"
    assert card.recommendation in ("Strong hire", "Hire")
    assert card.interview_completeness in ("complete", "partial")

    # Check competency evaluations
    evaluated = [c for c in card.competencies if c.status == "evaluated"]
    assert len(evaluated) > 0

    for c in evaluated:
        assert c.score is not None
        assert 1.0 <= c.score <= 5.0
        # Evidence should cite the candidate's actual words
        assert ("hr attrition" in c.evidence.lower() or "ibm" in c.evidence.lower() or
                "fastapi" in c.evidence.lower() or len(c.evidence) > 10)
        assert len(c.strengths) > 0
        assert len(c.missing_details) > 0
        assert len(c.improvement_suggestion) > 0
        # STAR example should exist
        assert "Situation:" in c.stronger_answer_example or "Action:" in c.stronger_answer_example

    # Verify key strengths and practice topics
    assert len(card.key_strengths) >= 1
    assert len(card.recommended_practice_topics) >= 1
    print("  ✓ Strong answers evaluation test passed.")


def test_evaluation_vague_answers_with_gaps():
    """Verify that vague answers receive developing scores with missing details and STAR guidance."""
    cfg = InterviewConfig(role="Junior Engineer", interview_type="mixed", max_questions=2)
    rubric = rubric_for("mixed")

    turns = [
        Turn(role="interviewer", text="Tell me about a difficult problem you solved.", stage="core", kind="message"),
        Turn(role="candidate", text="I learned and adapted to new technologies and it worked fine.", stage="core", kind="message"),
        Turn(role="interviewer", text="Can you specify which technologies?", stage="core", kind="followup"),
        Turn(role="candidate", text="I used python and fixed some bugs.", stage="core", kind="message"),
    ]

    card = build_scorecard(MockLLM(), rubric, turns, cfg)
    assert card.weighted_total < 65.0, f"Vague answers should not pass high hiring threshold, got {card.weighted_total}"

    evaluated = [c for c in card.competencies if c.status == "evaluated"]
    assert len(evaluated) > 0

    for c in evaluated:
        assert c.score <= 3.2
        # Missing details should identify the lack of technical specificity
        assert any("specific" in m.lower() or "technolog" in m.lower() or "metrics" in m.lower() for m in c.missing_details)
        assert len(c.improvement_suggestion) > 0
        # Stronger answer example should include placeholders like [Specify ...]
        assert "[" in c.stronger_answer_example and "]" in c.stronger_answer_example
    print("  ✓ Vague answers evaluation test passed.")


def test_evaluation_incomplete_interview_handling():
    """Verify that unprobed competencies are marked 'insufficient_evidence' and not penalized."""
    cfg = InterviewConfig(role="Data Engineer", interview_type="technical", max_questions=5)
    rubric = rubric_for("technical")

    # 1. Zero answers (aborted immediately)
    empty_turns = [
        Turn(role="interviewer", text="Hi, welcome to the interview.", stage="greeting", kind="message")
    ]
    card_empty = build_scorecard(MockLLM(), rubric, empty_turns, cfg)
    assert card_empty.weighted_total == 0.0
    assert card_empty.recommendation == "Insufficient Data"
    assert all(c.status == "insufficient_evidence" for c in card_empty.competencies)
    assert all(c.score is None for c in card_empty.competencies)
    assert "No competencies" in card_empty.scoring_methodology

    # 2. Only 1 short turn answered
    short_turns = [
        Turn(role="interviewer", text="Walk me through your background.", stage="greeting", kind="message"),
        Turn(role="candidate", text="I have 2 years of experience with Python.", stage="core", kind="message"),
    ]
    card_short = build_scorecard(MockLLM(), rubric, short_turns, cfg)
    # Competencies not probed must be insufficient_evidence, not 1.0 or failed
    unprobed = [c for c in card_short.competencies if c.status == "insufficient_evidence"]
    assert len(unprobed) > 0
    for u in unprobed:
        assert u.score is None
        assert "not" in u.explanation.lower() or "insufficient" in u.explanation.lower()

    # Methodology should explain normalization without arbitrary penalties
    assert "normalized" in card_short.scoring_methodology.lower() or "insufficient" in card_short.scoring_methodology.lower()
    print("  ✓ Incomplete interview handling test passed.")


def test_evaluation_anti_hallucination_guardrails():
    """Verify that the evaluation never fabricates metrics, employers, or unsupported claims."""
    cfg = InterviewConfig(role="Frontend Developer", interview_type="technical", max_questions=2)
    rubric = rubric_for("technical")

    candidate_text = "I built the customer dashboard UI with React and Tailwind CSS."
    turns = [
        Turn(role="interviewer", text="Tell me about your dashboard project.", stage="core", kind="message"),
        Turn(role="candidate", text=candidate_text, stage="core", kind="message"),
    ]

    card = build_scorecard(MockLLM(), rubric, turns, cfg)
    # Verify no fabricated metrics in evidence or STAR answer
    for c in card.competencies:
        # Evidence must only cite candidate text
        if c.status == "evaluated":
            assert "React" in c.evidence or "dashboard" in c.evidence
            # It must not invent a company name like Google or Amazon
            assert "Google" not in c.evidence and "Amazon" not in c.evidence
            # It must not invent false percentage improvements like 'increased revenue by 500%'
            assert "500%" not in c.evidence
            assert "increased revenue" not in c.evidence.lower()
    print("  ✓ Anti-hallucination guardrail test passed.")


def test_evaluation_scoring_consistency_and_weights():
    """Verify that rubric weights sum to 1.0 and identical transcripts yield consistent scores."""
    cfg = InterviewConfig(role="Site Reliability Engineer", interview_type="technical", max_questions=3)
    rubric = rubric_for("technical")

    turns = [
        Turn(role="interviewer", text="Describe a system outage you handled.", stage="core", kind="message"),
        Turn(
            role="candidate",
            text=(
                "We experienced a database failover cascade on our PostgreSQL cluster. "
                "I identified connection starvation in the PgBouncer pool, tuned max_client_conn, "
                "and recovered traffic in 12 minutes, saving our SLA."
            ),
            stage="core",
            kind="message",
        ),
    ]

    card1 = build_scorecard(MockLLM(), rubric, turns, cfg)
    card2 = build_scorecard(MockLLM(), rubric, turns, cfg)

    # Weights must sum to 1.0
    weights = sum(c.weight for c in card1.competencies)
    assert abs(weights - 1.0) < 1e-6, f"Rubric weights must sum to 1.0, got {weights}"

    # Scores must be bounded and identical
    assert 0 <= card1.weighted_total <= 100
    assert card1.weighted_total == card2.weighted_total
    assert card1.recommendation == card2.recommendation
    assert len(card1.competencies) == len(card2.competencies)
    print("  ✓ Scoring consistency and weights test passed.")


def test_evaluation_session_specific_isolation():
    """Verify that evaluations for two distinct candidate sessions never cross-pollinate."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        rubric = rubric_for("technical")

        # Session 1: ML Engineer
        cfg1 = InterviewConfig(role="ML Engineer", candidate_name="Alice", interview_type="technical")
        iid1 = store.create_interview(vars(cfg1))
        turns1 = [
            Turn(role="candidate", text="I trained an HR attrition machine learning model on IBM datasets.", stage="core"),
        ]
        card1 = build_scorecard(MockLLM(), rubric, turns1, cfg1)
        store.save_scorecard(iid1, card1.to_dict())

        # Session 2: Distributed Systems Engineer
        cfg2 = InterviewConfig(role="Systems Engineer", candidate_name="Bob", interview_type="technical")
        iid2 = store.create_interview(vars(cfg2))
        turns2 = [
            Turn(role="candidate", text="I built a Kafka distributed streaming pipeline handling 50k QPS.", stage="core"),
        ]
        card2 = build_scorecard(MockLLM(), rubric, turns2, cfg2)
        store.save_scorecard(iid2, card2.to_dict())

        # Retrieve and verify persistence & isolation
        saved1 = store.get_interview(iid1)
        saved2 = store.get_interview(iid2)

        sc1 = saved1.get("scorecard", {})
        sc2 = saved2.get("scorecard", {})

        assert "attrition" in json.dumps(sc1).lower() or "ibm" in json.dumps(sc1).lower()
        assert "kafka" not in json.dumps(sc1).lower()

        assert "kafka" in json.dumps(sc2).lower()
        assert "attrition" not in json.dumps(sc2).lower()

        print("  ✓ Session-specific evaluation isolation test passed.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_evaluation_model_failure_fallback():
    """Verify that an LLMError during scorecard generation triggers documented fallback behavior safely.
    
    Checks:
    1. The fallback produces a valid, complete ScoreCard instance without unhandled exceptions.
    2. Summary clearly documents the offline heuristic fallback and includes the error message.
    3. Competencies and scores adhere to rubric scale (1.0 to 5.0) or are safely unassessed.
    4. NO fabricated candidate evidence, employers, degrees, or metrics are introduced.
    """
    class FailingLLM:
        def chat(self, messages, *, task="question", **kwargs):
            if task == "score":
                raise LLMError("API Gateway Timeout: Provider connection reset during scorecard evaluation")
            return "Generic message"

    cfg = InterviewConfig(
        role="Backend Engineer",
        interview_type="technical",
        max_questions=3,
        candidate_name="Prakhar",
    )
    rubric = rubric_for("technical")

    candidate_text = (
        "I optimized PostgreSQL database connection pools using PgBouncer and tuned max_client_conn, "
        "recovering service latency in 12 minutes during a failover cascade."
    )
    turns = [
        Turn(role="interviewer", text="Describe a database outage you handled.", stage="core", kind="message"),
        Turn(role="candidate", text=candidate_text, stage="core", kind="message"),
    ]

    card = build_scorecard(FailingLLM(), rubric, turns, cfg)

    # 1. Valid ScoreCard returned
    assert isinstance(card, ScoreCard)
    assert 0.0 <= card.weighted_total <= 100.0
    assert card.recommendation in ("Strong hire", "Hire", "Borderline — lean no", "No hire", "Partial — Insufficient Data")
    assert card.interview_completeness in ("complete", "partial")

    # 2. Documented fallback behavior
    assert "offline evaluation generated" in card.summary.lower()
    assert "api gateway timeout" in card.summary.lower()

    # 3. Check competencies
    assert len(card.competencies) == len(rubric)
    for c in card.competencies:
        assert c.score is not None
        assert 1.0 <= c.score <= 5.0
        # 4. Anti-fabrication check: Evidence must be candidate speech prefix or N/A
        assert c.evidence in candidate_text or c.evidence == "N/A"
        # Must not fabricate fake company or fake metrics
        assert "Google" not in c.evidence and "Netflix" not in c.evidence
        assert "500%" not in c.evidence
        # Star improvement must use generic STAR template without fake metrics
        assert "Situation" in c.stronger_answer_example
        assert "Action" in c.stronger_answer_example
    print("  ✓ Model failure fallback test passed.")


def test_evaluation_report_cache_idempotency():
    """Verify that calling engine.report() multiple times reuses cached results and is idempotent.
    
    Verifies:
    1. First call invokes LLM chat for scoring exactly once and caches the result.
    2. Second call reuses the existing report without re-invoking the LLM (call count remains 1).
    3. The returned report payload is identical across both calls.
    """
    class CountingLLM(MockLLM):
        def __init__(self):
            super().__init__()
            self.score_chat_calls = 0

        def chat(self, messages, *, task="question", **kwargs):
            if task == "score":
                self.score_chat_calls += 1
            return super().chat(messages, task=task, **kwargs)

    llm = CountingLLM()
    cfg = InterviewConfig(
        role="Cloud Architect",
        interview_type="technical",
        max_questions=2,
        candidate_name="Prakhar",
    )
    engine = InterviewEngine(cfg, llm)
    engine.start()
    engine.respond("I designed a multi-region disaster recovery architecture on AWS using S3 cross-region replication.")

    assert llm.score_chat_calls == 0, "LLM scoring should not be called prior to report generation"

    # First call: Generates and caches scorecard
    report1 = engine.report()
    assert llm.score_chat_calls == 1, f"Expected 1 LLM score call, got {llm.score_chat_calls}"
    assert "scorecard" in report1
    assert "competencies" in report1["scorecard"]

    # Second call: Should reuse cache without re-invoking LLM
    report2 = engine.report()
    assert llm.score_chat_calls == 1, f"Expected LLM score calls to remain 1 after repeated report(), got {llm.score_chat_calls}"
    assert report1 == report2, "Second report call must return identical cached result"
    assert report1 is report2, "Second report call must return the exact cached object reference"

    print("  ✓ Report cache idempotency test passed.")


def test_evaluation_server_side_evidence_validation():
    """Verify that server-side validation rejects or flags unsupported/hallucinated evidence.
    
    Verifies:
    1. Direct unit verification via is_evidence_grounded for grounded and hallucinated cases.
    2. Verbatim candidate evidence is validated and preserved (evidence_verified=True).
    3. Hallucinated evidence (fabricated metrics, unmentioned technologies, fake quotes) is flagged as unsupported.
    4. Valid evaluator feedback (explanations, strengths, improvement suggestions) is preserved.
    5. No fabricated replacement evidence is introduced into the scorecard.
    """
    cand_transcript = "I built the customer dashboard UI with React and Tailwind CSS, improving load speed."

    # Direct unit tests for grounding validation
    assert is_evidence_grounded(
        "Candidate stated: 'I built the customer dashboard UI with React and Tailwind CSS'",
        cand_transcript,
    ) is True
    assert is_evidence_grounded(
        "Built dashboard UI with React and Tailwind CSS",
        cand_transcript,
    ) is True
    assert is_evidence_grounded(
        'Candidate stated: "I worked at Google building Spanner for 10 years"',
        cand_transcript,
    ) is False
    assert is_evidence_grounded(
        "Candidate managed Apache Spark clusters on AWS Kubernetes handling 100M daily events with 99.999% SLA.",
        cand_transcript,
    ) is False
    assert is_evidence_grounded("", cand_transcript) is False

    cfg = InterviewConfig(role="Frontend Developer", interview_type="technical", max_questions=2)
    rubric = rubric_for("technical")

    candidate_turn = Turn(
        role="candidate",
        text=cand_transcript,
        stage="core",
        kind="message",
    )
    candidate_turns = [candidate_turn]

    # Model returns:
    # 1. fundamentals: Grounded quote citing React & Tailwind CSS
    # 2. experience: Fabricated claim about Apache Spark & Kubernetes handling 100M events with 99.999% SLA
    # 3. depth: Fabricated quote with attribution 'Candidate stated: "I worked at Google building Spanner for 10 years"'
    mock_data = {
        "summary": "Evaluation with mixed grounded and hallucinated claims.",
        "key_strengths": ["React development"],
        "priority_improvements": ["Technical depth"],
        "recommended_practice_topics": ["Frontend architecture"],
        "competencies": [
            {
                "id": "fundamentals",
                "status": "evaluated",
                "score": 4.0,
                "explanation": "Demonstrated strong knowledge of modern frontend frameworks.",
                "evidence": "Candidate stated: 'I built the customer dashboard UI with React and Tailwind CSS'",
                "strengths": ["Clear frontend framework experience"],
                "missing_details": ["Could provide details on state management"],
                "improvement_suggestion": "Discuss component architecture trade-offs",
                "stronger_answer_example": "Situation: Building dashboard. Action: Used React. Result: Improved load speed.",
            },
            {
                "id": "experience",
                "status": "evaluated",
                "score": 4.5,
                "explanation": "Candidate articulated complex distributed systems experience.",
                "evidence": "Candidate managed Apache Spark clusters on AWS Kubernetes handling 100M daily events with 99.999% SLA.",
                "strengths": ["High-scale data processing"],
                "missing_details": ["Could detail cluster failover testing"],
                "improvement_suggestion": "Detail distributed consensus algorithms",
                "stronger_answer_example": "Situation: Large data scale. Action: Optimized Spark. Result: Met SLA.",
            },
            {
                "id": "depth",
                "status": "evaluated",
                "score": 4.2,
                "explanation": "Deep industry experience in large tech companies.",
                "evidence": 'Candidate stated: "I worked at Google building Spanner for 10 years as a Principal Engineer"',
                "strengths": ["Industry pedigree"],
                "missing_details": ["Detail specific database consensus protocols"],
                "improvement_suggestion": "Explain distributed transaction concurrency",
                "stronger_answer_example": "Situation: At Google. Action: Built Spanner. Result: Scaled global DB.",
            },
        ],
    }

    card = validate_and_build_scorecard(mock_data, rubric, candidate_turns, cfg)
    by_id = {c.id: c for c in card.competencies}

    # 1. Grounded evidence must be verified
    c_fund = by_id["fundamentals"]
    assert c_fund.evidence_verified is True
    assert "react" in c_fund.evidence.lower()
    assert "[Unsupported" not in c_fund.evidence

    # 2. Fabricated claim about Spark / AWS Kubernetes must be flagged as unsupported
    c_exp = by_id["experience"]
    assert c_exp.evidence_verified is False
    assert "[Unsupported: quote or claim not found in candidate transcript]" == c_exp.evidence
    # 3. Valid feedback is preserved
    assert len(c_exp.strengths) > 0
    assert "distributed systems" in c_exp.explanation.lower()
    assert len(c_exp.improvement_suggestion) > 0
    # 4. No fabricated replacement evidence introduced
    assert "spark" not in c_exp.evidence.lower()
    assert "99.999%" not in c_exp.evidence

    # 5. Fabricated quote about Google Spanner must be flagged as unsupported
    c_depth = by_id["depth"]
    assert c_depth.evidence_verified is False
    assert "[Unsupported: quote or claim not found in candidate transcript]" == c_depth.evidence
    assert "google" not in c_depth.evidence.lower()
    assert "spanner" not in c_depth.evidence.lower()
    # Valid feedback preserved
    assert len(c_depth.strengths) > 0
    assert len(c_depth.improvement_suggestion) > 0

    print("  ✓ Server-side evidence validation test passed.")


if __name__ == "__main__":
    tests = [
        test_evaluation_strong_answers_with_evidence,
        test_evaluation_vague_answers_with_gaps,
        test_evaluation_incomplete_interview_handling,
        test_evaluation_anti_hallucination_guardrails,
        test_evaluation_scoring_consistency_and_weights,
        test_evaluation_session_specific_isolation,
        test_evaluation_model_failure_fallback,
        test_evaluation_report_cache_idempotency,
        test_evaluation_server_side_evidence_validation,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase 4 evaluation engine tests passed successfully!")
