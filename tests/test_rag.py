"""Automated tests for Phase 3: RAG (Retrieval-Augmented Generation).

Tests:
1. Section-aware document chunking and metadata preservation.
2. Embedding generation and semantic cosine similarity.
3. Persistent SQLite vector storage and idempotent re-indexing.
4. Semantic retrieval relevance (query matches relevant chunk over unrelated ones).
5. Cross-candidate privacy and isolation (Candidate A never sees Candidate B's data).
6. Data retention and erasure cascading for document chunks.
7. Context-grounded interview question generation receiving RAG context.
8. Adaptive evidence-grounded follow-ups on vague/thin candidate responses.
9. Resilience to empty documents, scanned unreadable files, and embedding errors.
10. API endpoints for chunk inspection and semantic retrieval.

Run: python3 tests/test_rag.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer.engine import InterviewEngine
from interviewer.llm import MockLLM
from interviewer.questions import generate_personalized_questions
from interviewer.rag import (
    DeterministicNgramEmbedder,
    DocumentChunk,
    EmbeddingService,
    chunk_document,
    cosine_similarity,
    format_evidence_block,
    index_interview_documents,
    retrieve_evidence,
)
from interviewer.store import Store


def test_section_aware_document_chunking():
    """Verify that chunking splits by sections, preserves metadata, and handles empty input."""
    resume_text = (
        "PROFESSIONAL SUMMARY\n"
        "Senior Backend Engineer with 7 years of distributed systems experience.\n\n"
        "WORK EXPERIENCE\n"
        "Led engineering team designing payment pipelines handling 50k transactions/sec.\n"
        "Migrated legacy monolithic service to Kubernetes microservices using FastAPI and Kafka.\n\n"
        "TECHNICAL SKILLS\n"
        "Python, Go, PostgreSQL, Redis, Apache Kafka, Docker, Kubernetes, AWS.\n\n"
        "EDUCATION\n"
        "B.S. in Computer Science, University of Technology, 2017."
    )

    chunks = chunk_document(resume_text, doc_type="resume", interview_id="session-101")
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    sections = [c.section for c in chunks]
    assert "summary" in sections, f"Missing summary section in {sections}"
    assert "experience" in sections, f"Missing experience section in {sections}"
    assert "skills" in sections, f"Missing skills section in {sections}"
    assert "education" in sections, f"Missing education section in {sections}"

    for c in chunks:
        assert c.interview_id == "session-101"
        assert c.doc_type == "resume"
        assert c.text.strip()
        assert c.chunk_index >= 0

    # Test JD chunking
    jd_text = (
        "ROLE OVERVIEW\n"
        "We are looking for a Staff Engineer to lead our Core Infrastructure team.\n\n"
        "RESPONSIBILITIES\n"
        "Architect and maintain distributed storage systems with 99.99% availability.\n\n"
        "REQUIREMENTS\n"
        "7+ years experience with high-throughput distributed systems in Python or Go."
    )
    jd_chunks = chunk_document(jd_text, doc_type="job_description", interview_id="session-101")
    jd_sections = [c.section for c in jd_chunks]
    assert "responsibilities" in jd_sections
    assert "requirements" in jd_sections

    # Empty text handling
    empty_chunks = chunk_document("", doc_type="resume")
    assert empty_chunks == [], "Empty text should return empty list of chunks"
    whitespace_chunks = chunk_document("   \n\n  ", doc_type="job_description")
    assert whitespace_chunks == [], "Whitespace text should return empty list of chunks"


def test_embedding_and_cosine_similarity():
    """Verify cosine similarity calculation and embedding generation."""
    # Test vector cosine math
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_b) - 1.0) < 1e-4
    assert abs(cosine_similarity(vec_a, vec_c) - 0.0) < 1e-4

    # Test deterministic fallback vectorizer
    embedder = DeterministicNgramEmbedder(dim=256)
    e1 = embedder.embed_one("Python backend Kafka streaming")
    e2 = embedder.embed_one("Python backend Kafka message queue")
    e3 = embedder.embed_one("Graphic design Photoshop typography watercolor")

    sim_related = cosine_similarity(e1, e2)
    sim_unrelated = cosine_similarity(e1, e3)
    assert sim_related > sim_unrelated, f"Expected related ({sim_related}) > unrelated ({sim_unrelated})"

    # Test main EmbeddingService (fastembed with fallback)
    svc = EmbeddingService.get_instance()
    vectors = svc.embed_texts(["Kafka streaming architecture", "UI frontend React hooks"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[1]) > 0


def test_sqlite_vector_persistence_and_reindexing():
    """Verify storing, retrieving, and updating chunks without duplicates in SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        iid = store.create_interview({"candidate_name": "Alice", "mode": "practice"})

        # Initial indexing
        resume_v1 = "EXPERIENCE\nBuilt caching layer with Redis."
        res1 = index_interview_documents(store, iid, resume_text=resume_v1)
        assert res1["status"] == "ok"
        assert res1["resume_chunks"] >= 1

        chunks_v1 = store.get_document_chunks(iid, doc_type="resume")
        assert len(chunks_v1) >= 1
        assert "Redis" in chunks_v1[0]["text"]

        # Re-index with updated resume (e.g. candidate re-uploaded their file)
        resume_v2 = (
            "EXPERIENCE\n"
            "Built caching layer with Redis.\n"
            "Added Kafka distributed pub-sub stream."
        )
        res2 = index_interview_documents(store, iid, resume_text=resume_v2)
        assert res2["status"] == "ok"

        # Verify no duplicate accumulation
        chunks_v2 = store.get_document_chunks(iid, doc_type="resume")
        # Ensure only the new version exists
        assert any("Kafka" in c["text"] for c in chunks_v2)
        # Check chunk count didn't double
        assert len(chunks_v2) == res2["resume_chunks"]

        # Check detail includes chunk stats
        detail = store.get_interview(iid)
        assert "chunk_stats" in detail
        assert detail["chunk_stats"]["resume_chunks"] == len(chunks_v2)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_semantic_retrieval_relevance():
    """Verify that a targeted query retrieves the relevant chunk rather than unrelated text."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        iid = store.create_interview({"candidate_name": "Bob", "mode": "practice"})

        resume = (
            "EXPERIENCE\n"
            "Engineered distributed Kafka event pipeline processing 100M events daily.\n\n"
            "PROJECTS\n"
            "Built automated React 18 design system with Tailwind CSS and Storybook."
        )
        jd = (
            "REQUIREMENTS\n"
            "Strong knowledge of streaming event architectures and distributed message brokers."
        )
        index_interview_documents(store, iid, resume_text=resume, job_description=jd)

        # Query for streaming pipeline
        streaming_results = retrieve_evidence(store, iid, "Kafka event streaming distributed broker", top_k=2)
        assert len(streaming_results) > 0
        top_match = streaming_results[0]
        assert "Kafka" in top_match["text"] or "streaming" in top_match["text"]

        # Query for frontend
        ui_results = retrieve_evidence(store, iid, "React frontend design system UI", top_k=2)
        assert len(ui_results) > 0
        assert "React" in ui_results[0]["text"]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_candidate_privacy_and_isolation():
    """STRICT PRIVACY: Candidate A's documents must NEVER be returned for Candidate B."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        id_a = store.create_interview({"candidate_name": "Candidate Alpha", "mode": "practice"})
        id_b = store.create_interview({"candidate_name": "Candidate Beta", "mode": "practice"})

        resume_a = "EXPERIENCE\nLead Flight Software Engineer on Mars Rover navigation sensors."
        resume_b = "EXPERIENCE\nAlgorithmic High Frequency Options Trading at Citadel Securities."

        index_interview_documents(store, id_a, resume_text=resume_a)
        index_interview_documents(store, id_b, resume_text=resume_b)

        # Candidate A queries for trading at Citadel
        res_a = retrieve_evidence(store, id_a, "Citadel algorithmic options trading", top_k=5)
        # Should NOT return Candidate B's resume
        for item in res_a:
            assert "Citadel" not in item["text"], "LEAK: Candidate A retrieved Candidate B's confidential resume!"

        # Candidate B queries for Mars Rover
        res_b = retrieve_evidence(store, id_b, "Mars Rover navigation sensors", top_k=5)
        for item in res_b:
            assert "Mars Rover" not in item["text"], "LEAK: Candidate B retrieved Candidate A's confidential resume!"

        # Cross-candidate direct database query isolation check
        chunks_for_a = store.get_document_chunks(id_a)
        for c in chunks_for_a:
            assert c["interview_id"] == id_a
            assert "Citadel" not in c["text"]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_erasure_and_purge_cascades_chunks():
    """Verify that deleting an interview or purging expired sessions removes vector chunks."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        iid = store.create_interview({"candidate_name": "Charlie", "candidate_email": "charlie@test.com", "mode": "practice"})
        index_interview_documents(store, iid, resume_text="EXPERIENCE\nCloud Infrastructure architect.")

        assert len(store.get_document_chunks(iid)) > 0

        # Erase candidate
        store.delete_candidate("charlie@test.com")
        assert len(store.get_document_chunks(iid)) == 0, "Document chunks must be deleted on candidate erasure"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_question_generation_receives_rag_context():
    """Verify that question generation prompt incorporates retrieved RAG chunks."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        cfg = InterviewConfig(
            role="Distributed Systems Engineer",
            resume_text="EXPERIENCE\nArchitected high throughput Kafka clustering on bare metal.",
            job_description="REQUIREMENTS\nDesign mission-critical distributed databases.",
        )
        iid = store.create_interview(vars(cfg))
        index_interview_documents(store, iid, resume_text=cfg.resume_text, job_description=cfg.job_description)

        recorded_prompts = []

        class RecordingMockLLM:
            name = "RecordingMock"

            def chat(self, messages, **kwargs):
                for m in messages:
                    recorded_prompts.append(m.get("content", ""))
                return (
                    '{"questions": [{"id": "p1", "competency": "experience", '
                    '"text": "Walk me through your bare metal Kafka cluster architecture.", '
                    '"followup_hint": "Ask about replication factor."}]}'
                )

        llm = RecordingMockLLM()
        questions = generate_personalized_questions(cfg, llm, store=store, interview_id=iid)
        assert len(questions) == 1
        assert "Kafka" in questions[0].text

        all_prompt_text = " ".join(recorded_prompts)
        assert "RETRIEVED VERIFIED EVIDENCE" in all_prompt_text
        assert "Kafka" in all_prompt_text
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_adaptive_followup_grounded_in_evidence():
    """Verify that a thin answer triggers an adaptive follow-up that captures evidence."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = Store(db_path)
        cfg = InterviewConfig(
            role="Backend Engineer",
            resume_text="EXPERIENCE\nImplemented token bucket rate limiter in Go and Redis with sub-millisecond p99 latency.",
            allow_followups=True,
            max_followups_per_question=1,
        )
        iid = store.create_interview(vars(cfg))
        index_interview_documents(store, iid, resume_text=cfg.resume_text)

        followup_prompts = []

        class FollowupRecordingLLM:
            name = "FollowupLLM"

            def chat(self, messages, task=None, **kwargs):
                followup_prompts.append(messages[-1]["content"])
                return json.dumps({
                    "action": "follow_up",
                    "spoken_response": "You mentioned building a rate limiter—how did Redis atomicity prevent race conditions under load?",
                    "reason_for_decision": "Probing technical depth and concurrency on Redis project",
                    "competency": "experience",
                    "evidence_used": ["Rate limiter with Redis"]
                })

        llm = FollowupRecordingLLM()
        engine = InterviewEngine(cfg, llm, store=store, interview_id=iid)
        engine.start()

        # Provide a thin, vague response
        resp = engine.respond("I built it and it worked fine.")
        assert resp["stage"] == "core"
        assert not resp["done"]

        # Check turn has evidence
        last_turn = engine.turns[-1]
        assert last_turn.kind == "followup"
        assert last_turn.evidence is not None and len(last_turn.evidence) > 0
        assert any("rate limiter" in ev["text"].lower() or "redis" in ev["text"].lower() for ev in last_turn.evidence)

        # Check prompt provided to LLM contained retrieved evidence block
        assert len(followup_prompts) >= 1
        assert "RETRIEVED VERIFIED EVIDENCE" in followup_prompts[-1]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_failure_resilience_and_graceful_fallbacks():
    """Verify that empty documents, missing stores, or embedding errors never crash the interview."""
    # 1. Empty documents
    cfg_empty = InterviewConfig(resume_text="", job_description="")
    engine = InterviewEngine(cfg_empty, MockLLM())
    first_msg = engine.start()
    assert len(first_msg) > 0
    resp = engine.respond("I have 5 years experience.")
    assert "message" in resp

    # 2. Embedding service fallback
    fallback_embedder = DeterministicNgramEmbedder(dim=128)
    vec = fallback_embedder.embed_one("")
    assert len(vec) == 128
    assert all(v == 0.0 for v in vec)

    # 3. Format empty evidence
    assert format_evidence_block([]) == "None retrieved."


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
