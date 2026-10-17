"""A small, extensible question bank.

In production you would generate or retrieve questions per role/JD (and that is
exactly where the LLM earns its keep). The bank keeps the prototype
deterministic and shows the shape the engine expects: every question maps to a
rubric competency and carries a follow-up hint.
"""
from dataclasses import dataclass


@dataclass
class Question:
    id: str
    competency: str
    text: str
    followup_hint: str
    evidence: list | None = None


BEHAVIORAL = [
    Question("b1", "ownership",
             "Walk me through a project you owned end to end. What was your specific contribution, and what was the outcome?",
             "Ask what they would do differently and how they measured success."),
    Question("b2", "problem_solving",
             "Tell me about a time you faced a hard problem with no obvious solution. How did you approach it?",
             "Probe the options they considered and why they chose one."),
    Question("b3", "collaboration",
             "Describe a situation where you disagreed with a teammate or stakeholder. How did you handle it?",
             "Ask what the other person's perspective was and what changed."),
    Question("b4", "communication",
             "Explain something complex you worked on as if I were non-technical.",
             "Ask them to compress it into two sentences."),
    Question("b5", "motivation",
             "Why this role, and what are you looking for in your next move?",
             "Probe what would make them leave within a year."),
]

TECHNICAL = [
    Question("t1", "fundamentals",
             "Pick a core concept in your field and explain it from first principles.",
             "Ask for an example where it breaks down."),
    Question("t2", "problem_solving",
             "How would you design a system to handle a problem relevant to this role? Talk me through your approach.",
             "Probe scaling, failure modes, and trade-offs."),
    Question("t3", "depth",
             "Tell me about the hardest bug you have debugged. What made it hard?",
             "Ask how they would prevent it from recurring."),
    Question("t4", "experience",
             "Which past project best shows your technical ability, and what did you personally build?",
             "Ask for the measurable impact."),
    Question("t5", "communication",
             "How do you decide between two technically valid approaches?",
             "Ask for a real example and the outcome."),
]

MIXED = [
    Question("m1", "communication",
             "Give me a two-minute overview of your background and what you are best at.",
             "Ask them to go deeper on the strongest point."),
    Question("m2", "problem_solving",
             "Describe a problem you solved that you are proud of. How did you break it down?",
             "Probe the alternatives they rejected."),
    Question("m3", "fundamentals",
             "What is an area you know deeply, and how would you explain its fundamentals to a beginner?",
             "Ask a probing follow-up to test real depth."),
    Question("m4", "ownership",
             "Tell me about a time you had to deliver under a tight deadline. What did you do?",
             "Ask what they cut and why."),
    Question("m5", "motivation",
             "What kind of work energises you, and why this role?",
             "Probe alignment with the role."),
    Question("m6", "collaboration",
             "How do you handle feedback you disagree with?",
             "Ask for a concrete instance."),
]


def questions_for(interview_type: str):
    return {"behavioral": BEHAVIORAL, "technical": TECHNICAL, "mixed": MIXED}[interview_type]


def generate_personalized_questions(config, llm, store=None, interview_id=None) -> list[Question]:
    """Generate interview questions tailored to the candidate's resume and job description.

    Strict grounding rules:
    - Questions MUST be relevant to the target job requirements.
    - Questions MUST map directly to the candidate's actual experience from their resume.
    - Uses RAG semantic retrieval when store and interview_id are present to find the most
      salient evidence chunks from the candidate's uploaded background.
    - The model MUST NOT invent experience, employers, degrees, achievements, or metrics.
    - If generation fails or returns empty, cleanly falls back to the static question bank.
    """
    import json
    resume = (config.resume_text or "").strip()
    jd = (config.job_description or "").strip()

    # If neither is provided, return default banked questions
    if not resume and not jd:
        return list(questions_for(config.interview_type))

    # Retrieve specific evidence chunks via RAG if available
    retrieved_resume_chunks = []
    retrieved_jd_chunks = []
    if store is not None and interview_id:
        try:
            from .rag import format_evidence_block, retrieve_evidence
            retrieved_resume_chunks = retrieve_evidence(
                store, interview_id,
                f"{config.role} past experience achievements technical skills",
                doc_type="resume", top_k=4,
            )
            retrieved_jd_chunks = retrieve_evidence(
                store, interview_id,
                f"{config.role} responsibilities requirements qualifications",
                doc_type="job_description", top_k=4,
            )
        except Exception:
            pass

    rag_resume_text = (
        format_evidence_block(retrieved_resume_chunks)
        if retrieved_resume_chunks else (resume[:3000] if resume else "(Not provided)")
    )
    rag_jd_text = (
        format_evidence_block(retrieved_jd_chunks)
        if retrieved_jd_chunks else (jd[:2000] if jd else "(Not provided)")
    )

    prompt = (
        f"You are an expert hiring bar raiser creating interview questions for a "
        f"{config.level}-level {config.role} position ({config.interview_type} focus).\n\n"
        f"--- TARGET JOB DESCRIPTION (RETRIEVED REQUIREMENTS) ---\n{rag_jd_text}\n\n"
        f"--- CANDIDATE RESUME (RETRIEVED VERIFIED EVIDENCE) ---\n{rag_resume_text}\n\n"
        "--- STRICT GROUNDING INSTRUCTIONS ---\n"
        "1. Never invent or hallucinate past employers, project names, metrics, achievements, or skills absent from the candidate's verified resume.\n"
        "2. Only ask about roles, skills, and projects explicitly stated in the retrieved candidate background.\n"
        "3. Map each question to a competency: ownership, problem_solving, collaboration, communication, fundamentals, depth, experience, or motivation.\n"
        "4. Include a concise follow-up hint for the interviewer for each question.\n"
        f"5. Generate between 3 and {config.max_questions} high-signal questions.\n\n"
        'Return JSON format only, matching this structure:\n'
        '{\n'
        '  "questions": [\n'
        '    {\n'
        '      "id": "p1",\n'
        '      "competency": "experience",\n'
        '      "text": "Specific question grounded in resume and job description...",\n'
        '      "followup_hint": "Follow-up guidance for probing..."\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    messages = [
        {"role": "system", "content": "You are a precise technical interviewer. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = llm.chat(messages, task="generate_questions", temperature=0.3, max_tokens=1500, json_mode=True)
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            start, end = raw.find("{"), raw.rfind("}")
            if 0 <= start < end:
                data = json.loads(raw[start : end + 1])

        if data and isinstance(data.get("questions"), list) and len(data["questions"]) > 0:
            result = []
            combined_evidence = (retrieved_resume_chunks + retrieved_jd_chunks)[:3]
            for i, q in enumerate(data["questions"]):
                if not isinstance(q, dict):
                    continue
                qid = q.get("id") or f"p{i+1}"
                comp = q.get("competency") or "problem_solving"
                qtext = (q.get("text") or "").strip()
                hint = (q.get("followup_hint") or "").strip()
                if qtext:
                    result.append(Question(
                        id=str(qid),
                        competency=str(comp),
                        text=qtext,
                        followup_hint=hint,
                        evidence=combined_evidence if combined_evidence else None,
                    ))
            if result:
                return result
    except Exception:
        pass

    # Graceful fallback to default bank if LLM generation fails or returns empty
    return list(questions_for(config.interview_type))
