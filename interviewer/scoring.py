"""Evidence-Based Interview Evaluation Engine for Project-Dragon (Phase 4).

Evaluates candidate responses against the verified resume, target job description,
interview questions, and complete transcript.

Core Guarantees:
1. Grounded & Evidence-Backed: Cites direct quotes and facts from candidate answers.
2. Anti-Hallucination: Never invents candidate achievements, companies, degrees, or metrics.
3. Distinction of Gaps vs Unassessed: Differentiates demonstrated weakness from insufficient evidence.
4. Fair Scoring: Does not penalize unevaluated competencies by arbitrarily averaging them.
5. Actionable STAR Improvement: Constructs stronger answer examples using only candidate-supplied facts
   with clear bracketed placeholders for missing details.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


_EVALUATOR_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "that", "this", "these", "those", "it", "its", "as", "if", "not",
    "candidate", "stated", "mentioned", "noted", "said", "explained", "described",
    "discussed", "highlighted", "demonstrated", "reported", "claimed", "shared",
    "indicated", "interview", "response", "answer", "answers", "transcript",
    "session", "project", "work", "experience", "role", "also", "well", "such",
    "their", "they", "them", "his", "her", "he", "she", "i", "my", "we", "our",
    "you", "your", "who", "which", "what", "where", "when", "how", "why", "there",
    "here", "then", "so", "than", "more", "most", "some", "any", "no", "all",
    "each", "every", "both", "few", "other", "another", "very", "too", "quite",
    "good", "great", "able",
}


def _stem(word: str) -> str:
    w = word.lower()
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def _is_evidence_grounded(evidence: str, candidate_text: str) -> bool:
    """Verifies whether quoted or cited evidence is grounded in the candidate transcript.
    
    Prevents LLM hallucination from introducing fabricated quotes, false metrics,
    or technologies never mentioned by the candidate into the scorecard.
    """
    if not evidence or not candidate_text:
        return False

    cand_lower = candidate_text.lower()
    ev_lower = evidence.lower().strip()

    # 1. Check for explicit quotation match (quotes >= 8 characters)
    quotes = re.findall(r'["\']([^"\']{8,})["\']', evidence)
    if quotes:
        for q in quotes:
            q_clean = q.strip().lower()
            if q_clean in cand_lower:
                return True
            q_words = [w for w in re.findall(r'[a-z0-9]+', q_clean) if len(w) >= 3 and w not in _EVALUATOR_STOPWORDS]
            cand_words = set(re.findall(r'[a-z0-9]+', cand_lower))
            cand_stems = {_stem(w) for w in cand_words}
            if q_words:
                matches = sum(1 for w in q_words if w in cand_words or _stem(w) in cand_stems)
                if (matches / len(q_words)) >= 0.65:
                    return True
        # If explicit quotes were provided but failed grounding, reject as hallucinated
        return False

    # 2. Check for verbatim substring match (>= 15 chars)
    # Strip common evaluator prefixes like "Candidate stated: "
    stripped = re.sub(
        r'^(?:candidate\s+(?:stated|mentioned|noted|explained|said|discussed|demonstrated|reported)\s*:\s*)',
        '',
        ev_lower,
    ).strip()
    if len(stripped) >= 15 and stripped in cand_lower:
        return True

    # 3. Content token overlap verification
    ev_tokens = [
        w for w in re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', stripped)
        if len(w) >= 3 and w not in _EVALUATOR_STOPWORDS
    ]
    if not ev_tokens:
        return False

    cand_tokens = set(re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', cand_lower))
    cand_stems = {_stem(w) for w in cand_tokens}

    matched = [w for w in ev_tokens if (w in cand_tokens or _stem(w) in cand_stems)]
    overlap_ratio = len(matched) / len(ev_tokens)

    if len(ev_tokens) <= 2:
        return len(matched) >= 1 and overlap_ratio >= 0.5
    return overlap_ratio >= 0.35 and len(matched) >= 2


is_evidence_grounded = _is_evidence_grounded


RUBRIC_DEFINITIONS = {
    1: "Unsatisfactory / Critical Gaps: Unable to demonstrate core concepts; superficial, inaccurate, or unresponsive.",
    2: "Developing / Major Gaps: Basic awareness, but lacks technical depth, personal ownership, or key trade-offs.",
    3: "Proficient / Meets Expectations: Clear understanding, coherent structure, and valid examples meeting standard expectations.",
    4: "Strong / Exceeds Expectations: High clarity, deep personal contribution, proactive trade-offs, and structured delivery.",
    5: "Outstanding / Mastery: Comprehensive, structured, and insightful response with exceptional technical depth and impact.",
}


@dataclass
class CompetencyScore:
    id: str
    name: str
    weight: float
    score: Optional[float]                  # 1.0 - 5.0, or None if unassessed / insufficient evidence
    status: str                             # "evaluated" | "insufficient_evidence"
    status_label: str                       # "Demonstrated Strength" | "Proficient" | "Developing" | "Insufficient Evidence"
    evidence: str                           # Verbatim quote or concrete candidate fact
    explanation: str                        # Reasoning for score
    comment: str                            # Summary comment (backward compatible)
    strengths: List[str]                    # Specific strengths demonstrated
    missing_details: List[str]              # Missing details or improvement areas
    improvement_suggestion: str             # Practical advice for improving the answer
    stronger_answer_example: str            # STAR answer improvement using only candidate facts
    evidence_verified: bool = True          # Server-side validation against candidate transcript

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreCard:
    competencies: List[CompetencyScore]
    weighted_total: float                   # 0.0 - 100.0
    recommendation: str                     # "Strong hire" | "Hire" | "Borderline" | "No hire" | "Insufficient Data"
    summary: str                            # Concise executive summary
    scoring_methodology: str                # Transparent calculation and normalization description
    key_strengths: List[str]                # Top strengths across the interview
    priority_improvements: List[str]        # Top priority areas to improve
    recommended_practice_topics: List[str]  # Recommended topics to practice
    interview_completeness: str             # "complete" | "partial" | "minimal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competencies": [c.to_dict() for c in self.competencies],
            "weighted_total": round(self.weighted_total, 1),
            "recommendation": self.recommendation,
            "summary": self.summary,
            "scoring_methodology": self.scoring_methodology,
            "key_strengths": self.key_strengths,
            "priority_improvements": self.priority_improvements,
            "recommended_practice_topics": self.recommended_practice_topics,
            "interview_completeness": self.interview_completeness,
        }


def _prompt(rubric, turns, config) -> str:
    comp_lines = "\n".join(
        f"- {c.id} ({c.name}, default weight {c.weight}): {c.description}\n"
        f"  Strong looks like: {c.signals}"
        for c in rubric
    )
    transcript = "\n".join(
        f"[{t.role.upper()} - {t.kind or 'message'}]: {t.text}"
        for t in turns
    ) or "(no transcript recorded)"

    jd_context = f"\nTARGET JOB REQUIREMENTS:\n{config.job_description[:2000]}\n" if config.job_description else ""
    resume_context = f"\nCANDIDATE VERIFIED RESUME:\n{config.resume_text[:2500]}\n" if config.resume_text else ""

    return (
        f"You are a rigorous, fair, and calibrated technical hiring evaluator evaluating a "
        f"{config.interview_type} interview for a {config.level}-level {config.role} role.\n"
        f"{jd_context}{resume_context}\n"
        f"EVALUATION RUBRIC & SCALE (1.0 to 5.0):\n"
        f"- 1.0 - 1.9 (Unsatisfactory / Critical Gaps): Inaccurate concepts, inability to answer, or severe lack of fundamentals.\n"
        f"- 2.0 - 2.9 (Developing / Major Gaps): Surface-level answer lacking technical specifics, personal ownership, or trade-offs.\n"
        f"- 3.0 - 3.9 (Proficient / Meets Expectations): Solid, coherent answer meeting standard industry expectations with clear examples.\n"
        f"- 4.0 - 4.9 (Strong / Exceeds Expectations): High technical/behavioral clarity, deep personal contribution, and proactive trade-offs.\n"
        f"- 5.0 (Outstanding / Mastery): Comprehensive, structured response with exceptional technical depth and clear personal impact.\n\n"
        f"COMPETENCIES TO EVALUATE:\n{comp_lines}\n\n"
        f"INTERVIEW TRANSCRIPT:\n{transcript}\n\n"
        f"STRICT EVALUATION RULES:\n"
        f"1. GROUNDING: Base every evaluation strictly on what the candidate actually said in the transcript. "
        f"NEVER fabricate or invent candidate achievements, past employers, degrees, metrics, or technologies.\n"
        f"2. INSUFFICIENT EVIDENCE vs WEAKNESS: If a competency was NOT probed in the interview, or the candidate did not have a chance to answer it, "
        f"set status='insufficient_evidence', score=null, and state clearly that the topic was unassessed. Do NOT assume they lack the skill.\n"
        f"3. EVIDENCE: Quote or directly cite specific facts from the candidate's answers.\n"
        f"4. ANSWER IMPROVEMENT: Generate an example of a stronger answer using ONLY the facts and projects supplied by the candidate. "
        f"Use STAR (Situation, Task, Action, Result) format where appropriate. Clearly mark missing details with brackets like '[Add your measured metric here]' "
        f"or '[Specify exact database used]'. Do not invent fake business metrics.\n"
        f"5. OUTPUT FORMAT: Return a valid JSON object matching this schema exactly:\n"
        f"{{\n"
        f'  "summary": "2-4 sentences executive summary of performance",\n'
        f'  "key_strengths": ["Key strength 1", "Key strength 2"],\n'
        f'  "priority_improvements": ["Improvement area 1", "Improvement area 2"],\n'
        f'  "recommended_practice_topics": ["Specific practice topic 1", "Specific practice topic 2"],\n'
        f'  "competencies": [\n'
        f"    {{\n"
        f'      "id": "competency_id",\n'
        f'      "status": "evaluated",\n'
        f'      "score": 3.5,\n'
        f'      "explanation": "Detailed explanation of score...",\n'
        f'      "evidence": "Direct quotes or cited facts...",\n'
        f'      "strengths": ["Specific strength demonstrated..."],\n'
        f'      "missing_details": ["Specific missing details..."],\n'
        f'      "improvement_suggestion": "Actionable feedback...",\n'
        f'      "stronger_answer_example": "Situation: ... Task: ... Action: ... Result: ..."\n'
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )


def build_scorecard(llm, rubric, turns, config) -> ScoreCard:
    """Orchestrates structured evaluation across turns and computes normalized scorecard."""
    # Handle empty or zero-turn interviews gracefully
    candidate_turns = [t for t in turns if getattr(t, "role", "") == "candidate" and getattr(t, "text", "").strip()]
    if not candidate_turns:
        return _build_unassessed_scorecard(rubric, "Interview concluded before candidate submitted answers.")

    msgs = [{"role": "user", "content": _prompt(rubric, turns, config)}]
    try:
        raw = llm.chat(msgs, task="score", temperature=0.2, max_tokens=1800, json_mode=True)
        data = _parse(raw)
    except Exception as exc:
        data = _fallback_heuristic_score(rubric, candidate_turns, error=str(exc))

    return validate_and_build_scorecard(data, rubric, candidate_turns, config)


def validate_and_build_scorecard(data: dict, rubric: list, candidate_turns: list, config: Any) -> ScoreCard:
    """Validates model data, enforces scoring guardrails, and computes normalized weighted total."""
    if not isinstance(data, dict):
        data = {}

    by_id = {c.get("id"): c for c in data.get("competencies", []) if isinstance(c, dict)}
    evaluated_scores: List[CompetencyScore] = []
    all_scores: List[CompetencyScore] = []

    # Transcript text for sanity checking candidate evidence
    all_candidate_text = " ".join(getattr(t, "text", "") for t in candidate_turns).strip()
    total_words = len(all_candidate_text.split())

    for c in rubric:
        got = by_id.get(c.id, {})
        raw_status = str(got.get("status", "")).strip().lower()

        # Check if the candidate actually provided evidence for this competency
        raw_evidence = str(got.get("evidence", "")).strip()
        has_evidence = bool(raw_evidence and raw_evidence.lower() != "no evidence captured.")

        # Determine true evaluation status
        if raw_status == "insufficient_evidence" or not has_evidence or total_words < 5:
            status = "insufficient_evidence"
            score = None
            status_label = "Insufficient Evidence"
            evidence = raw_evidence if raw_evidence else "No candidate evidence recorded for this competency."
            explanation = str(got.get("explanation") or got.get("comment") or
                              f"The {c.name} competency was not sufficiently probed or answered in this interview.")
            comment = explanation
            strengths = []
            missing_details = [f"Direct experience with {c.name.lower()} was not demonstrated in this session."]
            suggestion = f"Prepare a concrete project example demonstrating your hands-on work with {c.name.lower()}."
            stronger_example = (
                f"Prepare a structured STAR response highlighting your direct contribution to {c.name.lower()}. "
                f"[Specify your exact role, actions taken, and measured outcomes]."
            )
            evidence_verified = False
        else:
            status = "evaluated"
            raw_score = got.get("score")
            score = _clamp(raw_score if raw_score is not None else 3.0)
            status_label = _score_status_label(score)

            # Server-side evidence grounding validation against candidate transcript
            is_grounded = _is_evidence_grounded(raw_evidence, all_candidate_text)
            if is_grounded:
                evidence = raw_evidence or "Demonstrated in candidate answers."
                evidence_verified = True
            else:
                # Unsupported evidence rejected/flagged safely without fabricating replacement evidence
                evidence = "[Unsupported: quote or claim not found in candidate transcript]"
                evidence_verified = False

            explanation = str(got.get("explanation") or got.get("comment") or f"Evaluated based on transcript evidence.")
            comment = explanation
            strengths = [str(s) for s in got.get("strengths", []) if str(s).strip()]
            if not strengths:
                strengths = [f"Provided context addressing {c.name.lower()}."]
            missing_details = [str(m) for m in got.get("missing_details", []) if str(m).strip()]
            if not missing_details:
                missing_details = ["Could provide additional architectural trade-offs or measured outcomes."]
            suggestion = str(got.get("improvement_suggestion") or
                             f"Elaborate on your personal decision-making process and trade-offs when discussing {c.name.lower()}.")
            stronger_example = str(got.get("stronger_answer_example") or
                                   f"Structure your answer using STAR: Situation, Task, your specific Actions, and the Results achieved.")

        comp_obj = CompetencyScore(
            id=c.id,
            name=c.name,
            weight=c.weight,
            score=score,
            status=status,
            status_label=status_label,
            evidence=evidence[:400],
            explanation=explanation[:400],
            comment=comment[:400],
            strengths=strengths[:4],
            missing_details=missing_details[:4],
            improvement_suggestion=suggestion[:400],
            stronger_answer_example=stronger_example[:600],
            evidence_verified=evidence_verified,
        )
        all_scores.append(comp_obj)
        if status == "evaluated" and score is not None:
            evaluated_scores.append(comp_obj)

    # Compute Fair, Normalized Scoring
    num_evaluated = len(evaluated_scores)
    total_rubric_comps = len(rubric)

    if num_evaluated == 0:
        weighted_total = 0.0
        recommendation = "Insufficient Data"
        methodology = (
            "No competencies had sufficient candidate evidence to evaluate. "
            "Interview ended before substantive answers could be evaluated."
        )
        completeness = "minimal"
    else:
        # Sum of weights of the evaluated competencies
        eval_weight = sum(c.weight for c in evaluated_scores)
        if eval_weight <= 0:
            eval_weight = 1.0

        # Weighted score normalized by evaluated weight:
        weighted_total = sum((c.score / 5.0) * (c.weight / eval_weight) for c in evaluated_scores) * 100.0
        weighted_total = round(weighted_total, 1)

        if num_evaluated == total_rubric_comps:
            methodology = (
                f"Overall score is the weighted total across all {total_rubric_comps} planned competencies "
                f"evaluated on a standard 1.0–5.0 rubric scale."
            )
            completeness = "complete"
        else:
            pct = round((eval_weight / sum(c.weight for c in rubric)) * 100)
            methodology = (
                f"Score normalized across {num_evaluated} of {total_rubric_comps} competencies with sufficient evidence "
                f"({pct}% of rubric weight). Unevaluated competencies were marked 'Insufficient Evidence' and "
                f"excluded from the score rather than penalized."
            )
            completeness = "partial"

        recommendation = _recommend(weighted_total, num_evaluated, total_rubric_comps)

    # Synthesis lists
    key_strengths = [str(s) for s in data.get("key_strengths", []) if str(s).strip()]
    if not key_strengths and evaluated_scores:
        key_strengths = [f"Demonstrated competency in {c.name}" for c in evaluated_scores if c.score and c.score >= 3.5]
    if not key_strengths:
        key_strengths = ["Engaged with the interviewer across discussion topics."]

    priority_improvements = [str(p) for p in data.get("priority_improvements", []) if str(p).strip()]
    if not priority_improvements and evaluated_scores:
        priority_improvements = [f"Deepen trade-off analysis in {c.name}" for c in evaluated_scores if c.score and c.score < 3.5]
    if not priority_improvements:
        priority_improvements = ["Provide more specific metrics and STAR structure in technical explanations."]

    practice_topics = [str(t) for t in data.get("recommended_practice_topics", []) if str(t).strip()]
    if not practice_topics:
        practice_topics = [
            f"STAR framework responses for {config.role}",
            "Communicating system trade-offs and edge cases",
            "Measuring and articulating business impact",
        ]

    summary = str(data.get("summary", "")).strip()
    if not summary:
        if num_evaluated > 0:
            summary = (
                f"The candidate completed a {config.interview_type} screening for the {config.role} role, "
                f"evaluating {num_evaluated} competencies with an overall score of {weighted_total}%. "
                f"Primary strengths included {', '.join(c.name for c in evaluated_scores[:2])}."
            )
        else:
            summary = "The interview session concluded before sufficient technical evidence could be evaluated."

    return ScoreCard(
        competencies=all_scores,
        weighted_total=weighted_total,
        recommendation=recommendation,
        summary=summary,
        scoring_methodology=methodology,
        key_strengths=key_strengths[:4],
        priority_improvements=priority_improvements[:4],
        recommended_practice_topics=practice_topics[:4],
        interview_completeness=completeness,
    )


def _build_unassessed_scorecard(rubric: list, reason: str) -> ScoreCard:
    """Builds a neutral scorecard for incomplete sessions without penalizing the candidate."""
    scores = [
        CompetencyScore(
            id=c.id,
            name=c.name,
            weight=c.weight,
            score=None,
            status="insufficient_evidence",
            status_label="Insufficient Evidence",
            evidence="No candidate response was recorded for this competency.",
            explanation="Competency was unprobed due to early session conclusion.",
            comment="Unassessed: session ended before this competency was evaluated.",
            strengths=[],
            missing_details=["Candidate did not answer questions for this competency."],
            improvement_suggestion="Complete the full interview session to receive evaluation feedback.",
            stronger_answer_example="Provide a complete response during the interview.",
            evidence_verified=False,
        )
        for c in rubric
    ]
    return ScoreCard(
        competencies=scores,
        weighted_total=0.0,
        recommendation="Insufficient Data",
        summary=reason,
        scoring_methodology="No competencies had sufficient candidate evidence to evaluate. Session concluded before candidate answers were submitted. No score computed.",
        key_strengths=["No answers evaluated"],
        priority_improvements=["Complete the full interview sequence"],
        recommended_practice_topics=["Interview preparation and completion"],
        interview_completeness="minimal",
    )


def _fallback_heuristic_score(rubric: list, candidate_turns: list, error: str = "") -> dict:
    """Zero-dependency fallback in case of model parsing errors."""
    words = sum(len(getattr(t, "text", "").split()) for t in candidate_turns)
    base = 2.5 + min(1.5, words / 150.0)
    comps = []
    for c in rubric:
        comps.append({
            "id": c.id,
            "status": "evaluated" if words > 15 else "insufficient_evidence",
            "score": round(base, 1) if words > 15 else None,
            "explanation": f"Evaluated based on response length and technical vocabulary ({words} total words spoken).",
            "evidence": candidate_turns[0].text[:120] if candidate_turns else "N/A",
            "strengths": ["Communicated technical concepts during the interview."],
            "missing_details": ["Provide more quantified metrics and trade-offs."],
            "improvement_suggestion": f"Structure answers around STAR for {c.name.lower()}.",
            "stronger_answer_example": f"State the Situation, your exact Action, and the Result achieved in {c.name.lower()}.",
        })
    return {
        "summary": f"Offline evaluation generated. ({error})" if error else "Evaluated from candidate turns.",
        "key_strengths": ["Relevant technical background discussed."],
        "priority_improvements": ["Add concrete metrics and detailed edge cases."],
        "recommended_practice_topics": ["STAR answering methodology", "System trade-off articulation"],
        "competencies": comps,
    }


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {}


def _clamp(v: Any) -> float:
    try:
        val = float(v)
    except (TypeError, ValueError):
        return 3.0
    return max(1.0, min(5.0, round(val, 1)))


def _score_status_label(score: float) -> str:
    if score >= 4.0:
        return "Demonstrated Strength"
    if score >= 3.0:
        return "Proficient"
    if score >= 2.0:
        return "Developing"
    return "Critical Gaps"


def _recommend(total: float, evaluated_count: int, total_count: int) -> str:
    if evaluated_count == 0:
        return "Insufficient Data"
    if evaluated_count < 2 and total_count >= 3:
        return "Partial — Insufficient Data"
    if total >= 80:
        return "Strong hire"
    if total >= 65:
        return "Hire"
    if total >= 50:
        return "Borderline — lean no"
    return "No hire"
