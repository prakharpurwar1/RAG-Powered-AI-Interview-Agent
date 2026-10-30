"""Pluggable LLM clients.

The engine only ever calls ``chat(messages, task=...)``. Swap providers by
changing one line or one env var. ``MockLLM`` needs no key and no network, so
the entire interview flow can be exercised offline.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from typing import Optional


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def sanitize_secret_text(text: str) -> str:
    """Removes sensitive keys, tokens, and authorization credentials from logs/error strings."""
    if not text:
        return ""
    sanitized = str(text)

    # 1. Scrub exact matches of known environment secrets
    for env_var in ("LLM_API_KEY", "SARVAM_API_KEY", "DATABASE_URL", "TEST_DATABASE_URL", "DASHBOARD_TOKEN"):
        val = os.environ.get(env_var, "").strip()
        if val and len(val) >= 4:
            sanitized = sanitized.replace(val, "[REDACTED]")

    # 2. Scrub Bearer tokens
    sanitized = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9_\-\.]{6,}', 'Bearer [REDACTED]', sanitized)

    # 3. Scrub header-style key/value pairs (e.g. api-subscription-key: xyz, password 'xyz')
    sanitized = re.sub(
        r'(?i)(api[-_]?subscription[-_]?key|api[-_]?key|authorization|token|secret|password)\s*[:=\s]\s*["\']?[A-Za-z0-9_\-\.]{4,}["\']?',
        r'\1: [REDACTED]',
        sanitized,
    )

    # 4. Scrub embedded basic auth credentials in URLs (e.g. postgres://user:pass@host)
    sanitized = re.sub(r'://([^:]+):([^@]+)@', r'://\1:[REDACTED]@', sanitized)

    # 5. Scrub standard API key shapes
    sanitized = re.sub(r'\b(sk|sarvam)[-_][a-zA-Z0-9_\-]{8,}\b', '[REDACTED]', sanitized)

    return sanitized


def _log_llm_diagnostic(event: str, **fields):
    """Write structured diagnostic log to sys.stderr with secret sanitization."""
    rec = {"t": round(time.time(), 3), "event": event}
    for k, v in fields.items():
        if isinstance(v, str):
            rec[k] = sanitize_secret_text(v)
        else:
            rec[k] = v
    try:
        sys.stderr.write(json.dumps(rec) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _categorize_http_status(status_code: int) -> str:
    """Classify HTTP status codes into standardized error categories."""
    if status_code in (401, 403):
        return "authentication_failure"
    elif status_code in (402, 429):
        return "quota_error"
    elif status_code in (400, 422):
        return "invalid_request_error"
    elif 500 <= status_code < 600:
        return "upstream_server_error"
    return "http_error"


def _categorize_network_error(exc: Exception) -> str:
    """Classify network / URLError exceptions into standardized error categories."""
    err_str = str(exc).lower()
    if isinstance(exc, (socket.timeout, TimeoutError)) or "timed out" in err_str:
        return "timeout"
    reason = getattr(exc, "reason", None)
    if reason:
        reason_str = str(reason).lower()
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in reason_str:
            return "timeout"
        if "ssl" in reason_str or "certificate" in reason_str:
            return "ssl_error"
        if "connection refused" in reason_str:
            return "connection_refused"
    return "connection_error"


class LLMClient:
    """Base class. ``task`` is an engine hint: greeting|question|followup|decide_turn|score|wrapup."""

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MockLLM(LLMClient):
    """Deterministic, offline stand-in. Exercises every branch without a key."""

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        if task == "score":
            return json.dumps(_mock_scores(messages))
        if task == "decide_turn":
            return json.dumps(_mock_decide_turn(messages))
        if task == "followup":
            return ("Thanks. Can you make that concrete — what was your specific "
                    "role, and what was the measurable outcome?")
        if task == "wrapup":
            return ("That's everything from my side. Thanks for your time — "
                    "we'll be in touch about next steps.")
        if task == "generate_questions":
            return json.dumps(_mock_questions(messages))
        # greeting / question -> return empty so the engine uses its banked text
        return ""

    @property
    def name(self):
        return "MockLLM (offline)"


def _mock_decide_turn(messages):
    """Generates intelligent structured turn decisions for offline/testing mode.

    Understands the candidate's answer, references projects mentioned, asks targeted
    follow-ups for vague answers or missing measurable impact, handles greetings,
    clarification requests, questions about rule-based/AI architecture, and off-topic
    questions gracefully, and transitions to different competencies when answers are complete.
    """
    last_user_content = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_content = m.get("content", "")
            break

    # Extract candidate's latest message from prompt format
    match_ans = re.search(r'CANDIDATE LATEST (?:INPUT|MESSAGE|ANSWER)[^\n]*\n+"?(.*?)"?\s*(?:\n---|\nCURRENT|\n$)', last_user_content, re.DOTALL | re.IGNORECASE)
    if not match_ans:
        match_ans = re.search(r'CANDIDATE LATEST (?:INPUT|MESSAGE|ANSWER):\s*"?(.*?)"?\s*(?:\n---|\nCURRENT|\n$)', last_user_content, re.DOTALL | re.IGNORECASE)
    candidate_ans = match_ans.group(1).strip() if match_ans else ""
    if not candidate_ans and messages:
        # Fallback to inspecting raw text of earlier user message
        for m in reversed(messages[:-1]):
            if m.get("role") == "user":
                candidate_ans = m.get("content", "").strip()
                break

    ans_lower = candidate_ans.lower()
    words = candidate_ans.split()

    # Extract current question being discussed if present
    match_q = re.search(r'CURRENT QUESTION BEING DISCUSSED[^\n]*\n+"?(.*?)"?\s*(?:\n---|\nCURRENT|\n$)', last_user_content, re.DOTALL | re.IGNORECASE)
    if not match_q:
        match_q = re.search(r'CURRENT QUESTION BEING DISCUSSED:\s*"?(.*?)"?\s*(?:\n---|\nCURRENT|\n$)', last_user_content, re.DOTALL | re.IGNORECASE)
    current_q = match_q.group(1).strip() if match_q else "tell me about your technical background."

    # Extract cues passed by engine
    can_followup = "can follow up: true" in last_user_content.lower() or "can_followup: true" in last_user_content.lower()
    reached_max = "reached max questions: true" in last_user_content.lower() or "max questions reached: true" in last_user_content.lower()

    # Extract uncovered competencies from prompt if available
    comp_match = re.search(r'UNCOVERED COMPETENCIES REMAINING:\s*\[(.*?)\]', last_user_content, re.IGNORECASE)
    uncovered = []
    if comp_match:
        uncovered = [c.strip().strip("'\"") for c in comp_match.group(1).split(",") if c.strip().strip("'\"")]

    current_comp_match = re.search(r'CURRENT COMPETENCY BEING ASSESSED:\s*(\w+)', last_user_content, re.IGNORECASE)
    current_comp = current_comp_match.group(1).strip() if current_comp_match else "experience"

    # Identify project/skill references mentioned by candidate
    project_mentioned = None
    if "hr attrition" in ans_lower or "attrition" in ans_lower:
        project_mentioned = "HR attrition model"
    elif "ibm" in ans_lower:
        project_mentioned = "IBM dataset machine-learning pipeline"
    elif "money-management" in ans_lower or "money management" in ans_lower or "budget" in ans_lower:
        project_mentioned = "money-management application"
    elif "kafka" in ans_lower or "streaming" in ans_lower:
        project_mentioned = "Kafka event pipeline"
    elif "redis" in ans_lower or "cache" in ans_lower or "caching" in ans_lower:
        project_mentioned = "Redis distributed caching architecture"

    has_metrics = bool(re.search(r'\b(?:\d+%?|percent|users|latency|ms|roi|accuracy|f1|auc|qps|sla)\b', ans_lower))

    # -------------------------------------------------------------------------
    # Intent 1: Meta-question (e.g. "Are you rule-based?", AI process questions)
    # -------------------------------------------------------------------------
    is_meta = any(p in ans_lower for p in (
        "rule-based", "rule based", "are you rule", "are you an ai", "are you a bot",
        "are you ai", "are you human", "are you a real person", "how does this work",
        "how is this evaluated", "how are you evaluating", "how will i be scored",
        "interview process", "who created you"
    ))
    if is_meta and not project_mentioned:
        return {
            "action": "conversational",
            "spoken_response": (
                "I am an AI interviewer driven by a large language model with retrieval-augmented context "
                "that adapts dynamically to your background, rather than following a rigid rule-based script. "
                f"Returning to our focus: {current_q}"
            ),
            "reason_for_decision": "Candidate inquired about the AI interviewer architecture or process; answered transparently and guided back to current question.",
            "competency": current_comp,
            "evidence_used": [f"Candidate asked: '{candidate_ans[:80]}'"],
            "candidate_intent": "meta_question",
        }

    # -------------------------------------------------------------------------
    # Intent 2: Clarification request
    # -------------------------------------------------------------------------
    is_clarification = any(p in ans_lower for p in (
        "what do you mean", "can you clarify", "could you clarify", "clarify what you mean",
        "clarification", "could you explain what", "can you explain what", "don't understand",
        "didn't understand", "what is meant by", "meaning of", "can you give an example of what you mean",
        "what specifically", "what are you looking for", "looking for in this question", "what kind of details"
    ))
    if is_clarification and not project_mentioned:
        if "trade-off" in ans_lower or "tradeoff" in ans_lower:
            clarification_explanation = (
                "By technical trade-offs, I mean how you balanced competing architectural priorities—such as "
                "throughput versus latency, consistency versus availability, or delivery speed versus technical debt. "
                "How did you evaluate those considerations in your work?"
            )
        else:
            clarification_explanation = (
                f"To clarify, we are exploring your direct personal ownership and technical choices regarding {current_comp.replace('_', ' ')}. "
                "Could you walk me through a specific real-world example from your past experience?"
            )
        return {
            "action": "clarify",
            "spoken_response": clarification_explanation,
            "reason_for_decision": "Candidate requested clarification on question concepts; explained context clearly without revealing expected answers.",
            "competency": current_comp,
            "evidence_used": [f"Candidate asked: '{candidate_ans[:80]}'"],
            "candidate_intent": "clarification",
        }

    # -------------------------------------------------------------------------
    # Intent 3: Repeat / Rephrase request
    # -------------------------------------------------------------------------
    is_rephrase = any(p in ans_lower for p in (
        "repeat the question", "repeat that", "say that again", "rephrase", "could you repeat",
        "can you repeat", "can you restate", "didn't catch that", "pardon", "could you rephrase"
    ))
    if is_rephrase and not project_mentioned:
        return {
            "action": "rephrase",
            "spoken_response": f"Certainly, let me restate the question for you: {current_q}",
            "reason_for_decision": "Candidate asked to repeat or rephrase the question; restated clearly.",
            "competency": current_comp,
            "evidence_used": [f"Candidate requested: '{candidate_ans[:80]}'"],
            "candidate_intent": "rephrase",
        }

    # -------------------------------------------------------------------------
    # Intent 4: Greeting / Small Talk
    # -------------------------------------------------------------------------
    is_greeting = (
        len(words) <= 12 and
        any(ans_lower.startswith(w) or f" {w}" in ans_lower for w in (
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "thanks for having me", "nice to meet you", "how are you"
        )) and not project_mentioned and not has_metrics
    )
    if is_greeting:
        return {
            "action": "conversational",
            "spoken_response": f"Hello! Great to connect with you today. Whenever you're ready, let's turn to our question: {current_q}",
            "reason_for_decision": "Candidate offered a greeting; welcomed them warmly and directed to the active interview question.",
            "competency": current_comp,
            "evidence_used": [f"Candidate greeted: '{candidate_ans[:60]}'"],
            "candidate_intent": "greeting",
        }

    # -------------------------------------------------------------------------
    # Intent 5: Off-topic remark
    # -------------------------------------------------------------------------
    is_off_topic = (
        any(p in ans_lower for p in (
            "weather", "capital of", "what's for lunch", "who won", "favorite food",
            "tell me a joke", "recipe", "how old are you", "what time is it",
            "match", "champions league", "cricket", "football", "sports", "game yesterday"
        )) and not project_mentioned and not has_metrics
    )
    if is_off_topic:
        return {
            "action": "conversational",
            "spoken_response": f"I am focused on conducting your technical interview today. Let's return to the question at hand: {current_q}",
            "reason_for_decision": "Candidate asked an off-topic question; politely acknowledged and redirected to the interview agenda.",
            "competency": current_comp,
            "evidence_used": [f"Candidate asked: '{candidate_ans[:60]}'"],
            "candidate_intent": "off_topic",
        }

    # -------------------------------------------------------------------------
    # Intent 6: Candidate is answering the interview question
    # -------------------------------------------------------------------------
    # Rule 1: Finish if max questions reached and no follow-up needed
    if reached_max and not (can_followup and len(words) < 15):
        return {
            "action": "finish",
            "spoken_response": "That covers all of my questions. Do you have any questions for me about the team or the role?",
            "reason_for_decision": "Completed all planned interview competencies for this session.",
            "competency": "wrapup",
            "evidence_used": [f"Candidate concluded with: '{candidate_ans[:80]}...'"] if candidate_ans else ["Interview sequence complete"],
            "candidate_intent": "answer",
        }

    # Rule 2: Vague answer OR missing measurable impact -> Follow up (if allowed)
    is_vague = len(words) < 22 or any(m in ans_lower for m in ("learned and adapted", "i learned", "adapted to them", "it worked fine", "built the backend", "used python"))

    if can_followup and (is_vague or not has_metrics):
        if project_mentioned:
            if not has_metrics:
                spoken = f"You mentioned {project_mentioned}. Could you explain specifically how you evaluated the model or measured success on that deliverable, without assuming initial baseline metrics?"
                reason = f"Candidate described work on {project_mentioned} but did not provide how success or performance was evaluated."
            else:
                spoken = f"Regarding {project_mentioned}, what was your specific personal contribution to the system design, and how did you resolve the key technical trade-offs?"
                reason = f"Candidate introduced {project_mentioned}; following up to examine personal architectural ownership."
        else:
            if not has_metrics:
                spoken = "Can you make that concrete—what was your direct personal contribution, and how was the project outcome or impact measured?"
                reason = "Candidate response was brief and lacked measurable success criteria."
            else:
                spoken = "Could you walk me through one specific technical hurdle you encountered during that work and how you worked through it?"
                reason = "Probing technical depth and trade-offs on the candidate's stated experience."

        return {
            "action": "follow_up",
            "spoken_response": spoken,
            "reason_for_decision": reason,
            "competency": current_comp,
            "evidence_used": [f"Candidate stated: '{candidate_ans[:100]}'"],
            "candidate_intent": "answer",
        }

    # Rule 3: Sufficiently complete answer -> Advance to next competency
    next_comp = uncovered[0] if uncovered else "ownership"
    if next_comp == current_comp and len(uncovered) > 1:
        next_comp = uncovered[1]

    if project_mentioned:
        spoken = f"That gives helpful clarity on how you handled the {project_mentioned}. Shifting to {next_comp.replace('_', ' ')}: could you tell me about a time you had to take ownership of an ambiguous problem and make an architectural decision with incomplete information?"
        reason = f"Candidate provided a concrete explanation of {project_mentioned}; advancing to assess {next_comp}."
    else:
        spoken = f"Understood. Moving forward to {next_comp.replace('_', ' ')}: tell me about a project where you owned the delivery end-to-end from requirements to production deployment."
        reason = f"Candidate answered previous competency sufficiently; transitioning to {next_comp}."

    return {
        "action": "next_question",
        "spoken_response": spoken,
        "reason_for_decision": reason,
        "competency": next_comp,
        "evidence_used": [f"Candidate demonstrated experience in {current_comp}: '{candidate_ans[:100]}'"],
        "candidate_intent": "answer",
    }


def _mock_scores(messages):
    """Phase 4 Grounded Mock Evaluation: builds evidence-based scorecards from candidate transcript."""
    content = ""
    for m in messages:
        content += m.get("content", "") + "\n"

    # Extract candidate statements from transcript block
    candidate_lines = []
    in_transcript = False
    for line in content.split("\n"):
        if "INTERVIEW TRANSCRIPT:" in line:
            in_transcript = True
            continue
        if in_transcript:
            if line.startswith("CANDIDATE:") or "[CANDIDATE" in line:
                # Exclude non-answer conversational turns from candidate lines evaluated for technical score
                if any(k in line.lower() for k in ("clarification", "meta", "greeting", "rephrase", "off_topic", "conversational")):
                    continue
                # Strip prefix
                txt = line.split(":", 1)[-1].strip()
                if txt:
                    candidate_lines.append(txt)

    if not candidate_lines:
        # Fallback to inspecting any candidate text in content
        for line in content.split("\n"):
            if "[CANDIDATE" in line or line.startswith("CANDIDATE:"):
                if any(k in line.lower() for k in ("clarification", "meta", "greeting", "rephrase", "off_topic", "conversational")):
                    continue
                candidate_lines.append(line.split(":", 1)[-1].strip())

    combined_text = " ".join(candidate_lines).strip()
    words = len(combined_text.split())
    text_lower = combined_text.lower()

    # Incomplete / minimal interview
    if words < 8:
        return {
            "summary": "The candidate completed only a minimal portion of the interview. Insufficient data to evaluate core competencies.",
            "key_strengths": ["Attended the interview session"],
            "priority_improvements": ["Complete the full interview sequence to demonstrate technical depth"],
            "recommended_practice_topics": ["Mock interview completion", "Communication confidence"],
            "competencies": [
                {
                    "id": cid,
                    "status": "insufficient_evidence",
                    "score": None,
                    "explanation": "Competency was unprobed due to insufficient answers.",
                    "evidence": "No candidate evidence recorded.",
                    "strengths": [],
                    "missing_details": ["Did not answer questions on this topic."],
                    "improvement_suggestion": "Ensure you provide comprehensive answers to each question.",
                    "stronger_answer_example": "Situation: [Role and project context]. Action: [Your personal contribution]. Result: [Outcome achieved].",
                }
                for cid in ["communication", "problem_solving", "fundamentals", "ownership", "motivation", "experience", "depth", "collaboration"]
            ],
        }

    # Detect projects and metrics mentioned
    project = None
    if "hr attrition" in text_lower or "attrition" in text_lower:
        project = "HR attrition prediction model"
    elif "ibm" in text_lower:
        project = "IBM dataset machine-learning pipeline"
    elif "money-management" in text_lower or "money management" in text_lower or "budget" in text_lower:
        project = "money-management budgeting application"
    elif "kafka" in text_lower or "streaming" in text_lower:
        project = "Kafka distributed event pipeline"
    elif "redis" in text_lower or "caching" in text_lower or "billing" in text_lower:
        project = "Redis caching and billing architecture"

    has_metrics = bool(re.search(r'\b(?:\d+%?|percent|users|latency|ms|roi|accuracy|f1|auc|qps|sla)\b', text_lower))
    is_vague = any(p in text_lower for p in ["learned and adapted", "i learned", "adapted to them", "it worked fine"]) and not has_metrics

    # Evidence quote snippet
    quote = candidate_lines[0][:150] if candidate_lines else combined_text[:150]

    # Score tiers
    if has_metrics and not is_vague:
        base_score = 4.2
        quality = "strong"
    elif words >= 25 and not is_vague:
        base_score = 3.6
        quality = "proficient"
    else:
        base_score = 2.4
        quality = "developing"

    # Build competencies list
    comps = []
    # Core competencies commonly in rubrics
    rubric_ids = ["communication", "problem_solving", "fundamentals", "ownership", "motivation", "experience", "depth", "collaboration"]

    for idx, cid in enumerate(rubric_ids):
        # In a shorter interview, simulate that 3-4 competencies were evaluated and others unprobed
        if idx >= 4 and words < 40:
            comps.append({
                "id": cid,
                "status": "insufficient_evidence",
                "score": None,
                "explanation": f"The {cid.replace('_', ' ')} competency was not actively probed during this short interview session.",
                "evidence": "No candidate evidence recorded for this competency.",
                "strengths": [],
                "missing_details": [f"Direct evidence for {cid.replace('_', ' ')} was not gathered."],
                "improvement_suggestion": f"Prepare a structured STAR story showcasing {cid.replace('_', ' ')}.",
                "stronger_answer_example": f"Situation: At [Company], I led [Project]. Action: I addressed {cid.replace('_', ' ')} by [Your action]. Result: [Your outcome].",
            })
            continue

        c_score = round(base_score - (0.2 if idx % 2 == 1 else -0.1), 1)
        c_score = max(1.0, min(5.0, c_score))

        if quality == "strong":
            strengths = [
                f"Clearly explained technical contributions regarding {project or 'their system'}.",
                "Grounded answers with quantified outcomes and measurable performance.",
            ]
            missing = [
                "Could detail edge-case trade-offs and alternative architectures evaluated.",
            ]
            suggestion = "Discuss what failed during development and how you adapted the system architecture."
            star_example = (
                f"Situation: While developing the {project or 'project'}, we needed to ensure high reliability. "
                f"Task: My goal was to lead the technical implementation and measure system outcomes. "
                f"Action: I personally architected the core workflow using the tools mentioned. [To strengthen: detail why alternative patterns were ruled out]. "
                f"Result: The solution successfully delivered the performance outlined in my response. [To strengthen: compare against baseline before deployment]."
            )
        elif quality == "proficient":
            strengths = [
                f"Demonstrated familiarity with {project or 'relevant domain concepts'}.",
                "Structured communication was easy to follow and relevant.",
            ]
            missing = [
                "Lacks specific validation metrics and measured business impact.",
            ]
            suggestion = "Incorporate clear success metrics and explain how you validated outcomes without relying on assumptions."
            star_example = (
                f"Situation: During my work on the {project or 'project'}, we faced technical requirements to deliver. "
                f"Task: I was responsible for delivering the core feature set. "
                f"Action: I built the implementation using my stated technical stack. [To strengthen: elaborate on your personal decision-making trade-offs]. "
                f"Result: Delivered the functionality to production. [To strengthen: add your measured impact if recorded]."
            )
        else: # developing / vague
            strengths = [
                "Expressed willingness to learn and adapt to unfamiliar technical challenges.",
            ]
            missing = [
                "Did not specify the exact technologies used or candidate's personal contribution.",
                "Missing concrete details on how obstacles were overcome.",
            ]
            suggestion = "Avoid generic statements like 'I adapted'. Name the specific technology, the exact problem, and what YOU individually built."
            star_example = (
                f"Situation: When working on {project or 'the project'}, we encountered an unfamiliar technology roadblock. "
                f"Task: I needed to master [Specify the exact technology] to deliver the component. "
                f"Action: I systematically reviewed documentation, prototyped [Specify your prototype], and resolved [Specify the hurdle]. "
                f"Result: Unblocked delivery and deployed the component successfully. [To strengthen: state how long it took or team feedback received]."
            )

        comps.append({
            "id": cid,
            "status": "evaluated",
            "score": c_score,
            "explanation": f"Candidate demonstrated {quality} performance on {cid.replace('_', ' ')} based on transcript responses.",
            "evidence": f"Candidate stated: '{quote}'",
            "strengths": strengths,
            "missing_details": missing,
            "improvement_suggestion": suggestion,
            "stronger_answer_example": star_example,
        })

    key_strengths = [
        f"Grounded experience in {project}" if project else "Clear communication of engineering background",
        "Demonstrated technical ownership and adaptation to role requirements",
    ]
    priority_improvements = [
        "Include more concrete architectural trade-offs in technical answers",
        "Quantify project impact and validate model or system baselines",
    ]
    practice_topics = [
        f"STAR structured technical stories for {project or 'core projects'}",
        "System reliability, edge cases, and architectural trade-offs",
        "Articulating individual contributions versus team deliverables",
    ]

    return {
        "summary": (
            f"The candidate discussed their experience with {project or 'software engineering systems'}. "
            f"Demonstrated {quality} technical competency with clear potential; performance can be elevated "
            f"by articulating architectural trade-offs and concrete impact metrics."
        ),
        "key_strengths": key_strengths,
        "priority_improvements": priority_improvements,
        "recommended_practice_topics": practice_topics,
        "competencies": comps,
    }



def _mock_questions(messages):
    """Grounded mock questions for offline test environments."""
    return {
        "questions": [
            {
                "id": "p1",
                "competency": "experience",
                "text": "Looking at your resume, could you walk me through the key project you highlighted and your direct technical contributions to it?",
                "followup_hint": "Probe technical choices and measurable results mentioned in their background.",
            },
            {
                "id": "p2",
                "competency": "problem_solving",
                "text": "The job description emphasizes delivering reliable solutions. What was the most challenging technical roadblock in your past experience, and how did you resolve it?",
                "followup_hint": "Ask about alternative approaches considered and why they chose their solution.",
            },
            {
                "id": "p3",
                "competency": "ownership",
                "text": "Tell me about a time you took complete ownership of a deliverable from requirements through deployment in your past roles.",
                "followup_hint": "Probe how they handled timeline trade-offs and edge cases.",
            },
            {
                "id": "p4",
                "competency": "fundamentals",
                "text": "How do the core architectural patterns you've utilized in your previous work map to the requirements outlined in this target role?",
                "followup_hint": "Check depth in system design and data handling.",
            },
            {
                "id": "p5",
                "competency": "collaboration",
                "text": "Describe a scenario where you collaborated with cross-functional partners or navigated differing technical opinions on your team.",
                "followup_hint": "Probe stakeholder alignment and resolution.",
            },
            {
                "id": "p6",
                "competency": "motivation",
                "text": "Given your career trajectory so far, what specifically attracted you to this role and how does it fit your goals?",
                "followup_hint": "Probe role alignment and growth expectations.",
            },
        ]
    }


def _ssl_context():
    """Build SSL context using certifi CA bundle if available."""
    try:
        import certifi
        import ssl
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            import ssl
            return ssl.create_default_context()
        except Exception:
            return None


class OpenAICompatLLM(LLMClient):
    """Works with any OpenAI-compatible /chat/completions endpoint.

    Defaults to Sarvam AI (strong Indian-language support, India-hosted). Also
    fine for OpenAI, Together, Groq, vLLM, Ollama — just change base_url/model.
    """

    def __init__(self, api_key=None, base_url=None, model=None, auth_style=None, timeout=60):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("SARVAM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL",
                                                     "https://api.sarvam.ai/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "sarvam-105b-conversations")
        # Sarvam uses the 'api-subscription-key' header; OpenAI-style uses Bearer.
        self.auth_style = auth_style or os.environ.get("LLM_AUTH_STYLE", "api-subscription-key")
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("No API key. Set LLM_API_KEY or SARVAM_API_KEY, or use MockLLM.")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.auth_style == "api-subscription-key":
            h["api-subscription-key"] = self.api_key
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        ctx = _ssl_context()
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "ignore")[:400]
            sanitized_body = sanitize_secret_text(err_body)
            sanitized_url = sanitize_secret_text(self.base_url)
            cat = _categorize_http_status(e.code)
            _log_llm_diagnostic(
                "llm_request_failed",
                provider="sarvam",
                model=self.model,
                endpoint=sanitized_url,
                status_code=e.code,
                error_category=cat,
                error_response=sanitized_body,
                task=task,
            )
            raise LLMError(
                f"HTTP {e.code} ({cat}) from LLM provider ({sanitized_url}): {sanitized_body}",
                status_code=e.code,
            ) from e
        except urllib.error.URLError as e:
            cat = _categorize_network_error(e)
            sanitized_err = sanitize_secret_text(str(e.reason if hasattr(e, "reason") else e))
            sanitized_url = sanitize_secret_text(self.base_url)
            _log_llm_diagnostic(
                "llm_request_failed",
                provider="sarvam",
                model=self.model,
                endpoint=sanitized_url,
                status_code=None,
                error_category=cat,
                error_response=sanitized_err,
                task=task,
            )
            raise LLMError(
                f"{cat.replace('_', ' ').title()} connecting to LLM provider ({sanitized_url}): {sanitized_err}",
                status_code=None,
            ) from e

        if not isinstance(data, dict) or "choices" not in data or not data["choices"]:
            sanitized_url = sanitize_secret_text(self.base_url)
            err_detail = data.get("error", {}).get("message") if isinstance(data, dict) and isinstance(data.get("error"), dict) else str(data)[:200]
            sanitized_detail = sanitize_secret_text(str(err_detail))
            _log_llm_diagnostic(
                "llm_request_failed",
                provider="sarvam",
                model=self.model,
                endpoint=sanitized_url,
                status_code=200,
                error_category="invalid_response_schema",
                error_response=sanitized_detail,
                task=task,
            )
            raise LLMError(
                f"Unexpected response schema from LLM provider ({sanitized_url}): {sanitized_detail}"
            )

        choice_msg = data["choices"][0].get("message", {})
        content = choice_msg.get("content")
        if content is None:
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            refusal = choice_msg.get("refusal")
            sanitized_url = sanitize_secret_text(self.base_url)
            _log_llm_diagnostic(
                "llm_request_failed",
                provider="sarvam",
                model=self.model,
                endpoint=sanitized_url,
                status_code=200,
                error_category="empty_content_response",
                error_response=f"finish_reason: {finish_reason}, refusal: {refusal}",
                task=task,
            )
            raise LLMError(
                f"LLM provider ({sanitized_url}) returned empty content (finish_reason: {finish_reason}, refusal: {refusal})"
            )

        return str(content)

    @property
    def name(self):
        return f"OpenAICompatLLM({self.model} @ {self.base_url})"


def build_llm(mock: bool = False) -> LLMClient:
    """Factory: MockLLM if requested or no key present, else the real client."""
    has_key = bool(os.environ.get("LLM_API_KEY") or os.environ.get("SARVAM_API_KEY"))
    if mock or not has_key:
        return MockLLM()
    return OpenAICompatLLM()

