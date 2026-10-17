"""Pluggable LLM clients.

The engine only ever calls ``chat(messages, task=...)``. Swap providers by
changing one line or one env var. ``MockLLM`` needs no key and no network, so
the entire interview flow can be exercised offline.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMError(RuntimeError):
    pass


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
    follow-ups for vague answers or missing measurable impact, and transitions to
    different competencies when answers are sufficiently complete.
    """
    last_user_content = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_content = m.get("content", "")
            break

    # Extract candidate's latest answer from prompt format
    match_ans = re.search(r'CANDIDATE LATEST ANSWER:\s*"?(.*?)"?\s*(?:\n---|\nCURRENT|\n$)', last_user_content, re.DOTALL)
    candidate_ans = match_ans.group(1).strip() if match_ans else ""
    if not candidate_ans and messages:
        # Fallback to inspecting raw text of earlier user message
        for m in reversed(messages[:-1]):
            if m.get("role") == "user":
                candidate_ans = m.get("content", "").strip()
                break

    ans_lower = candidate_ans.lower()
    words = candidate_ans.split()

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

    # Rule 1: Finish if max questions reached and no follow-up needed
    if reached_max and not (can_followup and len(words) < 15):
        return {
            "action": "finish",
            "spoken_response": "That covers all of my questions. Do you have any questions for me about the team or the role?",
            "reason_for_decision": "Completed all planned interview competencies for this session.",
            "competency": "wrapup",
            "evidence_used": [f"Candidate concluded with: '{candidate_ans[:80]}...'"] if candidate_ans else ["Interview sequence complete"],
        }

    # Rule 2: Vague answer OR missing measurable impact -> Follow up (if allowed)
    is_vague = len(words) < 22 or any(m in ans_lower for m in ("learned and adapted", "i learned", "adapted to them", "it worked fine", "built the backend", "used python"))
    has_metrics = bool(re.search(r'\b(?:\d+%?|percent|users|latency|ms|roi|accuracy|f1|auc|qps|sla)\b', ans_lower))

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
                # Strip prefix
                txt = line.split(":", 1)[-1].strip()
                if txt:
                    candidate_lines.append(txt)

    if not candidate_lines:
        # Fallback to inspecting any candidate text in content
        for line in content.split("\n"):
            if "[CANDIDATE" in line or line.startswith("CANDIDATE:"):
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


class OpenAICompatLLM(LLMClient):
    """Works with any OpenAI-compatible /chat/completions endpoint.

    Defaults to Sarvam AI (strong Indian-language support, India-hosted). Also
    fine for OpenAI, Together, Groq, vLLM, Ollama — just change base_url/model.
    """

    def __init__(self, api_key=None, base_url=None, model=None, auth_style=None, timeout=60):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("SARVAM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL",
                                                     "https://api.sarvam.ai/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "sarvam-105b")
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "ignore")[:400]
            raise LLMError(f"HTTP {e.code} from LLM provider ({self.base_url}): {err_body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Network error connecting to LLM provider ({self.base_url}): {e}") from e
        return data["choices"][0]["message"]["content"]

    @property
    def name(self):
        return f"OpenAICompatLLM({self.model} @ {self.base_url})"


def build_llm(mock: bool = False) -> LLMClient:
    """Factory: MockLLM if requested or no key present, else the real client."""
    has_key = bool(os.environ.get("LLM_API_KEY") or os.environ.get("SARVAM_API_KEY"))
    if mock or not has_key:
        return MockLLM()
    return OpenAICompatLLM()

