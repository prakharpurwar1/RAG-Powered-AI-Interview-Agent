"""The interviewer engine: a small, explicit state machine.

Stages: greeting -> core (N questions, optional follow-up) ->
candidate_questions -> wrapup -> done.

The engine owns the *flow*; the LLM only phrases turns and scores at the end.
That separation is what keeps interviews consistent, resumable and auditable —
and it means a flaky model can never derail the structure of the interview.
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import InterviewConfig
from .llm import LLMClient, LLMError, sanitize_secret_text
from .questions import generate_personalized_questions, questions_for
from .rubric import rubric_for
from .scoring import build_scorecard


def _log_diagnostic(event: str, **fields):
    rec = {"t": round(time.time(), 3), "event": event}
    rec.update(fields)
    sys.stderr.write(json.dumps(rec) + "\n")
    sys.stderr.flush()


ALLOWED_ACTIONS = {"follow_up", "next_question", "finish", "clarify", "rephrase", "conversational"}
ALLOWED_INTENTS = {"answer", "greeting", "clarification", "rephrase", "meta_question", "off_topic"}


@dataclass
class TurnDecision:
    action: str                  # "follow_up" | "next_question" | "finish" | "clarify" | "rephrase" | "conversational"
    spoken_response: str
    reason_for_decision: str
    competency: str
    evidence_used: list[str]
    candidate_intent: str = "answer" # "answer" | "greeting" | "clarification" | "rephrase" | "meta_question" | "off_topic"


def validate_turn_decision(data: dict) -> TurnDecision:
    """Validate that the model output strictly conforms to the structured decision schema."""
    if not isinstance(data, dict):
        raise ValueError("Model decision must be a JSON object dictionary.")

    raw_action = str(data.get("action", "")).strip().lower()
    raw_intent = str(data.get("candidate_intent", "")).strip().lower()

    # Normalize action synonyms
    action = raw_action
    if action in {"greeting", "meta", "off_topic", "redirect", "small_talk", "divert"}:
        action = "conversational"
    elif action in {"clarification"}:
        action = "clarify"
    elif action in {"repeat", "rephrase_question"}:
        action = "rephrase"

    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action '{action}'. Must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}.")

    spoken = str(data.get("spoken_response", "")).strip()
    if not spoken:
        raise ValueError("spoken_response must be a non-empty string.")

    reason = str(data.get("reason_for_decision", "")).strip()
    comp = str(data.get("competency", "")).strip()

    raw_ev = data.get("evidence_used", [])
    if isinstance(raw_ev, list):
        ev_list = [str(x).strip() for x in raw_ev if str(x).strip()]
    elif isinstance(raw_ev, str) and raw_ev.strip():
        ev_list = [raw_ev.strip()]
    else:
        ev_list = []

    intent = raw_intent
    if intent not in ALLOWED_INTENTS:
        if raw_action in {"small_talk", "greeting"}:
            intent = "greeting"
        elif raw_action in {"meta", "process"}:
            intent = "meta_question"
        elif raw_action in {"divert", "off_topic", "redirect"}:
            intent = "off_topic"
        elif action == "clarify":
            intent = "clarification"
        elif action == "rephrase":
            intent = "rephrase"
        elif action == "conversational":
            intent = "meta_question"
        else:
            intent = "answer"

    return TurnDecision(
        action=action,
        spoken_response=spoken,
        reason_for_decision=reason,
        competency=comp,
        evidence_used=ev_list,
        candidate_intent=intent,
    )


@dataclass
class Turn:
    role: str                      # "interviewer" | "candidate"
    text: str
    stage: str
    question_id: Optional[str] = None
    kind: str = "message"          # message | followup
    evidence: Optional[list] = None # Retrieved document evidence chunks


class InterviewEngine:
    def __init__(self, config: InterviewConfig, llm: LLMClient, seed=None, store=None,
                 interview_id=None, session_id=None):
        self.config = config.validate()
        self.llm = llm
        self.store = store
        self.interview_id = interview_id
        self.session_id = session_id

        # Generate initial grounded questions or fallback bank
        if self.config.resume_text or self.config.job_description:
            qs = generate_personalized_questions(
                self.config, self.llm, store=self.store, interview_id=self.interview_id
            )
        else:
            qs = list(questions_for(self.config.interview_type))

        if self.config.randomize_order:
            random.Random(seed).shuffle(qs)

        self.questions = qs[: self.config.max_questions]
        self.rubric = rubric_for(self.config.interview_type)
        self.competencies_pool = [c.id for c in self.rubric]
        self.covered_competencies: list[str] = []

        self.stage = "greeting"
        self.q_index = -1
        self.questions_asked = 0
        self.followups_used = 0
        self.current_competency = self.questions[0].competency if self.questions else "experience"
        self.turns: list[Turn] = []
        self.finished = False
        self._report_cache = None

    def _retrieve(self, query: str, doc_type: Optional[str] = None, top_k: int = 3) -> list:
        """Fetch grounded RAG chunks strictly isolated by candidate interview_id."""
        if self.store is not None and self.interview_id:
            try:
                from .rag import retrieve_evidence
                return retrieve_evidence(
                    self.store, self.interview_id, query, doc_type=doc_type, top_k=top_k
                )
            except Exception:
                pass
        return []

    def _current_question_text(self) -> str:
        """Find the text of the current question or follow-up being addressed."""
        for t in reversed(self.turns):
            if t.role == "interviewer" and t.kind in ("message", "followup", "rephrase", "clarify"):
                return t.text
        first_q = self.questions[0] if self.questions else None
        return first_q.text if first_q else "Tell me about your technical background and experience."

    # ---- prompt helpers -------------------------------------------------
    def _system_prompt(self) -> str:
        c = self.config
        persona = (
            f"You are a professional, senior technical interviewer at {c.company} conducting a "
            f"{c.interview_type} interview for a {c.level}-level {c.role} role. "
            f"Speak as an attentive, discerning human interviewer. Ask ONE question or follow-up at a time. "
            f"Keep each spoken turn to 1-3 conversational sentences. Never reveal scoring mechanics, "
            f"and never answer your own questions."
        )
        if c.mode == "practice":
            persona += (" This is a practice session: after a weak answer you may give a brief, kind nudge, "
                        "but still challenge the candidate to explain their personal contribution.")
        else:
            persona += (" This is a real screening interview: stay neutral, professional, and rigorous.")

        if c.job_description:
            persona += f"\n\nTARGET JOB REQUIREMENTS:\n{c.job_description[:2000]}"
        if c.resume_text:
            persona += f"\n\nCANDIDATE'S VERIFIED RESUME:\n{c.resume_text[:3000]}"

        persona += (
            "\n\nGROUNDING RULES:\n"
            "1. Ground questions strictly in the verified background. Never invent experience or projects not mentioned in the resume or job description.\n"
            "2. Deeply understand what the candidate actually says. Refer to their stated projects, "
            "technologies, and experiences directly.\n"
            "3. If the candidate's answer is vague or lacks measurable outcomes, probe for clarification.\n"
            "4. If their answer is complete, move to an uncovered competency."
        )
        return persona

    def _history(self):
        msgs = [{"role": "system", "content": self._system_prompt()}]
        for t in self.turns:
            msgs.append({
                "role": "assistant" if t.role == "interviewer" else "user",
                "content": t.text,
            })
        return msgs

    def _record(self, role, text, *, kind="message", question_id=None, evidence=None):
        self.turns.append(Turn(role=role, text=text, stage=self.stage,
                               question_id=question_id, kind=kind, evidence=evidence))

    # ---- flow -----------------------------------------------------------
    def start(self) -> str:
        first_q = self.questions[0] if self.questions else None
        first = first_q.text if first_q else "Tell me about yourself and your background."
        self.stage = "core"
        self.q_index = 0
        self.questions_asked = 1
        self.current_competency = first_q.competency if first_q else (
            self.competencies_pool[0] if self.competencies_pool else "experience"
        )
        self.covered_competencies = [self.current_competency]
        self.followups_used = 0

        greeting = (
            f"Hi {self.config.candidate_name}, thanks for making the time. "
            f"I'm looking forward to our conversation about the {self.config.role} role. "
            f"Let's start: {first}"
        )
        try:
            lead = self.llm.chat(
                self._history() + [{
                    "role": "user",
                    "content": (f"Begin the interview: greet the candidate briefly, then ask "
                                f"this opening question: {first}"),
                }],
                task="greeting", max_tokens=220,
            ).strip()
            message = lead or greeting
        except LLMError as err:
            _log_diagnostic(
                "model_greeting_error",
                session_id=self.session_id or "",
                interview_id=self.interview_id or "",
                error=sanitize_secret_text(str(err)),
            )
            raise
        except Exception as exc:
            sanitized_exc = sanitize_secret_text(str(exc))
            _log_diagnostic(
                "model_greeting_error",
                session_id=self.session_id or "",
                interview_id=self.interview_id or "",
                error=sanitized_exc,
            )
            raise LLMError(f"Model failed to generate greeting: {sanitized_exc}") from exc

        ev = first_q.evidence if (first_q and first_q.evidence) else self._retrieve(
            f"{self.config.role} background skills summary", top_k=2
        )
        self._record("interviewer", message, kind="message",
                     question_id=first_q.id if first_q else self.current_competency,
                     evidence=ev if ev else None)
        return message

    def respond(self, user_text: str) -> dict:
        if self.finished:
            return {"message": "", "stage": "done", "done": True, "report": self.report()}

        self._record("candidate", user_text)

        # --- wrap-up phase: candidate asked their questions -------------
        if self.stage == "candidate_questions":
            try:
                wrap = self.llm.chat(
                    self._history() + [{"role": "user", "content":
                        "Wrap up the interview warmly in 1-2 sentences."}],
                    task="wrapup", max_tokens=160,
                ).strip()
            except Exception:
                wrap = ""
            if not wrap:
                wrap = ("Thanks for those questions — and for your time today. "
                        "We'll be in touch about next steps.")
            self._record("interviewer", wrap, kind="message")
            self.stage = "done"
            self.finished = True
            return {"message": wrap, "stage": "done", "done": True, "report": self.report()}

        # --- RAG semantic retrieval strictly for this session -----------
        rag_query = f"{self.current_competency} {user_text}"
        ev = self._retrieve(rag_query, top_k=3)
        sources = [
            {"chunk_id": c.get("chunk_id"), "doc_type": c.get("doc_type"),
             "section": c.get("section"), "chunk_idx": c.get("chunk_index")}
            for c in ev
        ]
        _log_diagnostic(
            "rag_retrieval",
            session_id=self.session_id or "",
            interview_id=self.interview_id or "",
            query=rag_query[:100],
            chunks_retrieved=len(ev),
            sources=sources,
            retrieved_context_included=bool(ev),
        )

        # --- Evaluate state for model decision -------------------------
        can_followup = (
            self.config.allow_followups and
            self.followups_used < self.config.max_followups_per_question
        )
        reached_max_questions = (self.questions_asked >= self.config.max_questions)
        uncovered = [c for c in self.competencies_pool if c not in self.covered_competencies]

        from .rag import format_evidence_block
        ev_block = format_evidence_block(ev) if ev else "(No specific document chunks retrieved)"

        current_q_str = self._current_question_text()
        prev_questions = [t.text for t in self.turns if t.role == "interviewer"]
        prev_q_str = "\n".join(f"- {q}" for q in prev_questions) if prev_questions else "(None)"

        decision_prompt = (
            "You are an expert AI interviewer conducting a live screening.\n\n"
            f"--- CANDIDATE LATEST MESSAGE ---\n\"{user_text}\"\n\n"
            f"--- CURRENT QUESTION BEING DISCUSSED ---\n\"{current_q_str}\"\n\n"
            f"--- CURRENT COMPETENCY BEING ASSESSED ---\n{self.current_competency}\n\n"
            f"--- INTERVIEW PROGRESS STATE ---\n"
            f"- Main Questions Asked: {self.questions_asked} of {self.config.max_questions}\n"
            f"- Follow-ups Used on Current Question: {self.followups_used} of {self.config.max_followups_per_question}\n"
            f"- Can Follow Up: {'True' if can_followup else 'False'}\n"
            f"- Reached Max Questions: {'True' if reached_max_questions else 'False'}\n"
            f"- Covered Competencies: {self.covered_competencies}\n"
            f"- Uncovered Competencies Remaining: {uncovered}\n\n"
            f"--- PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT) ---\n{prev_q_str}\n\n"
            f"--- RETRIEVED VERIFIED EVIDENCE (RAG) ---\n{ev_block}\n\n"
            "--- CANDIDATE INTENT & DECISION INSTRUCTIONS ---\n"
            "First, classify the candidate's intent from their latest message:\n"
            "1. GREETING / SMALL TALK (e.g. 'hello', 'good morning', 'thanks for having me'):\n"
            "   - action='conversational', candidate_intent='greeting'.\n"
            "   - Greet warmly in 1 brief sentence, then gently guide them to address the current question.\n"
            "2. CLARIFICATION REQUEST (e.g. 'what do you mean by trade-offs?', 'can you explain what you're asking?'):\n"
            "   - action='clarify', candidate_intent='clarification'.\n"
            "   - Explain the question or concept clearly WITHOUT giving away the expected answer or solution, then invite them to respond.\n"
            "3. REPEAT OR REPHRASE REQUEST (e.g. 'can you repeat the question?', 'could you rephrase that?'):\n"
            "   - action='rephrase', candidate_intent='rephrase'.\n"
            "   - Restate or simplify the current question clearly and invite them to answer.\n"
            "4. META-QUESTION / PROCESS QUESTION (e.g. 'are you rule-based?', 'are you an AI?', 'how is this evaluated?'):\n"
            "   - action='conversational', candidate_intent='meta_question'.\n"
            "   - Answer directly and truthfully in 1-2 sentences (e.g. explain that you are an AI interviewer driven by an LLM with RAG context rather than a rigid rule-based script), then guide them back to the current question.\n"
            "5. OFF-TOPIC REMARK (unrelated to the interview):\n"
            "   - action='conversational', candidate_intent='off_topic'.\n"
            "   - Briefly acknowledge and politely redirect back to the current question.\n"
            "6. INTERVIEW ANSWER (the candidate is answering the question):\n"
            "   - candidate_intent='answer'.\n"
            "   - If they mentioned a specific project, technology, or decision, refer to it directly.\n"
            "   - If the answer is vague or lacks measurable outcomes and Can Follow Up is True: action='follow_up'. Ask ONE targeted follow-up.\n"
            "   - If the answer explains what they did but lacks measurable impact or outcomes and Can Follow Up is True: action='follow_up'. Ask how success was measured without assuming metrics exist.\n"
            "   - If the answer is sufficiently complete, or if Can Follow Up is False:\n"
            "     - If Reached Max Questions is True: action='finish'.\n"
            "     - Otherwise: action='next_question'. Move to an uncovered competency from the list and ask a fresh grounded question.\n\n"
            "DO NOT repeat or rephrase any question from the PREVIOUSLY ASKED QUESTIONS list when moving to a new competency.\n"
            "Return JSON ONLY matching this structure:\n"
            "{\n"
            "  \"candidate_intent\": \"greeting\" | \"clarification\" | \"rephrase\" | \"meta_question\" | \"off_topic\" | \"answer\",\n"
            "  \"action\": \"conversational\" | \"clarify\" | \"rephrase\" | \"follow_up\" | \"next_question\" | \"finish\",\n"
            "  \"spoken_response\": \"1-3 conversational spoken sentences to the candidate\",\n"
            "  \"reason_for_decision\": \"Clear explanation of why this action and intent were chosen\",\n"
            "  \"competency\": \"competency name\",\n"
            "  \"evidence_used\": [\"facts or quotes from candidate message and retrieved documents\"]\n"
            "}"
        )

        # Omit the last candidate turn from _history() here because decision_prompt directly frames the candidate's latest message
        prior_turns = self.turns[:-1] if self.turns and self.turns[-1].role == "candidate" else self.turns
        messages = [{"role": "system", "content": self._system_prompt()}] + [
            {"role": "assistant" if t.role == "interviewer" else "user", "content": t.text}
            for t in prior_turns
        ] + [{"role": "user", "content": decision_prompt}]

        try:
            raw_decision = self.llm.chat(
                messages, task="decide_turn", temperature=0.3, max_tokens=400, json_mode=True
            )
            data = None
            try:
                data = json.loads(raw_decision)
            except Exception:
                start_idx, end_idx = raw_decision.find("{"), raw_decision.rfind("}")
                if 0 <= start_idx < end_idx:
                    data = json.loads(raw_decision[start_idx : end_idx + 1])

            if not data or not isinstance(data, dict):
                raise ValueError("Model output did not contain valid JSON object")

            decision = validate_turn_decision(data)
        except LLMError as err:
            _log_diagnostic("model_decision_error",
                            session_id=self.session_id or "",
                            interview_id=self.interview_id or "",
                            error=str(err))
            # Do not silently mask model errors with generic canned questions!
            raise
        except Exception as exc:
            _log_diagnostic("model_decision_error",
                            session_id=self.session_id or "",
                            interview_id=self.interview_id or "",
                            error=str(exc))
            raise LLMError(f"Model failed to return a valid structured decision: {exc}") from exc

        # Guardrails: enforce engine constraints for answer actions
        if decision.action == "follow_up" and not can_followup:
            if reached_max_questions:
                decision.action = "finish"
            else:
                decision.action = "next_question"
                if not decision.competency or decision.competency == self.current_competency:
                    decision.competency = uncovered[0] if uncovered else "ownership"

        if decision.action == "next_question" and reached_max_questions:
            decision.action = "finish"

        _log_diagnostic(
            "model_decision",
            session_id=self.session_id or "",
            interview_id=self.interview_id or "",
            action=decision.action,
            candidate_intent=decision.candidate_intent,
            competency=decision.competency,
            reason_for_decision=decision.reason_for_decision,
            evidence_used=decision.evidence_used,
            retrieved_context_included=bool(ev),
        )

        # Apply action
        if decision.action in ("clarify", "rephrase", "conversational"):
            # Mark candidate turn with detected non-answer intent so it's not scored as technical evidence
            if self.turns and self.turns[-1].role == "candidate":
                self.turns[-1].kind = decision.candidate_intent
            self._record("interviewer", decision.spoken_response, kind=decision.action,
                         question_id=self.current_competency, evidence=ev if ev else None)
            return {
                "message": decision.spoken_response,
                "stage": self.stage,
                "done": False,
                "evidence": ev,
                "decision": asdict(decision),
            }

        elif decision.action == "follow_up":
            if self.turns and self.turns[-1].role == "candidate":
                self.turns[-1].kind = "answer"
            self.followups_used += 1
            # Main question counter does NOT advance merely because a follow-up was asked!
            self._record("interviewer", decision.spoken_response, kind="followup",
                         question_id=self.current_competency, evidence=ev if ev else None)
            return {
                "message": decision.spoken_response,
                "stage": self.stage,
                "done": False,
                "evidence": ev,
                "decision": asdict(decision),
            }

        elif decision.action == "next_question":
            if self.turns and self.turns[-1].role == "candidate":
                self.turns[-1].kind = "answer"
            if decision.competency in self.covered_competencies and uncovered:
                decision.competency = uncovered[0]
            self.followups_used = 0
            self.questions_asked += 1
            self.q_index += 1
            self.current_competency = decision.competency or (uncovered[0] if uncovered else "depth")
            if self.current_competency not in self.covered_competencies:
                self.covered_competencies.append(self.current_competency)
            self._record("interviewer", decision.spoken_response, kind="message",
                         question_id=self.current_competency, evidence=ev if ev else None)
            return {
                "message": decision.spoken_response,
                "stage": self.stage,
                "done": False,
                "evidence": ev,
                "decision": asdict(decision),
            }

        else: # finish
            if self.turns and self.turns[-1].role == "candidate":
                self.turns[-1].kind = "answer"
            self.stage = "candidate_questions"
            msg = decision.spoken_response
            if not any(w in msg.lower() for w in ("questions", "anything", "ask me")):
                msg = f"{msg} That covers my questions. Do you have any questions for me?"
            self._record("interviewer", msg, kind="message")
            return {
                "message": msg,
                "stage": self.stage,
                "done": False,
                "decision": asdict(decision),
            }

    def report(self) -> dict:
        if self._report_cache is not None:
            return self._report_cache

        # Check if report already exists in store to prevent duplicate LLM calls
        if self.store is not None and self.interview_id:
            row = self.store.get_interview(self.interview_id)
            if row and row.get("evaluation_report"):
                try:
                    sc = json.loads(row["evaluation_report"])
                    self._report_cache = {
                        "config": asdict(self.config),
                        "transcript": [asdict(t) for t in self.turns],
                        "scorecard": sc,
                        "candidate_name": self.config.candidate_name,
                        "role": self.config.role,
                        "level": self.config.level,
                    }
                    return self._report_cache
                except Exception:
                    pass

        card = build_scorecard(self.llm, self.rubric, self.turns, self.config)
        self._report_cache = {
            "config": asdict(self.config),
            "transcript": [asdict(t) for t in self.turns],
            "scorecard": card.to_dict(),
            "candidate_name": self.config.candidate_name,
            "role": self.config.role,
            "level": self.config.level,
        }
        return self._report_cache

    @classmethod
    def rehydrate_from_store(
        cls,
        store: Any,
        interview_id: str,
        session_id: Optional[str] = None,
        llm: Optional[LLMClient] = None,
    ) -> Optional["InterviewEngine"]:
        """Reconstruct InterviewEngine state fully from persisted SQLite records."""
        row = store.get_interview(interview_id)
        if not row:
            return None

        cfg = InterviewConfig(
            mode=row.get("mode") or "practice",
            interview_type=row.get("interview_type") or "mixed",
            role=row.get("role") or "Software Engineer",
            level=row.get("level") or "mid",
            candidate_name=row.get("candidate_name") or "Candidate",
            candidate_email=row.get("candidate_email") or "",
            company=row.get("company") or "Acme Corp",
            language=row.get("language") or "en-IN",
            consent=bool(row.get("consent")),
            consent_version=row.get("consent_version") or "1.0",
            resume_text=row.get("resume_text") or "",
            job_description=row.get("job_description") or "",
        )

        if llm is None:
            from .llm import build_llm
            llm = build_llm(mock=True if row.get("llm") == "MockLLM" else False)

        engine = cls(
            config=cfg,
            llm=llm,
            seed=interview_id,
            store=store,
            interview_id=interview_id,
            session_id=session_id or row.get("session_id") or interview_id,
        )

        # Restore turns
        engine.turns = []
        for t in row.get("turns", []):
            engine.turns.append(
                Turn(
                    role=t.get("role", "interviewer"),
                    text=t.get("text", ""),
                    stage=t.get("stage", "core"),
                    question_id=t.get("question_id"),
                    kind=t.get("kind", "message"),
                    evidence=t.get("evidence"),
                )
            )

        # Reconstruct engine state from turns and status
        if row.get("status") == "completed":
            engine.finished = True
            engine.stage = "done"
        elif engine.turns:
            last_turn = engine.turns[-1]
            engine.stage = last_turn.stage or "core"
            if engine.stage == "core":
                questions_count = sum(1 for t in engine.turns if t.role == "interviewer" and t.kind == "message")
                engine.questions_asked = max(1, questions_count)
                engine.q_index = engine.questions_asked - 1
                followup_count = sum(1 for t in engine.turns if t.role == "interviewer" and t.kind == "followup")
                engine.followups_used = followup_count

        # Restore cached report if already scored
        if row.get("evaluation_report"):
            try:
                engine._report_cache = {
                    "config": asdict(engine.config),
                    "transcript": [asdict(t) for t in engine.turns],
                    "scorecard": json.loads(row["evaluation_report"]),
                    "candidate_name": engine.config.candidate_name,
                    "role": engine.config.role,
                    "level": engine.config.level,
                }
            except Exception:
                pass

        return engine

    def update_documents(self, resume_text: Optional[str] = None, job_description: Optional[str] = None):
        """Update active document context in memory."""
        if resume_text is not None:
            self.config.resume_text = resume_text
        if job_description is not None:
            self.config.job_description = job_description


