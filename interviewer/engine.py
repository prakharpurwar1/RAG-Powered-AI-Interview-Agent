"""The interviewer engine: a small, explicit state machine.

Stages: greeting -> core (N questions, optional follow-up) ->
candidate_questions -> wrapup -> done.

The engine owns the *flow*; the LLM only phrases turns and scores at the end.
That separation is what keeps interviews consistent, resumable and auditable —
and it means a flaky model can never derail the structure of the interview.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Optional

from .config import InterviewConfig
from .llm import LLMClient
from .questions import questions_for
from .rubric import rubric_for
from .scoring import build_scorecard

_ACKS = [
    "Thanks.", "Got it.", "Okay, that's helpful.", "Right.",
    "Understood.", "Interesting — thanks.",
]

_THIN_MARKERS = ("i don't know", "i dont know", "not sure", "no idea", "idk", "skip")


@dataclass
class Turn:
    role: str                      # "interviewer" | "candidate"
    text: str
    stage: str
    question_id: Optional[str] = None
    kind: str = "message"          # message | followup


class InterviewEngine:
    def __init__(self, config: InterviewConfig, llm: LLMClient, seed=None):
        self.config = config.validate()
        self.llm = llm
        # Integrity: shuffle question order per candidate so a shared "script"
        # is useless. Seeded, so a given candidate's order is reproducible.
        qs = list(questions_for(config.interview_type))
        if config.randomize_order:
            random.Random(seed).shuffle(qs)
        self.questions = qs[: config.max_questions]
        self.rubric = rubric_for(config.interview_type)
        self.stage = "greeting"
        self.q_index = -1
        self.followups_used = 0
        self.turns: list[Turn] = []
        self.finished = False
        self._report_cache = None

    # ---- prompt helpers -------------------------------------------------
    def _system_prompt(self) -> str:
        c = self.config
        persona = (
            f"You are a professional interviewer at {c.company} conducting a "
            f"{c.interview_type} interview for a {c.level}-level {c.role} role. "
            f"Speak as a warm but rigorous human interviewer. Ask ONE question at a "
            f"time. Keep each turn to 1-3 sentences. Never reveal scores, and never "
            f"answer your own questions."
        )
        if c.mode == "practice":
            persona += (" This is a practice session: after a weak answer you may give "
                        "a brief, kind nudge, but still let the candidate do the work.")
        else:
            persona += (" This is a real screening interview: stay neutral and do not "
                        "coach the candidate.")
        return persona

    def _history(self):
        msgs = [{"role": "system", "content": self._system_prompt()}]
        for t in self.turns:
            msgs.append({
                "role": "assistant" if t.role == "interviewer" else "user",
                "content": t.text,
            })
        return msgs

    def _record(self, role, text, *, kind="message", question_id=None):
        self.turns.append(Turn(role=role, text=text, stage=self.stage,
                               question_id=question_id, kind=kind))

    def _is_thin(self, text: str) -> bool:
        low = text.lower()
        if any(m in low for m in _THIN_MARKERS):
            return True
        return len(text.split()) < 25

    # ---- flow -----------------------------------------------------------
    def start(self) -> str:
        first = self.questions[0].text if self.questions else "Tell me about yourself."
        self.stage = "core"
        self.q_index = 0
        greeting = (
            f"Hi {self.config.candidate_name}, thanks for making the time. "
            f"I'm looking forward to our conversation about the {self.config.role} role. "
            f"Let's start: {first}"
        )
        # Let a real LLM warm it up; MockLLM returns "" and we keep the template.
        lead = self.llm.chat(
            self._history() + [{
                "role": "user",
                "content": (f"Begin the interview: greet the candidate briefly, then ask "
                            f"this first question: {first}"),
            }],
            task="greeting", max_tokens=220,
        ).strip()
        message = lead or greeting
        self._record("interviewer", message, kind="message",
                     question_id=self.questions[0].id if self.questions else None)
        return message

    def respond(self, user_text: str) -> dict:
        if self.finished:
            return {"message": "", "stage": "done", "done": True, "report": self.report()}

        self._record("candidate", user_text)

        # --- wrap-up phase: candidate asked their questions -------------
        if self.stage == "candidate_questions":
            wrap = self.llm.chat(
                self._history() + [{"role": "user", "content":
                    "Wrap up the interview warmly in 1-2 sentences."}],
                task="wrapup", max_tokens=160,
            ).strip()
            if not wrap:
                wrap = ("Thanks for those — and for your time today. "
                        "We'll be in touch about next steps.")
            self._record("interviewer", wrap, kind="message")
            self.stage = "done"
            self.finished = True
            return {"message": wrap, "stage": "done", "done": True, "report": self.report()}

        # --- decide: follow up, or advance ------------------------------
        current = (self.questions[self.q_index]
                   if 0 <= self.q_index < len(self.questions) else None)

        if (current is not None and self.config.allow_followups
                and self.followups_used < self.config.max_followups_per_question
                and self._is_thin(user_text)):
            self.followups_used += 1
            follow = self.llm.chat(
                self._history() + [{"role": "user", "content":
                    f"Ask ONE brief follow-up about: {current.text} "
                    f"Hint: {current.followup_hint}. Return only the question."}],
                task="followup", max_tokens=120,
            ).strip()
            if not follow:
                follow = ("Can you make that concrete — what was your specific role, "
                          "and what was the measurable outcome?")
            self._record("interviewer", follow, kind="followup", question_id=current.id)
            return {"message": follow, "stage": self.stage, "done": False}

        # --- advance to the next question -------------------------------
        self.followups_used = 0
        self.q_index += 1
        if self.q_index < len(self.questions):
            q = self.questions[self.q_index]
            ack = _ACKS[(self.q_index - 1) % len(_ACKS)]
            msg = f"{ack} {q.text}".strip()
            self._record("interviewer", msg, kind="message", question_id=q.id)
            return {"message": msg, "stage": self.stage, "done": False}

        # --- out of questions -> invite the candidate's questions -------
        self.stage = "candidate_questions"
        msg = "That covers my questions. Do you have any questions for me?"
        self._record("interviewer", msg, kind="message")
        return {"message": msg, "stage": self.stage, "done": False}

    def report(self) -> dict:
        if self._report_cache is None:
            card = build_scorecard(self.llm, self.rubric, self.turns, self.config)
            self._report_cache = {
                "config": asdict(self.config),
                "transcript": [asdict(t) for t in self.turns],
                "scorecard": card.to_dict(),
            }
        return self._report_cache
