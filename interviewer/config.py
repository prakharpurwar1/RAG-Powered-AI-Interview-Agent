"""Configuration objects for the AI interviewer."""
from dataclasses import dataclass


@dataclass
class InterviewConfig:
    # "practice" (coaches the interviewee) or "hiring" (screens candidates)
    mode: str = "practice"
    # "behavioral" | "technical" | "mixed"
    interview_type: str = "mixed"
    role: str = "Software Engineer"
    level: str = "mid"                 # junior | mid | senior
    candidate_name: str = "Candidate"
    candidate_email: str = ""
    company: str = "the company"

    # --- consent / compliance ---
    consent: bool = False              # recorded consent (required for hiring mode)
    consent_version: str = "1.0"       # which consent text was agreed to
    retention_days: int = 90           # auto-purge after this many days

    language: str = "en-IN"
    max_questions: int = 6
    allow_followups: bool = True
    max_followups_per_question: int = 1

    # --- integrity ---
    randomize_order: bool = False      # shuffle questions per candidate
    proctoring: bool = True            # collect integrity signals

    def validate(self) -> "InterviewConfig":
        if self.mode not in {"practice", "hiring"}:
            raise ValueError("mode must be 'practice' or 'hiring'")
        if self.interview_type not in {"behavioral", "technical", "mixed"}:
            raise ValueError("interview_type must be behavioral/technical/mixed")
        if self.level not in {"junior", "mid", "senior"}:
            raise ValueError("level must be junior/mid/senior")
        if self.mode == "hiring" and not self.consent:
            raise ValueError("hiring mode requires recorded consent (consent=true)")
        self.max_questions = max(1, min(20, int(self.max_questions)))
        self.retention_days = max(1, min(3650, int(self.retention_days)))
        return self
