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

    # --- Phase 2: Resume & Job Description ---
    resume_text: str = ""              # Extracted readable resume text
    job_description: str = ""          # Target job description text

    def validate(self) -> "InterviewConfig":
        if not self.mode or self.mode not in {"practice", "hiring"}:
            raise ValueError("mode must be 'practice' or 'hiring'")
        if not self.interview_type or self.interview_type not in {"behavioral", "technical", "mixed"}:
            raise ValueError("interview_type must be behavioral/technical/mixed")
        if not str(self.role or "").strip():
            raise ValueError("role is required")
        if not self.level or self.level not in {"junior", "mid", "senior"}:
            raise ValueError("level must be junior/mid/senior")
        if not str(self.candidate_name or "").strip():
            raise ValueError("candidate_name is required")
        if not str(self.language or "").strip():
            raise ValueError("language is required")
        if self.mode == "hiring" and not self.consent:
            raise ValueError("hiring mode requires recorded consent (consent=true)")
        try:
            mq = int(self.max_questions)
            if mq <= 0:
                raise ValueError()
            self.max_questions = max(1, min(20, mq))
        except (ValueError, TypeError):
            raise ValueError("max_questions must be a positive integer")
        self.retention_days = max(1, min(3650, int(self.retention_days)))
        self.resume_text = str(self.resume_text or "").strip()
        self.job_description = str(self.job_description or "").strip()
        self.candidate_email = str(self.candidate_email or "").strip()
        if self.candidate_email:
            import re
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", self.candidate_email):
                raise ValueError("candidate_email must be a valid email address")
        return self
