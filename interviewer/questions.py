"""A small, extensible question bank.

In production you would generate or retrieve questions per role/JD (and that is
exactly where the LLM earns its keep). The bank keeps the prototype
deterministic and shows the shape the engine expects: every question maps to a
rubric competency and carries a follow-up hint.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    competency: str
    text: str
    followup_hint: str


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
