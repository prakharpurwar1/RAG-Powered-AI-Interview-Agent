"""Rubrics: the scoring backbone of a structured interview.

Every competency has a fixed weight and a 1-5 scale. Keeping the rubric external
to the LLM is what makes scores comparable across candidates and auditable later.
Swap these lists for your own competency model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Competency:
    id: str
    name: str
    weight: float
    description: str
    signals: str   # what a strong answer looks like


BEHAVIORAL = [
    Competency("communication", "Communication", 0.20,
               "Clarity, structure and conciseness of answers.",
               "Uses a STAR-like structure; easy to follow; no rambling."),
    Competency("ownership", "Ownership & Accountability", 0.20,
               "Takes responsibility for outcomes, not just tasks.",
               "Says 'I' for their own contribution; owns mistakes; drives to a result."),
    Competency("problem_solving", "Problem Solving", 0.20,
               "How they break down and resolve ambiguous problems.",
               "Frames the problem; weighs options; explains trade-offs."),
    Competency("collaboration", "Collaboration & Influence", 0.20,
               "Working with others, handling conflict, persuading.",
               "Concrete examples of aligning people; handles disagreement well."),
    Competency("motivation", "Motivation & Fit", 0.20,
               "Why this role/company; self-awareness and growth.",
               "Specific, honest reasons; realistic about strengths and gaps."),
]

TECHNICAL = [
    Competency("fundamentals", "Technical Fundamentals", 0.30,
               "Core knowledge for the role.",
               "Correct, precise use of concepts; no hand-waving."),
    Competency("problem_solving", "Problem Solving", 0.25,
               "Approach to technical problems.",
               "Clarifies requirements; considers edge cases; iterates."),
    Competency("depth", "Depth & Trade-offs", 0.20,
               "Goes beyond surface answers.",
               "Discusses alternatives, complexity, real constraints."),
    Competency("communication", "Technical Communication", 0.15,
               "Explaining technical ideas clearly.",
               "Explains to a non-expert; structured; checks understanding."),
    Competency("experience", "Relevant Experience", 0.10,
               "Hands-on evidence from past work.",
               "Concrete projects, their specific role, measurable impact."),
]

MIXED = [
    Competency("communication", "Communication", 0.20,
               "Clarity and structure of answers.", "Easy to follow; structured."),
    Competency("problem_solving", "Problem Solving", 0.25,
               "Analytical approach.", "Frames the problem; weighs options; decides."),
    Competency("fundamentals", "Technical Fundamentals", 0.25,
               "Core knowledge.", "Correct and precise."),
    Competency("ownership", "Ownership & Impact", 0.15,
               "Drives outcomes.", "Owns results; measurable impact."),
    Competency("motivation", "Motivation & Fit", 0.15,
               "Reason for the role.", "Specific and honest."),
]


def rubric_for(interview_type: str):
    return {"behavioral": BEHAVIORAL, "technical": TECHNICAL, "mixed": MIXED}[interview_type]
