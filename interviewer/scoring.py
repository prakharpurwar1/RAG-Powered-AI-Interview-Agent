"""Turn a transcript into a defensible scorecard.

The LLM proposes per-competency scores *with evidence*; this module owns the
weighting and the totals, so the arithmetic is transparent and testable and the
model cannot quietly invent a final number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class CompetencyScore:
    id: str
    name: str
    weight: float
    score: float          # 1-5
    evidence: str
    comment: str


@dataclass
class ScoreCard:
    competencies: list
    weighted_total: float   # 0-100
    recommendation: str
    summary: str

    def to_dict(self):
        return {
            "competencies": [vars(c) for c in self.competencies],
            "weighted_total": round(self.weighted_total, 1),
            "recommendation": self.recommendation,
            "summary": self.summary,
        }


def _prompt(rubric, turns, config):
    comp_lines = "\n".join(
        f"- {c.id} ({c.name}): {c.description} Strong looks like: {c.signals}"
        for c in rubric
    )
    transcript = "\n".join(f"{t.role.upper()}: {t.text}" for t in turns) or "(no transcript)"
    return (
        f"You are an expert interviewer scoring a {config.interview_type} interview "
        f"for a {config.level}-level {config.role} role.\n\n"
        f"Score each competency from 1 (poor) to 5 (excellent) using ONLY evidence "
        f"from the transcript. Be strict and fair; do not invent evidence. If a "
        f"competency was not probed, say so in the comment and score it 3.\n\n"
        f"Competencies:\n{comp_lines}\n\n"
        f"Transcript:\n{transcript}\n\n"
        'Return JSON exactly like: {"competencies": [{"id": "...", "score": 4, '
        '"evidence": "...", "comment": "..."}], "summary": "2-3 sentences"}'
    )


def build_scorecard(llm, rubric, turns, config) -> ScoreCard:
    msgs = [{"role": "user", "content": _prompt(rubric, turns, config)}]
    raw = llm.chat(msgs, task="score", temperature=0.2, max_tokens=1200, json_mode=True)
    data = _parse(raw)

    by_id = {c.get("id"): c for c in data.get("competencies", []) if isinstance(c, dict)}
    scores = []
    for c in rubric:
        got = by_id.get(c.id, {})
        scores.append(CompetencyScore(
            id=c.id,
            name=c.name,
            weight=c.weight,
            score=_clamp(got.get("score", 3.0)),
            evidence=str(got.get("evidence", "No evidence captured."))[:300],
            comment=str(got.get("comment", ""))[:300],
        ))

    total = sum((s.score / 5.0) * s.weight for s in scores) * 100.0
    return ScoreCard(
        competencies=scores,
        weighted_total=total,
        recommendation=_recommend(total),
        summary=(str(data.get("summary", ""))[:600] or "No summary produced."),
    )


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        # tolerate code fences / prose wrapped around the JSON object
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {}


def _clamp(v) -> float:
    try:
        v = float(v)
    except Exception:
        return 3.0
    return max(1.0, min(5.0, v))


def _recommend(total: float) -> str:
    if total >= 80:
        return "Strong hire"
    if total >= 65:
        return "Hire"
    if total >= 50:
        return "Borderline — lean no"
    return "No hire"
