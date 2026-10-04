"""Integrity signals for remote interviews.

Collects lightweight proctoring events (tab switches, pastes, suspiciously fast
answers) and turns them into a 0-100 integrity score plus human-readable flags.

Design stance: this is *advisory*. It flags sessions for human review; it never
auto-rejects a candidate. Integrity signals are noisy and biased, so the rule is
the same as for scoring -- surface the evidence, let a person decide.
"""
from __future__ import annotations

import time

# weight subtracted per occurrence, and a cap on how many count
RULES = {
    "paste":      {"weight": 15, "cap": 3, "label": "Text pasted into an answer"},
    "blur":       {"weight": 5,  "cap": 3, "label": "Switched away from the interview tab"},
    "fast":       {"weight": 10, "cap": 2, "label": "Unusually fast, lengthy answer"},
    "copy":       {"weight": 8,  "cap": 2, "label": "Copied text from the interview"},
    "devtools":   {"weight": 10, "cap": 1, "label": "Developer tools / inspector opened"},
}


class IntegrityMonitor:
    """Accumulates signals for one session. Cheap, thread-safe enough for a demo."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events = []

    def record(self, kind: str, detail=None):
        if not self.enabled or kind not in RULES:
            return
        self.events.append({"kind": kind, "detail": detail, "at": round(time.time(), 2)})

    def _counts(self):
        counts = {}
        for e in self.events:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        return counts

    def flags(self) -> list:
        out = []
        for kind, n in self._counts().items():
            rule = RULES[kind]
            counted = min(n, rule["cap"])
            out.append({
                "kind": kind,
                "label": rule["label"],
                "count": n,
                "penalty": rule["weight"] * counted,
            })
        out.sort(key=lambda f: -f["penalty"])
        return out

    def score(self) -> float:
        penalty = sum(f["penalty"] for f in self.flags())
        return float(max(0, min(100, 100 - penalty)))

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "score": self.score(),
            "flags": self.flags(),
            "events": len(self.events),
            "detail": self.events[-50:],
        }
