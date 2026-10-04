"""Terminal runner — handy for quick testing and for scripting interviews."""
from __future__ import annotations

import argparse

from .config import InterviewConfig
from .engine import InterviewEngine
from .llm import build_llm


def run(argv=None):
    ap = argparse.ArgumentParser(description="Run an AI interview in the terminal.")
    ap.add_argument("--mock", action="store_true", help="offline, no API key needed")
    ap.add_argument("--mode", default="practice", choices=["practice", "hiring"])
    ap.add_argument("--type", dest="interview_type", default="mixed",
                    choices=["behavioral", "technical", "mixed"])
    ap.add_argument("--role", default="Software Engineer")
    ap.add_argument("--level", default="mid", choices=["junior", "mid", "senior"])
    ap.add_argument("--name", default="Candidate")
    ap.add_argument("--questions", type=int, default=6)
    args = ap.parse_args(argv)

    cfg = InterviewConfig(
        mode=args.mode, interview_type=args.interview_type, role=args.role,
        level=args.level, candidate_name=args.name, max_questions=args.questions,
    )
    engine = InterviewEngine(cfg, build_llm(mock=args.mock))

    print(f"\n=== AI Interviewer ({engine.llm.name}) ===")
    print("Type your answers. 'quit' to stop early.\n")
    print(f"Interviewer: {engine.start()}\n")

    while not engine.finished:
        try:
            reply = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if reply.lower() in {"quit", "exit"}:
            break
        out = engine.respond(reply)
        print(f"\nInterviewer: {out['message']}\n")

    sc = engine.report()["scorecard"]
    print("=" * 62)
    print(f"RESULT: {sc['recommendation']}   (weighted score {sc['weighted_total']}/100)")
    print("-" * 62)
    for c in sc["competencies"]:
        print(f"  {c['name']:<28} {c['score']:>3}/5   (weight {c['weight']})")
    print("-" * 62)
    print(f"{sc['summary']}\n")
