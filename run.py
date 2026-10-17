#!/usr/bin/env python3
"""Entry point.

    python run.py                      # FastAPI web UI + dashboard at http://127.0.0.1:8000
    python run.py cli [--mock]         # interview in the terminal
    python run.py purge                # delete interviews past their retention window
    python run.py export --email a@b   # data access request -> JSON on stdout
    python run.py export --all         # every interview -> JSON on stdout
    python run.py --port=9000 --db=my.db
"""
import argparse
import json
import os
import sys

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

from interviewer.cli import run as run_cli


def _store():
    from interviewer.store import Store
    return Store(os.environ.get("INTERVIEW_DB", "interviews.db"))


def main():
    argv = sys.argv[1:]

    if argv and argv[0] == "cli":
        return run_cli(argv[1:])

    if argv and argv[0] == "purge":
        n = _store().purge_expired()
        print(f"purged {n} expired interview(s)")
        return

    if argv and argv[0] == "export":
        ap = argparse.ArgumentParser(prog="run.py export")
        ap.add_argument("--email", default=None)
        ap.add_argument("--all", action="store_true")
        a = ap.parse_args(argv[1:])
        st = _store()
        if a.email:
            data = st.export_candidate(a.email)
        else:
            data = [st.get_interview(r["id"]) for r in st.list_interviews(sort="recent")]
        print(json.dumps(data, indent=2))
        return

    # Support cloud environment variables (Render, Railway, Heroku)
    env_port = int(os.environ.get("PORT", 8000))
    env_host = os.environ.get("HOST", "0.0.0.0")
    host, port, db = env_host, env_port, None
    for arg in argv:
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg.startswith("--db="):
            db = arg.split("=", 1)[1]

    if db:
        os.environ["INTERVIEW_DB"] = db

    # Prefer FastAPI + Uvicorn for production async performance
    try:
        import uvicorn
        from interviewer.app import app
        print(f"⚡ Starting FastAPI backend at http://{host}:{port} (Swagger docs: http://{host}:{port}/docs)")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        from interviewer.server import serve
        print(f"Starting standard Python server at http://{host}:{port}")
        serve(host=host, port=port, db=db)


if __name__ == "__main__":
    main()
