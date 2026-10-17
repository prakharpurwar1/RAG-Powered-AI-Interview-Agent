"""Tests for the HTTP API, voice endpoints and reviewer flow.

Run:  python tests/test_server.py
"""
import base64
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# force offline voice + no API keys before importing the server
os.environ["VOICE_MOCK"] = "1"
os.environ.pop("LLM_API_KEY", None)
os.environ.pop("SARVAM_API_KEY", None)

from interviewer import server as srv  # noqa: E402


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def _start_server():
    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db)
    os.environ["INTERVIEW_DB"] = db
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port, db


def test_end_to_end_http():
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        # capabilities
        _, body, _ = _get(base + "/api/capabilities")
        assert json.loads(body)["voice"]["mock"] is True

        # start (practice, offline)
        _, body, _ = _post(base + "/api/start",
                           {"mock": True, "candidate_name": "Prakhar",
                            "interview_type": "mixed", "max_questions": 2})
        start = json.loads(body)
        sid, iid = start["session_id"], start["interview_id"]
        assert iid

        # answer through to the end
        answer = ("I led the migration and owned the outcome; we cut errors by 40 "
                  "percent and I mentored two juniors along the way.")
        done = None
        for _ in range(12):
            _, body, _ = _post(base + "/api/message", {"session_id": sid, "message": answer})
            done = json.loads(body)
            if done.get("done"):
                break
        assert done["done"] is True

        # voice: transcribe (mock) and speak (mock -> wav)
        _, body, _ = _post(base + "/api/transcribe",
                           {"audio_b64": base64.b64encode(b"\x00\x10" * 100).decode()})
        assert "transcript" in json.loads(body)
        st, audio, ctype = _post(base + "/api/speak", {"text": "Hello there"})
        assert st == 200 and ctype == "audio/wav" and audio[:4] == b"RIFF"

        # dashboard data
        _, body, _ = _get(base + "/api/interviews?sort=score")
        assert len(json.loads(body)["interviews"]) == 1
        _, body, _ = _get(base + f"/api/interviews/{iid}")
        detail = json.loads(body)
        assert detail["status"] == "completed" and len(detail["turns"]) > 0

        # human review
        _, body, _ = _post(base + f"/api/interviews/{iid}/review",
                           {"reviewer": "me", "notes": "good", "human_score": 4.2,
                            "human_recommendation": "Hire"})
        assert json.loads(body)["review_status"] == "reviewed"

        # exports
        _, body, ctype = _get(base + "/api/export.csv")
        assert b"candidate" in body and "csv" in ctype
        _, body, _ = _get(base + "/api/export.json")
        assert isinstance(json.loads(body), list)

        # pages serve
        _, body, _ = _get(base + "/")
        assert b"<html" in body.lower()
        _, body, _ = _get(base + "/dashboard")
        assert b"<html" in body.lower()
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_hiring_requires_consent():
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            _post(base + "/api/start", {"mock": True, "mode": "hiring"})
            assert False, "expected 500 for missing consent"
        except urllib.error.HTTPError as e:
            assert e.code == 500
        # with consent it succeeds
        _, body, _ = _post(base + "/api/start",
                           {"mock": True, "mode": "hiring", "consent": True})
        assert json.loads(body)["interview_id"]
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_ws_streaming_bridge():
    httpd, port, db = _start_server()
    try:
        from interviewer.streaming import WebSocketClient
        c = WebSocketClient(f"ws://127.0.0.1:{port}/api/stream-asr", timeout=10).connect()
        assert c.recv_json()["event"] == "ready"
        c.send_text(json.dumps({
            "event": "audio_input",
            "audio": base64.b64encode(b"\x00\x10" * 160).decode(),
        }))
        ev = c.recv_json()
        assert "event" in ev
        c.send_text(json.dumps({"event": "end"}))
        c.close()
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_health_and_capabilities():
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        _, body, _ = _get(base + "/api/health")
        h = json.loads(body)
        assert h["status"] == "ok" and "sessions" in h
        _, body, _ = _get(base + "/api/capabilities")
        assert json.loads(body)["proctoring"] is True
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_integrity_signal_persisted():
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        _, body, _ = _post(base + "/api/start",
                           {"mock": True, "mode": "hiring", "consent": True,
                            "max_questions": 2})
        s = json.loads(body)
        sid, iid = s["session_id"], s["interview_id"]
        _post(base + "/api/signal", {"session_id": sid, "kind": "paste"})

        answer = ("I led the design and owned delivery; we cut latency by 40 percent "
                  "and I mentored two engineers along the way.")
        for _ in range(12):
            _, b, _ = _post(base + "/api/message", {"session_id": sid, "message": answer})
            if json.loads(b).get("done"):
                break

        _, body, _ = _get(base + f"/api/interviews/{iid}")
        d = json.loads(body)
        assert d["integrity_score"] is not None and d["integrity_score"] < 100
        assert any(f["kind"] == "paste" for f in d["flags"])
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_dashboard_auth_gate():
    os.environ["DASHBOARD_TOKEN"] = "secret"
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            _get(base + "/api/interviews")
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        st, _, _ = _get(base + "/api/interviews?token=secret")
        assert st == 200
        st, _, _ = _get(base + "/")            # candidate UI stays open
        assert st == 200
        st, _, _ = _get(base + "/api/health")  # health stays open
        assert st == 200
    finally:
        httpd.shutdown()
        os.environ.pop("DASHBOARD_TOKEN", None)
        try:
            os.unlink(db)
        except OSError:
            pass


def test_document_endpoints_and_grounded_start():
    import io, docx
    # Generate docx in memory
    doc = docx.Document()
    doc.add_paragraph("Devin Patel - Staff Infrastructure Engineer")
    doc.add_paragraph("Specialized in Kubernetes, Kafka, and distributed data stores.")
    buf = io.BytesIO()
    doc.save(buf)
    b64_doc = base64.b64encode(buf.getvalue()).decode()

    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        # 1. Parse valid DOCX
        st, body, _ = _post(base + "/api/documents/parse", {
            "filename": "devin_resume.docx",
            "content_b64": b64_doc,
        })
        assert st == 200
        res = json.loads(body)
        assert res["status"] == "ok"
        assert "Devin Patel" in res["text"]
        assert res["word_count"] > 5

        # 2. Unsupported extension error
        try:
            _post(base + "/api/documents/parse", {
                "filename": "resume.png",
                "content_b64": b64_doc,
            })
            assert False, "expected 400 for unsupported extension"
        except urllib.error.HTTPError as e:
            assert e.code == 400

        # 3. Validate text JD
        st, body, _ = _post(base + "/api/documents/validate-text", {
            "text": "Seeking Staff Infrastructure Engineer to scale our Kubernetes clusters.",
            "label": "Job description",
        })
        assert st == 200
        assert "Kubernetes" in json.loads(body)["text"]

        # 4. Start interview with resume and JD
        st, body, _ = _post(base + "/api/start", {
            "mock": True,
            "candidate_name": "Devin Patel",
            "interview_type": "technical",
            "role": "Staff Infrastructure Engineer",
            "resume_text": res["text"],
            "job_description": "Scale Kubernetes and Kafka infrastructure.",
        })
        assert st == 200
        start_res = json.loads(body)
        iid = start_res["interview_id"]

        # Verify persisted resume and JD in store
        detail = srv._store().get_interview(iid)
        assert "Devin Patel" in detail["resume_text"]
        assert "Kubernetes" in detail["job_description"]

        # 5. Verify RAG chunks indexed via API
        st, body, _ = _get(base + f"/api/interviews/{iid}/chunks")
        assert st == 200
        chunk_data = json.loads(body)
        assert chunk_data["count"] >= 1
        assert any("Kubernetes" in c["text"] or "Devin" in c["text"] for c in chunk_data["chunks"])

        # 6. Verify RAG retrieval endpoint
        st, body, _ = _post(base + f"/api/interviews/{iid}/retrieve", {
            "query": "Kubernetes cluster scaling",
            "top_k": 2,
        })
        assert st == 200
        ret_data = json.loads(body)
        assert len(ret_data["results"]) >= 1
        assert "score" in ret_data["results"][0]
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
