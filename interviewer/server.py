"""Web server + JSON API + reviewer dashboard + WebSocket voice bridge.

Standard library only.

Hardening in this build
-----------------------
* optional dashboard auth  (set ``DASHBOARD_TOKEN``)
* request size cap, per-IP rate limit on session creation
* session TTL sweep, health endpoint
* structured JSON logs for every consequential action
* retention sweep on startup and an on-demand purge
* integrity signals recorded per session and stored with the interview

Routes
------
  GET  /                       chat UI
  GET  /dashboard              reviewer dashboard (auth)
  GET  /api/capabilities       what voice/LLM is available
  GET  /api/health             liveness + counters

  POST /api/start              begin an interview (persists it)
  POST /api/message            one candidate turn (persists the turn)
  POST /api/report             transcript + scorecard (persists scorecard + integrity)
  POST /api/signal             integrity event from the client (blur / paste / …)

  POST /api/transcribe         audio -> text   (Sarvam Saaras REST)
  POST /api/speak              text -> audio   (Sarvam Bulbul, HTTP stream)
  WS   /api/stream-asr         live partial transcripts (Sarvam Realtime bridge)

  GET  /api/interviews         list (filter + sort)          (auth)
  GET  /api/interviews/<id>    detail                        (auth)
  POST /api/interviews/<id>/review   human override + notes  (auth)
  POST /api/interviews/<id>/delete   remove (retention)      (auth)
  GET  /api/candidate?email=   everything held about a person(auth)
  POST /api/candidate/delete   erasure request               (auth)
  GET  /api/stats              dashboard summary             (auth)
  GET  /api/export.csv|.json   audit export                  (auth)
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import InterviewConfig
from .documents import DocumentError, extract_document_text, validate_text_document
from .engine import InterviewEngine
from .integrity import IntegrityMonitor
from .llm import build_llm, LLMError, sanitize_secret_text
from .streaming import (OP_CLOSE, OP_TEXT, _Buffered, build_streaming_asr,
                        is_websocket_request, read_frame, send_frame, server_handshake)
from .store import Store, StoreError
from .voice import build_voice

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")

STARTED_AT = time.time()
MAX_BODY = int(os.environ.get("MAX_BODY_BYTES", 8 * 1024 * 1024))
SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", 6 * 3600))
START_RATE = int(os.environ.get("START_RATE_PER_MIN", 60))

_SESSIONS = {}
_LOCK = threading.Lock()
_STORE = None
_STORE_PATH = None
_VOICE = None
_VOICE_MOCK = None
_RATE = {}


def _log(event: str, **fields):
    rec = {"t": round(time.time(), 3), "event": event}
    rec.update(fields)
    sys.stderr.write(json.dumps(rec) + "\n")
    sys.stderr.flush()


def _store():
    global _STORE, _STORE_PATH
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
        if _STORE is None or _STORE_PATH != db_url:
            _STORE = Store(database_url=db_url)
            _STORE_PATH = db_url
        return _STORE
    path = os.environ.get("INTERVIEW_DB", "interviews.db")
    if _STORE is None or _STORE_PATH != path:
        _STORE = Store(path)
        _STORE_PATH = path
    return _STORE


def _voice():
    global _VOICE, _VOICE_MOCK
    mock = bool(os.environ.get("VOICE_MOCK"))
    if _VOICE is None or _VOICE_MOCK != mock:
        _VOICE = build_voice(mock=mock)
        _VOICE_MOCK = mock
    return _VOICE


def _sweep_sessions():
    now = time.time()
    with _LOCK:
        dead = [sid for sid, s in _SESSIONS.items() if now - s.get("last", now) > SESSION_TTL]
        for sid in dead:
            _SESSIONS.pop(sid, None)
    if dead:
        _log("sessions_swept", count=len(dead))


def _rate_ok(ip):
    now = time.time()
    bucket = [t for t in _RATE.get(ip, []) if now - t < 60]
    if len(bucket) >= START_RATE:
        _RATE[ip] = bucket
        return False
    bucket.append(now)
    _RATE[ip] = bucket
    return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    # ---- low-level helpers ---------------------------------------------
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            raise ValueError(f"request body too large ({length} bytes)")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def _query(self):
        if "?" not in self.path:
            return {}
        out = {}
        for pair in self.path.split("?", 1)[1].split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k] = v
        return out

    def _dash_ok(self, query):
        token = os.environ.get("DASHBOARD_TOKEN", "")
        if not token:
            return True
        given = query.get("token") or ""
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            given = auth[7:]
        return given == token

    def _file(self, name):
        try:
            with open(os.path.join(_WEB_DIR, name), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        except FileNotFoundError:
            self._send(500, {"error": f"{name} not found"})

    # ---- GET ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        query = self._query()
        _sweep_sessions()

        if path == "/api/stream-asr":
            return self._ws_bridge()

        if path in ("/", "/index.html"):
            return self._file("index.html")
        if path in ("/dashboard", "/dashboard.html"):
            if not self._dash_ok(query):
                return self._send(401, {"error": "unauthorized"})
            return self._file("dashboard.html")

        if path == "/api/capabilities":
            return self._send(200, {"voice": _voice().capabilities(),
                                    "llm": build_llm().name,
                                    "proctoring": True})
        if path in ("/api/health", "/health", "/healthz"):
            with _LOCK:
                n = len(_SESSIONS)
            return self._send(200, {"status": "ok", "uptime_s": round(time.time() - STARTED_AT, 1),
                                    "sessions": n, "db": _store().path})


        if path.startswith("/api/") and not self._dash_ok(query):
            return self._send(401, {"error": "unauthorized"})

        if path == "/api/interviews":
            return self._send(200, {"interviews":
                                    _store().list_interviews(query.get("status"),
                                                             query.get("sort", "score"))})
        if path == "/api/stats":
            return self._send(200, _store().stats())
        if path == "/api/candidate":
            email = query.get("email", "")
            return self._send(200, {"candidate": email,
                                    "interviews": _store().export_candidate(email)})
        if path.startswith("/api/interviews/"):
            parts = path.split("/")
            iid = parts[3]
            sub = parts[4] if len(parts) > 4 else ""
            if sub == "chunks":
                chunks = _store().get_document_chunks(iid, doc_type=query.get("doc_type"))
                cleaned = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
                return self._send(200, {"interview_id": iid, "chunks": cleaned, "count": len(cleaned)})
            detail = _store().get_interview(iid)
            return self._send(200, detail) if detail else self._send(404, {"error": "not found"})
        if path == "/api/export.csv":
            return self._send(200, self._export_csv().encode(), "text/csv",
                              {"Content-Disposition": "attachment; filename=interviews.csv"})
        if path == "/api/export.json":
            return self._send(200, self._export_json().encode(), "application/json",
                              {"Content-Disposition": "attachment; filename=interviews.json"})

        return self._send(404, {"error": "not found"})

    # ---- POST -----------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        query = self._query()
        _sweep_sessions()
        try:
            payload = self._body()
        except ValueError as e:
            return self._send(413, {"error": str(e)})

        try:
            if path == "/api/start":
                return self._start(payload)
            if path == "/api/message":
                return self._message(payload)
            if path == "/api/report":
                return self._report(payload)
            if path == "/api/signal":
                return self._signal(payload)
            if path == "/api/transcribe":
                return self._transcribe(payload)
            if path == "/api/speak":
                return self._speak(payload)
            if path == "/api/documents/parse":
                return self._parse_document(payload)
            if path == "/api/documents/validate-text":
                return self._validate_text_document(payload)

            if path.startswith("/api/") and not self._dash_ok(query):
                return self._send(401, {"error": "unauthorized"})

            if path == "/api/candidate/delete":
                n = _store().delete_candidate(payload.get("email", ""))
                _log("candidate_erased", email=payload.get("email"), count=n)
                return self._send(200, {"deleted": n})
            if path.startswith("/api/interviews/"):
                parts = path.split("/")
                iid, action = parts[3], (parts[4] if len(parts) > 4 else "")
                if action == "retrieve":
                    from .rag import retrieve_evidence
                    results = retrieve_evidence(
                        _store(), iid, payload.get("query", ""),
                        doc_type=payload.get("doc_type"),
                        top_k=int(payload.get("top_k", 3)),
                    )
                    return self._send(200, {"interview_id": iid, "query": payload.get("query", ""), "results": results})
                if action == "review":
                    return self._review(iid, payload)
                if action == "delete":
                    _store().delete_interview(iid)
                    _log("interview_deleted", interview=iid)
                    return self._send(200, {"deleted": iid})
        except Exception as e:
            _log("error", path=path, message=str(e))
            return self._send(500, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    # ---- interview endpoints -------------------------------------------
    def _start(self, payload):
        ip = self.client_address[0]
        if not _rate_ok(ip):
            _log("rate_limited", ip=ip)
            return self._send(429, {"error": "too many interviews started; slow down"})

        allowed = InterviewConfig.__dataclass_fields__.keys()
        cfg = InterviewConfig(**{k: v for k, v in payload.items() if k in allowed})
        # hiring defaults to a randomised question order (anti-cheating)
        if cfg.mode == "hiring" and payload.get("randomize_order") is None:
            cfg.randomize_order = True
        cfg.validate()

        sid = os.urandom(8).hex()
        try:
            st = _store()
            iid = st.create_interview(vars(cfg), llm="")
        except StoreError as err:
            sanitized_err = sanitize_secret_text(str(err))
            _log("db_start_error", error=sanitized_err)
            return self._send(503, {"error": "Database service unavailable. Please check database connectivity."})

        # Phase 3: Index Resume & Job Description via RAG Vector Storage
        if cfg.resume_text or cfg.job_description:
            try:
                from .rag import index_interview_documents
                index_interview_documents(st, iid, resume_text=cfg.resume_text, job_description=cfg.job_description)
            except Exception as exc:
                _log("rag_indexing_error", interview=iid, error=str(exc))

        engine = InterviewEngine(cfg, build_llm(mock=bool(payload.get("mock", False))), seed=iid, store=st, interview_id=iid, session_id=sid)
        monitor = IntegrityMonitor(enabled=cfg.proctoring)
        with _LOCK:
            _SESSIONS[sid] = {"engine": engine, "iid": iid, "persisted": 0,
                              "monitor": monitor, "created": time.time(),
                              "last": time.time(), "prompt_at": time.time()}
        try:
            msg = engine.start()
        except LLMError as err:
            with _LOCK:
                _SESSIONS.pop(sid, None)
            upstream_status = getattr(err, "status_code", None)
            sanitized_msg = sanitize_secret_text(str(err))
            _log("llm_start_error", session=sid, interview=iid, upstream_status=upstream_status, error=sanitized_msg)
            status_info = f" (HTTP {upstream_status})" if upstream_status else ""
            return self._send(502, {"error": f"Interview model error: Upstream AI service request failed{status_info}."})
        self._persist_new_turns(sid)
        _log("interview_started", interview=iid, mode=cfg.mode, type=cfg.interview_type)
        return self._send(200, {"session_id": sid, "interview_id": iid, "message": msg,
                                "stage": engine.stage, "llm": engine.llm.name,
                                "proctoring": cfg.proctoring})

    def _message(self, payload):
        sess = _SESSIONS.get(payload.get("session_id"))
        if not sess:
            return self._send(404, {"error": "unknown session"})
        sess["last"] = time.time()

        text = payload.get("message", "")
        # integrity: a long, correct-looking answer arriving suspiciously fast
        elapsed = time.time() - sess.get("prompt_at", time.time())
        if elapsed < 6 and len(text.split()) > 40:
            sess["monitor"].record("fast", detail=round(elapsed, 1))

        try:
            out = sess["engine"].respond(text)
        except LLMError as err:
            upstream_status = getattr(err, "status_code", None)
            sanitized_msg = sanitize_secret_text(str(err))
            _log("llm_generation_error", session=payload.get("session_id"), interview_id=sess["iid"], upstream_status=upstream_status, error=sanitized_msg)
            status_info = f" (HTTP {upstream_status})" if upstream_status else ""
            return self._send(502, {"error": f"Interview model error: Upstream AI service request failed{status_info}."})
        sess["prompt_at"] = time.time()
        self._persist_new_turns(payload["session_id"])
        if out.get("done"):
            self._finalize(sess)
            out["interview_id"] = sess["iid"]
        return self._send(200, out)

    def _report(self, payload):
        sess = _SESSIONS.get(payload.get("session_id"))
        if not sess:
            return self._send(404, {"error": "unknown session"})
        rep = sess["engine"].report()
        self._finalize(sess, rep=rep)
        rep["interview_id"] = sess["iid"]
        return self._send(200, rep)

    def _finalize(self, sess, rep=None):
        rep = rep or sess["engine"].report()
        _store().save_scorecard(sess["iid"], rep["scorecard"])
        summary = sess["monitor"].summary()
        _store().save_integrity(sess["iid"], summary)
        _log("interview_completed", interview=sess["iid"],
             score=rep["scorecard"].get("weighted_total"),
             integrity=summary["score"], flags=len(summary["flags"]))

    def _signal(self, payload):
        sess = _SESSIONS.get(payload.get("session_id"))
        if not sess:
            return self._send(404, {"error": "unknown session"})
        sess["last"] = time.time()
        sess["monitor"].record(payload.get("kind", ""), detail=payload.get("detail"))
        return self._send(200, {"ok": True, "score": sess["monitor"].score()})

    def _persist_new_turns(self, sid):
        sess = _SESSIONS.get(sid)
        if not sess:
            return
        turns = sess["engine"].turns
        while sess["persisted"] < len(turns):
            _store().add_turn(sess["iid"], sess["persisted"], vars(turns[sess["persisted"]]))
            sess["persisted"] += 1

    # ---- voice endpoints ------------------------------------------------
    def _transcribe(self, payload):
        audio = payload.get("audio_b64", "")
        if not audio:
            return self._send(400, {"error": "audio_b64 required"})
        try:
            data = base64.b64decode(audio)
        except Exception:
            return self._send(400, {"error": "invalid base64 audio"})
        text = _voice().transcribe(data, filename=payload.get("filename", "audio.webm"),
                                   language_code=payload.get("language", "en-IN"))
        return self._send(200, {"transcript": text})

    def _speak(self, payload):
        text = payload.get("text", "").strip()
        if not text:
            return self._send(400, {"error": "text required"})
        audio = _voice().synthesize_stream(text, language_code=payload.get("language", "en-IN"),
                                           speaker=payload.get("speaker", "shubh"))
        return self._send(200, audio, "audio/wav")

    # ---- document endpoints (Phase 2) -----------------------------------
    def _parse_document(self, payload):
        filename = payload.get("filename", "")
        b64_content = payload.get("content_b64", "")
        if not filename or not b64_content:
            return self._send(400, {"error": "filename and content_b64 are required."})
        try:
            content = base64.b64decode(b64_content)
            text, meta = extract_document_text(filename, content)
            return self._send(200, {
                "status": "ok",
                "filename": meta["filename"],
                "format": meta["format"],
                "size_bytes": meta["size_bytes"],
                "char_count": meta["char_count"],
                "word_count": meta["word_count"],
                "text": text,
            })
        except DocumentError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def _validate_text_document(self, payload):
        text = payload.get("text", "")
        label = payload.get("label", "Job description")
        try:
            cleaned = validate_text_document(text, doc_label=label)
            return self._send(200, {
                "status": "ok",
                "text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            })
        except DocumentError as e:
            return self._send(400, {"error": str(e)})

    # ---- reviewer endpoints ---------------------------------------------
    def _review(self, iid, payload):
        if not _store().get_interview(iid):
            return self._send(404, {"error": "not found"})
        hs = payload.get("human_score")
        try:
            hs = float(hs) if hs not in (None, "") else None
        except Exception:
            hs = None
        _store().save_review(iid, payload.get("reviewer", "reviewer"),
                             payload.get("notes", ""), hs,
                             payload.get("human_recommendation", ""))
        _log("interview_reviewed", interview=iid, reviewer=payload.get("reviewer"))
        return self._send(200, _store().get_interview(iid))

    def _export_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "created_at", "candidate", "email", "role", "type", "level",
                    "status", "score", "recommendation", "integrity", "flags",
                    "human_score", "human_recommendation", "review_status", "reviewer",
                    "consent", "consent_version", "retention_until"])
        for r in _store().list_interviews(sort="recent"):
            flags = r.get("flags") or ""
            if isinstance(flags, str):
                try:
                    flags = ";".join(f.get("label", "") for f in json.loads(flags))
                except (ValueError, TypeError):
                    flags = ""
            w.writerow([r["id"], r["created_at"], r["candidate_name"], r["candidate_email"],
                        r["role"], r["interview_type"], r["level"], r["status"],
                        r["weighted_total"], r["recommendation"], r["integrity_score"], flags,
                        r["human_score"], r["human_recommendation"], r["review_status"],
                        r["reviewer"], r["consent"], r["consent_version"], r["retention_until"]])
        return buf.getvalue()

    def _export_json(self):
        return json.dumps([_store().get_interview(r["id"])
                           for r in _store().list_interviews(sort="recent")], indent=2)

    # ---- WebSocket bridge ----------------------------------------------
    def _ws_bridge(self):
        if not is_websocket_request(self.headers):
            return self._send(400, {"error": "expected websocket upgrade"})
        conn = self.connection
        if not server_handshake(self.headers, conn):
            return
        self.close_connection = True
        reader = _Buffered(conn)
        asr = build_streaming_asr()
        try:
            asr.connect()
            send_frame(conn, json.dumps({"event": "ready"}).encode(), OP_TEXT)
            while True:
                try:
                    opcode, payload, _fin = read_frame(reader)
                except Exception:
                    break
                if opcode == OP_CLOSE:
                    break
                if opcode != OP_TEXT:
                    continue
                msg = json.loads(payload.decode() or "{}")
                ev = msg.get("event")
                if ev == "audio_input":
                    asr.send_audio(base64.b64decode(msg.get("audio", "")))
                    try:
                        send_frame(conn, json.dumps(asr.next_event()).encode(), OP_TEXT)
                    except Exception:
                        pass
                elif ev == "flush":
                    asr.flush()
                    try:
                        send_frame(conn, json.dumps(asr.next_event()).encode(), OP_TEXT)
                    except Exception:
                        pass
                elif ev == "end":
                    break
        except Exception as e:
            try:
                send_frame(conn, json.dumps({"event": "error", "message": str(e)}).encode(), OP_TEXT)
            except Exception:
                pass
        finally:
            try:
                asr.close()
            except Exception:
                pass


def serve(host=None, port=None, db=None):
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))
    if db:
        os.environ["INTERVIEW_DB"] = db
    purged = _store().purge_expired()
    if purged:
        _log("retention_purge", removed=purged)
    httpd = ThreadingHTTPServer((host, port), Handler)
    _log("server_start", host=host, port=port, db=_store().path, purged=purged,
         auth=bool(os.environ.get("DASHBOARD_TOKEN")))
    print(f"\n  AI Interviewer      ->  http://{host}:{port}")
    print(f"  Reviewer dashboard  ->  http://{host}:{port}/dashboard")
    if os.environ.get("DASHBOARD_TOKEN"):
        print("  Dashboard auth      ->  ON (append ?token=…)")
    print("  (Ctrl+C to stop)\n")
    httpd.serve_forever()
