"""FastAPI Application for AI Interviewer Bot.

Production-grade asynchronous API with Swagger docs, WebSocket streaming,
CORS, Pydantic validation, and SQLite persistence.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import InterviewConfig
from .documents import DocumentError, extract_document_text, validate_text_document
from .engine import InterviewEngine
from .integrity import IntegrityMonitor
from .llm import build_llm, LLMError
from .store import Store
from .streaming import build_streaming_asr
from .voice import build_voice

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
STARTED_AT = time.time()
SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", 6 * 3600))
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_STORE: Optional[Store] = None
_STORE_PATH: Optional[str] = None
_VOICE = None
_VOICE_MOCK: Optional[bool] = None


def _log(event: str, **fields):
    rec = {"t": round(time.time(), 3), "event": event}
    rec.update(fields)
    sys.stderr.write(json.dumps(rec) + "\n")
    sys.stderr.flush()


def get_store() -> Store:
    global _STORE, _STORE_PATH
    is_serverless = bool(
        os.environ.get("NETLIFY")
        or os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    )
    default_db = "/tmp/interviews.db" if is_serverless else "interviews.db"
    path = os.environ.get("INTERVIEW_DB", default_db)
    if _STORE is None or _STORE_PATH != path:
        _STORE = Store(path)
        _STORE_PATH = path
    return _STORE


def get_voice():
    global _VOICE, _VOICE_MOCK
    mock = bool(os.environ.get("VOICE_MOCK"))
    if _VOICE is None or _VOICE_MOCK != mock:
        _VOICE = build_voice(mock=mock)
        _VOICE_MOCK = mock
    return _VOICE


def sweep_sessions():
    now = time.time()
    dead = [sid for sid, s in _SESSIONS.items() if now - s.get("last", now) > SESSION_TTL]
    for sid in dead:
        _SESSIONS.pop(sid, None)


def evict_sessions(interview_id: str = "", email: str = ""):
    """Evict in-memory sessions when an interview or candidate data is deleted."""
    to_del = []
    store = get_store()
    for sid, s in _SESSIONS.items():
        if interview_id and s.get("iid") == interview_id:
            to_del.append(sid)
        elif email:
            row = store.get_interview(s.get("iid", ""))
            if row and row.get("candidate_email") == email:
                to_del.append(sid)
    for sid in to_del:
        _SESSIONS.pop(sid, None)


def get_or_rehydrate_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session from memory or gracefully rehydrate from SQLite store."""
    sweep_sessions()
    if not session_id:
        return None

    sess = _SESSIONS.get(session_id)
    if sess:
        return sess

    # Attempt rehydration from SQLite store
    store = get_store()
    row = store.get_interview_by_session_id(session_id)
    if not row:
        return None

    iid = row["id"]
    engine = InterviewEngine.rehydrate_from_store(store, iid, session_id=session_id)
    if not engine:
        return None

    monitor = IntegrityMonitor(enabled=bool(engine.config.proctoring))
    if row.get("flags"):
        for flag in row.get("flags", []):
            kind = flag.get("kind")
            cnt = int(flag.get("count", 1))
            for _ in range(cnt):
                monitor.record(kind)

    sess = {
        "engine": engine,
        "iid": iid,
        "persisted": len(engine.turns),
        "monitor": monitor,
        "created": row.get("created_at") or time.time(),
        "last": time.time(),
        "prompt_at": time.time(),
    }
    _SESSIONS[session_id] = sess
    _log("session_rehydrated", session_id=session_id, interview_id=iid, turns=len(engine.turns))
    return sess


def check_dashboard_auth(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    expected = os.environ.get("DASHBOARD_TOKEN", "")
    if not expected:
        return True
    given = token or ""
    if authorization and authorization.startswith("Bearer "):
        given = authorization[7:].strip()
    if given != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: invalid or missing dashboard token",
        )
    return True



# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    mode: str = "practice"
    interview_type: str = "mixed"
    role: str = "Software Engineer"
    level: str = "mid"
    candidate_name: str = "Candidate"
    candidate_email: Optional[str] = ""
    language: str = "en-IN"
    max_questions: int = 6
    mock: bool = False
    consent: bool = False
    randomize_order: Optional[bool] = None
    proctoring: bool = True
    resume_text: Optional[str] = ""
    job_description: Optional[str] = ""


class ValidateTextRequest(BaseModel):
    text: str
    label: Optional[str] = "Job description"


class MessageRequest(BaseModel):
    session_id: str
    message: str


class ReportRequest(BaseModel):
    session_id: str


class SignalRequest(BaseModel):
    session_id: str
    kind: str
    detail: Optional[Any] = None


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "en-IN"
    filename: str = "audio.webm"


class SpeakRequest(BaseModel):
    text: str
    language: str = "en-IN"
    speaker: str = "shubh"


class ReviewRequest(BaseModel):
    reviewer: Optional[str] = "Reviewer"
    notes: Optional[str] = ""
    human_score: Optional[Any] = None
    human_recommendation: Optional[str] = ""


class CandidateDeleteRequest(BaseModel):
    email: str


class RagRetrieveRequest(BaseModel):
    query: str
    doc_type: Optional[str] = None
    top_k: int = 3


class ResumeSessionRequest(BaseModel):
    session_id: str


class UpdateDocumentsRequest(BaseModel):
    session_id: Optional[str] = None
    resume_text: Optional[str] = None
    job_description: Optional[str] = None
    doc_type: Optional[str] = None



# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: sweep expired interviews
    try:
        purged = get_store().purge_expired()
        if purged:
            sys.stderr.write(f"[Startup] Purged {purged} expired interview(s)\n")
    except Exception as e:
        sys.stderr.write(f"[Startup Error] {e}\n")
    yield
    # Shutdown logic if needed


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Interviewer API",
    description="Asynchronous conversational AI interviewer backend with real-time speech and scoring",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for external frontend or local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Frontend Static Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(_WEB_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html")


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    dash_path = os.path.join(_WEB_DIR, "dashboard.html")
    return FileResponse(dash_path, media_type="text/html")


# ---------------------------------------------------------------------------
# Core Interview API
# ---------------------------------------------------------------------------
@app.get("/api/capabilities", tags=["System"])
async def get_capabilities():
    """Returns audio, voice, and LLM capabilities of the current server."""
    return {
        "voice": get_voice().capabilities(),
        "llm": build_llm().name,
        "proctoring": True,
    }


@app.get("/api/health", tags=["System"])
@app.get("/health", include_in_schema=False)
@app.get("/healthz", include_in_schema=False)
async def get_health():
    """Healthcheck endpoint for monitoring uptime and active sessions."""
    sweep_sessions()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "sessions": len(_SESSIONS),
        "db": get_store().path,
    }



# ---------------------------------------------------------------------------
# Document Processing Endpoints (Phase 2)
# ---------------------------------------------------------------------------
@app.post("/api/documents/parse", tags=["Documents"])
async def parse_document(
    request: Request,
    file: Optional[UploadFile] = None,
):
    """Upload and extract text from a Resume or Job Description (PDF or DOCX).
    Supports multipart/form-data with 'file', or JSON with 'filename' and 'content_b64'.
    """
    filename = ""
    content = b""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("file")
        if not uploaded or not hasattr(uploaded, "read"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No document file was uploaded. Please provide a file.",
            )
        filename = getattr(uploaded, "filename", "") or "document.pdf"
        content = await uploaded.read()
    elif "application/json" in content_type:
        body = await request.json()
        filename = body.get("filename", "document.pdf")
        b64_data = body.get("content_b64", "")
        if not b64_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'content_b64' in JSON payload.",
            )
        try:
            content = base64.b64decode(b64_data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 encoding: {exc}",
            )
    else:
        if file and file.filename:
            filename = file.filename
            content = await file.read()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported Content-Type. Please upload via multipart/form-data or application/json.",
            )

    try:
        text, meta = extract_document_text(filename, content)
        return {
            "status": "ok",
            "filename": meta["filename"],
            "format": meta["format"],
            "size_bytes": meta["size_bytes"],
            "char_count": meta["char_count"],
            "word_count": meta["word_count"],
            "text": text,
        }
    except DocumentError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing '{filename}': {exc}",
        )


@app.post("/api/documents/validate-text", tags=["Documents"])
async def validate_document_text(req: ValidateTextRequest):
    """Validate and clean plain text input for job descriptions."""
    try:
        cleaned = validate_text_document(req.text, doc_label=req.label or "Job description")
        return {
            "status": "ok",
            "text": cleaned,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
        }
    except DocumentError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@app.post("/api/start", tags=["Interview"])
async def start_interview(req: StartSessionRequest):
    """Initializes a new interview session."""
    sweep_sessions()
    data = req.model_dump()
    allowed = InterviewConfig.__dataclass_fields__.keys()
    cfg_args = {k: v for k, v in data.items() if k in allowed}
    cfg = InterviewConfig(**cfg_args)

    if cfg.mode == "hiring" and req.randomize_order is None:
        cfg.randomize_order = True
    try:
        cfg.validate()
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    sid = os.urandom(8).hex()
    store = get_store()
    iid = store.create_interview(vars(cfg), llm="", session_id=sid)

    # Phase 3: Index Resume & Job Description via RAG Vector Storage
    if cfg.resume_text or cfg.job_description:
        try:
            from .rag import index_interview_documents
            index_interview_documents(store, iid, resume_text=cfg.resume_text, job_description=cfg.job_description)
        except Exception as exc:
            _log("rag_indexing_error", interview_id=iid, error=str(exc))

    llm = build_llm(mock=req.mock)
    engine = InterviewEngine(cfg, llm, seed=iid, store=store, interview_id=iid, session_id=sid)
    monitor = IntegrityMonitor(enabled=cfg.proctoring)

    _SESSIONS[sid] = {
        "engine": engine,
        "iid": iid,
        "persisted": 0,
        "monitor": monitor,
        "created": time.time(),
        "last": time.time(),
        "prompt_at": time.time(),
    }

    first_message = engine.start()

    # Persist opening message
    turns = engine.turns
    while _SESSIONS[sid]["persisted"] < len(turns):
        idx = _SESSIONS[sid]["persisted"]
        store.add_turn(iid, idx, vars(turns[idx]))
        _SESSIONS[sid]["persisted"] += 1

    return {
        "session_id": sid,
        "interview_id": iid,
        "message": first_message,
        "stage": engine.stage,
        "llm": engine.llm.name,
        "proctoring": cfg.proctoring,
    }


@app.post("/api/session/resume", tags=["Interview"])
@app.get("/api/session/{session_id}", tags=["Interview"])
async def resume_session(
    session_id: Optional[str] = None,
    req: Optional[ResumeSessionRequest] = None,
):
    """Resume an active or completed interview session after browser refresh or server restart."""
    sid = (req.session_id if req else None) or session_id or ""
    sess = get_or_rehydrate_session(sid)
    if not sess:
        row = get_store().get_interview_by_session_id(sid)
        if row:
            return {
                "status": "ok",
                "resumed": True,
                "session_id": sid,
                "interview_id": row["id"],
                "completed": row.get("status") == "completed",
                "scorecard": row.get("scorecard"),
                "turns": row.get("turns", []),
                "config": {
                    "role": row.get("role"),
                    "level": row.get("level"),
                    "candidate_name": row.get("candidate_name"),
                    "interview_type": row.get("interview_type"),
                },
            }
        raise HTTPException(status_code=404, detail="Session not found or expired")

    engine = sess["engine"]
    iid = sess["iid"]
    last_turn = engine.turns[-1] if engine.turns else None
    from dataclasses import asdict
    return {
        "status": "ok",
        "resumed": True,
        "session_id": sid,
        "interview_id": iid,
        "stage": engine.stage,
        "completed": engine.finished or engine.stage == "done",
        "message": last_turn.text if last_turn and last_turn.role == "interviewer" else "",
        "turns": [asdict(t) for t in engine.turns],
        "proctoring": engine.config.proctoring,
        "candidate_name": engine.config.candidate_name,
        "role": engine.config.role,
        "level": engine.config.level,
        "interview_type": engine.config.interview_type,
    }


@app.post("/api/session/{session_id}/documents", tags=["Interview"])
async def update_session_documents(session_id: str, req: UpdateDocumentsRequest):
    """Replace or remove uploaded resume/job description for an active session."""
    sess = get_or_rehydrate_session(session_id)
    store = get_store()
    iid = sess["iid"] if sess else None
    if not iid:
        row = store.get_interview_by_session_id(session_id)
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        iid = row["id"]

    from .rag import index_interview_documents

    store.update_interview_documents(
        iid,
        resume_text=req.resume_text,
        job_description=req.job_description,
    )

    idx_res = index_interview_documents(
        store,
        iid,
        resume_text=req.resume_text,
        job_description=req.job_description,
        doc_type=req.doc_type,
    )

    if sess and sess.get("engine"):
        sess["engine"].update_documents(
            resume_text=req.resume_text,
            job_description=req.job_description,
        )

    return {
        "status": "ok",
        "session_id": session_id,
        "interview_id": iid,
        "indexing": idx_res,
    }


@app.post("/api/message", tags=["Interview"])
async def send_message(req: MessageRequest):
    """Processes candidate's response turn and returns the interviewer's next question."""
    sess = get_or_rehydrate_session(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    sess["last"] = time.time()
    text = req.message.strip()

    # Integrity velocity check
    elapsed = time.time() - sess.get("prompt_at", time.time())
    if elapsed < 6 and len(text.split()) > 40:
        sess["monitor"].record("fast", detail=round(elapsed, 1))

    try:
        out = sess["engine"].respond(text)
    except LLMError as err:
        _log("llm_generation_error", session_id=req.session_id, interview_id=sess["iid"], error=str(err))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Interview model error: {err}",
        )
    sess["prompt_at"] = time.time()

    # Persist turns
    store = get_store()
    turns = sess["engine"].turns
    while sess["persisted"] < len(turns):
        idx = sess["persisted"]
        store.add_turn(sess["iid"], idx, vars(turns[idx]))
        sess["persisted"] += 1

    if out.get("done"):
        rep = sess["engine"].report()
        store.save_scorecard(sess["iid"], rep["scorecard"])
        integrity_summary = sess["monitor"].summary()
        store.save_integrity(sess["iid"], integrity_summary)
        out["interview_id"] = sess["iid"]

    return out


@app.post("/api/report", tags=["Interview"])
async def generate_report(req: ReportRequest):
    """Generates and finalizes the candidate scorecard and evaluation report idempotently."""
    sess = get_or_rehydrate_session(req.session_id)
    store = get_store()

    if not sess:
        # Check if completed in store
        row = store.get_interview_by_session_id(req.session_id)
        if row and row.get("scorecard"):
            return {
                "scorecard": row["scorecard"],
                "interview_id": row["id"],
                "candidate_name": row.get("candidate_name", ""),
                "role": row.get("role", ""),
                "level": row.get("level", ""),
            }
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # If already scored in SQLite, return cached scorecard directly
    row = store.get_interview(sess["iid"])
    if row and row.get("scorecard"):
        return {
            "scorecard": row["scorecard"],
            "interview_id": sess["iid"],
            "candidate_name": row.get("candidate_name", ""),
            "role": row.get("role", ""),
            "level": row.get("level", ""),
        }

    rep = sess["engine"].report()
    store.save_scorecard(sess["iid"], rep["scorecard"])
    integrity_summary = sess["monitor"].summary()
    store.save_integrity(sess["iid"], integrity_summary)
    rep["interview_id"] = sess["iid"]
    return rep


@app.post("/api/signal", tags=["Interview"])
async def record_signal(req: SignalRequest):
    """Records proctoring telemetry signals (blur, copy, paste)."""
    sess = get_or_rehydrate_session(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    sess["last"] = time.time()
    sess["monitor"].record(req.kind, detail=req.detail)
    return {"ok": True, "score": sess["monitor"].score()}



# ---------------------------------------------------------------------------
# Voice & Audio Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/transcribe", tags=["Voice"])
async def transcribe_audio(req: TranscribeRequest):
    """Transcribes base64-encoded audio chunk to text using configured ASR."""
    try:
        data = base64.b64decode(req.audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload")

    text = get_voice().transcribe(
        data, filename=req.filename, language_code=req.language
    )
    return {"transcript": text}


@app.post("/api/speak", tags=["Voice"])
async def synthesize_speech(req: SpeakRequest):
    """Synthesizes text into streaming audio WAV bytes."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio_bytes = get_voice().synthesize_stream(
        text, language_code=req.language, speaker=req.speaker
    )
    return Response(content=audio_bytes, media_type="audio/wav")


@app.websocket("/api/stream-asr")
async def websocket_asr(ws: WebSocket):
    """Real-time streaming speech-to-text WebSocket endpoint."""
    await ws.accept()
    asr = build_streaming_asr()
    try:
        asr.connect()
        await ws.send_json({"event": "ready"})
        while True:
            data = await ws.receive_text()
            msg = json.loads(data or "{}")
            ev = msg.get("event")
            if ev == "audio_input":
                audio_raw = base64.b64decode(msg.get("audio", ""))
                asr.send_audio(audio_raw)
                try:
                    event_data = asr.next_event()
                    if event_data:
                        await ws.send_json(event_data)
                except Exception:
                    pass
            elif ev == "flush":
                asr.flush()
                try:
                    event_data = asr.next_event()
                    if event_data:
                        await ws.send_json(event_data)
                except Exception:
                    pass
            elif ev == "end":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        sys.stderr.write(f"[WebSocket ASR Error] {e}\n")
    finally:
        try:
            asr.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reviewer & Dashboard Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/stats", tags=["Dashboard"])
async def get_dashboard_stats(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    return get_store().stats()


@app.get("/api/interviews", tags=["Dashboard"])
async def list_interviews(
    status: Optional[str] = Query(None),
    sort: str = Query("score"),
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    return {"interviews": get_store().list_interviews(status, sort)}


@app.get("/api/interviews/{interview_id}", tags=["Dashboard"])
async def get_interview_detail(
    interview_id: str,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    detail = get_store().get_interview(interview_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Interview not found")
    return detail


@app.post("/api/interviews/{interview_id}/review", tags=["Dashboard"])
async def review_interview(
    interview_id: str,
    req: ReviewRequest,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    store = get_store()
    if not store.get_interview(interview_id):
        raise HTTPException(status_code=404, detail="Interview not found")

    hs = req.human_score
    try:
        hs_float = float(hs) if hs not in (None, "") else None
    except Exception:
        hs_float = None

    store.save_review(
        interview_id,
        req.reviewer or "Reviewer",
        req.notes or "",
        hs_float,
        req.human_recommendation or "",
    )
    return store.get_interview(interview_id)


@app.post("/api/interviews/{interview_id}/delete", tags=["Dashboard"])
async def delete_interview(
    interview_id: str,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    get_store().delete_interview(interview_id)
    evict_sessions(interview_id=interview_id)
    return {"deleted": interview_id}


@app.get("/api/interviews/{interview_id}/chunks", tags=["RAG"])
async def get_interview_chunks(
    interview_id: str,
    doc_type: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    """Inspect stored document chunks and metadata for an interview session."""
    check_dashboard_auth(token, authorization)
    chunks = get_store().get_document_chunks(interview_id, doc_type=doc_type)
    cleaned = []
    for c in chunks:
        c_copy = dict(c)
        c_copy.pop("embedding", None)  # Omit large raw vector for clean display
        cleaned.append(c_copy)
    return {"interview_id": interview_id, "chunks": cleaned, "count": len(cleaned)}


@app.post("/api/interviews/{interview_id}/retrieve", tags=["RAG"])
async def retrieve_interview_chunks(
    interview_id: str,
    req: RagRetrieveRequest,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    """Retrieve semantically similar evidence chunks for the current candidate session."""
    check_dashboard_auth(token, authorization)
    from .rag import retrieve_evidence
    results = retrieve_evidence(
        get_store(), interview_id, req.query, doc_type=req.doc_type, top_k=req.top_k
    )
    return {"interview_id": interview_id, "query": req.query, "results": results}


@app.get("/api/candidate", tags=["Compliance"])
async def export_candidate_data(
    email: str = Query(...),
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    return {"candidate": email, "interviews": get_store().export_candidate(email)}


@app.post("/api/candidate/delete", tags=["Compliance"])
async def delete_candidate_data(
    req: CandidateDeleteRequest,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    evict_sessions(email=req.email)
    n = get_store().delete_candidate(req.email)
    return {"deleted": n}



@app.get("/api/export.csv", tags=["Export"])
async def export_csv(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "id",
            "created_at",
            "candidate",
            "email",
            "role",
            "type",
            "level",
            "status",
            "score",
            "recommendation",
            "integrity",
            "flags",
            "human_score",
            "human_recommendation",
            "review_status",
            "reviewer",
            "consent",
            "consent_version",
            "retention_until",
        ]
    )
    for r in get_store().list_interviews(sort="recent"):
        flags = r.get("flags") or ""
        if isinstance(flags, str):
            try:
                flags = ";".join(f.get("label", "") for f in json.loads(flags))
            except Exception:
                flags = ""
        w.writerow(
            [
                r.get("id"),
                r.get("created_at"),
                r.get("candidate_name"),
                r.get("candidate_email"),
                r.get("role"),
                r.get("interview_type"),
                r.get("level"),
                r.get("status"),
                r.get("weighted_total"),
                r.get("recommendation"),
                r.get("integrity_score"),
                flags,
                r.get("human_score"),
                r.get("human_recommendation"),
                r.get("review_status"),
                r.get("reviewer"),
                r.get("consent"),
                r.get("consent_version"),
                r.get("retention_until"),
            ]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=interviews.csv"},
    )


@app.get("/api/export.json", tags=["Export"])
async def export_json(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    check_dashboard_auth(token, authorization)
    data = [
        get_store().get_interview(r["id"])
        for r in get_store().list_interviews(sort="recent")
    ]
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": "attachment; filename=interviews.json"},
    )
