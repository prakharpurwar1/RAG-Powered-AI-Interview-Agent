# AI Interviewer — end-to-end prototype

An AI that runs an interview from greeting to scored report, in **text or voice**,
for **two audiences**:

- **Practice** — coaches *you*: warm tone, gentle nudges, a report you can learn from.
- **Hiring** — screens *candidates*: neutral tone, a fixed rubric, comparable scores,
  a reviewer dashboard, integrity signals, an audit trail, and retention controls.

Zero third-party dependencies — pure Python standard library. Runs offline out of the box.

---

## Quick start

```bash
cd interview-bot

python run.py cli --mock          # terminal interview, no key needed
python run.py                     # web UI      -> http://127.0.0.1:8000
                                  # dashboard   -> http://127.0.0.1:8000/dashboard

python run.py purge               # delete interviews past their retention window
python run.py export --email a@b  # data access request -> JSON on stdout
```

## What's inside

| Area | Status | Where |
|------|--------|-------|
| Text interview, state machine, rubric scoring | done | `engine.py`, `scoring.py`, `rubric.py` |
| Offline mock LLM (no key, no network) | done | `llm.py` |
| Server-side voice: ASR (REST) + TTS (REST + HTTP stream) | done | `voice.py` |
| Real-time streaming ASR over WebSocket | done | `streaming.py` |
| Barge-in: stop the interviewer when you speak | done | `web/index.html` |
| SQLite persistence (interviews, turns, scores, reviews) | done | `store.py` |
| Reviewer dashboard: rank, filter, transcript, human override | done | `web/dashboard.html` |
| Integrity: randomised order + proctoring signals + score | done | `integrity.py` |
| Retention, consent records, data access & erasure | done | `store.py`, `run.py` |
| Hardening: dashboard auth, health, size caps, rate limit, audit logs | done | `server.py` |

## Use a real LLM

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://api.sarvam.ai/v1"   # default (Sarvam AI)
export LLM_MODEL="sarvam-105b"                   # or sarvam-105b-conversations
export LLM_AUTH_STYLE="api-subscription-key"     # Sarvam; "bearer" for OpenAI
python run.py
```

Any OpenAI-compatible endpoint works (Sarvam, OpenAI, Together, Groq, vLLM, Ollama).

## Voice

Three ways, in order of setup cost:

1. **Browser (no key)** — microphone + speech synthesis, in the UI.
2. **Server REST (one key)** — push-to-talk: record → `POST /api/transcribe` (Saaras) →
   text; the reply is spoken via `POST /api/speak` (Bulbul, HTTP stream).
   ```bash
   export SARVAM_API_KEY="your-key"
   ```
3. **Real-time streaming** — the 📡 button streams mic audio (PCM16) over a WebSocket to
   `/api/stream-asr`, bridging to Sarvam Realtime ASR for partial transcripts.

**Barge-in** (on by default) listens while the interviewer is speaking and stops the audio
the moment you start talking. It uses echo cancellation, but on loudspeakers a very short
reply can still self-trigger — headphones are the reliable option. Force the offline voice
mock with `VOICE_MOCK=1`.

## Integrity & anti-cheating

Hiring mode defaults to a **randomised question order**, so a shared "script" is useless.
The client reports lightweight **proctoring signals** — tab switches, pastes, copies — and
the server flags **unusually fast, lengthy answers**. These combine into a 0-100
**integrity score** shown on the dashboard.

It is deliberately *advisory*: it flags sessions for human review, it never auto-rejects.
Integrity signals are noisy and biased, so the same rule as scoring applies — surface the
evidence, let a person decide.

## Compliance & retention

- **Consent** is required for hiring mode and recorded with a timestamp and a version.
- **Retention** — every interview gets a `retention_until` date (`retention_days`, default
  90). `python run.py purge` deletes the expired ones; the server also sweeps on startup.
- **Data access** — `GET /api/candidate?email=…` returns everything held about a person.
- **Erasure** — `POST /api/candidate/delete` removes it all; the dashboard can delete a
  single interview.
- Audio is never stored server-side; only the transcript is kept.

## Production hardening

| Control | How |
|---------|-----|
| Dashboard auth | set `DASHBOARD_TOKEN`; then append `?token=…` (or `Authorization: Bearer`) |
| Health check | `GET /api/health` |
| Request size cap | `MAX_BODY_BYTES` (default 8 MB) → 413 |
| Rate limit on session creation | `START_RATE_PER_MIN` (default 60) → 429 |
| Session TTL | `SESSION_TTL_SECONDS` (default 6 h), swept automatically |
| Audit logging | structured JSON events on stderr (start, complete, review, delete, purge) |

---

## Architecture

```
        ┌────────────┐   text / audio / ws   ┌──────────────────────┐
        │  Web UI /  │ ────────────────────▶ │   InterviewEngine    │
        │  Dashboard │ ◀──────────────────── │  (explicit state     │
        └────────────┘   question / report   │   machine)           │
              ▲  ▲                          └───────┬──────────────┘
     browser  │  │ browser                          │ prompt + task
     speech   │  │ mic                    ┌─────────▼─────────┐
     synth    │  └───────────────────────▶│   LLMClient        │
              │                           │  (Sarvam/OpenAI/   │
        ┌─────┴─────────┐                 │   Mock)            │
        │  VoiceIO      │                 └─────────┬─────────┘
        │ REST + WS     │                           │ evidence
        │ Saaras/Bulbul │                 ┌─────────▼─────────┐
        └───────┬───────┘                 │  scoring.py        │
                │                         │  weighted rubric   │
        ┌───────▼───────┐                 └─────────┬─────────┘
        │  Sarvam APIs  │                           │
        └───────────────┘        ┌──────────────────▼──────────────────┐
                                 │  store.py (SQLite)                   │
                                 │  + integrity.py   + dashboard/audit  │
                                 └──────────────────────────────────────┘
```

**The key design choice:** the engine owns the flow; the LLM only phrases turns and
scores at the end. That keeps interviews consistent, resumable and auditable.

| Module | Job |
|--------|-----|
| `config.py` | settings (mode, type, role, level, consent, retention, integrity) |
| `questions.py` | question bank, one per competency (+ follow-up hint) |
| `rubric.py` | competencies, weights, "what good looks like" |
| `engine.py` | state machine: greeting → core → wrap-up → done (+ randomised order) |
| `llm.py` | pluggable client + offline `MockLLM` |
| `scoring.py` | LLM proposes evidence; code owns weights & totals |
| `integrity.py` | proctoring signals → flags → 0-100 score |
| `voice.py` | server-side ASR/TTS (REST + HTTP stream) with mock |
| `streaming.py` | WebSocket codec + Sarvam Realtime ASR bridge |
| `store.py` | SQLite persistence + reviews + retention + erasure |
| `server.py` | HTTP API, dashboard routes, WS bridge, hardening |
| `web/index.html` | chat UI with voice and barge-in |
| `web/dashboard.html` | reviewer dashboard |

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/start` | begin an interview (persists it) |
| POST | `/api/message` | one candidate turn |
| POST | `/api/report` | transcript + scorecard |
| POST | `/api/signal` | integrity event (blur / paste / copy) |
| POST | `/api/transcribe` | audio → text (Saaras) |
| POST | `/api/speak` | text → audio (Bulbul) |
| WS | `/api/stream-asr` | live partial transcripts |
| GET | `/api/health` | liveness + counters |
| GET | `/api/interviews?status=&sort=` | list (filter + rank) *(auth)* |
| GET | `/api/interviews/<id>` | detail (transcript + scores + integrity + review) *(auth)* |
| POST | `/api/interviews/<id>/review` | human override + notes *(auth)* |
| POST | `/api/interviews/<id>/delete` | delete one (retention) *(auth)* |
| GET | `/api/candidate?email=` | everything held about a person *(auth)* |
| POST | `/api/candidate/delete` | erasure request *(auth)* |
| GET | `/api/export.csv` · `/api/export.json` | audit export *(auth)* |
| GET | `/api/stats` | dashboard summary *(auth)* |

## Tests

```bash
python tests/test_engine.py      # flow + scoring
python tests/test_store.py       # persistence, retention, erasure, migration
python tests/test_integrity.py   # integrity scoring
python tests/test_streaming.py   # WebSocket codec + handshake
python tests/test_server.py      # HTTP API, voice, reviewer, WS bridge, auth
# or:  python -m pytest -q
```

## Extending it

- **Better questions** — replace the bank with LLM generation from a job description.
- **More competencies** — edit `rubric.py`; weights must sum to 1.
- **Scale storage** — swap SQLite for Postgres; move sessions to Redis.
- **Identity checks** — add a webcam snapshot or document check to the integrity monitor.
- **Monitoring** — ship the JSON logs to your observability stack.
