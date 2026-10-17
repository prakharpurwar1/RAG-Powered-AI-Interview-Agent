# Production Deployment & Hosting Guide

This guide details the production architecture, hosting options, persistence strategies, and environment configurations for the **RAG-Powered AI Interview Agent**.

---

## 🏗️ Production Architecture Overview

The system is designed as a modern asynchronous FastAPI backend paired with a static HTML5/CSS3 client:
- **Backend Service**: FastAPI with Uvicorn ASGI server, asynchronous endpoints, SQLite persistence (with WAL mode), FastEmbed semantic embeddings (or offline deterministic fallback), and an engine state machine.
- **WebSocket Streaming**: `/api/stream-asr` delivers real-time binary audio frames for streaming speech-to-text.
- **Frontend**: Clean static assets (`web/index.html` and `web/dashboard.html`) served directly by FastAPI or via reverse proxy.

### ⚠️ Critical Hosting Prerequisite: Persistent Long-Running Container vs. Serverless

| Feature | Long-Running Container (Render / Railway / Fly / Docker) | Serverless (Netlify / Vercel / AWS Lambda) |
| :--- | :--- | :--- |
| **Real-time WebSockets (`/api/stream-asr`)** | ✅ Fully supported | ❌ Incompatible (Serverless functions terminate connections) |
| **Persistent SQLite & RAG Chunks** | ✅ Supported with persistent volume mount | ❌ Ephemeral (`/tmp` is wiped on cold starts and scaling) |
| **FastEmbed / ONNX Runtime** | ✅ Runs in-process with memory caching | ⚠️ High cold start latency / bundle size limitations |
| **Recommendation** | **PRIMARY RECOMMENDED ARCHITECTURE** | Useful only for static demo without persistent records |

---

## 🌟 Option 1: Render (Recommended Container Hosting)

Render runs full long-running Python processes with full WebSocket support and persistent disks.

### 1. Blueprint Deployment (`render.yaml`)
The repository includes a ready-to-use [`render.yaml`](render.yaml) blueprint:
1. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.
2. Connect your GitHub repository: `prakharpurwar1/RAG-Powered-AI-Interview-Agent`.
3. Render automatically configures:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn interviewer.main:app --host 0.0.0.0 --port $PORT`
   - **Healthcheck Path:** `/health`

### 2. Ensuring Persistent Storage on Render
* **Free Plan Note:** Render's free tier uses an ephemeral container disk (data is reset when the instance spins down or redeploys).
* **For Permanent Persistence:**
  1. In your Web Service settings on Render, navigate to **Disks** → **Add Disk**.
  2. Set **Mount Path** to `/var/data` (e.g. 1GB disk).
  3. Under **Environment Variables**, set:
     ```bash
     INTERVIEW_DB=/var/data/interviews.db
     ```
  All candidate interviews, resumes, job descriptions, vector embeddings, and scorecards will now survive all server restarts, scaling, and redeployments permanently.

---

## 🚀 Option 2: Railway (Fastest Automated Deployment)

1. Go to [railway.app](https://railway.app) and click **New Project** → **Deploy from GitHub repo**.
2. Select `RAG-Powered-AI-Interview-Agent`.
3. Railway automatically detects [`Procfile`](Procfile) and [`requirements.txt`](requirements.txt).
4. Add a Persistent Volume:
   - Go to your service **Data / Volumes** tab → **Add Volume**.
   - Mount path: `/app/data`
   - Set environment variable: `INTERVIEW_DB=/app/data/interviews.db`
5. In **Settings** → **Networking**, click **Generate Domain**.

---

## 🐳 Option 3: Docker Deployment

The repository includes an optimized [`Dockerfile`](Dockerfile):

```bash
# Build Docker image
docker build -t interview-bot .

# Run with persistent host volume for SQLite database
docker run -d \
  --name interview-bot \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e INTERVIEW_DB=/app/data/interviews.db \
  -e SARVAM_API_KEY=your_key_here \
  interview-bot
```

---

## ⚡ Serverless Deployments (Netlify / Vercel)

If deploying to Netlify or Vercel:
* [`netlify.toml`](netlify.toml) and [`vercel.json`](vercel.json) configure serverless redirects.
* **Limitations to be aware of:**
  - Real-time WebSockets (`/api/stream-asr`) will be disabled; audio must use push-to-talk `/api/transcribe`.
  - The SQLite database resides in `/tmp/interviews.db`, which is ephemeral and will not persist across independent Lambda worker invocations unless connected to an external cloud database (e.g. PostgreSQL or Turso).

---

## 🔐 Environment Variables & Secrets Reference

Configure these in your cloud provider's dashboard under **Environment Variables**:

| Variable | Required? | Default | Description |
| :--- | :--- | :--- | :--- |
| `INTERVIEW_DB` | Recommended | `interviews.db` | Absolute path to SQLite file (set to persistent mount like `/var/data/interviews.db`). |
| `SARVAM_API_KEY` | Optional | `""` | API key for Sarvam AI studio voice (TTS/ASR) and chat. Leave blank for offline mock mode. |
| `LLM_API_KEY` | Optional | `""` | API key for LLM generation (e.g. Sarvam / OpenAI). |
| `LLM_BASE_URL` | Optional | `https://api.sarvam.ai/v1` | LLM endpoint URL. |
| `LLM_MODEL` | Optional | `sarvam-105b` | Model name. |
| `DASHBOARD_TOKEN` | Optional | `""` | Secret token to secure `/dashboard` and reviewer API endpoints. |
| `SESSION_TTL_SECONDS` | Optional | `21600` (6 hrs) | Inactivity timeout before purging in-memory session cache. |
| `PORT` | Set by host | `8000` | Port for Uvicorn server. |

---

## 🔍 Healthcheck & Verification Endpoints

- **Liveness & Health Probe**: `GET /health` or `GET /api/health`
- **Capabilities Probe**: `GET /api/capabilities`
- **Candidate Interface**: `GET /`
- **Reviewer Dashboard**: `GET /dashboard` (pass `?token=...` if `DASHBOARD_TOKEN` is enabled)
- **Interactive OpenAPI Documentation**: `GET /docs`
