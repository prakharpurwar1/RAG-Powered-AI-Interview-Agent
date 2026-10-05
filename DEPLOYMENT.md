# AI Interviewer Deployment Guide (Free Hosting Options)

This guide walks you through deploying your **AI Interviewer** web app and FastAPI backend to free hosting platforms.

---

## ⚡ Option 1: Netlify (Free Frontend + Serverless FastAPI)

Netlify provides fast, reliable, 100% free hosting for the static UI and Python serverless API functions using Mangum.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Netlify serverless deployment configuration"
   git push origin main
   ```

2. **Deploy on Netlify**:
   - Go to [netlify.com](https://netlify.com) and log in with GitHub.
   - Click **Add new site** → **Import an existing project**.
   - Select your GitHub repository.
   - Netlify will automatically detect [`netlify.toml`](netlify.toml):
     - **Publish directory:** `web`
     - **Functions directory:** `netlify/functions`
   - Click **Deploy Site**.

3. **Your Live App**:
   - Netlify generates an instant public URL (e.g. `https://your-bot-name.netlify.app`).
   - Web UI: `https://your-bot-name.netlify.app/`
   - Reviewer Dashboard: `https://your-bot-name.netlify.app/dashboard`
   - Swagger Docs: `https://your-bot-name.netlify.app/docs`

---

## ▲ Option 2: Vercel (Free Serverless Python + Static UI)

Vercel provides zero-config Python ASGI deployments.

### Steps:
1. Go to [vercel.com](https://vercel.com) and sign in with GitHub.
2. Click **Add New…** → **Project**.
3. Import your GitHub repository.
4. Vercel automatically reads [`vercel.json`](vercel.json) and deploys immediately.

---

## 🌟 Option 3: Render (Free Web Service Container)

Render hosts full long-running Python processes with WebSockets.

### Steps:
1. Go to [render.com](https://render.com) and click **New +** → **Web Service**.
2. Connect your repo and configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn interviewer.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free`
3. Click **Create Web Service**.

---

## 🚀 Option 4: Railway (Fast, Free Starter Credits)

Railway automatically detects Python/FastAPI and deploys in seconds.

### Steps:
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically detect the [`Procfile`](Procfile) and [`requirements.txt`](requirements.txt) and deploy.
5. Under **Settings** → **Networking**, click **Generate Domain**.

---

## 💻 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python run.py
```

* **Candidate Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Reviewer Dashboard:** [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
* **Interactive API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
