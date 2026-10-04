# AI Interviewer Deployment Guide (Free Hosting Options)

This guide walks you through deploying your **AI Interviewer** web app and FastAPI backend to free hosting platforms.

---

## 🌟 Option 1: Render (Recommended — 100% Free Web Service)

Render offers a generous **free tier** for web services that connects directly to your GitHub repository.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of AI Interviewer"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/interview-bot.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com) and create a free account.
   - Click **New +** → **Web Service**.
   - Select **Build and deploy from a Git repository** and pick your repo.
   - Configure settings:
     - **Name**: `ai-interview-bot`
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn interviewer.main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: `Free`
   - (Optional) Under **Environment Variables**, add:
     - `LLM_API_KEY`: *(your OpenAI / Sarvam / Groq API key)*
     - `SARVAM_API_KEY`: *(optional for studio voice)*
     - `DASHBOARD_TOKEN`: *(optional secret token to protect reviewer dashboard)*
   - Click **Create Web Service**.

3. **Your Live App**:
   - Render gives you a free URL like: `https://ai-interview-bot.onrender.com`
   - Candidate Web UI: `https://ai-interview-bot.onrender.com/`
   - Reviewer Dashboard: `https://ai-interview-bot.onrender.com/dashboard`
   - Interactive Swagger API Docs: `https://ai-interview-bot.onrender.com/docs`

---

## 🚀 Option 2: Railway (Fast, Free Starter Credits)

Railway automatically detects Python/FastAPI and deploys in seconds.

### Steps:
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically detect the [`Procfile`](Procfile) and [`requirements.txt`](requirements.txt) and deploy.
5. In your Railway service settings under **Networking**, click **Generate Domain**.

---

## 🐳 Option 3: Hugging Face Spaces (100% Free Docker Container Hosting)

Hugging Face provides free 24/7 Docker spaces (2 vCPUs, 16GB RAM).

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set **SDK** to **Docker** (Blank).
3. Clone the space repo or push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/interview-bot
   git push hf main
   ```
4. Hugging Face will build your [`Dockerfile`](Dockerfile) and provide a permanent live web link.

---

## 💻 Running Locally with FastAPI

To run the FastAPI server locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python run.py
```

* **Candidate Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Reviewer Dashboard:** [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
* **Interactive API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
