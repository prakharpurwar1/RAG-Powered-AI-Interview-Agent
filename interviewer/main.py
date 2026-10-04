"""Main entry point for ASGI servers (Uvicorn / Gunicorn / Hypercorn)."""
import os
import uvicorn
from .app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("interviewer.main:app", host=host, port=port, reload=False)
