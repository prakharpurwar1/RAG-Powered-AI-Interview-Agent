"""Main entry point for ASGI servers (Uvicorn / Gunicorn / Hypercorn)."""
import os
import sys

# Ensure repository root is on sys.path if executed directly
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from interviewer.app import app
except ImportError:
    from .app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("interviewer.main:app", host=host, port=port, reload=False)
