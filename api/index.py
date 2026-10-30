"""Vercel Serverless Function entry point.

Exposes the FastAPI application instance for Vercel's Python runtime.
Ensures repository root is included in sys.path so modules in 'interviewer'
can be cleanly resolved without import errors.
"""
import os
import sys

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from interviewer.app import app

__all__ = ["app"]
