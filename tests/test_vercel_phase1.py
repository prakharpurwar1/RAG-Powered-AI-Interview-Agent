"""Phase 1 tests: Vercel entrypoint, routing, dashboard authentication, and configuration format.

Run: python -m pytest tests/test_vercel_phase1.py -v
"""
import json
import os
import re
import sys
import pytest
from starlette.testclient import TestClient

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.index import app as vercel_app
import interviewer.main as main_module


@pytest.fixture
def client():
    return TestClient(vercel_app)


# ===========================================================================
# 1. Entrypoint and Import Tests
# ===========================================================================

def test_vercel_entrypoint_exposes_fastapi_app():
    """Verify that api/index.py imports cleanly and exports a valid FastAPI instance."""
    assert vercel_app is not None
    assert vercel_app.title == "AI Interviewer API"
    assert len(vercel_app.routes) > 0


def test_interviewer_main_imports_without_relative_import_error():
    """Verify that interviewer/main.py can be loaded directly without parent package errors."""
    assert hasattr(main_module, "app")
    assert main_module.app is not None
    assert main_module.app.title == "AI Interviewer API"


# ===========================================================================
# 2. Routing and Static Asset Delivery Tests
# ===========================================================================

def test_candidate_ui_routes_served_without_auth(client):
    """Verify that candidate interface (/) is served publicly as text/html."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "text/html" in resp_root.headers.get("content-type", "")
    assert "AI Interviewer" in resp_root.text

    resp_index = client.get("/index.html")
    assert resp_index.status_code == 200
    assert "text/html" in resp_index.headers.get("content-type", "")
    assert "AI Interviewer" in resp_index.text


def test_api_routes_accessible(client):
    """Verify core API endpoints work through the Vercel entry point."""
    resp_cap = client.get("/api/capabilities")
    assert resp_cap.status_code == 200
    data = resp_cap.json()
    assert "voice" in data
    assert "llm" in data

    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    assert resp_health.json().get("status") == "ok"


def test_docs_and_openapi_schema(client):
    """Verify Swagger documentation and OpenAPI schema are accessible."""
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200

    resp_schema = client.get("/openapi.json")
    assert resp_schema.status_code == 200
    assert "openapi" in resp_schema.json()


# ===========================================================================
# 3. Reviewer Dashboard Authentication Tests
# ===========================================================================

def test_dashboard_auth_gate_enforcement(client, monkeypatch):
    """Verify that dashboard page and API endpoints require authentication when DASHBOARD_TOKEN is set."""
    token = "secret-reviewer-token-12345"
    monkeypatch.setenv("DASHBOARD_TOKEN", token)

    # 1. Candidate interface remains unblocked
    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200

    # 2. Accessing /dashboard without token is rejected
    resp_unauth_page = client.get("/dashboard")
    assert resp_unauth_page.status_code == 401

    resp_unauth_html = client.get("/dashboard.html")
    assert resp_unauth_html.status_code == 401

    # 3. Accessing /dashboard with valid token query param succeeds
    resp_auth_page = client.get(f"/dashboard?token={token}")
    assert resp_auth_page.status_code == 200
    assert "text/html" in resp_auth_page.headers.get("content-type", "")
    assert "Dashboard" in resp_auth_page.text

    # 4. Accessing /dashboard with valid Bearer header succeeds
    resp_auth_bearer = client.get("/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert resp_auth_bearer.status_code == 200

    # 5. Dashboard API endpoints enforce token
    resp_api_unauth = client.get("/api/interviews")
    assert resp_api_unauth.status_code == 401

    resp_api_auth = client.get(f"/api/interviews?token={token}")
    assert resp_api_auth.status_code == 200
    assert "interviews" in resp_api_auth.json()

    resp_stats_auth = client.get("/api/stats", headers={"Authorization": f"Bearer {token}"})
    assert resp_stats_auth.status_code == 200


def test_dashboard_open_when_no_token_configured(client, monkeypatch):
    """Verify that dashboard is accessible in open development mode when DASHBOARD_TOKEN is empty."""
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    resp_page = client.get("/dashboard")
    assert resp_page.status_code == 200
    assert "text/html" in resp_page.headers.get("content-type", "")

    resp_api = client.get("/api/interviews")
    assert resp_api.status_code == 200


# ===========================================================================
# 4. Modern vercel.json Configuration Validation
# ===========================================================================

def test_vercel_json_format_and_rules():
    """Verify vercel.json uses valid modern Vercel schema and no legacy builds."""
    vpath = os.path.join(ROOT_DIR, "vercel.json")
    with open(vpath, "r", encoding="utf-8") as f:
        vcfg = json.load(f)

    # Must NOT use deprecated builds
    assert "builds" not in vcfg, "Legacy 'builds' property should not be present in modern vercel.json"
    assert "routes" not in vcfg, "Legacy 'routes' property should not be present in modern vercel.json"

    # Must configure functions with maxDuration
    assert "functions" in vcfg
    assert "api/index.py" in vcfg["functions"]
    assert vcfg["functions"]["api/index.py"].get("maxDuration", 0) >= 30

    # Must contain rewrites
    assert "rewrites" in vcfg
    rewrites = vcfg["rewrites"]
    assert any(r.get("source") == "/api/(.*)" and r.get("destination") == "/api/index.py" for r in rewrites)
    assert any(r.get("source") == "/dashboard" and r.get("destination") == "/api/index.py" for r in rewrites)
    assert any(r.get("source") == "/" and r.get("destination") == "/api/index.py" for r in rewrites)


# ===========================================================================
# 5. Frontend Secrets Security Audit
# ===========================================================================

def test_frontend_does_not_expose_secrets():
    """Verify frontend HTML files do not contain hardcoded API keys or secret credentials."""
    for filename in ["index.html", "dashboard.html"]:
        fpath = os.path.join(ROOT_DIR, "web", filename)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for exposed API keys or credentials
        assert not re.search(r"['\"][a-zA-Z0-9_-]{20,}['\"]\s*(?:#|//)?\s*(?:api[_-]?key|secret|token)", content, re.IGNORECASE)
        assert "sk-" not in content
        assert "SARVAM_API_KEY=" not in content
        assert "LLM_API_KEY=" not in content
