"""Automated tests for Phase 1: Robust LLM Error Handling.

Verifies:
1. Successful startup in mock mode returns HTTP 200.
2. Successful startup with working LLM returns HTTP 200.
3. Upstream LLM HTTP error (e.g. HTTP 401, 429, 500) returns HTTP 502 Bad Gateway, not unhandled HTTP 500.
4. Upstream network error / timeout returns HTTP 502 Bad Gateway.
5. Failed session is cleaned up and not left lingering in active memory.
6. Secret sanitization prevents API keys, tokens, and authorization credentials from leaking.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from starlette.testclient import TestClient

from interviewer.app import _SESSIONS, app
from interviewer.llm import LLMClient, LLMError, OpenAICompatLLM, sanitize_secret_text


class WorkingFakeLLM(LLMClient):
    """Stand-in LLM returning a custom generated response."""

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        if task == "greeting":
            return "Welcome to the screening interview! Let's discuss your engineering background."
        if task == "decide_turn":
            return json.dumps({
                "action": "follow_up",
                "spoken_response": "Could you elaborate on the architecture trade-offs?",
                "reason_for_decision": "Probing technical depth",
                "competency": "fundamentals",
                "evidence_used": ["Candidate mentioned caching"],
            })
        return "Fake response"

    @property
    def name(self):
        return "WorkingFakeLLM"


class FailingFakeLLM(LLMClient):
    """Stand-in LLM that simulates an upstream provider failure."""

    def __init__(self, status_code: int = 401, message: str = "Invalid API key provided: secret_test_key_12345"):
        self.status_code = status_code
        self.message = message

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        raise LLMError(f"HTTP {self.status_code} from LLM provider: {self.message}", status_code=self.status_code)

    @property
    def name(self):
        return "FailingFakeLLM"


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Startup Success & Mock Mode
# ---------------------------------------------------------------------------

def test_start_mock_mode_success(client):
    """Verify that starting an interview in mock mode succeeds with HTTP 200."""
    resp = client.post("/api/start", json={
        "mock": True,
        "candidate_name": "Prakhar Purwar",
        "role": "Backend Engineer",
        "level": "mid",
        "interview_type": "mixed",
        "max_questions": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "interview_id" in data
    assert data["stage"] == "core"
    assert "Prakhar Purwar" in data["message"]
    assert "MockLLM" in data["llm"]


def test_start_real_mode_success_with_working_llm(client):
    """Verify that starting an interview with a working LLM returns HTTP 200 and the model lead."""
    with patch("interviewer.app.build_llm", return_value=WorkingFakeLLM()):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Jane Doe",
            "role": "Staff Engineer",
            "level": "senior",
            "interview_type": "technical",
            "max_questions": 2,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Welcome to the screening interview! Let's discuss your engineering background."
    assert data["stage"] == "core"
    assert data["llm"] == "WorkingFakeLLM"


# ---------------------------------------------------------------------------
# Tests: Upstream LLM Failure Handling (HTTP 502 vs HTTP 500)
# ---------------------------------------------------------------------------

def test_start_upstream_llm_auth_failure_returns_502(client):
    """Verify that upstream 401 auth failure returns HTTP 502 Bad Gateway instead of unhandled 500."""
    with patch("interviewer.app.build_llm", return_value=FailingFakeLLM(status_code=401)):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Test Candidate",
            "role": "Data Engineer",
            "interview_type": "mixed",
        })
    assert resp.status_code == 502
    data = resp.json()
    assert "detail" in data
    assert "Interview model error" in data["detail"]
    assert "HTTP 401" in data["detail"]
    # Ensure raw secret is not leaked in response
    assert "secret_test_key_12345" not in json.dumps(data)


def test_start_upstream_llm_rate_limit_returns_502(client):
    """Verify that upstream 429 rate limit returns HTTP 502 Bad Gateway."""
    with patch("interviewer.app.build_llm", return_value=FailingFakeLLM(status_code=429, message="Rate limit exceeded")):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Test Candidate",
            "role": "Data Engineer",
            "interview_type": "mixed",
        })
    assert resp.status_code == 502
    data = resp.json()
    assert "HTTP 429" in data["detail"]


def test_start_upstream_network_error_returns_502(client):
    """Verify that network connection timeout / URLError returns HTTP 502 Bad Gateway."""
    network_error_llm = MagicMock()
    network_error_llm.chat.side_effect = LLMError("Network error connecting to LLM provider: timed out", status_code=None)
    network_error_llm.name = "NetworkErrorLLM"

    with patch("interviewer.app.build_llm", return_value=network_error_llm):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Test Candidate",
            "role": "Systems Engineer",
            "interview_type": "technical",
        })
    assert resp.status_code == 502
    data = resp.json()
    assert "Interview model error" in data["detail"]


def test_failed_start_session_evicted_from_memory(client):
    """Verify that when start fails due to LLMError, the half-initialized session is not left in memory."""
    initial_sessions = set(_SESSIONS.keys())
    with patch("interviewer.app.build_llm", return_value=FailingFakeLLM(status_code=503)):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Test Candidate",
            "role": "Backend Engineer",
            "interview_type": "mixed",
        })
    assert resp.status_code == 502
    # Ensure no new orphaned session was retained in _SESSIONS
    new_sessions = set(_SESSIONS.keys()) - initial_sessions
    assert len(new_sessions) == 0


def test_message_upstream_llm_failure_returns_502(client):
    """Verify that when sending a candidate message encounters an LLMError, HTTP 502 is returned."""
    # First start a session in mock mode
    start_resp = client.post("/api/start", json={
        "mock": True,
        "candidate_name": "Test Candidate",
        "role": "DevOps Engineer",
        "interview_type": "mixed",
    })
    sid = start_resp.json()["session_id"]

    # Now simulate an LLMError on the next turn
    with patch.object(_SESSIONS[sid]["engine"].llm, "chat", side_effect=LLMError("HTTP 500 from LLM provider: Internal error", status_code=500)):
        msg_resp = client.post("/api/message", json={
            "session_id": sid,
            "message": "I worked on Kubernetes migrations and CI/CD pipelines.",
        })
    assert msg_resp.status_code == 502
    assert "HTTP 500" in msg_resp.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: Secret & Credential Sanitization
# ---------------------------------------------------------------------------

def test_sanitize_secret_text_redacts_api_keys_and_tokens(monkeypatch):
    """Verify that sanitize_secret_text redacts environment keys, Bearer tokens, and auth headers."""
    monkeypatch.setenv("SARVAM_API_KEY", "real_sarvam_secret_998877")
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin:super_secret_pw@db.neon.tech/main")

    # Raw error string containing multiple sensitive components
    raw_error = (
        "HTTP 401: api-subscription-key: real_sarvam_secret_998877 failed. "
        "Authorization: Bearer my_bearer_token_xyz123, "
        "db connection: postgresql://admin:super_secret_pw@db.neon.tech/main, "
        "sk-live_0123456789abcdef"
    )

    clean = sanitize_secret_text(raw_error)

    # Asserts that none of the secret substrings appear in the sanitized output
    assert "real_sarvam_secret_998877" not in clean
    assert "super_secret_pw" not in clean
    assert "my_bearer_token_xyz123" not in clean
    assert "sk-live_0123456789abcdef" not in clean
    assert "[REDACTED]" in clean


def test_openai_compat_llm_sanitizes_upstream_http_error(monkeypatch):
    """Verify that OpenAICompatLLM scrubs secret tokens echoed back in urllib HTTPError."""
    monkeypatch.setenv("SARVAM_API_KEY", "secret_key_123456")
    client = OpenAICompatLLM(api_key="secret_key_123456", base_url="https://api.sarvam.ai/v1")

    # Simulate an HTTPError with an error body containing the key
    err_stream = io.BytesIO(b'{"error": "Unauthorized key: secret_key_123456"}')
    http_err = urllib.error.HTTPError(
        url="https://api.sarvam.ai/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=err_stream,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hello"}])

    err_str = str(exc_info.value)
    assert exc_info.value.status_code == 401
    assert "secret_key_123456" not in err_str
    assert "[REDACTED]" in err_str


def test_server_module_start_llm_error_returns_502():
    """Verify that interviewer.server Handler returns HTTP 502 when engine.start fails with LLMError."""
    import tempfile
    import threading
    from http.server import ThreadingHTTPServer
    from interviewer import server as srv

    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db)
    os.environ["INTERVIEW_DB"] = db
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    try:
        with patch("interviewer.server.build_llm", return_value=FailingFakeLLM(status_code=401)):
            req = urllib.request.Request(
                base + "/api/start",
                data=json.dumps({"mock": False, "candidate_name": "Test", "role": "Dev"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=10)
            assert exc_info.value.code == 502
            body = json.loads(exc_info.value.read().decode())
            assert "Interview model error" in body["error"]
            assert "HTTP 401" in body["error"]
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_openai_compat_llm_uses_ssl_context_and_certifi():
    """Verify that OpenAICompatLLM constructs and passes a secure SSL context with certifi."""
    from interviewer.llm import _ssl_context
    ctx = _ssl_context()
    assert ctx is not None

    client = OpenAICompatLLM(api_key="test_key_123")
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "Hello from mock provider"}}]
    }).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "Hello from mock provider"
        assert mock_urlopen.call_count == 1
        _, kwargs = mock_urlopen.call_args
        assert "context" in kwargs
        assert kwargs["context"] is not None


def test_openai_compat_llm_handles_empty_or_malformed_choices():
    """Verify that if LLM provider returns a 200 OK with no choices or empty choices list, an LLMError is raised."""
    client = OpenAICompatLLM(api_key="test_key_123")
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({"choices": []}).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert "Unexpected response schema" in str(exc_info.value)


def test_openai_compat_llm_handles_null_content_with_llm_error():
    """Verify that if LLM provider returns content: null (e.g. reasoning token exhaustion), an informative LLMError is raised."""
    client = OpenAICompatLLM(api_key="test_key_123")
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({
        "choices": [{
            "finish_reason": "length",
            "message": {"content": None, "role": "assistant"}
        }]
    }).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert "returned empty content" in str(exc_info.value)
        assert "finish_reason: length" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Diagnostic Logging & Granular Error Categorization
# ---------------------------------------------------------------------------

def test_llm_diagnostic_logging_for_auth_failure(capsys):
    """Verify that OpenAICompatLLM logs structured diagnostic info for auth failure and redacts secrets."""
    client = OpenAICompatLLM(api_key="super_secret_auth_token_999", base_url="https://api.sarvam.ai/v1")
    err_stream = io.BytesIO(b'{"error": "Invalid api-subscription-key super_secret_auth_token_999"}')
    http_err = urllib.error.HTTPError(
        url="https://api.sarvam.ai/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=err_stream,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}], task="greeting")

    assert exc_info.value.status_code == 401
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip().startswith("{")]
    diag_logs = [log for log in log_lines if log.get("event") == "llm_request_failed"]
    assert len(diag_logs) == 1
    diag = diag_logs[0]
    assert diag["provider"] == "sarvam"
    assert diag["status_code"] == 401
    assert diag["error_category"] == "authentication_failure"
    assert diag["task"] == "greeting"
    # Ensure sensitive credentials are sanitized in both log and exception
    assert "super_secret_auth_token_999" not in json.dumps(diag)
    assert "[REDACTED]" in diag["error_response"]
    assert "super_secret_auth_token_999" not in str(exc_info.value)


def test_llm_diagnostic_logging_for_quota_and_rate_limit(capsys):
    """Verify that OpenAICompatLLM categorizes 429 and 402 as quota_error with diagnostic logging."""
    client = OpenAICompatLLM(api_key="test_key_123", base_url="https://api.sarvam.ai/v1")
    err_stream = io.BytesIO(b'{"error": {"code": 429, "message": "Rate limit exceeded or credits exhausted"}}')
    http_err = urllib.error.HTTPError(
        url="https://api.sarvam.ai/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=err_stream,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}], task="question")

    assert exc_info.value.status_code == 429
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip().startswith("{")]
    diag_logs = [log for log in log_lines if log.get("event") == "llm_request_failed"]
    assert len(diag_logs) == 1
    diag = diag_logs[0]
    assert diag["status_code"] == 429
    assert diag["error_category"] == "quota_error"
    assert "credits exhausted" in diag["error_response"]


def test_llm_diagnostic_logging_for_invalid_request(capsys):
    """Verify that OpenAICompatLLM categorizes 400 and 422 as invalid_request_error."""
    client = OpenAICompatLLM(api_key="test_key_123", base_url="https://api.sarvam.ai/v1")
    err_stream = io.BytesIO(b'{"error": "Unsupported parameter: response_format"}')
    http_err = urllib.error.HTTPError(
        url="https://api.sarvam.ai/v1/chat/completions",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=err_stream,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}], task="generate_questions")

    assert exc_info.value.status_code == 400
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip().startswith("{")]
    diag_logs = [log for log in log_lines if log.get("event") == "llm_request_failed"]
    assert len(diag_logs) == 1
    diag = diag_logs[0]
    assert diag["status_code"] == 400
    assert diag["error_category"] == "invalid_request_error"
    assert diag["task"] == "generate_questions"


def test_llm_diagnostic_logging_for_timeout(capsys):
    """Verify that socket/network timeouts are categorized as timeout."""
    import socket
    client = OpenAICompatLLM(api_key="test_key_123", base_url="https://api.sarvam.ai/v1")
    url_err = urllib.error.URLError(socket.timeout("The read operation timed out"))

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}], task="greeting")

    assert exc_info.value.status_code is None
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip().startswith("{")]
    diag_logs = [log for log in log_lines if log.get("event") == "llm_request_failed"]
    assert len(diag_logs) == 1
    diag = diag_logs[0]
    assert diag["status_code"] is None
    assert diag["error_category"] == "timeout"
    assert "timed out" in diag["error_response"].lower()


def test_llm_diagnostic_logging_for_connection_error(capsys):
    """Verify that network connection failures are categorized as connection_error."""
    client = OpenAICompatLLM(api_key="test_key_123", base_url="https://api.sarvam.ai/v1")
    url_err = urllib.error.URLError("[Errno 8] nodename nor servname provided, or not known")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}], task="greeting")

    assert exc_info.value.status_code is None
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip().startswith("{")]
    diag_logs = [log for log in log_lines if log.get("event") == "llm_request_failed"]
    assert len(diag_logs) == 1
    diag = diag_logs[0]
    assert diag["error_category"] == "connection_error"


def test_api_start_does_not_expose_upstream_payload_or_secrets_to_browser(client):
    """Verify that /api/start returns a sanitized HTTP 502 without exposing internal secrets or raw upstream error payloads to the browser."""
    secret_key = "sarvam_live_secret_key_abcdef"
    sensitive_upstream_body = f"Provider rejected request for key: {secret_key}. Database connection host: postgres://user:pw@internal-host."
    fake_failing_llm = MagicMock()
    fake_failing_llm.chat.side_effect = LLMError(f"HTTP 401 from LLM provider: {sensitive_upstream_body}", status_code=401)
    fake_failing_llm.name = "FailingLLM"

    with patch("interviewer.app.build_llm", return_value=fake_failing_llm):
        resp = client.post("/api/start", json={
            "mock": False,
            "candidate_name": "Test Candidate",
            "role": "Staff Engineer",
            "interview_type": "technical",
        })

    assert resp.status_code == 502
    data = resp.json()
    assert "detail" in data
    # The client response must be concise and safe:
    assert data["detail"] == "Interview model error: Upstream AI service request failed (HTTP 401)."
    # Crucially, the sensitive key and internal body must NOT appear anywhere in the response JSON:
    assert secret_key not in json.dumps(data)
    assert "internal-host" not in json.dumps(data)


