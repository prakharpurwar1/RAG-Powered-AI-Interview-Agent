"""Tests for configuration validation, missing fields handling, and frontend index.html constraints.

Run: python -m pytest tests/test_validation.py
"""
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer.config import InterviewConfig
from interviewer import server as srv


def _post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def _start_server():
    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db)
    os.environ["INTERVIEW_DB"] = db
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port, db


# ===========================================================================
# 1. InterviewConfig Direct Validation Tests
# ===========================================================================

def test_valid_submission_without_resume_and_jd():
    """Verify that selecting all required fields allows interview to start without resume or JD."""
    cfg = InterviewConfig(
        mode="practice",
        interview_type="technical",
        role="Frontend Engineer",
        level="mid",
        candidate_name="Alice Candidate",
        language="en-IN",
        max_questions=6,
        resume_text="",
        job_description="",
    )
    validated = cfg.validate()
    assert validated.role == "Frontend Engineer"
    assert validated.resume_text == ""
    assert validated.job_description == ""


def test_valid_submission_with_optional_docs_and_email():
    """Verify that optional candidate email, resume, and JD are accepted and preserved."""
    cfg = InterviewConfig(
        mode="practice",
        interview_type="mixed",
        role="Backend Engineer",
        level="senior",
        candidate_name="Bob Engineer",
        candidate_email="bob@example.com",
        language="en-IN",
        max_questions=4,
        resume_text="Extensive distributed systems experience with Python and Go.",
        job_description="Seeking a Senior Backend Engineer to build scalable microservices.",
    )
    validated = cfg.validate()
    assert validated.candidate_email == "bob@example.com"
    assert "distributed systems" in validated.resume_text
    assert "microservices" in validated.job_description


def test_missing_role_raises():
    """Verify that empty or whitespace-only role raises a validation error."""
    with pytest.raises(ValueError, match="role is required"):
        InterviewConfig(role="").validate()
    with pytest.raises(ValueError, match="role is required"):
        InterviewConfig(role="   ").validate()


def test_missing_candidate_name_raises():
    """Verify that empty candidate full name raises a validation error."""
    with pytest.raises(ValueError, match="candidate_name is required"):
        InterviewConfig(candidate_name="").validate()
    with pytest.raises(ValueError, match="candidate_name is required"):
        InterviewConfig(candidate_name="   ").validate()


def test_missing_or_invalid_mode_raises():
    """Verify that empty or invalid mode raises a validation error."""
    with pytest.raises(ValueError, match="mode must be 'practice' or 'hiring'"):
        InterviewConfig(mode="").validate()
    with pytest.raises(ValueError, match="mode must be 'practice' or 'hiring'"):
        InterviewConfig(mode="unknown_mode").validate()


def test_missing_or_invalid_interview_type_raises():
    """Verify that empty or invalid assessment focus raises a validation error."""
    with pytest.raises(ValueError, match="interview_type must be behavioral/technical/mixed"):
        InterviewConfig(interview_type="").validate()
    with pytest.raises(ValueError, match="interview_type must be behavioral/technical/mixed"):
        InterviewConfig(interview_type="other").validate()


def test_missing_or_invalid_level_raises():
    """Verify that empty or invalid seniority level raises a validation error."""
    with pytest.raises(ValueError, match="level must be junior/mid/senior"):
        InterviewConfig(level="").validate()
    with pytest.raises(ValueError, match="level must be junior/mid/senior"):
        InterviewConfig(level="lead").validate()


def test_missing_language_raises():
    """Verify that empty interview language raises a validation error."""
    with pytest.raises(ValueError, match="language is required"):
        InterviewConfig(language="").validate()


def test_invalid_max_questions_raises():
    """Verify that zero, negative, or invalid max_questions raises a validation error."""
    with pytest.raises(ValueError, match="max_questions must be a positive integer"):
        InterviewConfig(max_questions=0).validate()
    with pytest.raises(ValueError, match="max_questions must be a positive integer"):
        InterviewConfig(max_questions=-1).validate()


def test_invalid_email_format_raises_only_when_entered():
    """Verify email format validation is enforced only when an email is provided."""
    # Empty email is optional and valid
    cfg_empty_email = InterviewConfig(candidate_email="")
    assert cfg_empty_email.validate().candidate_email == ""

    # Valid email is accepted
    cfg_valid_email = InterviewConfig(candidate_email="test.user@company.co")
    assert cfg_valid_email.validate().candidate_email == "test.user@company.co"

    # Invalid email raises
    with pytest.raises(ValueError, match="candidate_email must be a valid email address"):
        InterviewConfig(candidate_email="not-an-email").validate()


# ===========================================================================
# 2. HTTP Server API Endpoint Validation Tests
# ===========================================================================

def test_api_start_with_all_required_fields_and_no_docs():
    """Server successfully starts an interview when all required fields are provided without docs."""
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        st, body, _ = _post(base + "/api/start", {
            "mock": True,
            "mode": "practice",
            "interview_type": "technical",
            "role": "Full Stack Engineer",
            "level": "senior",
            "candidate_name": "Devin Patel",
            "language": "en-IN",
            "max_questions": 4,
            "resume_text": "",
            "job_description": "",
        })
        assert st == 200
        data = json.loads(body)
        assert data["session_id"]
        assert data["interview_id"]
        assert "message" in data
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


def test_api_start_rejects_missing_required_fields():
    """Server rejects /api/start requests when required fields are empty strings."""
    httpd, port, db = _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        # Empty role
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/api/start", {
                "mock": True,
                "mode": "practice",
                "interview_type": "technical",
                "role": "",
                "level": "mid",
                "candidate_name": "John Doe",
                "language": "en-IN",
            })

        # Empty candidate_name
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/api/start", {
                "mock": True,
                "mode": "practice",
                "interview_type": "technical",
                "role": "Frontend Engineer",
                "level": "mid",
                "candidate_name": "",
                "language": "en-IN",
            })

        # Empty interview_type
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/api/start", {
                "mock": True,
                "mode": "practice",
                "interview_type": "",
                "role": "Frontend Engineer",
                "level": "mid",
                "candidate_name": "John Doe",
                "language": "en-IN",
            })

        # Empty level
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/api/start", {
                "mock": True,
                "mode": "practice",
                "interview_type": "mixed",
                "role": "Frontend Engineer",
                "level": "",
                "candidate_name": "John Doe",
                "language": "en-IN",
            })

        # Empty language
        with pytest.raises(urllib.error.HTTPError):
            _post(base + "/api/start", {
                "mock": True,
                "mode": "practice",
                "interview_type": "mixed",
                "role": "Frontend Engineer",
                "level": "mid",
                "candidate_name": "John Doe",
                "language": "",
            })
    finally:
        httpd.shutdown()
        try:
            os.unlink(db)
        except OSError:
            pass


# ===========================================================================
# 3. Frontend HTML / Validation Structure Tests
# ===========================================================================

def test_frontend_html_placeholders_and_initial_values():
    """Verify web/index.html includes neutral placeholders and does not preselect values."""
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Purpose & Tone dropdown has neutral placeholder and no preselected value
    assert '<select id="mode">' in html
    assert re.search(r'<select id="mode">\s*<option value="" disabled selected>Select purpose and tone</option>', html)
    assert 'value="practice" selected' not in html

    # 2. Assessment Focus dropdown has neutral placeholder
    assert '<select id="interview_type">' in html
    assert re.search(r'<select id="interview_type">\s*<option value="" disabled selected>Select assessment focus</option>', html)
    assert 'value="mixed" selected' not in html

    # 3. Seniority Level dropdown has neutral placeholder and no preselected mid-level
    assert '<select id="level">' in html
    assert re.search(r'<select id="level">\s*<option value="" disabled selected>Select seniority level</option>', html)
    assert 'value="mid" selected' not in html

    # 4. Question Count dropdown has neutral placeholder and no preselected 6 questions
    assert '<select id="max_questions">' in html
    assert re.search(r'<select id="max_questions">\s*<option value="" disabled selected>Select question count</option>', html)
    assert 'value="6" selected' not in html

    # 5. Interview Language dropdown has neutral placeholder and no preselected English
    assert '<select id="language">' in html
    assert re.search(r'<select id="language">\s*<option value="" disabled selected>Select interview language</option>', html)
    assert 'value="en-IN" selected' not in html

    # 6. Target Role input is empty by default (no "Software Engineer")
    assert '<input id="role" type="text" value=""' in html

    # 7. No role suggestion chip has 'active' class by default
    chips = re.findall(r'<span class="chip[^"]*"', html)
    assert len(chips) > 0
    for chip in chips:
        assert 'active' not in chip

    # 8. Individual error message containers exist for each required field
    for field_id in ["mode", "interview_type", "role", "level", "max_questions", "candidate_name", "candidate_email", "language", "consent"]:
        assert f'id="err_{field_id}"' in html

    # 9. No silent fallback defaults in JS startBtn payload
    assert 'const roleVal = $("role").value.trim() || "Software Engineer"' not in html
    assert 'const levelVal = $("level").value || "mid"' not in html
    assert 'const modeVal = $("mode").value || "practice"' not in html
