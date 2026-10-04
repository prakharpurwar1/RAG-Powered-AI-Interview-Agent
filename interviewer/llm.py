"""Pluggable LLM clients.

The engine only ever calls ``chat(messages, task=...)``. Swap providers by
changing one line or one env var. ``MockLLM`` needs no key and no network, so
the entire interview flow can be exercised offline.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Base class. ``task`` is an engine hint: greeting|question|followup|score|wrapup."""

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MockLLM(LLMClient):
    """Deterministic, offline stand-in. Exercises every branch without a key."""

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        if task == "score":
            return json.dumps(_mock_scores(messages))
        if task == "followup":
            return ("Thanks. Can you make that concrete — what was your specific "
                    "role, and what was the measurable outcome?")
        if task == "wrapup":
            return ("That's everything from my side. Thanks for your time — "
                    "we'll be in touch about next steps.")
        # greeting / question -> return empty so the engine uses its banked text
        return ""

    @property
    def name(self):
        return "MockLLM (offline)"


def _mock_scores(messages):
    """Heuristic scores from transcript length — enough to show the shape."""
    text = " ".join(m.get("content", "") for m in messages)
    words = len(text.split())
    base = 3.0 + min(1.5, words / 400.0)
    return {
        "competencies": [
            {"id": "communication", "score": round(base, 1),
             "evidence": "Answers were structured.",
             "comment": "Clear enough; could be tighter."},
            {"id": "problem_solving", "score": round(max(2.0, base - 0.3), 1),
             "evidence": "Outlined an approach.",
             "comment": "Would like more trade-off detail."},
            {"id": "fundamentals", "score": round(max(2.0, base - 0.2), 1),
             "evidence": "Referenced relevant concepts.",
             "comment": "Depth not fully demonstrated."},
            {"id": "ownership", "score": round(max(2.0, base - 0.4), 1),
             "evidence": "Described outcomes.",
             "comment": "Impact could be more measurable."},
            {"id": "motivation", "score": round(max(2.0, base - 0.5), 1),
             "evidence": "Gave reasons for interest.",
             "comment": "Reasonably specific."},
        ],
        "summary": "Offline mock evaluation. Run with a real LLM for genuine scoring.",
    }


class OpenAICompatLLM(LLMClient):
    """Works with any OpenAI-compatible /chat/completions endpoint.

    Defaults to Sarvam AI (strong Indian-language support, India-hosted). Also
    fine for OpenAI, Together, Groq, vLLM, Ollama — just change base_url/model.
    """

    def __init__(self, api_key=None, base_url=None, model=None, auth_style=None, timeout=60):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL",
                                                     "https://api.sarvam.ai/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "sarvam-105b")
        # Sarvam uses the 'api-subscription-key' header; OpenAI-style uses Bearer.
        self.auth_style = auth_style or os.environ.get("LLM_AUTH_STYLE", "api-subscription-key")
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("No API key. Set LLM_API_KEY or use MockLLM.")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.auth_style == "api-subscription-key":
            h["api-subscription-key"] = self.api_key
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages, *, task="question", temperature=0.6,
             max_tokens=800, json_mode=False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:400]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Network error: {e}") from e
        return data["choices"][0]["message"]["content"]

    @property
    def name(self):
        return f"OpenAICompatLLM({self.model} @ {self.base_url})"


def build_llm(mock: bool = False) -> LLMClient:
    """Factory: MockLLM if requested or no key present, else the real client."""
    if mock or not os.environ.get("LLM_API_KEY"):
        return MockLLM()
    return OpenAICompatLLM()
