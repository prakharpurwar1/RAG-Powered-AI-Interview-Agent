"""Server-side voice I/O (Sarvam Saaras + Bulbul), with an offline mock.

Two paths are offered:

  * REST  -- simple and robust; good for push-to-talk clips up to 30 s.
      POST https://api.sarvam.ai/speech-to-text      (Saaras, model saaras:v4)
      POST https://api.sarvam.ai/text-to-speech      (Bulbul, model bulbul:v3)

  * HTTP-stream -- same TTS, but the audio is returned as one binary body so
    playback can start sooner; good for low-latency turn-taking.
      POST https://api.sarvam.ai/text-to-speech/stream

For *true* partial transcripts while the user is still speaking, use
``streaming.SarvamStreamingASR`` (WebSocket). Auth is the ``api-subscription-key``
header; the base URL is ``https://api.sarvam.ai`` (no ``/v1`` prefix).

When no key is configured, ``VoiceIO(mock=True)`` (or the automatic fallback)
returns a scripted transcript and a short silent WAV, so the whole voice path is
exercisable offline.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import urllib.error
import urllib.request
import uuid

SARVAM_BASE = "https://api.sarvam.ai"


def silent_wav(seconds=0.4, sample_rate=16000) -> bytes:
    """A tiny, valid mono PCM16 WAV (silence) — used by the mock and by tests."""
    n = int(seconds * sample_rate)
    data = b"\x00\x00" * n
    byte_rate = sample_rate * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


class VoiceIO:
    def __init__(self, api_key=None, mock=False):
        self.api_key = (api_key
                        or os.environ.get("SARVAM_API_KEY")
                        or os.environ.get("LLM_API_KEY", ""))
        self.mock = mock or not self.api_key

    # -- capabilities -----------------------------------------------------
    def capabilities(self) -> dict:
        return {
            "server_voice": not self.mock,
            "mock": self.mock,
            "asr": "saaras:v4 (REST) + saaras:v3-realtime (WebSocket)",
            "tts": "bulbul:v3 (REST + HTTP stream)",
        }

    # -- Speech to text (Saaras, REST) ------------------------------------
    def transcribe(self, audio_bytes: bytes, filename="audio.webm",
                   language_code="en-IN", model="saaras:v4") -> str:
        if self.mock:
            return "This is a mock transcript from the offline voice engine."
        fields = {"model": model, "language_code": language_code}
        body, content_type = _multipart(fields, "file", filename, audio_bytes)
        data = self._post("/speech-to-text", body, content_type)
        return data.get("transcript", "")

    # -- Text to speech (Bulbul, REST) ------------------------------------
    def synthesize(self, text: str, language_code="en-IN",
                   speaker="shubh", model="bulbul:v3") -> bytes:
        if self.mock:
            return silent_wav()
        payload = _tts_payload(text, language_code, speaker, model)
        data = self._post("/text-to-speech", json.dumps(payload).encode(),
                          "application/json")
        audios = data.get("audios") or []
        return base64.b64decode(audios[0]) if audios else b""

    # -- Text to speech (Bulbul, HTTP stream) -----------------------------
    def synthesize_stream(self, text: str, language_code="en-IN",
                          speaker="shubh", model="bulbul:v3") -> bytes:
        """Single-POST streaming variant: returns the audio body directly."""
        if self.mock:
            return silent_wav()
        payload = _tts_payload(text, language_code, speaker, model)
        return self._post_raw("/text-to-speech/stream",
                              json.dumps(payload).encode(), "application/json")

    # -- HTTP plumbing ----------------------------------------------------
    def _headers(self, content_type):
        return {"api-subscription-key": self.api_key, "Content-Type": content_type}

    def _post(self, path, body, content_type):
        raw = self._post_raw(path, body, content_type)
        return json.loads(raw.decode())

    def _post_raw(self, path, body, content_type):
        req = urllib.request.Request(SARVAM_BASE + path, data=body,
                                     headers=self._headers(content_type), method="POST")
        try:
            from .llm import _ssl_context
            ctx = _ssl_context()
        except Exception:
            ctx = None
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Sarvam HTTP {e.code}: "
                               f"{e.read().decode('utf-8', 'ignore')[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Sarvam network error: {e}") from e


def _tts_payload(text, language_code, speaker, model):
    # Both key spellings have appeared in Sarvam's docs; send both to be safe.
    return {
        "text": text[:2500],
        "language_code": language_code,
        "target_language_code": language_code,
        "speaker": speaker,
        "model": model,
    }


def _multipart(fields: dict, file_field: str, filename: str, content: bytes):
    boundary = uuid.uuid4().hex
    lines = []
    for k, v in fields.items():
        lines += [f"--{boundary}", f'Content-Disposition: form-data; name="{k}"', "", str(v)]
    lines += [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"',
        "Content-Type: application/octet-stream",
        "",
    ]
    body = ("\r\n".join(lines).encode() + b"\r\n" + content
            + f"\r\n--{boundary}--\r\n".encode())
    return body, f"multipart/form-data; boundary={boundary}"


def build_voice(mock=False) -> VoiceIO:
    return VoiceIO(mock=mock)
