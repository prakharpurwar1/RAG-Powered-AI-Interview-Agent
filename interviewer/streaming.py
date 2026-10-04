"""Real-time voice, with no third-party dependencies.

This module provides:
  * a tiny RFC 6455 WebSocket frame codec (client *and* server side),
  * a WebSocket client used to talk to Sarvam's streaming endpoints,
  * ``SarvamStreamingASR`` -- live partial transcripts via
    ``wss://api.sarvam.ai/speech-to-text-realtime/ws``,
  * ``MockStreamingASR`` -- an offline stand-in so the whole path is testable.

The HTTP voice path (REST ASR + REST/HTTP-stream TTS) lives in ``voice.py``;
this file is for the low-latency streaming path.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
from urllib.parse import urlencode, urlparse

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Handshake helpers
# --------------------------------------------------------------------------
def accept_key(sec_key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((sec_key + WS_GUID).encode()).digest()
    ).decode()


def is_websocket_request(headers) -> bool:
    upgrade = (headers.get("Upgrade") or "").lower()
    connection = (headers.get("Connection") or "").lower()
    return upgrade == "websocket" and "upgrade" in connection


def server_handshake(headers, conn) -> bool:
    """Complete the server side of the handshake on a raw socket. Returns True on success."""
    key = headers.get("Sec-WebSocket-Key")
    if not key:
        return False
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n"
    )
    conn.sendall(response.encode())
    return True


# --------------------------------------------------------------------------
# Frame codec
# --------------------------------------------------------------------------
def encode_frame(payload: bytes, opcode: int = OP_TEXT, mask: bool = True) -> bytes:
    header = bytearray()
    header.append(0x80 | opcode)                       # FIN + opcode
    length = len(payload)
    if mask:
        header.append(0x80 | (126 if length >= 126 and length < 65536
                              else 127 if length >= 65536 else length))
    else:
        header.append(126 if length >= 126 and length < 65536
                      else 127 if length >= 65536 else length)
    if length >= 65536:
        header += struct.pack(">Q", length)
    elif length >= 126:
        header += struct.pack(">H", length)
    if mask:
        mask_key = os.urandom(4)
        header += mask_key
        payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(length))
    return bytes(header) + payload


class _Buffered:
    """Small buffered reader over a socket (handles bytes left after handshake)."""

    def __init__(self, sock, initial: bytes = b""):
        self.sock = sock
        self.buf = bytearray(initial)

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise WebSocketError("connection closed")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def read_frame(reader: _Buffered):
    """Read one frame. Returns (opcode, payload_bytes, fin)."""
    b1, b2 = reader.read(2)
    fin = bool(b1 & 0x80)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack(">H", reader.read(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", reader.read(8))[0]
    mask_key = reader.read(4) if masked else None
    payload = reader.read(length) if length else b""
    if mask_key:
        payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(len(payload)))
    return opcode, payload, fin


def send_frame(conn, payload: bytes, opcode: int = OP_TEXT, mask: bool = False):
    """Server-side send (unmasked by default)."""
    conn.sendall(encode_frame(payload, opcode=opcode, mask=mask))


# --------------------------------------------------------------------------
# WebSocket client
# --------------------------------------------------------------------------
class WebSocketClient:
    def __init__(self, url: str, headers: dict | None = None, timeout: int = 30):
        u = urlparse(url)
        self.secure = u.scheme == "wss"
        self.host = u.hostname
        self.port = u.port or (443 if self.secure else 80)
        self.path = u.path + (("?" + u.query) if u.query else "")
        self.headers = headers or {}
        self.timeout = timeout
        self.sock = None
        self.reader = None

    def connect(self) -> "WebSocketClient":
        raw = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.secure:
            raw = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        )
        for k, v in self.headers.items():
            req += f"{k}: {v}\r\n"
        req += "\r\n"
        raw.sendall(req.encode())

        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = raw.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed during handshake")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in status:
            raise WebSocketError(f"handshake failed: {status}")

        self.sock = raw
        self.reader = _Buffered(raw, rest)
        return self

    def send_text(self, text: str):
        self.sock.sendall(encode_frame(text.encode(), OP_TEXT, mask=True))

    def send_binary(self, data: bytes):
        self.sock.sendall(encode_frame(data, OP_BIN, mask=True))

    def recv(self):
        """Returns (opcode, payload_bytes). Handles ping/pong and close transparently."""
        while True:
            opcode, payload, _fin = read_frame(self.reader)
            if opcode == OP_PING:
                self.sock.sendall(encode_frame(payload, OP_PONG, mask=True))
                continue
            if opcode == OP_PONG:
                continue
            return opcode, payload

    def recv_json(self) -> dict:
        opcode, payload = self.recv()
        if opcode == OP_CLOSE:
            raise WebSocketError("closed by peer")
        try:
            return json.loads(payload.decode())
        except Exception:
            return {"event": "binary", "size": len(payload)}

    def close(self):
        try:
            if self.sock:
                self.sock.sendall(encode_frame(b"", OP_CLOSE, mask=True))
        except Exception:
            pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Sarvam streaming ASR
# --------------------------------------------------------------------------
class SarvamStreamingASR:
    """Live transcription over Sarvam's Realtime WebSocket.

    Feed it mono PCM16 audio; read partial and final transcripts as events.
    """

    def __init__(self, api_key=None, language_code="en-IN", model="saaras:v3-realtime",
                 sample_rate=16000, stream_type="balanced", mode="transcribe",
                 timeout=30):
        self.api_key = (api_key or os.environ.get("SARVAM_API_KEY")
                        or os.environ.get("LLM_API_KEY", ""))
        if not self.api_key:
            raise WebSocketError("No Sarvam API key for streaming ASR.")
        self.params = {
            "language_code": language_code,
            "model": model,
            "stream_type": stream_type,
            "mode": mode,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
        }
        self.timeout = timeout
        self.ws = None

    def connect(self):
        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{urlencode(self.params)}"
        self.ws = WebSocketClient(url, headers={"api-subscription-key": self.api_key},
                                  timeout=self.timeout).connect()
        return self

    def send_audio(self, pcm16: bytes):
        self.ws.send_text(json.dumps({
            "event": "audio_input",
            "audio": base64.b64encode(pcm16).decode(),
        }))

    def flush(self):
        self.ws.send_text(json.dumps({"event": "flush"}))

    def next_event(self) -> dict:
        return self.ws.recv_json()

    def close(self):
        try:
            self.ws.send_text(json.dumps({"event": "end"}))
        except Exception:
            pass
        self.ws.close()


class MockStreamingASR:
    """Offline stand-in: turns bytes of audio into a scripted partial + final.

    Lets the full streaming path be exercised with no key and no network.
    """

    def __init__(self, *args, **kwargs):
        self._sent = 0

    def connect(self):
        return self

    def send_audio(self, pcm16: bytes):
        self._sent += len(pcm16)

    def flush(self):
        pass

    def next_event(self) -> dict:
        return {"event": "transcript.partial", "text": "…(streaming mock)"}

    def close(self):
        pass


def build_streaming_asr(mock: bool = False, **kwargs):
    if mock or not (os.environ.get("SARVAM_API_KEY") or os.environ.get("LLM_API_KEY")):
        return MockStreamingASR()
    return SarvamStreamingASR(**kwargs)
