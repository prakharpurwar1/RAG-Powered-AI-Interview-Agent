"""Tests for the WebSocket codec and the streaming ASR bridge pieces.

Run:  python tests/test_streaming.py
"""
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interviewer import streaming as S


def test_frame_roundtrip_masked():
    payload = b"hello world"
    a, b = socket.socketpair()
    try:
        b.sendall(S.encode_frame(payload, S.OP_TEXT, mask=True))
        op, data, fin = S.read_frame(S._Buffered(a))
        assert op == S.OP_TEXT and data == payload and fin
    finally:
        a.close(); b.close()


def test_frame_large_payload():
    payload = b"x" * 70000                     # forces the 64-bit length path
    a, b = socket.socketpair()
    try:
        b.sendall(S.encode_frame(payload, S.OP_BIN, mask=True))
        op, data, _ = S.read_frame(S._Buffered(a))
        assert data == payload
    finally:
        a.close(); b.close()


def test_accept_key_known_vector():
    # RFC 6455 example key -> accept value
    assert S.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_handshake_and_echo_over_real_socket():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    errors = []

    def run():
        try:
            conn, _ = srv.accept()
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += conn.recv(4096)
            head = buf.split(b"\r\n\r\n")[0].decode()
            headers = {}
            for line in head.split("\r\n")[1:]:
                k, _, v = line.partition(":")
                headers[k.strip()] = v.strip()
            assert S.server_handshake(headers, conn), "server handshake failed"
            reader = S._Buffered(conn)
            op, data, _ = S.read_frame(reader)
            S.send_frame(conn, b"echo:" + data, S.OP_TEXT)
            conn.close()
        except Exception as e:                  # surface thread failures
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    c = S.WebSocketClient(f"ws://127.0.0.1:{port}/x").connect()
    c.send_text("ping")
    op, payload = c.recv()
    assert payload == b"echo:ping", payload
    c.close()
    t.join(timeout=5)
    srv.close()
    assert not errors, errors


def test_mock_streaming_asr_is_usable():
    asr = S.MockStreamingASR().connect()
    asr.send_audio(b"\x00" * 320)
    asr.flush()
    assert "partial" in asr.next_event()["event"]
    asr.close()


def test_build_streaming_asr_defaults_to_mock_without_key(monkeypatch_env=None):
    for k in ("SARVAM_API_KEY", "LLM_API_KEY"):
        os.environ.pop(k, None)
    assert isinstance(S.build_streaming_asr(), S.MockStreamingASR)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
