"""Range-resume of scripts/download_weights.py against a local HTTP server.

Needs httpx + huggingface_hub (the backend venv has them); skipped with a
plain stdlib Python:  .venv/bin/python -m unittest tests.test_download_resume
"""
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    import httpx
    import download_weights as dw
except ImportError:  # pragma: no cover
    dw = None

PAYLOAD = bytes(range(256)) * 400  # 100 KB


class Handler(BaseHTTPRequestHandler):
    ranges = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/hub/file":  # hub redirects to the "CDN"
            self.send_response(302)
            self.send_header("Location", "/cdn/file")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        Handler.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipIf(dw is None, "needs httpx + huggingface_hub (backend venv)")
class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        Handler.ranges = []

    def test_resumes_partial_file(self):
        dest = os.path.join(tempfile.mkdtemp(), "model.bin")
        with open(dest + ".part", "wb") as f:
            f.write(PAYLOAD[:30000])
        with httpx.Client() as client:
            dw.download_file(client, self.base + "/hub/file", {}, dest,
                             expected_size=len(PAYLOAD))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertEqual(Handler.ranges, ["bytes=30000-"])
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_oversized_part_restarts(self):
        dest = os.path.join(tempfile.mkdtemp(), "model.bin")
        with open(dest + ".part", "wb") as f:
            f.write(b"x" * (len(PAYLOAD) + 10))
        with httpx.Client() as client:
            dw.download_file(client, self.base + "/hub/file", {}, dest,
                             expected_size=len(PAYLOAD))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertEqual(Handler.ranges, [None])


if __name__ == "__main__":
    unittest.main()
