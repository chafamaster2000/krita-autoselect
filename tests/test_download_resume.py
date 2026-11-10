"""Range-resume of scripts/download_weights.py against a local HTTP server.

Needs httpx + huggingface_hub (the backend venv has them); skipped with a
plain stdlib Python:  .venv/bin/python -m unittest tests.test_download_resume
"""
import contextlib
import io
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

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


class Forbidden(BaseHTTPRequestHandler):
    """The hub with a valid token whose license wasn't accepted."""
    hits = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        Forbidden.hits += 1
        body = b'{"error": "Access to model facebook/sam3 is restricted"}'
        self.send_response(403)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipIf(dw is None, "needs httpx + huggingface_hub (backend venv)")
class GatedTest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Forbidden)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        Forbidden.hits = 0
        sleeps = mock.patch.object(dw.time, "sleep",
                                   side_effect=AssertionError("retried"))
        sleeps.start()
        self.addCleanup(sleeps.stop)

    def test_403_is_not_retried(self):
        dest = os.path.join(tempfile.mkdtemp(), "config.json")
        with httpx.Client() as client, \
                self.assertRaises(httpx.HTTPStatusError) as cm:
            dw.download_file(client, self.base + "/f", {}, dest, 10)
        self.assertEqual(cm.exception.response.status_code, 403)
        self.assertEqual(Forbidden.hits, 1)

    def test_main_shows_gated_help_and_writes_no_marker(self):
        target = tempfile.mkdtemp()
        out = io.StringIO()
        with mock.patch.object(dw, "list_files",
                               return_value=[("config.json", 10)]), \
                mock.patch.object(dw, "hf_hub_url",
                                  lambda repo, name: self.base + "/" + name), \
                contextlib.redirect_stdout(out):
            self.assertEqual(dw.main([target]), 1)
        self.assertIn(dw.GATED_HELP, out.getvalue())
        self.assertFalse(os.path.exists(os.path.join(target, dw.DONE_MARKER)))


@unittest.skipIf(dw is None, "needs httpx + huggingface_hub (backend venv)")
class MarkerTest(unittest.TestCase):
    """The plugin trusts the folder only once every file is in."""

    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def run_main(self, target, argv=()):
        names = [("config.json", len(PAYLOAD)), ("tokenizer.json", len(PAYLOAD))]
        with mock.patch.object(dw, "list_files", return_value=names), \
                mock.patch.object(dw, "hf_hub_url",
                                  lambda repo, name: self.base + "/hub/file"), \
                contextlib.redirect_stdout(io.StringIO()):
            return dw.main([target, *argv])

    def test_marker_only_after_every_file(self):
        sys.path.insert(0, os.path.join(ROOT, "krita-plugin",
                                        "kritaautoselect"))
        import backend as bk
        self.assertEqual(dw.DONE_MARKER, bk.WEIGHTS_DONE_MARKER)
        target = tempfile.mkdtemp()
        marker = os.path.join(target, dw.DONE_MARKER)
        self.assertEqual(self.run_main(target, ["--only", "config.json"]), 0)
        self.assertFalse(os.path.exists(marker))  # partial by request
        self.assertEqual(self.run_main(target), 0)
        self.assertTrue(os.path.exists(marker))
        self.assertTrue(bk.weights_complete(target))


if __name__ == "__main__":
    unittest.main()
