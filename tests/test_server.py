"""HTTP contract tests for the krita-autoselect daemon.

A FakeEngine replaces the real SAM 3 engine, so these run with stdlib only —
no torch, no GPU, no weights. They pin the request/response contract that
`kri select sam` depends on.
"""
import base64
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402


class FakeEngine:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or {
            "width": 640, "height": 480, "count": 1,
            "instances": [{"index": 0, "score": 0.97, "box": [10, 10, 50, 50]}],
            "mask_b64": base64.b64encode(b"fakemaskpng").decode(),
        }
        self.error = error

    def info(self):
        return {"model": "fake/sam3", "loaded": False,
                "device": None, "unload_after_s": 300.0}

    def segment(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class ServerTest(unittest.TestCase):
    def _serve(self, engine):
        srv = server.make_server(engine, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def _post(self, url, path, payload, headers=None):
        req = urllib.request.Request(
            url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_reports_engine_info(self):
        url = self._serve(FakeEngine())
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            data = json.loads(r.read())
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "krita-autoselect")
        self.assertFalse(data["loaded"])

    def test_segment_passes_params_and_returns_mask(self):
        engine = FakeEngine()
        url = self._serve(engine)
        status, data = self._post(url, "/segment", {
            "image_b64": "aW1n", "text": "the red car",
            "points": [[10, 20]], "point_labels": [1],
            "threshold": 0.6, "combine": "union",
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("mask_b64", data)
        call = engine.calls[-1]
        self.assertEqual(call["text"], "the red car")
        self.assertEqual(call["points"], [[10, 20]])
        self.assertEqual(call["threshold"], 0.6)
        self.assertEqual(call["combine"], "union")
        self.assertFalse(call["list_only"])

    def test_segment_combine_index_is_int(self):
        engine = FakeEngine()
        url = self._serve(engine)
        status, _ = self._post(url, "/segment", {
            "image_b64": "aW1n", "text": "car", "combine": "2"})
        self.assertEqual(status, 200)
        self.assertEqual(engine.calls[-1]["combine"], 2)

    def test_segment_within_passthrough(self):
        engine = FakeEngine()
        url = self._serve(engine)
        status, _ = self._post(url, "/segment", {
            "image_b64": "aW1n", "text": "hand", "within": [10, 20, 300, 200]})
        self.assertEqual(status, 200)
        self.assertEqual(engine.calls[-1]["within"], [10, 20, 300, 200])

    def test_segment_within_alone_is_valid_prompt(self):
        engine = FakeEngine()
        url = self._serve(engine)
        status, _ = self._post(url, "/segment", {
            "image_b64": "aW1n", "within": [0, 0, 50, 50]})
        self.assertEqual(status, 200)

    def test_segment_requires_some_prompt(self):
        url = self._serve(FakeEngine())
        status, data = self._post(url, "/segment", {"image_b64": "aW1n"})
        self.assertEqual(status, 400)
        self.assertIn("prompt", data["error"])

    def test_segment_requires_image(self):
        url = self._serve(FakeEngine())
        status, data = self._post(url, "/segment", {"text": "car"})
        self.assertEqual(status, 400)
        self.assertIn("image_b64", data["error"])

    def test_segment_validates_points_shape(self):
        url = self._serve(FakeEngine())
        status, data = self._post(url, "/segment", {
            "image_b64": "aW1n", "points": [[1, 2, 3]]})
        self.assertEqual(status, 400)
        self.assertIn("points", data["error"])

    def test_engine_error_returns_500_with_message(self):
        url = self._serve(FakeEngine(error=RuntimeError("no weights, see docs")))
        status, data = self._post(url, "/segment",
                                  {"image_b64": "aW1n", "text": "car"})
        self.assertEqual(status, 500)
        self.assertIn("no weights", data["error"])

    def test_browser_origin_rejected(self):
        url = self._serve(FakeEngine())
        status, data = self._post(url, "/segment",
                                  {"image_b64": "aW1n", "text": "car"},
                                  headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_invalid_json_is_400(self):
        url = self._serve(FakeEngine())
        req = urllib.request.Request(url + "/segment", data=b"not json",
                                     method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 400)

    def test_unknown_endpoint_404(self):
        url = self._serve(FakeEngine())
        status, _ = self._post(url, "/nope", {})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
