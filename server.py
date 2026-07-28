#!/usr/bin/env python3
"""krita-autoselect — local SAM 3 segmentation daemon.

Segments any PNG/JPEG by text prompt ("the red car"), click points and/or a
bounding box, and returns a canvas-sized grayscale mask. Built for `kri
select sam` (Krita selections) but Krita-independent: POST an image, get a
mask.

HTTP on 127.0.0.1 only, stdlib server, one endpoint:
  GET  /health   — daemon + model state
  POST /segment  — {image_b64, text?, points?, point_labels?, box?,
                    threshold?, mask_threshold?, combine?, list_only?}

Env:
  AUTOSELECT_PORT            port (default 5679)
  AUTOSELECT_MODEL           HF model id (default facebook/sam3)
  AUTOSELECT_WEIGHTS_PATH    local model dir (skips Hugging Face entirely)
  AUTOSELECT_DEVICE          cuda | cpu (default: auto)
  AUTOSELECT_DTYPE           bfloat16 | float16 | float32 (default: auto)
  AUTOSELECT_UNLOAD_AFTER_S  idle seconds before freeing VRAM; 0 = resident
                             (default 300)
  AUTOSELECT_COMFYUI_URL     ComfyUI to ask for VRAM before loading
                             (default http://127.0.0.1:8001; empty disables)
  AUTOSELECT_TOKEN           optional shared token (X-Autoselect-Token)
  HF_TOKEN                   standard Hugging Face auth for gated weights
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 5679
MAX_BODY_BYTES = 96 * 1024 * 1024
AUTH_TOKEN = os.environ.get("AUTOSELECT_TOKEN", "")

VALID_PROMPT_KEYS = ("text", "points", "box")


class BadRequest(ValueError):
    """Client error — message is returned with HTTP 400."""


def parse_segment_params(payload):
    """Validate /segment JSON into kwargs for engine.segment()."""
    if not isinstance(payload, dict):
        raise BadRequest("Body must be a JSON object")
    if not payload.get("image_b64"):
        raise BadRequest("image_b64 is required")
    if not any(payload.get(k) for k in VALID_PROMPT_KEYS):
        raise BadRequest("At least one prompt is required: text, points or box")

    points = payload.get("points")
    if points is not None:
        if (not isinstance(points, list)
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise BadRequest("points must be a list of [x, y] pairs")
    box = payload.get("box")
    if box is not None and (not isinstance(box, list) or len(box) != 4):
        raise BadRequest("box must be [x, y, width, height]")

    combine = payload.get("combine", "union")
    if combine != "union":
        try:
            combine = int(combine)
        except (TypeError, ValueError):
            raise BadRequest("combine must be 'union' or an instance index")

    try:
        threshold = float(payload.get("threshold", 0.5))
        mask_threshold = float(payload.get("mask_threshold", 0.5))
    except (TypeError, ValueError):
        raise BadRequest("threshold/mask_threshold must be numbers")

    return {
        "image_b64": payload["image_b64"],
        "text": payload.get("text") or None,
        "points": points,
        "point_labels": payload.get("point_labels"),
        "box": box,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "combine": combine,
        "list_only": bool(payload.get("list_only", False)),
    }


class AutoselectHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # our own prints are enough

    def _reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        """Same policy as kritamcp: reject anything browser-originated (CSRF
        / DNS-rebinding against a localhost server) + optional shared token."""
        if self.headers.get("Origin") or self.headers.get("Referer"):
            self._reply({"error": "Forbidden: browser-origin request rejected"}, 403)
            return False
        if AUTH_TOKEN and self.headers.get("X-Autoselect-Token") != AUTH_TOKEN:
            self._reply({"error": "Unauthorized"}, 401)
            return False
        return True

    def do_GET(self):
        if not self._auth_ok():
            return
        if self.path == "/health":
            info = self.server.engine.info()
            self._reply({"status": "ok", "service": "krita-autoselect", **info})
        else:
            self._reply({"error": "Unknown endpoint"}, 404)

    def do_POST(self):
        if not self._auth_ok():
            return
        if self.path != "/segment":
            self._reply({"error": "Unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply({"error": "Invalid Content-Length"}, 400)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._reply({"error": "Invalid JSON"}, 400)
            return
        try:
            kwargs = parse_segment_params(payload)
        except BadRequest as e:
            self._reply({"error": str(e)}, 400)
            return
        try:
            result = self.server.engine.segment(**kwargs)
        except Exception as e:
            self._reply({"error": str(e)}, 500)
            return
        self._reply({"status": "ok", **result})


def make_server(engine, port=DEFAULT_PORT, host="127.0.0.1"):
    server = ThreadingHTTPServer((host, port), AutoselectHandler)
    server.engine = engine
    return server


def main():
    from engine import Sam3Engine

    engine = Sam3Engine(
        model_id=os.environ.get("AUTOSELECT_MODEL", "facebook/sam3"),
        weights_path=os.environ.get("AUTOSELECT_WEIGHTS_PATH", ""),
        device=os.environ.get("AUTOSELECT_DEVICE") or None,
        dtype=os.environ.get("AUTOSELECT_DTYPE") or None,
        unload_after_s=float(os.environ.get("AUTOSELECT_UNLOAD_AFTER_S", "300")),
        comfyui_url=os.environ.get("AUTOSELECT_COMFYUI_URL",
                                   "http://127.0.0.1:8001"),
    )
    port = int(os.environ.get("AUTOSELECT_PORT", DEFAULT_PORT))
    server = make_server(engine, port)
    print(f"[autoselect] listening on http://127.0.0.1:{port} "
          f"(model loads on first request)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[autoselect] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
