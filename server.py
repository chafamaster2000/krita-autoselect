#!/usr/bin/env python3
"""krita-autoselect — local SAM 3 segmentation daemon.

Segments any PNG/JPEG by text prompt ("the red car"), click points and/or a
bounding box, and returns a canvas-sized grayscale mask. Built for `kri
select sam` (Krita selections) but Krita-independent: POST an image, get a
mask.

HTTP on 127.0.0.1 only, stdlib server:
  GET  /health        — daemon + model state, plus "capabilities" and
                        "api_version" so clients detect /segment_many
  POST /segment       — {image_b64, text?, points?, point_labels?, box?,
                         threshold?, mask_threshold?, combine?, list_only?}
  POST /segment_many  — {image_b64, prompts: [text | {text?, box?, points?,
                         point_labels?, threshold?, combine?, within?}],
                         masks?: "instances"|"union"|"none",
                         mask_threshold?}: one image, many prompts; the
                         backbone runs once and the text-only prompts go
                         through the detector in batches

Box convention (kept for compatibility): boxes IN (box, within) are
[x, y, width, height]; boxes OUT (instances[].box) are [x1, y1, x2, y2].

Env:
  AUTOSELECT_PORT            port (default 5679)
  AUTOSELECT_MODEL           HF model id (default facebook/sam3)
  AUTOSELECT_WEIGHTS_PATH    local model dir (skips Hugging Face entirely)
  AUTOSELECT_DEVICE          cuda | mps | cpu (default: auto)
  AUTOSELECT_DTYPE           bfloat16 | float16 | float32 (default: auto)
  AUTOSELECT_UNLOAD_AFTER_S  idle seconds before freeing VRAM; 0 = resident
                             (default 300)
  AUTOSELECT_COMFYUI_URL     ComfyUI to ask for VRAM before loading
                             (default http://127.0.0.1:8001; empty disables)
  AUTOSELECT_TOKEN           optional shared token (X-Autoselect-Token)
  AUTOSELECT_FEATURE_CACHE   images whose vision features stay cached (4)
  AUTOSELECT_PROMPT_BATCH    text prompts per detector forward (16)
  AUTOSELECT_PARENT_WATCH    "stdin": exit when stdin reaches EOF — set by a
                             parent (the Krita plugin) that keeps the pipe
                             open, so the daemon dies with it even on a crash
  HF_TOKEN                   standard Hugging Face auth for gated weights
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 5679
IS_WINDOWS = os.name == "nt"
MAX_BODY_BYTES = 96 * 1024 * 1024
AUTH_TOKEN = os.environ.get("AUTOSELECT_TOKEN", "")

VALID_PROMPT_KEYS = ("text", "points", "box")
# Clients (kri, autopainter) feature-detect with these, not with a 404.
API_VERSION = 2
CAPABILITIES = ("segment", "segment_many")
MASK_MODES = ("instances", "union", "none")
MAX_PROMPTS = 256


class BadRequest(ValueError):
    """Client error — message is returned with HTTP 400."""


def parse_segment_params(payload):
    """Validate /segment JSON into kwargs for engine.segment()."""
    if not isinstance(payload, dict):
        raise BadRequest("Body must be a JSON object")
    if not payload.get("image_b64"):
        raise BadRequest("image_b64 is required")
    if not any(payload.get(k) for k in VALID_PROMPT_KEYS + ("within",)):
        raise BadRequest("At least one prompt is required: text, points or box")

    points = payload.get("points")
    if points is not None:
        if (not isinstance(points, list)
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise BadRequest("points must be a list of [x, y] pairs")
    box = payload.get("box")
    if box is not None and (not isinstance(box, list) or len(box) != 4):
        raise BadRequest("box must be [x, y, width, height]")
    within = payload.get("within")
    if within is not None and (not isinstance(within, list) or len(within) != 4):
        raise BadRequest("within must be [x, y, width, height]")

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
        "within": within,
    }


def parse_segment_many_params(payload):
    """Validate /segment_many JSON. Returns (image_b64, prompts, masks,
    mask_threshold); each prompt is a dict of engine.segment() kwargs
    without image_b64/list_only/mask_threshold."""
    if not isinstance(payload, dict):
        raise BadRequest("Body must be a JSON object")
    if not payload.get("image_b64"):
        raise BadRequest("image_b64 is required")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise BadRequest("prompts must be a non-empty list")
    if len(prompts) > MAX_PROMPTS:
        raise BadRequest(f"at most {MAX_PROMPTS} prompts per request")
    out = []
    for i, p in enumerate(prompts):
        if isinstance(p, str):
            p = {"text": p}
        if not isinstance(p, dict):
            raise BadRequest(f"prompts[{i}] must be text or an object")
        try:
            kw = parse_segment_params(dict(p, image_b64="-"))
        except BadRequest as e:
            raise BadRequest(f"prompts[{i}]: {e}")
        for k in ("image_b64", "list_only", "mask_threshold"):
            kw.pop(k)
        out.append(kw)
    masks = payload.get("masks", "instances")
    if masks not in MASK_MODES:
        raise BadRequest(f"masks must be one of {list(MASK_MODES)}")
    try:
        mask_threshold = float(payload.get("mask_threshold", 0.5))
    except (TypeError, ValueError):
        raise BadRequest("mask_threshold must be a number")
    return payload["image_b64"], out, masks, mask_threshold


def _mask_png_b64(mask):
    import base64
    import io
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
        buf, "PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_segment_many(engine, image_b64, prompts, masks, mask_threshold):
    """Engine call + encoding.

    instances: one PNG per instance, cropped to its bbox, offset in mask_xy.
    union: one canvas-sized PNG per prompt, honouring its `combine`.
    none: only counts, scores and boxes.
    """
    import numpy as np
    image = engine.decode_image(image_b64)
    width, height = image.size
    results, timing = engine.segment_many(
        image, prompts, mask_threshold=mask_threshold,
        crop=(masks == "instances"))
    out = []
    for p, (instances, np_masks) in zip(prompts, results):
        item = {"text": p.get("text"), "count": len(instances),
                "instances": [dict(i) for i in instances]}
        if masks == "instances":
            for inst, m in zip(item["instances"], np_masks):
                if m.size:
                    inst["mask_b64"] = _mask_png_b64(m)
        else:
            for inst in item["instances"]:
                inst.pop("mask_xy", None)
            if masks == "union" and instances:
                combine = p.get("combine", "union")
                if combine == "union":
                    chosen = np_masks
                elif 0 <= combine < len(np_masks):
                    chosen = [np_masks[combine]]
                else:
                    chosen = []
                if chosen:
                    final = np.zeros((height, width), bool)
                    for m in chosen:
                        final |= m
                    item["mask_b64"] = _mask_png_b64(final)
        out.append(item)
    return {"width": width, "height": height, "results": out,
            "timing": timing}


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
            self._reply({"status": "ok", "service": "krita-autoselect",
                         "api_version": API_VERSION,
                         "capabilities": list(CAPABILITIES), **info})
        else:
            self._reply({"error": "Unknown endpoint"}, 404)

    def do_POST(self):
        if not self._auth_ok():
            return
        if self.path not in ("/segment", "/segment_many"):
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
            if self.path == "/segment_many":
                args = parse_segment_many_params(payload)
                result = run_segment_many(self.server.engine, *args)
            else:
                kwargs = parse_segment_params(payload)
                result = self.server.engine.segment(**kwargs)
        except BadRequest as e:
            self._reply({"error": str(e)}, 400)
            return
        except Exception as e:
            print(f"[autoselect] segment failed: {type(e).__name__}: {e}",
                  flush=True)
            self._reply({"error": str(e)}, 500)
            return
        self._reply({"status": "ok", **result})


class _Server(ThreadingHTTPServer):
    def server_bind(self):
        # http.server sets SO_REUSEADDR. On Windows that option lets a second
        # socket bind a port that is still LISTENING, so a second daemon (an
        # orphan, run_daemon.ps1, another Krita) would silently share 5679
        # instead of failing with "address in use". Windows doesn't need it
        # for TIME_WAIT either (asyncio, tornado and twisted skip it there).
        self.allow_reuse_address = not IS_WINDOWS
        super().server_bind()


def make_server(engine, port=DEFAULT_PORT, host="127.0.0.1"):
    server = _Server((host, port), AutoselectHandler)
    server.engine = engine
    return server


def _exit_on_stdin_eof():
    """Parent watch: the parent holds our stdin open and never writes. When
    it exits (cleanly or not) the OS closes the pipe and we get EOF."""
    try:
        while os.read(sys.stdin.fileno(), 4096):
            pass
    except OSError:
        pass
    print("[autoselect] parent closed stdin — exiting", flush=True)
    os._exit(0)


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
        feature_cache=int(os.environ.get("AUTOSELECT_FEATURE_CACHE", "4")),
        prompt_batch=int(os.environ.get("AUTOSELECT_PROMPT_BATCH", "16")),
    )
    port = int(os.environ.get("AUTOSELECT_PORT", DEFAULT_PORT))
    if os.environ.get("AUTOSELECT_PARENT_WATCH") == "stdin":
        threading.Thread(target=_exit_on_stdin_eof, daemon=True).start()
    try:
        server = make_server(engine, port)
    except OSError as e:
        print(f"[autoselect] error: cannot listen on 127.0.0.1:{port}: {e} "
              f"(address already in use? set AUTOSELECT_PORT)", flush=True)
        sys.exit(3)
    print(f"[autoselect] listening on http://127.0.0.1:{port} "
          f"(model loads on first request)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[autoselect] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
