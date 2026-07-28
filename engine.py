"""SAM 3 engine for krita-autoselect: lazy model lifecycle + inference.

The model is loaded on first use and unloaded after a configurable idle
period so the GPU stays free for image generation (ComfyUI / Krita AI
Diffusion) the rest of the time. All torch/transformers imports happen
inside methods — importing this module is cheap and dependency-free, which
keeps the HTTP layer testable without ML installed.
"""
import base64
import io
import json
import threading
import time
import urllib.request

DEFAULT_MODEL = "facebook/sam3"
# Model weights (~3.4 GB fp32 / ~1.7 GB bf16) + activations headroom.
VRAM_NEEDED_BYTES = 6 * 1024 ** 3
WATCHDOG_INTERVAL_S = 30.0
# A click becomes a tiny positive "exemplar box" for the PCS model: SAM 3 in
# transformers prompts with text and/or boxes, not raw points.
POINT_BOX_MIN_RADIUS_PX = 4
POINT_BOX_RADIUS_FRACTION = 0.005

GATED_WEIGHTS_HELP = (
    "Could not download SAM 3 weights. The checkpoint is gated on Hugging "
    "Face: (1) accept the license at https://huggingface.co/facebook/sam3 "
    "with your HF account, (2) authenticate on this machine (set HF_TOKEN "
    "or run `hf auth login`), and (3) if you use a fine-grained token, "
    "enable 'Access public gated repositories' in its settings at "
    "https://huggingface.co/settings/tokens. Alternatively set "
    "AUTOSELECT_WEIGHTS_PATH to a local directory with the downloaded model."
)


class EngineError(RuntimeError):
    """Error whose message is safe to return to the HTTP client."""


class Sam3Engine:
    def __init__(self, model_id=DEFAULT_MODEL, weights_path=None, device=None,
                 dtype=None, unload_after_s=300.0, comfyui_url=""):
        self.model_id = model_id
        self.weights_path = weights_path or None
        self.device = device          # None = auto (cuda if available)
        self.dtype_name = dtype       # None = auto (bfloat16 on cuda)
        self.unload_after_s = float(unload_after_s)
        self.comfyui_url = comfyui_url.rstrip("/") if comfyui_url else ""
        self._lock = threading.Lock()  # serializes load/unload/inference
        self._model = None
        self._processor = None
        self._resolved_device = None
        self._last_used = 0.0
        self._watchdog_started = False

    # ----- lifecycle -----

    def info(self):
        return {
            "model": self.model_id,
            "loaded": self._model is not None,
            "device": self._resolved_device,
            "unload_after_s": self.unload_after_s,
        }

    def _free_comfyui_vram(self):
        """Best-effort: ask ComfyUI to drop its cached models before we load."""
        if not self.comfyui_url:
            return
        try:
            body = json.dumps({"unload_models": True, "free_memory": True}).encode()
            req = urllib.request.Request(
                self.comfyui_url + "/free", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
            print("[autoselect] asked ComfyUI to free VRAM")
            time.sleep(2.0)  # give it a moment to actually release
        except Exception as e:
            print(f"[autoselect] ComfyUI free failed (continuing): {e}")

    def _ensure_vram(self, torch, device):
        if not device.startswith("cuda"):
            return
        free, _total = torch.cuda.mem_get_info()
        if free < VRAM_NEEDED_BYTES:
            self._free_comfyui_vram()
            torch.cuda.empty_cache()

    def _load_locked(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype_name:
            dtype = getattr(torch, self.dtype_name)
        else:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        self._ensure_vram(torch, device)
        source = self.weights_path or self.model_id
        t0 = time.time()
        try:
            processor = Sam3Processor.from_pretrained(source)
            model = Sam3Model.from_pretrained(source, dtype=dtype)
        except Exception as e:
            msg = str(e)
            if any(k in msg.lower() for k in ("gated", "401", "403", "token",
                                              "authorized", "authentication")):
                raise EngineError(GATED_WEIGHTS_HELP) from e
            error = f"Could not load SAM 3 from '{source}': {e}"
            if not self.weights_path:
                # transformers often swallows the underlying 401/403 into a
                # generic message — for hub loads, always include the fix.
                error += f" | Hint: {GATED_WEIGHTS_HELP}"
            raise EngineError(error) from e
        model = model.to(device).eval()
        self._model, self._processor = model, processor
        self._resolved_device = device
        self._dtype = dtype
        print(f"[autoselect] SAM 3 loaded on {device} ({dtype}) "
              f"in {time.time() - t0:.1f}s")
        if not self._watchdog_started:
            self._watchdog_started = True
            threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def unload(self):
        with self._lock:
            self._unload_locked()

    def _unload_locked(self):
        if self._model is None:
            return
        self._model = None
        self._processor = None
        try:
            import torch
            if self._resolved_device and self._resolved_device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[autoselect] SAM 3 unloaded (idle)")

    def _watchdog_loop(self):
        while True:
            time.sleep(WATCHDOG_INTERVAL_S)
            if self.unload_after_s <= 0:
                continue
            with self._lock:
                idle = time.time() - self._last_used
                if self._model is not None and idle > self.unload_after_s:
                    self._unload_locked()

    # ----- inference -----

    def _prompt_boxes(self, width, height, points, point_labels, box):
        """Merge click points (as tiny exemplar boxes) and an explicit box
        into (boxes_xyxy, labels). Returns (None, None) if empty."""
        boxes, labels = [], []
        if box:
            x, y, w, h = [float(v) for v in box]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
        if points:
            r = max(POINT_BOX_MIN_RADIUS_PX,
                    round(POINT_BOX_RADIUS_FRACTION * max(width, height)))
            plabels = point_labels or [1] * len(points)
            if len(plabels) != len(points):
                raise EngineError("point_labels length must match points")
            for (px, py), lab in zip(points, plabels):
                boxes.append([float(px) - r, float(py) - r,
                              float(px) + r, float(py) + r])
                labels.append(1 if int(lab) else 0)
        return (boxes, labels) if boxes else (None, None)

    def segment(self, image_b64, text=None, points=None, point_labels=None,
                box=None, threshold=0.5, mask_threshold=0.5,
                combine="union", list_only=False):
        """Run SAM 3 and compose the final mask.

        Returns {"instances": [{index, score, box}], "width", "height",
        "count", "mask_b64"? } — mask_b64 is a canvas-sized grayscale PNG,
        omitted when list_only or when nothing matched.
        """
        import numpy as np
        import torch
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            image = image.convert("RGB")
        except Exception as e:
            raise EngineError(f"Invalid image payload: {e}") from e
        width, height = image.size

        with self._lock:
            if self._model is None:
                self._load_locked()
            self._last_used = time.time()
            model, processor = self._model, self._processor

            boxes, labels = self._prompt_boxes(
                width, height, points, point_labels, box)
            kwargs = {}
            if text:
                kwargs["text"] = text
            if boxes:
                kwargs["input_boxes"] = [boxes]
                kwargs["input_boxes_labels"] = [labels]
            inputs = processor(images=image, return_tensors="pt", **kwargs)
            inputs = inputs.to(model.device)
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(self._dtype)

            t0 = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs, threshold=float(threshold),
                mask_threshold=float(mask_threshold),
                target_sizes=[(height, width)],
            )[0]
            self._last_used = time.time()

        masks = results["masks"]
        scores = results["scores"]
        bxs = results["boxes"]
        order = sorted(range(len(scores)),
                       key=lambda i: float(scores[i]), reverse=True)
        instances = []
        np_masks = []
        for rank, i in enumerate(order):
            instances.append({
                "index": rank,
                "score": round(float(scores[i]), 4),
                "box": [round(float(v), 1) for v in bxs[i]],
            })
            np_masks.append(np.asarray(masks[i].cpu()).astype(bool))
        print(f"[autoselect] segment: {len(instances)} instance(s) "
              f"in {time.time() - t0:.2f}s")

        out = {"width": width, "height": height,
               "count": len(instances), "instances": instances}
        if list_only or not instances:
            return out

        if combine == "union":
            chosen = np_masks
        else:
            idx = int(combine)
            if idx < 0 or idx >= len(np_masks):
                raise EngineError(
                    f"instance {idx} out of range (have {len(np_masks)})")
            chosen = [np_masks[idx]]
        final = np.zeros((height, width), dtype=bool)
        for m in chosen:
            final |= m
        png = io.BytesIO()
        Image.fromarray(final.astype(np.uint8) * 255, mode="L").save(png, "PNG")
        out["mask_b64"] = base64.b64encode(png.getvalue()).decode("ascii")
        return out
