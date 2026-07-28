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
        self._tracker = None            # PVS head (click-to-object, SAM2-style)
        self._tracker_processor = None
        self._resolved_device = None
        self._last_used = 0.0
        self._watchdog_started = False

    # ----- lifecycle -----

    def info(self):
        return {
            "model": self.model_id,
            "loaded": self._model is not None,
            "tracker_loaded": self._tracker is not None,
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

    def _load_tracker_locked(self):
        """Load the PVS head (Sam3Tracker): click/box → the exact object.

        Same checkpoint as the detector; loaded independently and only when a
        visual-only prompt arrives, so text users never pay for it."""
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        device = self._resolved_device or self.device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype_name:
            dtype = getattr(torch, self.dtype_name)
        else:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        self._ensure_vram(torch, device)
        source = self.weights_path or self.model_id
        t0 = time.time()
        try:
            processor = Sam3TrackerProcessor.from_pretrained(source)
            model = Sam3TrackerModel.from_pretrained(source, dtype=dtype)
        except Exception as e:
            error = f"Could not load SAM 3 tracker from '{source}': {e}"
            if not self.weights_path:
                error += f" | Hint: {GATED_WEIGHTS_HELP}"
            raise EngineError(error) from e
        model = model.to(device).eval()
        self._tracker, self._tracker_processor = model, processor
        self._resolved_device = device
        self._dtype = dtype
        print(f"[autoselect] SAM 3 tracker loaded on {device} ({dtype}) "
              f"in {time.time() - t0:.1f}s")
        if not self._watchdog_started:
            self._watchdog_started = True
            threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def unload(self):
        with self._lock:
            self._unload_locked()

    def _unload_locked(self):
        if self._model is None and self._tracker is None:
            return
        self._model = None
        self._processor = None
        self._tracker = None
        self._tracker_processor = None
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
                loaded = self._model is not None or self._tracker is not None
                if loaded and idle > self.unload_after_s:
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

        Routing: a text prompt → concept mode (PCS, every matching instance;
        points/box refine the concept). Visual-only prompt → tracker mode
        (PVS, SAM2-style: the exact object under the clicks / in the box).

        Returns {"instances": [{index, score, box}], "width", "height",
        "count", "mask_b64"? } — mask_b64 is a canvas-sized grayscale PNG,
        omitted when list_only or when nothing matched.
        """
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            image = image.convert("RGB")
        except Exception as e:
            raise EngineError(f"Invalid image payload: {e}") from e

        if text:
            instances, np_masks = self._segment_concept(
                image, text, points, point_labels, box,
                threshold, mask_threshold)
        else:
            instances, np_masks = self._segment_visual(
                image, points, point_labels, box)
        return self._compose(image.size, instances, np_masks,
                             combine, list_only)

    def _segment_concept(self, image, text, points, point_labels, box,
                         threshold, mask_threshold):
        """PCS: all instances of a concept (text + optional exemplar refiners)."""
        import numpy as np
        import torch

        width, height = image.size
        with self._lock:
            if self._model is None:
                self._load_locked()
            self._last_used = time.time()
            model, processor = self._model, self._processor

            boxes, labels = self._prompt_boxes(
                width, height, points, point_labels, box)
            kwargs = {"text": text}
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

        scores = results["scores"]
        order = sorted(range(len(scores)),
                       key=lambda i: float(scores[i]), reverse=True)
        instances, np_masks = [], []
        for rank, i in enumerate(order):
            instances.append({
                "index": rank,
                "score": round(float(scores[i]), 4),
                "box": [round(float(v), 1) for v in results["boxes"][i]],
            })
            np_masks.append(np.asarray(results["masks"][i].cpu()).astype(bool))
        print(f"[autoselect] concept: {len(instances)} instance(s) "
              f"in {time.time() - t0:.2f}s")
        return instances, np_masks

    def _segment_visual(self, image, points, point_labels, box):
        """PVS (tracker): the one object under the clicks / in the box."""
        import numpy as np
        import torch

        with self._lock:
            if self._tracker is None:
                self._load_tracker_locked()
            self._last_used = time.time()
            model, processor = self._tracker, self._tracker_processor

            kwargs = {}
            n_points = len(points) if points else 0
            if points:
                kwargs["input_points"] = [[[list(p) for p in points]]]
                kwargs["input_labels"] = [[[
                    1 if int(l) else 0
                    for l in (point_labels or [1] * len(points))]]]
            if box:
                x, y, w, h = [float(v) for v in box]
                kwargs["input_boxes"] = [[[x, y, x + w, y + h]]]
            inputs = processor(images=image, return_tensors="pt", **kwargs)
            inputs = inputs.to(model.device)
            if torch.is_tensor(inputs.get("pixel_values")):
                inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

            t0 = time.time()
            with torch.no_grad():
                # Con un solo punto conviene el multimask (3 candidatas, gana
                # la de mejor IoU); con refinamiento multi-punto la API
                # recomienda una sola máscara.
                if n_points > 1:
                    outputs = model(**inputs, multimask_output=False)
                else:
                    outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks.float().cpu(), inputs["original_sizes"])[0]
            self._last_used = time.time()

        # (num_objects=1, num_masks, H, W) ranked by predicted IoU — keep best.
        ious = outputs.iou_scores.float().cpu().numpy().reshape(-1)
        best = int(ious.argmax())
        mask = np.asarray(masks[0][best]).astype(bool)
        ys, xs = np.where(mask)
        if not len(xs):
            print("[autoselect] visual: empty mask")
            return [], []
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() + 1), float(ys.max() + 1)]
        print(f"[autoselect] visual: 1 object (iou {ious[best]:.2f}) "
              f"in {time.time() - t0:.2f}s")
        return [{"index": 0, "score": round(float(ious[best]), 4),
                 "box": bbox}], [mask]

    def _compose(self, size, instances, np_masks, combine, list_only):
        import numpy as np
        from PIL import Image

        width, height = size
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
