"""SAM 3 engine for krita-autoselect: lazy model lifecycle + inference.

The model is loaded on first use and unloaded after a configurable idle
period so the GPU stays free for image generation (ComfyUI / Krita AI
Diffusion) the rest of the time. All torch/transformers imports happen
inside methods — importing this module is cheap and dependency-free, which
keeps the HTTP layer testable without ML installed.
"""
import base64
import hashlib
import io
import json
import os
import sys
import threading
import time
import urllib.request
from collections import OrderedDict

# Apple Silicon: ops without an MPS kernel run on the CPU instead of raising.
# PyTorch reads this once, when torch is imported (lazily, below).
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL = "facebook/sam3"
# Model weights (~3.4 GB fp32 / ~1.7 GB bf16) + activations headroom.
VRAM_NEEDED_BYTES = 6 * 1024 ** 3
WATCHDOG_INTERVAL_S = 30.0
# A click becomes a tiny positive "exemplar box" for the PCS model: SAM 3 in
# transformers prompts with text and/or boxes, not raw points.
POINT_BOX_MIN_RADIUS_PX = 4
POINT_BOX_RADIUS_FRACTION = 0.005
# Vision features of the last few images stay on the device, so follow-up
# prompts on the same canvas (refinement clicks, a batch of part prompts,
# retries) skip the backbone. One entry is ~60 MB in bf16 at 1008 px.
DEFAULT_FEATURE_CACHE = 4
# Text prompts per detector forward in segment_many(). Halved on OOM.
DEFAULT_PROMPT_BATCH = 16

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
                 dtype=None, unload_after_s=300.0, comfyui_url="",
                 feature_cache=DEFAULT_FEATURE_CACHE,
                 prompt_batch=DEFAULT_PROMPT_BATCH):
        self.model_id = model_id
        self.weights_path = weights_path or None
        self.device = device          # None = auto (cuda, mps, else cpu)
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
        self._loading = None            # "detector" | "tracker" while loading
        self.feature_cache_size = max(0, int(feature_cache))
        self.prompt_batch = max(1, int(prompt_batch))
        self._features = OrderedDict()  # image key -> vision embeddings
        self._batch_cap = None          # batch size that fit after an OOM
        self.cache_hits = 0
        self.cache_misses = 0

    # ----- lifecycle -----

    def info(self):
        return {
            "model": self.model_id,
            "loaded": self._model is not None,
            "loading": self._loading is not None,
            "tracker_loaded": self._tracker is not None,
            "device": self._resolved_device,
            "unload_after_s": self.unload_after_s,
            "feature_cache": {"size": len(self._features),
                              "max": self.feature_cache_size,
                              "hits": self.cache_hits,
                              "misses": self.cache_misses},
            "prompt_batch": self.prompt_batch,
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

    @staticmethod
    def _free_vram_nvidia_smi():
        """Free VRAM in bytes WITHOUT creating a CUDA context (None if
        nvidia-smi is unavailable). When another process (ComfyUI) holds
        almost all VRAM, even torch's context creation inside
        mem_get_info() fails with OOM — so measure from outside first."""
        import shutil
        import subprocess
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        try:
            # stdin=DEVNULL: our stdin is the parent-watch pipe, with a
            # thread blocked reading it. On Windows that pending synchronous
            # ReadFile locks the pipe's file object, so a child inheriting it
            # stalls in its CRT startup (GetFileType) until the 10 s timeout.
            out = subprocess.run(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=10, creationflags=0x08000000 if os.name == "nt" else 0)
            return int(out.stdout.split()[0]) * 1024 ** 2
        except Exception:
            return None

    def _ensure_vram(self, torch, device):
        if not device.startswith("cuda"):
            return
        free = self._free_vram_nvidia_smi()
        if free is None:
            try:
                free, _total = torch.cuda.mem_get_info()
            except Exception as e:  # no room even for a CUDA context
                print(f"[autoselect] cannot query VRAM ({e})")
                free = 0
        if free < VRAM_NEEDED_BYTES:
            print(f"[autoselect] only {free / 1024 ** 3:.1f} GB VRAM free")
            self._free_comfyui_vram()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _auto_device(torch):
        """cuda, else Apple Silicon's GPU (mps), else cpu. mps keeps the
        float32 default: bf16 needs macOS 14+ and not every SAM 3 op has a
        half-precision MPS kernel; unified memory makes fp32 affordable."""
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _check_torch_supported(torch):
        """transformers silently disables a torch it deems too old, and then
        fails with a misleading "PyTorch not found". Intel Macs always hit
        this: PyTorch's last x86_64 macOS build is 2.2."""
        try:
            from transformers.utils import is_torch_available
        except ImportError:
            return
        if is_torch_available():
            return
        import platform
        msg = (f"The installed PyTorch {torch.__version__} is too old for "
               "this transformers/SAM 3.")
        if sys.platform == "darwin" and platform.machine() == "x86_64":
            msg += (" PyTorch stopped publishing Intel (x86_64) macOS builds "
                    "after 2.2, so SAM 3 cannot run on an Intel Mac (or an "
                    "x86_64 Python under Rosetta): use an Apple Silicon Mac, "
                    "or run this daemon on another machine.")
        else:
            msg += " Upgrade torch in the daemon's environment."
        raise EngineError(msg)

    def _load_locked(self):
        self._loading = "detector"
        try:
            self._load_detector_locked()
        finally:
            self._loading = None

    def _load_detector_locked(self):
        import torch
        self._check_torch_supported(torch)
        from transformers import Sam3Model, Sam3Processor

        device = self.device or self._auto_device(torch)
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
        self._loading = "tracker"
        try:
            self._load_tracker_model_locked()
        finally:
            self._loading = None

    def _load_tracker_model_locked(self):
        """Load the PVS head (Sam3Tracker): click/box → the exact object.

        Same checkpoint as the detector; loaded independently and only when a
        visual-only prompt arrives, so text users never pay for it."""
        import torch
        self._check_torch_supported(torch)
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        device = (self._resolved_device or self.device
                  or self._auto_device(torch))
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
        self._features.clear()
        self._batch_cap = None
        try:
            import torch
            if self._resolved_device and self._resolved_device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif self._resolved_device and self._resolved_device.startswith("mps"):
                # Hand the cached unified memory back to macOS.
                torch.mps.empty_cache()
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

    @staticmethod
    def decode_image(image_b64):
        """base64 PNG/JPEG -> RGB PIL image (EngineError if it is not one)."""
        from PIL import Image
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            return image.convert("RGB")
        except Exception as e:
            raise EngineError(f"Invalid image payload: {e}") from e

    @staticmethod
    def decode_image_any(image_b64):
        """base64 image -> PIL image as is (alpha kept)."""
        from PIL import Image
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            image.load()
            return image
        except Exception as e:
            raise EngineError(f"Invalid image payload: {e}") from e

    @staticmethod
    def image_key(image):
        """Cache key of an RGB image: hash of its pixels (not of the file
        bytes, so a PNG and a JPEG re-encode of other pixels never collide
        and the same canvas re-sent by Krita hits)."""
        h = hashlib.sha1(image.tobytes())
        h.update(repr(image.size).encode())
        return h.hexdigest()

    def segment(self, image_b64, text=None, points=None, point_labels=None,
                box=None, threshold=0.5, mask_threshold=0.5,
                combine="union", list_only=False, within=None):
        """Run SAM 3 and compose the final mask.

        Routing: a text prompt → concept mode (PCS, every matching instance;
        points/box refine the concept). Visual-only prompt → tracker mode
        (PVS, SAM2-style: the exact object under the clicks / in the box).

        `within` = [x, y, w, h]: keep only instances mostly inside that
        region — the "drag a box + verify it's a hand" gesture (text mode).

        Returns {"instances": [{index, score, box}], "width", "height",
        "count", "mask_b64"? } — mask_b64 is a canvas-sized grayscale PNG,
        omitted when list_only or when nothing matched.
        """
        image = self.decode_image(image_b64)
        instances, np_masks = self.segment_arrays(
            image, text=text, points=points, point_labels=point_labels,
            box=box, threshold=threshold, mask_threshold=mask_threshold,
            within=within)
        return self._compose(image.size, instances, np_masks,
                             combine, list_only)

    def segment_arrays(self, image, text=None, points=None, point_labels=None,
                       box=None, threshold=0.5, mask_threshold=0.5,
                       within=None, key=None):
        """One prompt on a PIL image -> (instances, [bool HxW arrays])."""
        if text:
            instances, np_masks = self._segment_concept(
                image, text, points, point_labels, box,
                threshold, mask_threshold, key=key)
            if within:
                instances, np_masks = self._filter_within(
                    instances, np_masks, within)
            return instances, np_masks
        return self._segment_visual(image, points, point_labels, box or within)

    @staticmethod
    def _filter_within(instances, np_masks, within, min_overlap=0.5):
        """Keep instances whose bbox lies mostly inside `within` [x,y,w,h]."""
        wx1, wy1 = float(within[0]), float(within[1])
        wx2, wy2 = wx1 + float(within[2]), wy1 + float(within[3])
        kept_instances, kept_masks = [], []
        for inst, mask in zip(instances, np_masks):
            x1, y1, x2, y2 = inst["box"]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            ix1, iy1 = max(x1, wx1), max(y1, wy1)
            ix2, iy2 = min(x2, wx2), min(y2, wy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area >= min_overlap:
                inst = dict(inst, index=len(kept_instances))
                kept_instances.append(inst)
                kept_masks.append(mask)
        return kept_instances, kept_masks

    # ----- vision feature cache (lock held) -----

    def _ensure_detector_locked(self):
        if self._model is None:
            self._load_locked()
        self._last_used = time.time()

    def _vision_locked(self, image, key=None):
        """Vision embeddings of `image` (batch of 1), from the LRU cache or
        computed now. Returns (embeds, cache_hit)."""
        import torch
        key = key or self.image_key(image)
        cached = self._features.get(key)
        if cached is not None:
            self._features.move_to_end(key)
            self.cache_hits += 1
            return cached, True
        self.cache_misses += 1
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._model.device)
        pixel_values = pixel_values.to(self._dtype)
        with torch.no_grad():
            embeds = self._model.get_vision_features(pixel_values=pixel_values)
        if self.feature_cache_size:
            self._features[key] = embeds
            while len(self._features) > self.feature_cache_size:
                self._features.popitem(last=False)
        return embeds, False

    @staticmethod
    def _expand_vision(embeds, n):
        """The same image features as a batch of n (views, no copy)."""
        if n == 1:
            return embeds
        from transformers.models.sam3.modeling_sam3 import \
            Sam3VisionEncoderOutput

        def rep(ts):
            return tuple(t.expand(n, *t.shape[1:]) for t in ts)

        return Sam3VisionEncoderOutput(
            last_hidden_state=embeds.last_hidden_state.expand(
                n, *embeds.last_hidden_state.shape[1:]),
            fpn_hidden_states=rep(embeds.fpn_hidden_states),
            fpn_position_encoding=rep(embeds.fpn_position_encoding))

    @staticmethod
    def crop_mask(mask):
        """Full-size bool mask -> (crop, [x, y]) of its tight bbox. A batch
        of 60 prompts x several instances x a 7 MP canvas would otherwise
        hold gigabytes of mostly-empty masks."""
        import numpy as np
        rows = np.flatnonzero(mask.any(1))
        if not len(rows):
            return mask[:0, :0], [0, 0]
        cols = np.flatnonzero(mask.any(0))
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return mask[y0:y1, x0:x1].copy(), [int(x0), int(y0)]

    def _results_to_instances(self, results, crop=False):
        import numpy as np
        scores = results["scores"]
        order = sorted(range(len(scores)),
                       key=lambda i: float(scores[i]), reverse=True)
        instances, np_masks = [], []
        for rank, i in enumerate(order):
            inst = {
                "index": rank,
                "score": round(float(scores[i]), 4),
                "box": [round(float(v), 1) for v in results["boxes"][i]],
            }
            mask = np.asarray(results["masks"][i].cpu()).astype(bool)
            if crop:
                mask, inst["mask_xy"] = self.crop_mask(mask)
            instances.append(inst)
            np_masks.append(mask)
        return instances, np_masks

    def _segment_concept(self, image, text, points, point_labels, box,
                         threshold, mask_threshold, key=None):
        """PCS: all instances of a concept (text + optional exemplar refiners)."""
        import torch

        width, height = image.size
        with self._lock:
            self._ensure_detector_locked()
            model, processor = self._model, self._processor
            t0 = time.time()
            vision, hit = self._vision_locked(image, key)

            boxes, labels = self._prompt_boxes(
                width, height, points, point_labels, box)
            kwargs = {"text": text, "original_sizes": [[height, width]]}
            if boxes:
                kwargs["input_boxes"] = [boxes]
                kwargs["input_boxes_labels"] = [labels]
            inputs = processor(return_tensors="pt", **kwargs)
            inputs.pop("original_sizes", None)
            inputs = inputs.to(model.device)
            for k, value in inputs.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[k] = value.to(self._dtype)

            with torch.no_grad():
                outputs = model(vision_embeds=vision, **inputs)
            results = processor.post_process_instance_segmentation(
                outputs, threshold=float(threshold),
                mask_threshold=float(mask_threshold),
                target_sizes=[(height, width)],
            )[0]
            self._last_used = time.time()

        instances, np_masks = self._results_to_instances(results)
        print(f"[autoselect] concept: {len(instances)} instance(s) "
              f"in {time.time() - t0:.2f}s"
              f"{' (cached features)' if hit else ''}")
        return instances, np_masks

    def segment_many(self, image, prompts, mask_threshold=0.5, key=None,
                     crop=False):
        """Many prompts on ONE image in as few forwards as possible.

        The backbone runs once (or not at all, if the image is cached); all
        text-only prompts then go through the detector in batches of
        `prompt_batch` (halved on out-of-memory). Prompts with a box or
        points run one by one on the same cached features; visual-only
        prompts (no text) use the tracker.

        prompts: [{"text"?, "box"?, "points"?, "point_labels"?,
                   "threshold"?=0.5, "within"?}]
        crop: masks come cropped to their bbox, offset in inst["mask_xy"].
        Returns ([(instances, [bool masks])] in prompt order, timing dict).
        """
        import torch

        width, height = image.size
        timing = {"prompts": len(prompts)}
        results = [None] * len(prompts)
        plain = [i for i, p in enumerate(prompts)
                 if p.get("text") and not p.get("box") and not p.get("points")]
        t_start = time.time()
        if plain:
            with self._lock:
                self._ensure_detector_locked()
                t0 = time.time()
                vision, hit = self._vision_locked(image, key)
                if self._model.device.type == "cuda":
                    torch.cuda.synchronize()
                timing["vision_s"] = round(time.time() - t0, 3)
                timing["vision_cached"] = hit
                t0 = time.time()
                batch = min(self.prompt_batch, self._batch_cap or 1 << 30)
                pos = 0
                while pos < len(plain):
                    chunk = plain[pos:pos + batch]
                    try:
                        outs = self._detect_batch_locked(
                            vision, [prompts[i] for i in chunk],
                            width, height, mask_threshold, crop)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if batch == 1:
                            raise
                        batch = max(1, batch // 2)
                        self._batch_cap = batch
                        print(f"[autoselect] OOM: prompt batch -> {batch}")
                        continue
                    for i, out in zip(chunk, outs):
                        results[i] = out
                    pos += len(chunk)
                self._last_used = time.time()
                timing["detector_s"] = round(time.time() - t0, 3)
                timing["batch"] = batch
        t0 = time.time()
        for i, p in enumerate(prompts):
            if results[i] is not None:
                continue
            inst, masks = self.segment_arrays(
                image, text=p.get("text"), points=p.get("points"),
                point_labels=p.get("point_labels"), box=p.get("box"),
                threshold=float(p.get("threshold", 0.5)),
                mask_threshold=mask_threshold, within=p.get("within"),
                key=key)
            if crop:
                cropped = [self.crop_mask(m) for m in masks]
                for d, (_, xy) in zip(inst, cropped):
                    d["mask_xy"] = xy
                masks = [c for c, _ in cropped]
            results[i] = (inst, masks)
        if len(plain) < len(prompts):
            timing["single_s"] = round(time.time() - t0, 3)
        for i, p in enumerate(prompts):
            if p.get("within") and i in plain:
                results[i] = self._filter_within(*results[i], p["within"])
        timing["total_s"] = round(time.time() - t_start, 3)
        print(f"[autoselect] segment_many: {len(prompts)} prompt(s) in "
              f"{timing['total_s']:.2f}s {timing}")
        return results, timing

    def _detect_batch_locked(self, vision, prompts, width, height,
                             mask_threshold, crop=False):
        """One detector forward for several text prompts on one image."""
        import torch
        processor, model = self._processor, self._model
        n = len(prompts)
        text_inputs = processor(text=[p["text"] for p in prompts],
                                return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(vision_embeds=self._expand_vision(vision, n),
                            input_ids=text_inputs["input_ids"],
                            attention_mask=text_inputs["attention_mask"])
        min_thr = min(float(p.get("threshold", 0.5)) for p in prompts)
        per_prompt = processor.post_process_instance_segmentation(
            outputs, threshold=min_thr, mask_threshold=float(mask_threshold),
            target_sizes=[(height, width)] * n)
        out = []
        for p, res in zip(prompts, per_prompt):
            thr = float(p.get("threshold", 0.5))
            if thr > min_thr:
                keep = [j for j, s in enumerate(res["scores"])
                        if float(s) >= thr]
                res = {k: res[k][keep] for k in ("scores", "boxes", "masks")}
            out.append(self._results_to_instances(res, crop))
        return out

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
