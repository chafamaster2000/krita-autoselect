"""SAM 3 access for the pipeline + the rules that pick which instances count.

Two segmenters with the same interface, `many(image, prompts)` ->
([ [Instance, ...] per prompt ], timing):

- EngineSegmenter: the engine in this process (inside the daemon for
  /flats, or the CLI with --local).
- HttpSegmenter: a running daemon's POST /segment_many (keeps the model warm
  across CLI runs and shares the GPU with Krita's AI Select).
"""
import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


class Instance:
    """One SAM instance: score, box [x1,y1,x2,y2], mask crop at (x, y)."""

    __slots__ = ("index", "score", "box", "mask", "xy")

    def __init__(self, index, score, box, mask, xy):
        self.index, self.score, self.box = int(index), float(score), box
        self.mask, self.xy = mask, (int(xy[0]), int(xy[1]))

    @property
    def area(self):
        return int(self.mask.sum())

    def paste(self, canvas):
        """OR this instance into a full-size bool canvas."""
        x, y = self.xy
        h, w = self.mask.shape
        if h and w:
            canvas[y:y + h, x:x + w] |= self.mask


class SegmentError(RuntimeError):
    pass


class EngineSegmenter:
    def __init__(self, engine):
        self.engine = engine

    def many(self, image, prompts):
        results, timing = self.engine.segment_many(image, prompts, crop=True)
        out = []
        for inst, masks in results:
            out.append([Instance(d["index"], d["score"], d["box"], m,
                                 d.get("mask_xy", (0, 0)))
                        for d, m in zip(inst, masks)])
        return out, timing


def decode_mask_png(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.asarray(img.convert("L")) > 127


def encode_mask_png(mask):
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
        buf, "PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class HttpSegmenter:
    def __init__(self, url, token="", timeout=900.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, path, payload):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Autoselect-Token"] = self.token
        req = urllib.request.Request(self.url + path,
                                     data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                body = json.load(e)
            except Exception:
                body = {"error": f"HTTP {e.code}"}
            raise SegmentError(body.get("error") or f"HTTP {e.code}")
        except (urllib.error.URLError, OSError) as e:
            raise SegmentError(f"cannot reach {self.url}: {e}")

    def health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return json.load(r)
        except Exception:
            return None

    def many(self, image, prompts):
        buf = io.BytesIO()
        image.save(buf, "PNG", compress_level=1)
        t0 = time.time()
        res = self._post("/segment_many", {
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "prompts": prompts, "masks": "instances"})
        out = []
        for r in res["results"]:
            insts = []
            for d in r.get("instances", []):
                mask = decode_mask_png(d["mask_b64"]) if d.get("mask_b64") \
                    else np.zeros((0, 0), bool)
                insts.append(Instance(d["index"], d["score"], d["box"], mask,
                                      d.get("mask_xy", (0, 0))))
            out.append(insts)
        timing = dict(res.get("timing") or {})
        timing["http_s"] = round(time.time() - t0, 3)
        return out, timing


# ----- selection rules (pure) -----

def select_instances(instances, part, prompt, sil_area):
    """Apply a part's rules to one prompt's instances (sorted by score).

    Returns (kept, decisions) — decisions: [{index, score, area, used,
    reason?}] for the report. Rules, in order: `exclude` indices,
    min_score, max_area (fraction of the silhouette), `relative` to the
    best remaining score, then `instances` (keep the top N).
    """
    min_score = max(part["min_score"], float(prompt.get("min_score", 0)))
    relative = float(prompt.get("relative", part["relative"]))
    keep_n = prompt.get("instances", part["instances"])
    if keep_n == "best":
        keep_n = 1
    max_area = prompt.get("max_area", part["max_area"])
    exclude = set(prompt.get("exclude") or [])

    decisions, alive = [], []
    for inst in instances:
        d = {"index": inst.index, "score": round(inst.score, 3),
             "area": inst.area, "box": [round(v) for v in inst.box]}
        if inst.index in exclude:
            d["reason"] = "excluded by config"
        elif inst.score < min_score:
            d["reason"] = f"score < {min_score:g}"
        elif max_area and sil_area and inst.area > max_area * sil_area:
            d["reason"] = f"area > {max_area:g} of the silhouette"
        else:
            alive.append((inst, d))
        decisions.append(d)
    if alive and relative > 0:
        best = max(i.score for i, _ in alive)
        floor = best * relative
        kept = []
        for inst, d in alive:
            if inst.score < floor:
                d["reason"] = f"score < {relative:g} x best ({best:.2f})"
            else:
                kept.append((inst, d))
        alive = kept
    if keep_n != "all" and len(alive) > keep_n:
        for inst, d in alive[keep_n:]:
            d["reason"] = f"only the best {keep_n} kept"
        alive = alive[:keep_n]
    for _, d in alive:
        d["used"] = True
    for d in decisions:
        d.setdefault("used", False)
    return [i for i, _ in alive], decisions


def request_list(cfg, parts_prompts=None, silhouette=None):
    """SAM requests for a config: one per distinct text (threshold = the
    lowest any user of it accepts), plus one per box prompt. Returns
    (requests, key_of) where key_of(prompt_dict) -> request index."""
    reqs, index = [], {}

    def key(q):
        if q.get("box") is not None:
            return ("box", q.get("text") or "", tuple(q["box"]))
        return ("text", q["text"])

    def add(q, thr):
        k = key(q)
        if k in index:
            r = reqs[index[k]]
            r["threshold"] = min(r["threshold"], thr)
            return
        r = {"threshold": thr}
        if q.get("text"):
            r["text"] = q["text"]
        if q.get("box") is not None:
            r["box"] = list(q["box"])
        index[k] = len(reqs)
        reqs.append(r)

    for t in (silhouette if silhouette is not None
              else cfg["silhouette"]["prompts"]):
        add({"text": t}, 0.3)
    for part, prompts in (parts_prompts if parts_prompts is not None else
                          [(p, p["prompts"]) for p in cfg["parts"]]):
        for q in prompts:
            thr = min(0.3, part["min_score"], float(q.get("threshold", 1)))
            add(q, thr)
    return reqs, (lambda q: index[key(q)])
