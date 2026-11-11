"""/segment_many: request parsing, response encoding, the engine's batching
logic and its vision-feature LRU cache.

No torch, GPU or weights: the HTTP layer runs against a fake engine, and
the real Sam3Engine.segment_many runs with its model calls replaced
(_detect_batch_locked, _segment_concept/_segment_visual, the processor).
Needs numpy + Pillow (the daemon's own deps); skipped without them.
"""
import base64
import io
import json
import os
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import server  # noqa: E402
import engine  # noqa: E402

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None

W, H = 40, 30


def png_b64(w=W, h=H, color=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_png(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def rect_mask(x1, y1, x2, y2, w=W, h=H):
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def inst(i, box, score=0.9):
    return {"index": i, "score": score, "box": [float(v) for v in box]}


class ParseSegmentManyTest(unittest.TestCase):
    def parse(self, **payload):
        payload.setdefault("image_b64", "xx")
        return server.parse_segment_many_params(payload)

    def test_strings_become_text_prompts(self):
        image, prompts, masks, thr = self.parse(prompts=["glove", "boot"])
        self.assertEqual(image, "xx")
        self.assertEqual([p["text"] for p in prompts], ["glove", "boot"])
        self.assertEqual(masks, "instances")
        self.assertEqual(thr, 0.5)
        self.assertNotIn("image_b64", prompts[0])
        self.assertNotIn("list_only", prompts[0])
        self.assertNotIn("mask_threshold", prompts[0])

    def test_object_prompt_fields(self):
        _, prompts, masks, thr = self.parse(
            prompts=[{"text": "glove", "box": [1, 2, 3, 4], "threshold": 0.3,
                      "combine": "1", "within": [0, 0, 10, 10],
                      "points": [[5, 5]], "point_labels": [1]}],
            masks="union", mask_threshold="0.4")
        p = prompts[0]
        self.assertEqual(p["box"], [1, 2, 3, 4])
        self.assertEqual(p["threshold"], 0.3)
        self.assertEqual(p["combine"], 1)
        self.assertEqual(p["within"], [0, 0, 10, 10])
        self.assertEqual(p["points"], [[5, 5]])
        self.assertEqual((masks, thr), ("union", 0.4))

    def test_within_alone_is_a_prompt(self):
        _, prompts, _, _ = self.parse(prompts=[{"within": [0, 0, 5, 5]}])
        self.assertIsNone(prompts[0]["text"])

    def test_errors_name_the_prompt(self):
        cases = [
            ({"prompts": [{"text": "a"}, {"box": [1, 2, 3]}]},
             "prompts[1]: box must be [x, y, width, height]"),
            ({"prompts": ["a", {}]}, "prompts[1]: At least one prompt"),
            ({"prompts": ["a", 3]}, "prompts[1] must be text or an object"),
            ({"prompts": []}, "prompts must be a non-empty list"),
            ({"prompts": "glove"}, "prompts must be a non-empty list"),
            ({"prompts": ["a"] * 257}, "at most 256 prompts"),
            ({"prompts": ["a"], "masks": "all"}, "masks must be one of"),
            ({"prompts": ["a"], "mask_threshold": "x"},
             "mask_threshold must be a number"),
        ]
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(server.BadRequest) as cm:
                    self.parse(**payload)
                self.assertIn(message, str(cm.exception))

    def test_image_required_and_body_object(self):
        with self.assertRaisesRegex(server.BadRequest, "image_b64"):
            server.parse_segment_many_params({"prompts": ["a"]})
        with self.assertRaisesRegex(server.BadRequest, "JSON object"):
            server.parse_segment_many_params(["a"])

    def test_256_prompts_is_the_limit(self):
        _, prompts, _, _ = self.parse(prompts=["a"] * 256)
        self.assertEqual(len(prompts), 256)


class FakeManyEngine:
    """Engine stand-in for run_segment_many: two instances for "glove",
    none for anything else. Honours crop like the real engine."""

    decode_image = staticmethod(engine.Sam3Engine.decode_image)

    def __init__(self):
        self.calls = []

    def segment_many(self, image, prompts, mask_threshold=0.5, key=None,
                     crop=False):
        self.calls.append({"prompts": prompts, "crop": crop,
                           "mask_threshold": mask_threshold})
        out = []
        for p in prompts:
            if p.get("text") != "glove":
                out.append(([], []))
                continue
            masks = [rect_mask(2, 3, 6, 8), rect_mask(20, 10, 30, 20)]
            insts = [inst(0, [2, 3, 6, 8], 0.96), inst(1, [20, 10, 30, 20])]
            if crop:
                cropped = [engine.Sam3Engine.crop_mask(m) for m in masks]
                for d, (_, xy) in zip(insts, cropped):
                    d["mask_xy"] = xy
                masks = [c for c, _ in cropped]
            out.append((insts, masks))
        return out, {"prompts": len(prompts), "total_s": 0.01}


@unittest.skipIf(np is None, "needs numpy + Pillow")
class RunSegmentManyTest(unittest.TestCase):
    def run_many(self, prompts, masks):
        eng = FakeManyEngine()
        parsed = [server.parse_segment_params(
            dict({"text": p} if isinstance(p, str) else p, image_b64="-"))
            for p in prompts]
        return eng, server.run_segment_many(eng, png_b64(), parsed, masks, 0.5)

    def test_instances_are_cropped_pngs_with_offset(self):
        eng, out = self.run_many(["glove", "hat"], "instances")
        self.assertTrue(eng.calls[0]["crop"])
        self.assertEqual((out["width"], out["height"]), (W, H))
        glove, hat = out["results"]
        self.assertEqual((glove["text"], glove["count"]), ("glove", 2))
        self.assertEqual((hat["count"], hat["instances"]), (0, []))
        first = glove["instances"][0]
        self.assertEqual(first["mask_xy"], [2, 3])
        self.assertEqual(first["box"], [2.0, 3.0, 6.0, 8.0])
        crop = decode_png(first["mask_b64"])
        self.assertEqual(crop.shape, (5, 4))
        self.assertTrue((crop == 255).all())
        self.assertNotIn("mask_b64", glove)
        self.assertEqual(out["timing"]["prompts"], 2)

    def test_union_is_one_canvas_png_per_prompt(self):
        eng, out = self.run_many(["glove", "hat"], "union")
        self.assertFalse(eng.calls[0]["crop"])
        glove, hat = out["results"]
        full = decode_png(glove["mask_b64"])
        self.assertEqual(full.shape, (H, W))
        self.assertEqual(int((full > 0).sum()), 4 * 5 + 10 * 10)
        self.assertNotIn("mask_b64", glove["instances"][0])
        self.assertNotIn("mask_xy", glove["instances"][0])
        self.assertNotIn("mask_b64", hat)

    def test_union_honours_combine_index(self):
        _, out = self.run_many([{"text": "glove", "combine": 1}], "union")
        full = decode_png(out["results"][0]["mask_b64"])
        self.assertEqual(int((full > 0).sum()), 100)
        self.assertTrue(full[15, 25])
        self.assertFalse(full[5, 3])

    def test_union_combine_out_of_range_gives_no_mask(self):
        _, out = self.run_many([{"text": "glove", "combine": 7}], "union")
        self.assertNotIn("mask_b64", out["results"][0])
        self.assertEqual(out["results"][0]["count"], 2)

    def test_none_returns_only_metadata(self):
        _, out = self.run_many(["glove"], "none")
        item = out["results"][0]
        self.assertEqual(item["count"], 2)
        self.assertNotIn("mask_b64", item)
        for i in item["instances"]:
            self.assertEqual(set(i), {"index", "score", "box"})


class FakeProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, images=None, return_tensors=None, **kw):
        self.calls += 1
        return {"pixel_values": types.SimpleNamespace(
            to=lambda *_a, **_k: types.SimpleNamespace(
                to=lambda *_a, **_k: "pixels"))}


class FakeModel:
    device = types.SimpleNamespace(type="cpu")

    def __init__(self):
        self.vision_calls = 0

    def get_vision_features(self, pixel_values):
        self.vision_calls += 1
        return f"embeds#{self.vision_calls}"


def fake_torch():
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class OutOfMemoryError(RuntimeError):
        pass

    return types.SimpleNamespace(
        no_grad=_NoGrad,
        cuda=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError,
                                   empty_cache=lambda: None,
                                   synchronize=lambda: None))


@unittest.skipIf(np is None, "needs numpy + Pillow")
class EngineSegmentManyTest(unittest.TestCase):
    """Sam3Engine.segment_many with the model calls faked out."""

    def setUp(self):
        self.torch = fake_torch()
        patcher = mock.patch.dict(sys.modules, {"torch": self.torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.eng = engine.Sam3Engine(prompt_batch=2, unload_after_s=0)
        self.eng._model = FakeModel()
        self.eng._processor = FakeProcessor()
        self.eng._dtype = None
        self.batches = []
        self.singles = []
        # "glove": two instances, left and right; anything else: none.
        self.glove = [(inst(0, [2, 3, 6, 8], 0.96), rect_mask(2, 3, 6, 8)),
                      (inst(1, [20, 10, 30, 20], 0.7),
                       rect_mask(20, 10, 30, 20))]

        def detect(vision, prompts, width, height, mask_threshold, crop=False):
            self.batches.append([p["text"] for p in prompts])
            out = []
            for p in prompts:
                pairs = self.glove if p["text"] == "glove" else []
                thr = float(p.get("threshold", 0.5))
                pairs = [(dict(i), m) for i, m in pairs if i["score"] >= thr]
                insts, masks = [i for i, _ in pairs], [m for _, m in pairs]
                if crop:
                    cropped = [self.eng.crop_mask(m) for m in masks]
                    for d, (_, xy) in zip(insts, cropped):
                        d["mask_xy"] = xy
                    masks = [c for c, _ in cropped]
                out.append((insts, masks))
            return out

        def concept(image, text, points, point_labels, box, threshold,
                    mask_threshold, key=None):
            self.singles.append(("concept", text, box))
            return ([dict(i) for i, _ in self.glove],
                    [m for _, m in self.glove])

        def visual(image, points, point_labels, box):
            self.singles.append(("visual", None, box))
            return [inst(0, [0, 0, 5, 5])], [rect_mask(0, 0, 5, 5)]

        self.eng._detect_batch_locked = detect
        self.eng._segment_concept = concept
        self.eng._segment_visual = visual
        self.image = Image.new("RGB", (W, H), (1, 2, 3))

    def test_text_prompts_go_in_batches_of_prompt_batch(self):
        prompts = [{"text": t} for t in ("glove", "hat", "boot", "glove", "x")]
        results, timing = self.eng.segment_many(self.image, prompts)
        self.assertEqual(self.batches,
                         [["glove", "hat"], ["boot", "glove"], ["x"]])
        self.assertEqual([len(r[0]) for r in results], [2, 0, 0, 2, 0])
        self.assertEqual(timing["batch"], 2)
        self.assertEqual(timing["prompts"], 5)
        self.assertNotIn("single_s", timing)
        self.assertEqual(self.singles, [])

    def test_box_and_points_prompts_run_one_by_one(self):
        prompts = [{"text": "glove"},
                   {"text": "glove", "box": [1, 1, 5, 5]},
                   {"box": [0, 0, 5, 5]}]
        results, timing = self.eng.segment_many(self.image, prompts)
        self.assertEqual(self.batches, [["glove"]])
        self.assertEqual(self.singles, [("concept", "glove", [1, 1, 5, 5]),
                                        ("visual", None, [0, 0, 5, 5])])
        self.assertEqual([len(r[0]) for r in results], [2, 2, 1])
        self.assertIn("single_s", timing)

    def test_within_filters_batched_and_single_prompts(self):
        within = [0, 0, 10, 10]   # keeps the left glove only
        prompts = [{"text": "glove", "within": within},
                   {"text": "glove", "box": [1, 1, 5, 5], "within": within}]
        results, _ = self.eng.segment_many(self.image, prompts)
        for instances, masks in results:
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0]["box"], [2.0, 3.0, 6.0, 8.0])
            self.assertEqual(instances[0]["index"], 0)
            self.assertEqual(len(masks), 1)

    def test_within_alone_uses_the_tracker_box(self):
        results, _ = self.eng.segment_many(self.image,
                                           [{"within": [0, 0, 9, 9]}])
        self.assertEqual(self.singles, [("visual", None, [0, 0, 9, 9])])
        self.assertEqual(len(results[0][0]), 1)

    def test_crop_returns_tight_masks_with_offset(self):
        prompts = [{"text": "glove"}, {"text": "glove", "box": [1, 1, 5, 5]}]
        results, _ = self.eng.segment_many(self.image, prompts, crop=True)
        for instances, masks in results:
            self.assertEqual(instances[0]["mask_xy"], [2, 3])
            self.assertEqual(masks[0].shape, (5, 4))
            self.assertEqual(instances[1]["mask_xy"], [20, 10])
            self.assertEqual(masks[1].shape, (10, 10))

    def test_per_prompt_threshold(self):
        results, _ = self.eng.segment_many(
            self.image, [{"text": "glove", "threshold": 0.8},
                         {"text": "glove", "threshold": 0.3}])
        self.assertEqual([len(r[0]) for r in results], [1, 2])

    def test_oom_halves_the_batch_and_remembers_it(self):
        real = self.eng._detect_batch_locked
        oom = self.torch.cuda.OutOfMemoryError

        def flaky(vision, prompts, *a, **k):
            if len(prompts) > 1:
                raise oom("CUDA out of memory")
            return real(vision, prompts, *a, **k)

        self.eng._detect_batch_locked = flaky
        prompts = [{"text": "glove"}, {"text": "hat"}, {"text": "boot"}]
        results, timing = self.eng.segment_many(self.image, prompts)
        self.assertEqual(self.batches, [["glove"], ["hat"], ["boot"]])
        self.assertEqual(timing["batch"], 1)
        self.assertEqual(self.eng._batch_cap, 1)
        self.assertEqual([len(r[0]) for r in results], [2, 0, 0])

    def test_vision_runs_once_per_image(self):
        self.eng.segment_many(self.image, [{"text": "glove"}, {"text": "a"},
                                           {"text": "b"}])
        _, timing = self.eng.segment_many(self.image, [{"text": "glove"}])
        self.assertEqual(self.eng._model.vision_calls, 1)
        self.assertTrue(timing["vision_cached"])


@unittest.skipIf(np is None, "needs numpy + Pillow")
class FeatureCacheTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {"torch": fake_torch()})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.eng = engine.Sam3Engine(feature_cache=2)
        self.eng._model = FakeModel()
        self.eng._processor = FakeProcessor()
        self.eng._dtype = None

    def img(self, color):
        return Image.new("RGB", (8, 8), color)

    def test_lru_evicts_the_least_recently_used(self):
        a, b, c = self.img((1, 0, 0)), self.img((2, 0, 0)), self.img((3, 0, 0))
        ea, hit = self.eng._vision_locked(a)
        self.assertFalse(hit)
        self.eng._vision_locked(b)
        self.assertEqual(self.eng._vision_locked(a), (ea, True))  # a is fresh
        self.eng._vision_locked(c)                                 # evicts b
        self.assertEqual(len(self.eng._features), 2)
        self.assertTrue(self.eng._vision_locked(a)[1])
        self.assertFalse(self.eng._vision_locked(b)[1])
        self.assertEqual((self.eng.cache_hits, self.eng.cache_misses), (2, 4))
        info = self.eng.info()["feature_cache"]
        self.assertEqual(info, {"size": 2, "max": 2, "hits": 2, "misses": 4})

    def test_key_is_pixels_not_encoding(self):
        a = self.img((9, 9, 9))
        same = Image.open(io.BytesIO(base64.b64decode(png_b64(8, 8, (9, 9, 9)))))
        self.assertEqual(engine.Sam3Engine.image_key(a),
                         engine.Sam3Engine.image_key(same.convert("RGB")))
        other_size = Image.new("RGB", (16, 4), (9, 9, 9))
        self.assertNotEqual(engine.Sam3Engine.image_key(a),
                            engine.Sam3Engine.image_key(other_size))

    def test_size_zero_disables_the_cache(self):
        self.eng.feature_cache_size = 0
        a = self.img((1, 1, 1))
        self.eng._vision_locked(a)
        self.assertFalse(self.eng._vision_locked(a)[1])
        self.assertEqual(len(self.eng._features), 0)

    def test_unload_clears_the_cache(self):
        self.eng._vision_locked(self.img((5, 5, 5)))
        self.eng._batch_cap = 4
        self.eng.unload()
        self.assertEqual(len(self.eng._features), 0)
        self.assertIsNone(self.eng._batch_cap)


@unittest.skipIf(np is None, "needs numpy + Pillow")
class SegmentManyHttpTest(unittest.TestCase):
    """The endpoint and /health over real HTTP, fake engine."""

    def setUp(self):
        self.engine = FakeManyEngine()
        self.engine.info = lambda: {"model": "fake", "loaded": False}
        self.srv = server.make_server(self.engine, port=0)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    def post(self, path, payload):
        req = urllib.request.Request(self.url + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type":
                                              "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_health_advertises_segment_many(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            body = json.loads(r.read())
        self.assertEqual(body["api_version"], 2)
        self.assertEqual(body["capabilities"], ["segment", "segment_many"])
        self.assertEqual(body["model"], "fake")

    def test_segment_many_roundtrip(self):
        status, body = self.post("/segment_many", {
            "image_b64": png_b64(),
            "prompts": ["glove", {"text": "hat", "threshold": 0.2}],
            "masks": "none"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual([r["count"] for r in body["results"]], [2, 0])
        self.assertEqual(body["timing"]["prompts"], 2)

    def test_segment_many_bad_prompt_is_400(self):
        status, body = self.post("/segment_many", {
            "image_b64": png_b64(), "prompts": [{"text": "a"},
                                                {"box": [1, 2, 3]}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"],
                         "prompts[1]: box must be [x, y, width, height]")

    def test_flats_endpoint_is_gone(self):
        status, _ = self.post("/flats", {"image_b64": png_b64()})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
