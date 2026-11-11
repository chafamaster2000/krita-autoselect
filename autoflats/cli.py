"""Batch auto-flats, no Krita needed.

    python -m autoflats DRAWINGS/ -o FLATS/ --character santa
    python -m autoflats a.png b.jpg --config extra.yaml --local

Per drawing: <stem>.kra (Lineart in Multiply on top, a group with one
paint layer + selection mask per part), <stem>_preview.png,
<stem>_overlay.png (check view) and <stem>_report.json.

A file <stem>.flats.yaml next to a drawing is applied on top of the
character for that drawing only.

SAM runs in a daemon if one answers (--url, default the AI Select one) or
in this process (--local, or automatically when no daemon answers). While
the CPU computes the flats of drawing N, the GPU already segments N+1.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import config as cfgmod
from . import outputs, pipeline
from .segment import EngineSegmenter, HttpSegmenter, SegmentError

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
DEFAULT_URL = os.environ.get("AUTOSELECT_URL", "http://127.0.0.1:5679")


def find_inputs(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(IMAGE_EXTS) and \
                        not fn.endswith(("_preview.png", "_overlay.png")):
                    out.append(os.path.join(p, fn))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"autoflats: not found: {p}")
    return out


def drawing_override(path):
    stem = os.path.splitext(path)[0]
    for ext in (".flats.yaml", ".flats.yml", ".flats.json"):
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


def find_weights(start):
    """Local SAM 3 weights: models/sam3 in this checkout or any parent
    (a git worktree lives inside the repo), then the plugin's data folder.
    Empty = let transformers fetch facebook/sam3 from Hugging Face."""
    cands = []
    d = start
    for _ in range(6):
        cands.append(os.path.join(d, "models", "sam3"))
        d = os.path.dirname(d)
    cands.append(os.path.join(cfgmod.default_data_dir(), "models", "sam3"))
    for cand in cands:
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return ""


def make_local_engine(weights=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, here)
    from engine import Sam3Engine
    weights = weights or os.environ.get("AUTOSELECT_WEIGHTS_PATH", "") or \
        find_weights(here)
    return Sam3Engine(weights_path=weights, unload_after_s=0,
                      device=os.environ.get("AUTOSELECT_DEVICE") or None)


def pick_segmenter(args, log):
    if not args.local:
        seg = HttpSegmenter(args.url, token=os.environ.get("AUTOSELECT_TOKEN", ""))
        health = seg.health()
        if health and health.get("service") == "krita-autoselect":
            if "feature_cache" not in health:
                log(f"daemon at {args.url} is too old for /segment_many; "
                    f"using SAM in this process")
            else:
                log(f"SAM: daemon {args.url}")
                return seg
        elif args.url_explicit:
            raise SystemExit(f"autoflats: no daemon answers at {args.url}")
        else:
            log(f"no daemon at {args.url}: loading SAM 3 in this process")
    return EngineSegmenter(make_local_engine(args.weights))


def summary_line(rep):
    warns = [c for c in rep["checks"] if c["level"] == "warn"]
    return (f"{rep['image']}: {len(rep['layers'])} layers, "
            f"{rep['verdict']}"
            + (f" ({len(warns)} warnings: "
               + "; ".join(w['code'] for w in warns) + ")" if warns else "")
            + f" | SAM {rep['timings']['sam_s']:.1f}s, "
            f"flats {rep['timings']['flats_s']:.1f}s, "
            f"write {rep['timings'].get('write_s', 0):.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m autoflats",
        description="Auto-flat character line art into layered .kra files.")
    ap.add_argument("inputs", nargs="*", help="images or folders")
    ap.add_argument("-o", "--out", help="output folder (default: <input "
                    "folder>/flats)")
    ap.add_argument("-c", "--character", help="character file or library "
                    "name (default: the built-in taxonomy)")
    ap.add_argument("--config", action="append", default=[],
                    help="extra config file applied on top (repeatable)")
    ap.add_argument("--library", action="append", default=[],
                    help=f"character library folder (default "
                    f"{cfgmod.library_dir()})")
    ap.add_argument("--url", default=None, help="SAM daemon URL "
                    f"(default {DEFAULT_URL})")
    ap.add_argument("--local", action="store_true",
                    help="run SAM 3 in this process, not in a daemon")
    ap.add_argument("--weights", help="SAM 3 weights folder for --local "
                    "(default: models/sam3 here or in a parent, or the "
                    "AI Select data folder)")
    ap.add_argument("--ora", action="store_true", help="also write .ora")
    ap.add_argument("--parts-png", action="store_true",
                    help="also write one PNG per part")
    ap.add_argument("--no-masks", action="store_true",
                    help="no selection masks in the .kra")
    ap.add_argument("--no-kra", action="store_true")
    ap.add_argument("--labels", action="store_true",
                    help="also write <stem>_labels.png (part index per px)")
    ap.add_argument("--sequential", action="store_true",
                    help="don't overlap SAM of the next drawing with the "
                    "flats of this one (for timing comparisons)")
    ap.add_argument("--json", action="store_true",
                    help="print a JSON summary instead of text")
    ap.add_argument("--print-config", action="store_true",
                    help="print the merged config and exit")
    ap.add_argument("--new-character", metavar="NAME",
                    help="create a character file in the library and exit")
    args = ap.parse_args(argv)
    args.url_explicit = args.url is not None
    args.url = args.url or DEFAULT_URL
    log = (lambda m: print(m, file=sys.stderr, flush=True))
    libs = args.library or [cfgmod.library_dir()]
    loader = cfgmod.Loader(libs)

    if args.new_character:
        os.makedirs(libs[0], exist_ok=True)
        path = os.path.join(libs[0], args.new_character + ".yaml")
        if os.path.exists(path):
            raise SystemExit(f"autoflats: already exists: {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cfgmod.TEMPLATE.format(name=args.new_character))
        print(path)
        return 0

    inputs = find_inputs(args.inputs)
    try:
        base_cfg = cfgmod.load_config(args.character, args.config, loader)
        configs = {}
        for path in inputs:
            ov = drawing_override(path)
            configs[path] = cfgmod.load_config(
                args.character, args.config + ([ov] if ov else []), loader) \
                if ov else base_cfg
    except cfgmod.ConfigError as e:
        raise SystemExit(f"autoflats: config error: {e}")
    if args.print_config:
        print(json.dumps(base_cfg, indent=1, ensure_ascii=False))
        return 0
    if not inputs:
        raise SystemExit("autoflats: no images found")

    segmenter = pick_segmenter(args, log)
    t_batch = time.time()
    summaries = []

    def seg(path):
        img = Image.open(path)
        img.load()
        stem = os.path.splitext(os.path.basename(path))[0]
        return pipeline.segment_stage(img, configs[path], segmenter, stem, log)

    def finish(path, st):
        res = pipeline.finish(st)
        out_dir = args.out or os.path.join(os.path.dirname(path) or ".",
                                           "flats")
        t0 = time.perf_counter()
        paths = outputs.write_all(res, out_dir, st.name, kra=not args.no_kra,
                                  ora=args.ora, parts_png=args.parts_png,
                                  masks=not args.no_masks,
                                  labels=args.labels)
        res.report["timings"]["write_s"] = round(time.perf_counter() - t0, 3)
        with open(paths["report"], "w", encoding="utf-8") as f:
            json.dump(res.report, f, indent=1, ensure_ascii=False)
        return res.report

    errors = 0
    with ThreadPoolExecutor(max_workers=1) as gpu:
        pending = None
        for i, path in enumerate(inputs):
            fut = pending or gpu.submit(seg, path)
            pending = None
            try:
                st = fut.result()
            except (SegmentError, OSError) as e:
                log(f"{path}: SAM failed: {e}")
                errors += 1
                continue
            if i + 1 < len(inputs) and not args.sequential:
                pending = gpu.submit(seg, inputs[i + 1])
            try:
                rep = finish(path, st)
            except Exception as e:  # one bad drawing doesn't stop the batch
                log(f"{path}: failed: {type(e).__name__}: {e}")
                errors += 1
                continue
            summaries.append(rep)
            if not args.json:
                print(summary_line(rep), flush=True)
                print("   " + rep["outputs"]["overlay"], flush=True)
    total = round(time.time() - t_batch, 2)
    if args.json:
        print(json.dumps({
            "total_s": total, "errors": errors,
            "drawings": [{"image": r["image"], "verdict": r["verdict"],
                          "layers": [l["name"] for l in r["layers"]],
                          "warnings": [c["message"] for c in r["checks"]
                                       if c["level"] == "warn"],
                          "timings": r["timings"], "outputs": r["outputs"]}
                         for r in summaries]}, indent=1, ensure_ascii=False))
    else:
        print(f"{len(summaries)} drawing(s) in {total}s"
              + (f", {errors} failed" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
