"""One drawing: line art -> SAM (one batched call) -> flats -> checks.

`run()` returns a Result; `outputs.py` turns it into .kra / previews /
report. The segmenter is injected (in-process engine or HTTP daemon), so
this module never imports torch.
"""
import time

import numpy as np
from scipy import ndimage as ndi

from . import checks, flats, lineart
from .segment import request_list, select_instances


class Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _union(instances, shape):
    m = np.zeros(shape, bool)
    for inst in instances:
        inst.paste(m)
    return m


def _best(rep):
    return max([d["score"] for q in rep["prompts"] for d in q["instances"]
                if d["used"]] or [0.0])


def resolve_weak(cfg, raw_masks, rep_by_part, notes):
    """A `weak` part found on the same area as a same-slot part that SAM
    is surer about gives way (shirt 0.84 + skirt 0.92 beat dress 0.59).
    Only for parts marked weak: for pants-vs-shorts the score can be
    wrong (Santa), so those stay a warning for a person to settle."""
    parts = cfg["parts"]
    for i, part in enumerate(parts):
        if not part["weak"] or raw_masks[i] is None:
            continue
        mine = raw_masks[i]
        my_best = _best(rep_by_part[i])
        winners = []
        for j, other in enumerate(parts):
            if j == i or raw_masks[j] is None or \
                    not set(part["slot"]) & set(other["slot"]):
                continue
            inter = int((mine & raw_masks[j]).sum())
            smaller = min(int(mine.sum()), int(raw_masks[j].sum()))
            if smaller and inter > 0.5 * smaller and \
                    _best(rep_by_part[j]) > my_best:
                winners.append(other["name"])
        if winners:
            raw_masks[i] = None
            for q in rep_by_part[i]["prompts"]:
                for d in q["instances"]:
                    if d["used"]:
                        d["used"] = False
                        d["reason"] = "weak part: " + ", ".join(winners) + \
                            " scored higher on the same area"
            notes.append({"level": "info", "code": "weak_dropped",
                          "part": part["name"],
                          "message": f"{part['name']} ({my_best:.2f}) dropped: "
                                     f"{', '.join(winners)} scored higher on "
                                     f"the same area"})


def extend_silhouette(silhouette, cfg, kept_by_part, raw_masks, scale):
    """Add kept instances that touch the person (a held sword or staff, a
    cape) to the silhouette, in place. SAM's "person" stops at the body;
    without this they would count as background. Returns the part names
    that extended it."""
    if not silhouette.any():
        return []
    r = max(3, int(round(4 * scale)))
    near = ndi.binary_dilation(silhouette, iterations=r)
    added = []
    for part, kept in zip(cfg["parts"], kept_by_part):
        for inst in kept:
            x, y = inst.xy
            hh, ww = inst.mask.shape
            if not hh or not ww:
                continue
            region = near[y:y + hh, x:x + ww]
            if not (inst.mask & region).any():
                continue          # detached: background, not the character
            if (inst.mask & ~silhouette[y:y + hh, x:x + ww]).sum() < 0.05 * inst.area:
                continue
            inst.paste(silhouette)
            if part["name"] not in added:
                added.append(part["name"])
    return added


def gap_px(cfg, line, silhouette, scale):
    """(gap, leaks): `auto` closes line gaps only if the background leaks
    into the character through open outlines (clean ink -> 0)."""
    g = cfg.get("gap_close", "auto")
    if g == "auto":
        return flats.auto_gap(line, silhouette, scale)
    return int(g), {}


def run(image, cfg, segmenter, name="drawing", log=None):
    """image: PIL image of the line art. Returns a Result."""
    return finish(segment_stage(image, cfg, segmenter, name, log))


def segment_stage(image, cfg, segmenter, name="drawing", log=None):
    """GPU half: line art + every SAM request (+ the retry round) and the
    instance rules. Batch mode runs this for drawing N+1 while `finish`
    (CPU) handles drawing N."""
    t_all = time.perf_counter()
    times = {}
    log = log or (lambda msg: None)
    gray = lineart.to_gray(image)
    h, w = gray.shape
    thr = lineart.ink_threshold(gray, cfg.get("line_threshold", "auto"))
    line = gray < thr
    scale = lineart.scale_factor(gray.shape)
    rgb = lineart.on_white(image)
    times["load_s"] = round(time.perf_counter() - t_all, 3)

    # --- SAM: every prompt of the config in one batched call ---
    t0 = time.perf_counter()
    reqs, key_of = request_list(cfg)
    log(f"{name}: SAM, {len(reqs)} prompts")
    results, seg_timing = segmenter.many(rgb, reqs)
    seg_rounds = [seg_timing]

    sil_prompts = cfg["silhouette"]["prompts"]
    sil_inst = [i for t in sil_prompts
                for i in results[key_of({"text": t})] if i.score >= 0.3]
    sil_used = list(sil_prompts)
    retry = []                        # (part index, prompts) for round 2
    if not sil_inst and cfg["silhouette"]["alternatives"]:
        retry.append((None, [{"text": t} for t in
                             cfg["silhouette"]["alternatives"]]))
    sil_area_guess = None

    def part_select(part, prompts, res_of):
        kept, rep = [], []
        for q in prompts:
            insts = res_of(q)
            k, decisions = select_instances(insts, part, q, sil_area_guess)
            kept += k
            rep.append({"text": q.get("text") or f"box {q.get('box')}",
                        "instances": decisions})
        return kept, rep

    # silhouette area first (max_area rules need it)
    if sil_inst:
        sil_mask = _union(sil_inst, (h, w))
        sil_area_guess = int(sil_mask.sum())
    kept_by_part, rep_by_part = [], []
    for i, part in enumerate(cfg["parts"]):
        kept, rep = part_select(part, part["prompts"],
                                lambda q: results[key_of(q)])
        if not kept and part["alternatives"]:
            retry.append((i, part["alternatives"]))
        kept_by_part.append(kept)
        rep_by_part.append({"name": part["name"], "prompts": rep})

    if retry:
        # round 2 on the cached image features: only the alternatives
        alt_parts = [(cfg["parts"][i] if i is not None else
                      {"min_score": 0.3}, ps) for i, ps in retry]
        reqs2, key2 = request_list(cfg, alt_parts, silhouette=[])
        log(f"{name}: SAM retry, {len(reqs2)} alternative prompts")
        res2, t2 = segmenter.many(rgb, reqs2)
        seg_rounds.append(t2)
        for i, ps in retry:
            if i is None:
                for t in ps:
                    got = [x for x in res2[key2(t)] if x.score >= 0.3]
                    if got:
                        sil_inst, sil_used = got, [t["text"]]
                        break
                continue
            kept, rep = part_select(cfg["parts"][i], ps,
                                    lambda q: res2[key2(q)])
            for r in rep:
                r["alternative"] = True
            kept_by_part[i] = kept
            rep_by_part[i]["prompts"] += rep
    times["sam_s"] = round(time.perf_counter() - t0, 3)
    return Result(image=image, cfg=cfg, name=name, gray=gray, line=line,
                  thr=thr, scale=scale, times=times, t_all=t_all,
                  kept_by_part=kept_by_part, rep_by_part=rep_by_part,
                  sil_inst=sil_inst, sil_used=sil_used, seg_rounds=seg_rounds)


def finish(st):
    """CPU half: silhouette, flats, checks, report."""
    t_cpu = time.perf_counter()
    cfg, gray, line, scale = st.cfg, st.gray, st.line, st.scale
    h, w = gray.shape
    times, kept_by_part, rep_by_part = st.times, st.kept_by_part, st.rep_by_part
    sil_inst, sil_used, name, thr = st.sil_inst, st.sil_used, st.name, st.thr
    seg_rounds, image = st.seg_rounds, st.image
    raw_masks = [(_union(k, (h, w)) if k else None) for k in kept_by_part]
    notes = []
    resolve_weak(cfg, raw_masks, rep_by_part, notes)
    sil_found = bool(sil_inst)
    sil_added = []
    if sil_found:
        silhouette = _union(sil_inst, (h, w))
        sil_added = extend_silhouette(silhouette, cfg, kept_by_part,
                                      raw_masks, scale)
    else:
        # no person: the union of the parts, closed and hole-filled
        silhouette = np.zeros((h, w), bool)
        for m in raw_masks:
            if m is not None:
                silhouette |= m
        r = max(2, int(round(6 * scale)))
        silhouette = ndi.binary_fill_holes(
            ndi.binary_closing(silhouette, iterations=r))

    # --- flats ---
    t0 = time.perf_counter()
    gap, leaks = gap_px(cfg, line, silhouette, scale)
    fl = flats.compute(gray, line, raw_masks, silhouette, cfg["parts"],
                       scale=scale, gap=gap)
    geometry = flats.part_geometry(fl["label"], len(cfg["parts"]))
    times["flats_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    findings = notes + checks.run(cfg, rep_by_part, raw_masks, silhouette,
                                  fl["stats"], geometry,
                                  {"found": sil_found, "prompts": sil_used})
    times["checks_s"] = round(time.perf_counter() - t0, 3)

    parts_out = []
    for i, part in enumerate(cfg["parts"]):
        geo = geometry[i]
        entry = {"name": part["name"], "color": part["color"],
                 "category": part["category"], "index": i + 1,
                 "prompts": rep_by_part[i]["prompts"]}
        if geo:
            entry.update(geo)
        parts_out.append(entry)

    report = {
        "image": name,
        "size": [w, h],
        "config": {"name": cfg.get("name"), "sources": cfg.get("sources")},
        "lineart": {"threshold": thr, "gap_close_px": gap,
                    "leak_by_gap": {str(k): v for k, v in leaks.items()},
                    "scale": round(scale, 3),
                    **lineart.ink_stats(gray, line)},
        "silhouette": {"found": sil_found, "prompts": sil_used,
                       "extended_with": sil_added,
                       "pixels": int(silhouette.sum())},
        "sam": {"requests": sum(r.get("prompts", 0) for r in seg_rounds),
                "rounds": seg_rounds},
        "flats": {k: v for k, v in fl["stats"].items()},
        "flats_timings": fl["timings"],
        "layers": [p for p in parts_out if p.get("pixels")],
        "empty_parts": [p["name"] for p in parts_out if not p.get("pixels")],
        "checks": findings,
        "verdict": checks.verdict(findings),
        "timings": times,
    }
    times["cpu_s"] = round(time.perf_counter() - t_cpu, 3)
    return Result(gray=gray, line=line, label=fl["label"],
                  unlabeled=fl["unlabeled"], inside=fl["inside"],
                  silhouette=silhouette, parts=parts_out, cfg=cfg,
                  report=report, image=image)
