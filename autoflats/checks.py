"""Automatic quality checks: what a person caught by eye on the Santa POC.

Each check returns findings {level, code, message, part?, bbox?}; level is
"warn" (look at this before trusting the result) or "info". All
thresholds can be overridden from the config's `checks:` mapping.
"""
import numpy as np

DEFAULTS = {
    "low_score": 0.45,          # a kept part whose best instance is below
    "unlabeled_warn": 0.02,     # > 2 % of the silhouette with no SAM part
    "contested_warn": 0.03,     # > 3 % in regions two parts fight over
    "overlap_warn": 0.5,        # two same-category parts share > 50 %
    "outside_warn": 0.3,        # > 30 % of a part outside the silhouette
    "islands_warn": 0.05,       # > 5 % of a part removed as islands
    "fragments_info": 40,       # a part split in more pieces than this
    "dropped_area_info": 0.25,  # a dropped instance bigger than 25 % of kept
}


def finding(level, code, message, **extra):
    d = {"level": level, "code": code, "message": message}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


def _bbox_of(mask):
    rows = np.flatnonzero(mask.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(0))
    return [int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1),
            int(rows[-1] - rows[0] + 1)]


def run(cfg, part_reports, raw_masks, silhouette, flats_stats, geometry,
        silhouette_info):
    """part_reports: per part {name, prompts: [{text, instances: [...]}]}.
    raw_masks: per part bool mask (SAM union after the rules) or None."""
    th = dict(DEFAULTS)
    th.update(cfg.get("checks") or {})
    parts = cfg["parts"]
    out = []
    names = [p["name"] for p in parts]

    if not silhouette_info.get("found"):
        out.append(finding(
            "warn", "no_silhouette",
            "SAM did not find a person/character: the silhouette comes from "
            "the union of the parts, so the outline may be cut or leak. Add "
            "a `silhouette: {prompts: [...]}` that names this character."))

    for i, (part, rep) in enumerate(zip(parts, part_reports)):
        used = [d for q in rep["prompts"] for d in q["instances"] if d["used"]]
        dropped = [d for q in rep["prompts"] for d in q["instances"]
                   if not d["used"]]
        geo = geometry[i]
        if not used:
            if part["required"]:
                out.append(finding("warn", "missing",
                                   f"{part['name']}: nothing found "
                                   f"({', '.join(q['text'] for q in rep['prompts'])})",
                                   part=part["name"]))
            continue
        best = max(d["score"] for d in used)
        if best < th["low_score"]:
            out.append(finding(
                "warn", "low_score",
                f"{part['name']}: SAM is unsure (best score {best:.2f}); "
                f"check it or name it better in the config",
                part=part["name"], bbox=geo["bbox"] if geo else None))
        kept_area = sum(d["area"] for d in used)
        for q in rep["prompts"]:
            for d in q["instances"]:
                if not d["used"] and d["area"] > th["dropped_area_info"] * \
                        max(kept_area, 1) and "score <" in d.get("reason", "x") \
                        and d["score"] >= 0.3:
                    out.append(finding(
                        "info", "dropped_instance",
                        f"{part['name']}: '{q['text']}' #{d['index']} "
                        f"(score {d['score']}) dropped: {d['reason']}",
                        part=part["name"],
                        bbox=[d["box"][0], d["box"][1],
                              d["box"][2] - d["box"][0],
                              d["box"][3] - d["box"][1]]))
        if geo is None:
            out.append(finding(
                "info", "absorbed",
                f"{part['name']}: SAM found it but no pixels survived "
                f"(covered by higher-priority parts or cleaned as noise)",
                part=part["name"]))
            continue
        removed = flats_stats["islands_removed"][i]
        if removed > th["islands_warn"] * (geo["pixels"] + removed):
            out.append(finding(
                "warn", "islands",
                f"{part['name']}: {removed} px of small islands removed "
                f"({removed / (geo['pixels'] + removed):.0%}); SAM's mask "
                f"does not follow the lines here",
                part=part["name"], bbox=geo["bbox"]))
        if geo["components"] > th["fragments_info"] and not part["speck_exempt"]:
            out.append(finding(
                "info", "fragmented",
                f"{part['name']}: {geo['components']} separate pieces",
                part=part["name"], bbox=geo["bbox"]))

    # parts outside the silhouette (a background barrel read as a bag)
    if silhouette is not None and silhouette.any():
        for i, m in enumerate(raw_masks):
            if m is None:
                continue
            area = int(m.sum())
            if not area:
                continue
            outside = int((m & ~silhouette).sum())
            if outside > th["outside_warn"] * area:
                out.append(finding(
                    "warn", "outside_silhouette",
                    f"{names[i]}: {outside / area:.0%} of SAM's mask is "
                    f"outside the character (background object?)",
                    part=names[i], bbox=_bbox_of(m & ~silhouette)))

    # alternatives for the same body slot both found on the same area
    # (pants AND shorts, coat AND vest): priority picked one, a person
    # should say which in the character file. Containment between other
    # parts (laces in a shoe, a buckle on a belt) is normal and not checked.
    boxes = [_bbox_of(m) if m is not None else None for m in raw_masks]
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            if raw_masks[a] is None or raw_masks[b] is None:
                continue
            shared = set(parts[a]["slot"]) & set(parts[b]["slot"])
            if not shared:
                continue
            ba, bb = boxes[a], boxes[b]
            x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
            x1 = min(ba[0] + ba[2], bb[0] + bb[2])
            y1 = min(ba[1] + ba[3], bb[1] + bb[3])
            if x1 <= x0 or y1 <= y0:
                continue
            inter = int((raw_masks[a][y0:y1, x0:x1]
                         & raw_masks[b][y0:y1, x0:x1]).sum())
            smaller = min(int(raw_masks[a].sum()), int(raw_masks[b].sum()))
            if smaller and inter > th["overlap_warn"] * smaller:
                out.append(finding(
                    "warn", "overlap",
                    f"{names[a]} and {names[b]} both found on the same area "
                    f"({inter / smaller:.0%}); {names[b]} won by priority. "
                    f"Say which one this character wears (`remove:` the "
                    f"other in the character file)",
                    parts=[names[a], names[b]],
                    bbox=[x0, y0, x1 - x0, y1 - y0]))

    unl = flats_stats["unlabeled_frac"]
    if unl > th["unlabeled_warn"]:
        out.append(finding(
            "warn", "unlabeled",
            f"{unl:.1%} of the character has no SAM part; it was filled "
            f"from the neighbours (magenta in the overlay). Add a part for it",
            blobs=flats_stats["unlabeled_blobs"]))
    con = flats_stats["contested_frac"]
    if con > th["contested_warn"]:
        regs = [dict(r, parts=[names[p - 1] for p in r["parts"]])
                for r in flats_stats["contested_regions"]]
        out.append(finding(
            "info", "contested",
            f"{con:.1%} of the character follows SAM's edges instead of "
            f"the ink: open lines between "
            + ", ".join(sorted({" / ".join(r["parts"]) for r in regs}))
            + " (yellow boxes)", regions=regs))
    return out


def verdict(findings):
    return "review" if any(f["level"] == "warn" for f in findings) else "ok"
