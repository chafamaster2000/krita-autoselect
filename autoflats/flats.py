"""Flats from SAM masks + the line art (the santa-poc algorithm, generalized).

SAM knows WHAT each area is (beard, vest) but its edges don't follow the ink.
The closed areas of the line art do, so each closed region takes the part
that covers most of it; where a region mixes parts (an open line), SAM is
followed pixel by pixel. Small SAM noise is cleaned up, and finally the
colour is extended under the lines (nearest labelled pixel) so multiplying
the line art on top leaves no white halos.

Pure numpy/scipy. Input: grey image, ink mask, one bool mask per part in
priority order (later wins), silhouette mask. Output: label map
(0 = no part, i + 1 = parts[i]) plus everything the quality checks need.
"""
import time

import numpy as np
from scipy import ndimage as ndi

REGION_PURITY = 0.6     # a region goes whole to a part covering >= 60 %
SPECK_MAX = 3000        # px at the reference size
SPECK_SURROUND = 0.9    # a speck surrounded >= 90 % by one part joins it
MIXED_OPEN = 3          # px: opening of each part inside mixed regions
THICK_INK = 6           # px from paper: "thick ink" (solid black areas)
SIL_DILATE = 4          # px: ink this close to the silhouette is inside
CONTESTED_MIN = 0.4     # both top parts cover >= 40 % of a region
UNLABELED_BLOB = 0.002  # unlabeled blobs >= 0.2 % of the silhouette


class Timer:
    def __init__(self):
        self.times = {}
        self._t = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.times[name] = round(now - self._t, 4)
        self._t = now


def _bbox(sl):
    ys, xs = sl
    return [int(xs.start), int(ys.start), int(xs.stop - xs.start),
            int(ys.stop - ys.start)]


def _pad(sl, pad, shape):
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
                 for s, n in zip(sl, shape))


def close_gaps(line, gap):
    """Ink with gaps up to ~2*gap px bridged (morphological closing), for
    finding regions only: the bridges never show, they just stop one
    region from leaking into the next through an open line."""
    if gap <= 0:
        return line
    return line | ndi.binary_closing(line, iterations=int(gap),
                                     border_value=0)


def leak_fraction(line, silhouette):
    """Share of the silhouette's paper that sits in regions connected to
    the image border: with open outlines the background floods into the
    character. Clean ink gives ~0."""
    regions, n = ndi.label(~line)
    border = np.unique(np.concatenate(
        [regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    border = border[border > 0]
    paper_in = silhouette & ~line
    total = max(1, int(paper_in.sum()))
    if not len(border):
        return 0.0
    is_border = np.zeros(n + 1, bool)
    is_border[border] = True
    return float((paper_in & is_border[regions]).sum()) / total


def auto_gap(line, silhouette, scale, max_leak=0.03):
    """Smallest gap closing (0, then growing) that keeps the background out
    of the silhouette. Returns (gap_px, {gap: leak})."""
    tried = {}
    for g in (0, max(1, round(1.5 * scale)), max(2, round(3 * scale)),
              max(3, round(5 * scale))):
        if g in tried:
            continue
        tried[g] = round(leak_fraction(close_gaps(line, g), silhouette), 4)
        if tried[g] <= max_leak:
            return g, tried
    return min(tried, key=tried.get), tried


def sam_label_map(shape, part_masks):
    """Priority painting: 0 = nothing, i + 1 = part i; later parts win."""
    sam = np.zeros(shape, np.int16)
    for i, m in enumerate(part_masks):
        if m is not None:
            sam[m] = i + 1
    return sam


def compute(gray, line, part_masks, silhouette, parts, scale=1.0, gap=0):
    """Run the flats. `parts` = config parts (for per-part thresholds).

    Returns a dict: label, sam, inside, unlabeled (mask), stats, timings.
    """
    T = Timer()
    h, w = gray.shape
    k = len(parts) + 1
    area_scale = scale * scale

    sam = sam_label_map((h, w), part_masks)
    T.lap("1_sam_label_map")

    line_r = close_gaps(line, gap)
    regions, n = ndi.label(~line_r)
    border = np.unique(np.concatenate(
        [regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    idx = np.arange(1, n + 1)
    size = np.bincount(regions.ravel(), minlength=n + 1)
    in_sil = (np.bincount(regions.ravel(), weights=silhouette.ravel(),
                          minlength=n + 1) / np.maximum(size, 1))
    background = np.zeros(n + 1, bool)
    for r in border:
        if r and in_sil[r] < 0.5:
            background[r] = True
    background[1:] |= in_sil[1:] < 0.2
    background[0] = False
    T.lap("2_regions_and_background")

    votes = np.bincount(regions.ravel().astype(np.int64) * k + sam.ravel(),
                        minlength=(n + 1) * k).reshape(n + 1, k)
    labeled = votes[:, 1:]
    best = labeled.argmax(1) + 1
    region_total = np.maximum(votes.sum(1), 1)
    best_frac = labeled.max(1) / region_total
    whole = (best_frac >= REGION_PURITY) & ~background
    whole[0] = False
    label = np.where(whole, best, 0).astype(np.int16)[regions]
    mixed = (~whole[regions]) & (~background[regions]) & (regions > 0)
    label[mixed] = sam[mixed]
    # thin tongues SAM leaves in mixed regions (fur trim poking into the
    # vest through gaps in its outline): open each part there
    mixed_open = max(1, int(round(MIXED_OPEN * scale)))
    objs = ndi.find_objects(np.where(mixed, label, 0))
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sl = _pad(sl, 2 * mixed_open + 2, (h, w))
        sub = (label[sl] == i) & mixed[sl]
        thin = sub & ~ndi.binary_opening(sub, iterations=mixed_open)
        label[sl][thin] = 0
    # contested regions: two parts both claim a good share of one region
    second = np.sort(labeled, axis=1)[:, -2] / region_total if k > 2 else \
        np.zeros(n + 1)
    contested_ids = np.nonzero((best_frac >= CONTESTED_MIN)
                               & (second >= CONTESTED_MIN)
                               & ~background)[0]
    contested_ids = contested_ids[contested_ids > 0]
    T.lap("3_region_vote")

    sil_dil = max(2, int(round(SIL_DILATE * scale)))
    inside = ((regions > 0) & ~background[regions]) | (
        line_r & ndi.binary_dilation(silhouette, iterations=sil_dil))
    inside = ndi.binary_fill_holes(inside)
    T.lap("4_inside_silhouette")

    # small islands of a part that come mostly from mixed regions = SAM
    # noise -> unlabeled (filled by the neighbours below). A closed region
    # the vote gave whole is legitimate even if small (highlights).
    islands_removed = np.zeros(k, np.int64)
    objs = ndi.find_objects(label)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        min_island = parts[i - 1]["min_island"] * area_scale
        sub = label[sl] == i
        comp, nc = ndi.label(sub)
        if nc <= 1:
            continue
        sizes = np.bincount(comp.ravel(), minlength=nc + 1)
        from_mixed = np.bincount(comp.ravel(), weights=mixed[sl].ravel(),
                                 minlength=nc + 1) / np.maximum(sizes, 1)
        small = (sizes < min_island) & (from_mixed > 0.5)
        small[0] = False
        if small.any():
            drop = small[comp]
            islands_removed[i] += int(drop.sum())
            label[sl][drop] = 0
    T.lap("4b_island_cleanup")

    # specks enclosed by another part -> that part (the cream fleck inside
    # the red vest). Before filling the ink: once the ink is labelled the
    # speck would touch the big part through it.
    speck_max = SPECK_MAX * area_scale
    n_specks = 0
    objs = ndi.find_objects(label)
    for i, sl in enumerate(objs, start=1):
        if sl is None or parts[i - 1]["speck_exempt"]:
            continue
        psl = _pad(sl, 4, (h, w))
        lab = label[psl]          # view: writes go to `label`
        ink = line[psl]
        comp, nc = ndi.label(lab == i)
        if not nc:
            continue
        sizes = np.bincount(comp.ravel(), minlength=nc + 1)
        cobjs = ndi.find_objects(comp)
        for c in np.nonzero(sizes[1:] < speck_max)[0] + 1:
            csl = _pad(cobjs[c - 1], 4, lab.shape)
            m = comp[csl] == c
            ring = (ndi.binary_dilation(m, iterations=3) & ~m
                    & ~ink[csl] & (lab[csl] > 0))
            nb = np.bincount(lab[csl][ring], minlength=k)
            j = int(nb.argmax())
            if j not in (0, i) and nb[j] >= SPECK_SURROUND * nb.sum():
                lab[csl][m] = j
                n_specks += 1
    T.lap("4c_speck_cleanup")

    # interior pixels without a part (ink, no vote) -> nearest labelled one
    unlabeled = (sam == 0) & inside & ~line_r
    known = label > 0
    if known.any():
        _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
        filled = label[iy, ix]
    else:
        filled = label
    # thick ink (far from any paper): SAM decides if it says something —
    # otherwise a leg drawn almost solid black inherits a neighbour's label
    thick_ink = THICK_INK * scale
    thick = line & (ndi.distance_transform_edt(line) > thick_ink)
    filled = np.where(thick & (sam > 0), sam, filled)
    label = np.where(inside, filled, 0).astype(np.int16)
    T.lap("5_fill_under_lines")

    sil_area = max(1, int(inside.sum()))
    # big unlabeled blobs: SAM had no part there, the colour is a guess
    blobs = []
    comp, nc = ndi.label(unlabeled)
    if nc:
        sizes = np.bincount(comp.ravel(), minlength=nc + 1)
        big = np.nonzero(sizes[1:] >= UNLABELED_BLOB * sil_area)[0] + 1
        cobjs = ndi.find_objects(comp)
        for c in big[np.argsort(-sizes[big])][:8]:
            sl = cobjs[c - 1]
            got = np.bincount(label[sl][comp[sl] == c], minlength=k)
            blobs.append({"bbox": _bbox(sl), "pixels": int(sizes[c]),
                          "filled_as": int(got.argmax())})
    contested = []
    if len(contested_ids):
        robjs = ndi.find_objects(regions)
        order = contested_ids[np.argsort(-size[contested_ids])]
        for r in order:
            if size[r] < UNLABELED_BLOB * sil_area:
                break
            top2 = np.argsort(-labeled[r])[:2] + 1
            contested.append({
                "bbox": _bbox(robjs[r - 1]), "pixels": int(size[r]),
                "parts": [int(top2[0]), int(top2[1])],
                "shares": [round(float(labeled[r, top2[0] - 1] / region_total[r]), 2),
                           round(float(labeled[r, top2[1] - 1] / region_total[r]), 2)]})
            if len(contested) >= 8:
                break
    T.lap("6_measure")

    stats = {
        "regions": int(n),
        "silhouette_px": sil_area,
        "unlabeled_frac": round(float(unlabeled.sum()) / sil_area, 4),
        "mixed_frac": round(float((mixed & inside).sum()) / sil_area, 4),
        "contested_frac": round(float(size[contested_ids].sum()) / sil_area, 4)
        if len(contested_ids) else 0.0,
        "islands_removed": islands_removed[1:].tolist(),
        "specks_reassigned": n_specks,
        "unlabeled_blobs": blobs,
        "contested_regions": contested,
    }
    return {"label": label, "sam": sam, "inside": inside,
            "unlabeled": unlabeled, "stats": stats, "timings": T.times}


def part_geometry(label, n_parts):
    """Per part: pixels, bbox [x, y, w, h], connected components, and the
    centroid of its largest component (where the overlay writes its name)."""
    out = []
    objs = ndi.find_objects(label, max_label=n_parts)
    counts = np.bincount(label.ravel(), minlength=n_parts + 1)
    for i in range(1, n_parts + 1):
        sl = objs[i - 1] if i - 1 < len(objs) else None
        if sl is None or counts[i] == 0:
            out.append(None)
            continue
        sub = label[sl] == i
        comp, nc = ndi.label(sub)
        sizes = np.bincount(comp.ravel(), minlength=nc + 1)
        sizes[0] = 0
        big = int(sizes.argmax())
        cy, cx = ndi.center_of_mass(comp == big)
        out.append({"pixels": int(counts[i]), "bbox": _bbox(sl),
                    "components": int(nc),
                    "label_at": [int(cx + sl[1].start), int(cy + sl[0].start)]})
    return out
