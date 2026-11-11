"""Files from a pipeline Result: layered .kra, ORA, previews, report.

The .kra is written directly (no Krita needed): a zip with maindoc.xml and
each layer in Krita's tiled format ("VERSION 2", 64x64 tiles). Tiles are
stored with the raw-data flag instead of LZF, which Krita reads as is; the
zip's deflate does the compressing. Only tiles that hold pixels are
written, so a small part costs a few KB.

Layer stack (top to bottom):
    Lineart          paint layer, Multiply, locked   (the original drawing)
    Flats            group
      <part>         paint layer per part, highest priority on top
        Sel <part>   its selection mask
"""
import base64
import colorsys
import io
import json
import uuid
import zipfile
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 64
SRGB_PROFILE = "sRGB-elle-V2-srgbtrc.icc"


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# ----- rasters -----

def flats_rgb(result):
    """HxWx3: each part in its palette colour, white elsewhere."""
    h, w = result.label.shape
    lut = np.full((len(result.parts) + 1, 3), 255, np.uint8)
    for p in result.parts:
        lut[p["index"]] = hex_rgb(p["color"])
    return lut[result.label]


def preview(result):
    """What Krita shows: flats x line art (Multiply)."""
    flat = flats_rgb(result).astype(np.float32)
    out = flat * (result.gray[..., None].astype(np.float32) / 255.0)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def part_rgba_crop(result, part):
    """(RGBA PIL crop, x, y) of one part's layer."""
    x, y, w, h = part["bbox"]
    m = result.label[y:y + h, x:x + w] == part["index"]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[m] = (*hex_rgb(part["color"]), 255)
    return Image.fromarray(rgba, "RGBA"), x, y


def png_bytes(img, level=6):
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=level)
    return buf.getvalue()


# ----- .kra -----

def _tiles(arr, only_nonzero=True, origin=(0, 0)):
    """Krita tile stream for an HxWxC uint8 array (BGRA or alpha) whose
    top-left sits at `origin` (tile-aligned) in the image."""
    h, w, c = arr.shape
    ox, oy = origin
    th, tw = -(-h // TILE), -(-w // TILE)
    pad = np.zeros((th * TILE, tw * TILE, c), np.uint8)
    pad[:h, :w] = arr
    tiles = pad.reshape(th, TILE, tw, TILE, c).transpose(0, 2, 1, 3, 4)
    if only_nonzero:
        used = tiles.reshape(th, tw, -1).any(2)
    else:
        used = np.ones((th, tw), bool)
    size = TILE * TILE * c + 1
    chunks = []
    for ty, tx in zip(*np.nonzero(used)):
        chunks.append(b"%d,%d,LZF,%d\n" % (ox + tx * TILE, oy + ty * TILE,
                                            size))
        chunks.append(b"\x00")  # raw (uncompressed) tile data
        chunks.append(tiles[ty, tx].tobytes())
    head = (b"VERSION 2\nTILEWIDTH %d\nTILEHEIGHT %d\nPIXELSIZE %d\n"
            b"DATA %d\n" % (TILE, TILE, c, int(used.sum())))
    return head + b"".join(chunks)


def _bgra(rgb, alpha):
    h, w = alpha.shape
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0] = rgb[..., 2]
    out[..., 1] = rgb[..., 1]
    out[..., 2] = rgb[..., 0]
    out[..., 3] = alpha
    return out


def _uuid():
    return "{" + str(uuid.uuid4()) + "}"


def kra_bytes(result, doc_name="autoflats", masks=True, level=6):
    """The layered document as .kra bytes."""
    gray = result.gray
    h, w = gray.shape
    cfg = result.cfg
    files = []          # (archive name, bytes)
    counter = [1]

    def fname(prefix):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    layers_dir = f"{doc_name}/layers/"
    # line art: the original grey, opaque
    line_fn = fname("layer")
    rgb = np.repeat(gray[..., None], 3, axis=2)
    files.append((layers_dir + line_fn,
                  _tiles(_bgra(rgb, np.full_like(gray, 255)), False)))
    files.append((layers_dir + line_fn + ".defaultpixel", b"\0\0\0\0"))
    group_fn = fname("layer")
    part_xml = []
    for p in sorted([p for p in result.parts if p.get("pixels")],
                    key=lambda p: -p["index"]):
        x, y, bw, bh = p["bbox"]
        ax, ay = (x // TILE) * TILE, (y // TILE) * TILE
        m = result.label[ay:y + bh, ax:x + bw] == p["index"]
        color = np.zeros(m.shape + (3,), np.uint8)
        color[m] = hex_rgb(p["color"])
        alpha = m.astype(np.uint8) * 255
        lfn = fname("layer")
        files.append((layers_dir + lfn,
                      _tiles(_bgra(color, alpha), origin=(ax, ay))))
        files.append((layers_dir + lfn + ".defaultpixel", b"\0\0\0\0"))
        mask_xml = ""
        if masks:
            mfn = fname("mask")
            files.append((layers_dir + mfn + ".pixelselection",
                          _tiles(alpha[..., None], origin=(ax, ay))))
            files.append((layers_dir + mfn + ".pixelselection.defaultpixel",
                          b"\0"))
            mask_xml = (
                "<masks>"
                f'<mask nodetype="selectionmask" name={quoteattr("Sel " + p["name"])} '
                f'filename="{mfn}" visible="1" locked="0" active="0" '
                f'colorlabel="0" intimeline="0" x="0" y="0" uuid="{_uuid()}"/>'
                "</masks>")
        part_xml.append(
            f'<layer nodetype="paintlayer" name={quoteattr(p["name"])} '
            f'filename="{lfn}" colorspacename="RGBA" compositeop="normal" '
            f'opacity="255" visible="1" locked="0" collapsed="0" '
            f'channelflags="" channellockflags="" colorlabel="0" '
            f'onionskin="0" intimeline="0" x="0" y="0" uuid="{_uuid()}">'
            f"{mask_xml}</layer>")
    maindoc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE DOC PUBLIC '-//KDE//DTD krita 2.0//EN' "
        "'http://www.calligra.org/DTD/krita-2.0.dtd'>\n"
        '<DOC xmlns="http://www.calligra.org/DTD/krita" kritaVersion="5.2.0" '
        'syntaxVersion="2.0" editor="Krita">\n'
        f'<IMAGE name={quoteattr(doc_name)} mime="application/x-kra" '
        f'width="{w}" height="{h}" colorspacename="RGBA" '
        f'profile="{SRGB_PROFILE}" x-res="72" y-res="72" description="">\n'
        "<layers>"
        f'<layer nodetype="paintlayer" name={quoteattr(cfg["lineart_name"])} '
        f'filename="{line_fn}" colorspacename="RGBA" compositeop="multiply" '
        f'opacity="255" visible="1" locked="1" collapsed="0" channelflags="" '
        f'channellockflags="" colorlabel="0" onionskin="0" intimeline="0" '
        f'x="0" y="0" uuid="{_uuid()}"/>'
        f'<layer nodetype="grouplayer" name={quoteattr(cfg["group_name"])} '
        f'filename="{group_fn}" passthrough="0" colorspacename="RGBA" '
        f'compositeop="normal" opacity="255" visible="1" locked="0" '
        f'collapsed="0" channelflags="" colorlabel="0" intimeline="0" '
        f'x="0" y="0" uuid="{_uuid()}"><layers>'
        + "".join(part_xml) +
        "</layers></layer></layers>\n"
        '<ProjectionBackgroundColor ColorData="AAAAAA=="/>\n'
        "</IMAGE>\n</DOC>\n")
    docinfo = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE document-info PUBLIC '-//KDE//DTD document-info 1.1//EN' "
        "'http://www.calligra.org/DTD/document-info-1.1.dtd'>\n"
        '<document-info xmlns="http://www.calligra.org/DTD/document-info">'
        "<about><title>" + doc_name + "</title>"
        "<description>auto-flats</description></about></document-info>\n")
    merged = preview(result)
    thumb = merged.copy()
    thumb.thumbnail((256, 256))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=level) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/x-krita",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("maindoc.xml", maindoc)
        z.writestr("documentinfo.xml", docinfo)
        z.writestr("preview.png", png_bytes(thumb))
        for name, data in files:
            z.writestr(name, data)
        z.writestr("mergedimage.png", png_bytes(merged, 1),
                   compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


# ----- ORA (other apps: GIMP, MyPaint, ...; no selection masks) -----

def ora_bytes(result):
    gray = result.gray
    h, w = gray.shape
    cfg = result.cfg
    buf = io.BytesIO()
    stack = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<image version="0.0.5" w="{w}" h="{h}"><stack>',
             f'<layer name={quoteattr(cfg["lineart_name"])} '
             f'src="data/lineart.png" composite-op="svg:multiply" x="0" y="0"/>',
             f'<stack name={quoteattr(cfg["group_name"])}>']
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "image/openraster",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("data/lineart.png", png_bytes(Image.fromarray(gray, "L")))
        for p in sorted([p for p in result.parts if p.get("pixels")],
                        key=lambda p: -p["index"]):
            img, x, y = part_rgba_crop(result, p)
            src = f"data/part_{p['index']:02d}.png"
            z.writestr(src, png_bytes(img))
            stack.append(f'<layer name={quoteattr(p["name"])} src="{src}" '
                         f'x="{x}" y="{y}"/>')
        stack.append("</stack></stack></image>")
        z.writestr("stack.xml", "\n".join(stack))
        merged = preview(result)
        z.writestr("mergedimage.png", png_bytes(merged))
        thumb = merged.copy()
        thumb.thumbnail((256, 256))
        z.writestr("Thumbnails/thumbnail.png", png_bytes(thumb))
    return buf.getvalue()


# ----- overlay for checking -----

# Distinct, magenta-free colours for the check view (magenta is reserved
# for "no SAM part"), so parts that share a palette colour (skin, hands,
# face) still look different.
ID_COLORS = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (120, 200, 80), (160, 110, 60),
    (100, 100, 230), (255, 130, 110), (60, 200, 200), (170, 170, 40),
    (230, 80, 80), (140, 200, 255), (40, 120, 60), (255, 200, 140),
    (90, 60, 160), (180, 230, 120), (200, 120, 40), (20, 160, 200),
    (250, 170, 60), (110, 150, 110), (190, 60, 40), (150, 150, 220),
]


def id_colors(n):
    out = list(ID_COLORS[:n])
    i = 0
    while len(out) < n:
        h = (i * 0.61803398875) % 1.0
        i += 1
        if 0.75 < h < 0.95:      # skip magentas
            continue
        r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.85)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


FONT_NAMES = ("DejaVuSans.ttf", "NotoSans-Regular.ttf",
              "LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf",
              "Helvetica.ttc", "segoeui.ttf")


def _font(size):
    """A TrueType font with accents (layer names are Spanish: Cinturón);
    Pillow's built-in bitmap font lacks them."""
    for name in FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def overlay(result, max_side=1600):
    """Check view: parts in distinct colours with their names, magenta =
    filled without SAM, yellow boxes = contested regions, red boxes =
    warnings; legend and findings on the right."""
    h, w = result.label.shape
    ids = id_colors(len(result.parts))
    lut = np.full((len(result.parts) + 1, 3), 255, np.uint8)
    for p in result.parts:
        lut[p["index"]] = ids[p["index"] - 1]
    rgb = lut[result.label].astype(np.float32)
    rgb *= result.gray[..., None].astype(np.float32) / 255.0
    rgb = rgb.astype(np.uint8)
    # magenta stripes on interior pixels that had no SAM part
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = ((xx + yy) // max(2, int(6 * max(h, w) / 1600))) % 2 == 0
    rgb[result.unlabeled & stripes] = (255, 0, 255)
    f = min(1.0, max_side / max(h, w))
    img = Image.fromarray(rgb, "RGB")
    if f < 1:
        img = img.resize((int(w * f), int(h * f)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = _font(max(12, int(16 * img.width / 1000)))
    rep = result.report
    for f_ in rep["checks"]:
        for reg in f_.get("regions", []):
            x, y, bw, bh = [v * f for v in reg["bbox"]]
            draw.rectangle([x, y, x + bw, y + bh], outline=(230, 200, 0),
                           width=3)
        if f_["level"] == "warn" and f_.get("bbox"):
            x, y, bw, bh = [v * f for v in f_["bbox"]]
            draw.rectangle([x, y, x + bw, y + bh], outline=(220, 30, 30),
                           width=2)
    placed = []
    for p in sorted(result.parts, key=lambda p: -p.get("pixels", 0)):
        if not p.get("pixels"):
            continue
        x, y = p["label_at"][0] * f, p["label_at"][1] * f
        tw = draw.textlength(p["name"], font=font)
        th = font.size if hasattr(font, "size") else 12
        for _ in range(8):   # nudge down until it doesn't hit a label
            box = (x - tw / 2, y - th / 2, x + tw / 2, y + th / 2)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3]
                       and b[1] < box[3] for b in placed):
                break
            y += th + 2
        placed.append(box)
        draw.text((x, y), p["name"], fill=(0, 0, 0), font=font,
                  anchor="mm", stroke_width=2, stroke_fill=(255, 255, 255))
    # legend
    legend_w = max(360, img.width // 3)
    sheet = Image.new("RGB", (img.width + legend_w, img.height), "white")
    sheet.paste(img, (0, 0))
    d = ImageDraw.Draw(sheet)
    lf = _font(15)
    x0, y = img.width + 12, 10
    d.text((x0, y), f"{rep['image']}  [{rep['verdict']}]  "
           f"cfg: {rep['config']['name']}", fill=(0, 0, 0), font=lf)
    y += 24
    sil = max(1, rep["silhouette"]["pixels"])
    for p in sorted(result.parts, key=lambda p: -p["index"]):
        if not p.get("pixels"):
            continue
        d.rectangle([x0, y, x0 + 16, y + 16], fill=ids[p["index"] - 1],
                    outline=(0, 0, 0))
        d.rectangle([x0 + 20, y, x0 + 36, y + 16], fill=hex_rgb(p["color"]),
                    outline=(0, 0, 0))
        best = max([d_["score"] for q in p["prompts"] for d_ in q["instances"]
                    if d_["used"]] or [0])
        d.text((x0 + 42, y), f"{p['name']}  {p['pixels'] / sil:.1%}  "
               f"s={best:.2f}", fill=(0, 0, 0), font=lf)
        y += 20
    y += 8
    d.text((x0, y), "magenta = no SAM part (guessed)  yellow = contested",
           fill=(120, 0, 120), font=lf)
    y += 24
    for f_ in rep["checks"]:
        color = (200, 0, 0) if f_["level"] == "warn" else (90, 90, 90)
        text = f_["message"]
        while text and y < sheet.height - 18:
            line, text = _wrap(d, text, lf, legend_w - 24)
            d.text((x0, y), line, fill=color, font=lf)
            y += 18
        y += 4
    return sheet


def _wrap(draw, text, font, width):
    words = text.split()
    line = ""
    for i, word in enumerate(words):
        cand = (line + " " + word).strip()
        if draw.textlength(cand, font=font) > width and line:
            return line, " ".join(words[i:])
        line = cand
    return line, ""


# ----- everything for one drawing -----

def write_all(result, out_dir, stem, kra=True, ora=False, parts_png=False,
              masks=True, labels=False):
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    if labels:   # 8-bit label map (value = part index) for tests/tools
        p = os.path.join(out_dir, stem + "_labels.png")
        Image.fromarray(result.label.astype(np.uint8), "L").save(p)
        paths["labels"] = p
    if kra:
        p = os.path.join(out_dir, stem + ".kra")
        with open(p, "wb") as f:
            f.write(kra_bytes(result, doc_name=stem, masks=masks))
        paths["kra"] = p
    if ora:
        p = os.path.join(out_dir, stem + ".ora")
        with open(p, "wb") as f:
            f.write(ora_bytes(result))
        paths["ora"] = p
    p = os.path.join(out_dir, stem + "_preview.png")
    preview(result).save(p, compress_level=3)
    paths["preview"] = p
    p = os.path.join(out_dir, stem + "_overlay.png")
    overlay(result).save(p, compress_level=3)
    paths["overlay"] = p
    if parts_png:
        pdir = os.path.join(out_dir, stem + "_parts")
        os.makedirs(pdir, exist_ok=True)
        for part in result.parts:
            if part.get("pixels"):
                img, x, y = part_rgba_crop(result, part)
                img.save(os.path.join(pdir, f"{part['index']:02d}_{part['name']}.png"))
    result.report["outputs"] = paths
    p = os.path.join(out_dir, stem + "_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=1, ensure_ascii=False)
    paths["report"] = p
    return paths


def parts_payload(result):
    """Per-part RGBA crops (base64 PNG) for the Krita plugin / kri."""
    out = []
    for p in sorted([p for p in result.parts if p.get("pixels")],
                    key=lambda p: -p["index"]):
        img, x, y = part_rgba_crop(result, p)
        out.append({"name": p["name"], "color": p["color"],
                    "index": p["index"], "x": x, "y": y,
                    "width": img.width, "height": img.height,
                    "png_b64": base64.b64encode(png_bytes(img, 3)).decode()})
    return out
