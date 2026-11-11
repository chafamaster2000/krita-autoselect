"""Line art in: any PNG/JPEG (grey, coloured, transparent) -> grey + ink mask."""
import numpy as np
from PIL import Image

REFERENCE_PIXELS = 2865 * 2604   # the drawing the defaults were tuned on
THRESHOLD_RANGE = (90, 200)


def on_white(image):
    """RGB version of the drawing with transparency flattened onto white
    (what SAM should see; a plain convert("RGB") turns it black)."""
    if image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def to_gray(image):
    """PIL image -> uint8 HxW grey on white. Transparent pixels count as
    paper (line art on a transparent layer is the common case in Krita)."""
    if image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return np.asarray(image.convert("L"), dtype=np.uint8)


def otsu(gray):
    """Otsu's threshold on a uint8 image (pure numpy)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = 0
    return int(between.argmax())


def ink_threshold(gray, setting="auto"):
    """Grey level below which a pixel is ink. `auto` = Otsu clamped to a
    sane range (clean black ink lands ~128-140, a grey pencil sketch higher)."""
    if setting != "auto":
        return int(setting)
    lo, hi = THRESHOLD_RANGE
    return int(min(hi, max(lo, otsu(gray))))


def scale_factor(shape):
    """Linear size relative to the reference drawing (pixel sizes in the
    config are tuned for ~7.5 MP and scale with sqrt(area))."""
    h, w = shape[:2]
    return float(np.sqrt(h * w / REFERENCE_PIXELS))


def ink_stats(gray, line):
    """Numbers that describe the line art, for the report."""
    ink = gray[line]
    return {
        "ink_fraction": round(float(line.mean()), 4),
        "ink_mean_gray": int(ink.mean()) if ink.size else None,
        # mid-grey "ink": pencil, screentone or anti-aliased grey lines
        "gray_ink_fraction": round(float(((gray > 60) & line).sum()
                                         / max(1, line.sum())), 3),
    }
