"""Parts vocabulary: load, merge and validate auto-flats configs.

A config is the built-in taxonomy (default.yaml) plus any number of layers
on top, each one a character or drawing file that `extends` the previous:

    default.yaml  <-  characters/santa.yaml  <-  santa_02.flats.yaml

Every file is YAML (or JSON). The merge is by part name, and the part list
order is the priority (later = on top). Pure Python: no numpy, no torch.
"""
import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "default.yaml")
APP_DIR_NAME = "krita-autoselect"
EXTS = (".yaml", ".yml", ".json")

PART_FIELDS = {
    "name", "category", "prompts", "color", "min_score", "instances",
    "relative", "max_area", "min_island", "alternatives", "required",
    "speck_exempt", "above", "below", "enabled", "slot", "weak",
}
PROMPT_FIELDS = {"text", "box", "threshold", "instances", "relative",
                 "exclude", "max_area", "min_score"}
TOP_FIELDS = {
    "name", "extends", "description", "silhouette", "line_threshold",
    "gap_close", "group_name", "lineart_name", "parts", "remove", "only",
    "palette", "checks", "notes",
}
CATEGORIES = {"skin", "garment", "footwear", "accessory", "hair", "face",
              "object", "other"}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class ConfigError(ValueError):
    """A config file is malformed; the message says where and why."""


# ----- files -----

def parse_text(text, origin="<config>"):
    """YAML (if PyYAML is available) or JSON text -> dict."""
    text = text or ""
    if not text.strip():
        return {}
    try:
        import yaml
    except ImportError:
        yaml = None
    try:
        data = yaml.safe_load(text) if yaml else json.loads(text)
    except Exception as e:
        raise ConfigError(f"{origin}: cannot parse ({e})") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{origin}: the top level must be a mapping")
    return data


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_text(f.read(), origin=path)


def default_data_dir(platform=None, environ=None):
    """Same per-user folder the Krita plugin installs into."""
    environ = os.environ if environ is None else environ
    platform = platform or sys.platform
    home = environ.get("HOME") or os.path.expanduser("~")
    if platform.startswith("win"):
        base = environ.get("LOCALAPPDATA") or os.path.join(
            environ.get("USERPROFILE", home), "AppData", "Local")
    elif platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = environ.get("XDG_DATA_HOME") or os.path.join(
            home, ".local", "share")
    return os.path.join(base, APP_DIR_NAME)


def library_dir(environ=None):
    """Character library: AUTOSELECT_FLATS_DIR or <data dir>/flats/characters."""
    environ = os.environ if environ is None else environ
    return environ.get("AUTOSELECT_FLATS_DIR") or os.path.join(
        default_data_dir(environ=environ), "flats", "characters")


def find_in_library(name, dirs):
    for d in dirs:
        for ext in ("",) + EXTS:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


def list_library(dirs):
    """[(name, path)] of the character files in the library folders."""
    seen, out = set(), []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in EXTS and stem not in seen:
                seen.add(stem)
                out.append((stem, os.path.join(d, fn)))
    return out


class Loader:
    """Resolves `extends` names. `allow_paths` lets a file extend another
    by path (CLI); the daemon only accepts names from the library."""

    def __init__(self, library_dirs=None, allow_paths=True):
        self.library_dirs = list(library_dirs or [library_dir()])
        self.allow_paths = allow_paths

    def load(self, ref, base_dir=None):
        """(dict, origin) for a name or a path."""
        if ref in (None, "", "default"):
            return read_file(DEFAULT_PATH), "default"
        if self.allow_paths:
            cand = ref if os.path.isabs(ref) or base_dir is None else \
                os.path.join(base_dir, ref)
            if os.path.isfile(cand):
                return read_file(cand), os.path.abspath(cand)
        if re.match(r"^[\w .-]+$", ref):
            p = find_in_library(ref, self.library_dirs)
            if p:
                return read_file(p), p
        raise ConfigError(
            f"extends: '{ref}' not found (library: "
            f"{', '.join(self.library_dirs)})")


def _default_base():
    return merge({}, read_file(DEFAULT_PATH), "default")


def resolve(data, loader=None, origin="<config>", base_dir=None, _depth=0):
    """Fully merged, validated config for a dict that may `extends` others
    (a missing `extends` means `default`)."""
    loader = loader or Loader()
    if _depth > 8:
        raise ConfigError(f"{origin}: `extends` chain too deep (cycle?)")
    data = dict(data or {})
    parent_ref = data.pop("extends", None) or "default"
    if parent_ref == "default":
        base, chain = _default_base(), ["default"]
    else:
        parent, parent_origin = loader.load(parent_ref, base_dir)
        pdir = os.path.dirname(parent_origin)
        resolved = resolve(parent, loader, parent_origin, pdir, _depth + 1)
        chain = list(resolved.pop("sources"))
        base = resolved
    if not data:
        return finalize(base, chain)
    return finalize(merge(base, data, origin), chain + [origin])


def load_config(ref=None, overrides=(), loader=None):
    """CLI entry: a character name/path (None = default) plus per-drawing
    files applied on top, in order (their own `extends` is ignored: they
    sit on whatever came before)."""
    loader = loader or Loader()
    if ref in (None, "", "default"):
        cfg = resolve({}, loader, origin="default")
    else:
        data, origin = loader.load(ref)
        cfg = resolve(data, loader, origin, os.path.dirname(origin))
    for path in overrides:
        data = dict(read_file(path))
        data.pop("extends", None)
        sources = cfg.pop("sources")
        cfg = finalize(merge(cfg, data, path), sources + [path])
    return cfg


# ----- merge -----

def _part_list(parts, origin):
    """parts as a list of dicts (accepts a list or a {name: fields} map)."""
    if parts is None:
        return []
    if isinstance(parts, dict):
        out = []
        for name, fields in parts.items():
            fields = dict(fields or {})
            fields["name"] = name
            out.append(fields)
        return out
    if isinstance(parts, list):
        for p in parts:
            if not isinstance(p, dict) or not p.get("name"):
                raise ConfigError(f"{origin}: every part needs a `name`")
        return [dict(p) for p in parts]
    raise ConfigError(f"{origin}: `parts` must be a list or a mapping")


def merge(base, data, origin):
    """`data` on top of `base` (both raw dicts). Returns a new dict."""
    unknown = set(data) - TOP_FIELDS
    if unknown:
        raise ConfigError(f"{origin}: unknown keys {sorted(unknown)} "
                          f"(valid: {sorted(TOP_FIELDS)})")
    out = copy.deepcopy(base)
    parts = [dict(p) for p in out.get("parts", [])]
    for key in ("name", "description", "line_threshold", "gap_close",
                "group_name", "lineart_name", "notes"):
        if key in data:
            out[key] = data[key]
    if "silhouette" in data:
        sil = dict(out.get("silhouette") or {})
        sil.update(data["silhouette"] or {})
        out["silhouette"] = sil
    if "checks" in data:
        chk = dict(out.get("checks") or {})
        chk.update(data["checks"] or {})
        out["checks"] = chk

    names = [p["name"] for p in parts]
    for name in data.get("remove") or []:
        if name not in names:
            raise ConfigError(f"{origin}: remove: no part named '{name}' "
                              f"(have: {', '.join(names)})")
        parts = [p for p in parts if p["name"] != name]
        names = [p["name"] for p in parts]

    for new in _part_list(data.get("parts"), origin):
        bad = set(new) - PART_FIELDS
        if bad:
            raise ConfigError(f"{origin}: part '{new['name']}': unknown "
                              f"fields {sorted(bad)}")
        above, below = new.pop("above", None), new.pop("below", None)
        if new.get("enabled") is False:
            parts = [p for p in parts if p["name"] != new["name"]]
            names = [p["name"] for p in parts]
            continue
        new.pop("enabled", None)
        if new["name"] in names:
            i = names.index(new["name"])
            cur = parts.pop(i)
            cur.update(new)
            new = cur
            if above is None and below is None:
                parts.insert(i, new)
                names = [p["name"] for p in parts]
                continue
        names = [p["name"] for p in parts]
        if above is not None:
            if above not in names:
                raise ConfigError(f"{origin}: part '{new['name']}': above: "
                                  f"no part named '{above}'")
            parts.insert(names.index(above) + 1, new)
        elif below is not None:
            if below not in names:
                raise ConfigError(f"{origin}: part '{new['name']}': below: "
                                  f"no part named '{below}'")
            parts.insert(names.index(below), new)
        else:
            parts.append(new)
        names = [p["name"] for p in parts]

    if data.get("only"):
        keep = list(data["only"])
        missing = [n for n in keep if n not in names]
        if missing:
            raise ConfigError(f"{origin}: only: unknown parts {missing}")
        parts = [p for p in parts if p["name"] in keep]
        names = [p["name"] for p in parts]

    for name, color in (data.get("palette") or {}).items():
        if name not in names:
            raise ConfigError(f"{origin}: palette: no part named '{name}'")
        parts[names.index(name)]["color"] = color
    out["parts"] = parts
    return out


# ----- normalize -----

def _norm_prompt(p, part, origin):
    if isinstance(p, str):
        p = {"text": p}
    if not isinstance(p, dict):
        raise ConfigError(f"{origin}: part '{part}': a prompt must be text "
                          f"or a mapping")
    bad = set(p) - PROMPT_FIELDS
    if bad:
        raise ConfigError(f"{origin}: part '{part}': prompt fields "
                          f"{sorted(bad)} unknown (valid: "
                          f"{sorted(PROMPT_FIELDS)})")
    if not p.get("text") and not p.get("box"):
        raise ConfigError(f"{origin}: part '{part}': a prompt needs `text` "
                          f"or `box`")
    if p.get("box") is not None:
        box = p["box"]
        if not (isinstance(box, list) and len(box) == 4):
            raise ConfigError(f"{origin}: part '{part}': box must be "
                              f"[x, y, w, h]")
    q = dict(p)
    if "exclude" in q:
        q["exclude"] = [int(i) for i in q["exclude"]]
    return q


def _norm_instances(v, where):
    if v in (None, "all"):
        return "all"
    if v == "best":
        return 1
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: instances must be all, best or a number")
    if n < 1:
        raise ConfigError(f"{where}: instances must be >= 1")
    return n


def finalize(cfg, sources):
    """Validate and fill defaults. Returns the config the pipeline uses."""
    origin = sources[-1] if sources else "<config>"
    out = copy.deepcopy(cfg)
    out["sources"] = list(sources)
    out.setdefault("name", "default")
    sil = dict(out.get("silhouette") or {})
    sil["prompts"] = list(sil.get("prompts") or ["person"])
    sil["alternatives"] = list(sil.get("alternatives") or [])
    out["silhouette"] = sil
    out.setdefault("line_threshold", "auto")
    out.setdefault("gap_close", "auto")
    out.setdefault("group_name", "Flats")
    out.setdefault("lineart_name", "Lineart")
    out["checks"] = dict(out.get("checks") or {})
    for key in ("line_threshold", "gap_close"):
        v = out[key]
        if v != "auto" and not isinstance(v, (int, float)):
            raise ConfigError(f"{origin}: {key} must be auto or a number")
    seen = set()
    parts = []
    for p in out.get("parts") or []:
        name = p["name"]
        where = f"{origin}: part '{name}'"
        if name in seen:
            raise ConfigError(f"{where}: duplicated name")
        seen.add(name)
        bad = set(p) - PART_FIELDS
        if bad:
            raise ConfigError(f"{where}: unknown fields {sorted(bad)}")
        color = p.get("color", "#808080")
        if not isinstance(color, str) or not HEX.match(color):
            raise ConfigError(f"{where}: color must be #RRGGBB (got {color!r})")
        cat = p.get("category", "other")
        if cat not in CATEGORIES:
            raise ConfigError(f"{where}: category must be one of "
                              f"{sorted(CATEGORIES)}")
        prompts = [_norm_prompt(x, name, origin) for x in p.get("prompts") or []]
        alts = [_norm_prompt(x, name, origin) for x in p.get("alternatives") or []]
        if not prompts:
            raise ConfigError(f"{where}: needs at least one prompt")
        parts.append({
            "name": str(name),
            "category": cat,
            "prompts": prompts,
            "alternatives": alts,
            "color": color.upper(),
            "min_score": float(p.get("min_score", 0.3)),
            "instances": _norm_instances(p.get("instances"), where),
            "relative": float(p.get("relative", 0.0)),
            "max_area": float(p["max_area"]) if p.get("max_area") else None,
            "min_island": int(p.get("min_island", 600)),
            "required": bool(p.get("required", False)),
            "speck_exempt": bool(p.get("speck_exempt", False)),
            "weak": bool(p.get("weak", False)),
            "slot": ([p["slot"]] if isinstance(p.get("slot"), str)
                     else list(p.get("slot") or [])),
        })
    if not parts:
        raise ConfigError(f"{origin}: no parts left")
    out["parts"] = parts
    return out


def config_from_request(config=None, config_text=None, library_dirs=None):
    """Daemon side. `config_text`: YAML/JSON text, or a list of texts applied
    in order (a character file, then per-drawing overrides on top);
    `config`: a JSON object merged last. `extends` resolves by library name
    only, never a path."""
    loader = Loader(library_dirs, allow_paths=False)
    texts = config_text if isinstance(config_text, list) else \
        ([config_text] if config_text else [])
    layers = [parse_text(t, f"config_text[{i}]") for i, t in enumerate(texts)]
    if config:
        if not isinstance(config, dict):
            raise ConfigError("config must be a JSON object")
        layers.append(config)
    if not layers:
        return resolve({}, loader, origin="default")
    cfg = resolve(layers[0], loader, origin="config_text[0]")
    for i, data in enumerate(layers[1:], start=1):
        data = dict(data)
        data.pop("extends", None)
        sources = cfg.pop("sources")
        cfg = finalize(merge(cfg, data, f"config_text[{i}]"),
                       sources + [f"config_text[{i}]"])
    return cfg


def all_prompts(cfg):
    """Distinct text prompts of the config (silhouette first)."""
    seen, out = set(), []
    for t in cfg["silhouette"]["prompts"]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    for p in cfg["parts"]:
        for q in p["prompts"]:
            t = q.get("text")
            if t and not q.get("box") and t not in seen:
                seen.add(t)
                out.append(t)
    return out


TEMPLATE = """\
# Auto-flats character file. Everything here is optional except `extends`.
# Reuse it for every drawing of this character:
#   python -m autoflats DIBUJOS/ --character {name}
# or pick it in Krita (AI Select docker -> Auto-flats).
name: {name}
extends: default

# Parts this character does NOT have (layer names from the default list):
# remove: [Falda, Vestido, Capa, Armadura]

# Change or add parts. Order = priority: `above: X` puts a part on top of X.
# parts:
#   Chaleco: {{prompts: [vest, coat], color: "#C62828"}}
#   Guantes: {{instances: best}}
#   Medias: {{prompts: [sock cuff], above: Calzado}}

# Colours of this character, reused in every drawing:
# palette:
#   Piel: "#F2C29B"
#   Pelo: "#E4E4EA"
"""
