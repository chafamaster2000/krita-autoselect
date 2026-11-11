"""Build the self-contained Krita plugin (AI Select) from this repo.

The plugin carries the daemon (server.py, engine.py, download_weights.py,
requirements.txt) in kritaautoselect/daemon/. Those files are copied from
the repo root at build time — the root stays the only source of truth, and
the copies never live in git.

Usage:
  python scripts/build_plugin.py                 # -> dist/kritaautoselect-<ver>.zip
  python scripts/build_plugin.py --install       # copy into Krita's pykrita dir
  python scripts/build_plugin.py --install DIR   # ... or into DIR
  python scripts/build_plugin.py --plan-json     # install plan (JSON, no changes)
  python scripts/build_plugin.py --plan-json --uninstall   # uninstall plan
  python scripts/build_plugin.py --uninstall [DIR]  # remove it from pykrita

--plan-json is the component contract of the ecosystem installer
(autopainter/installer): install_component() returns the list of actions
and the installer shows, executes and records them. See the README,
"Instalador del ecosistema".

The zip is what Krita's Tools > Scripts > "Import Python Plugin from File..."
expects: kritaautoselect.desktop + kritaautoselect/ (with directory entries).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_SRC = os.path.join(ROOT, "krita-plugin")
NAME = "kritaautoselect"
# name inside daemon/ -> source path in the repo
DAEMON_SOURCES = {
    "server.py": os.path.join(ROOT, "server.py"),
    "engine.py": os.path.join(ROOT, "engine.py"),
    "download_weights.py": os.path.join(ROOT, "scripts", "download_weights.py"),
    "requirements.txt": os.path.join(ROOT, "requirements.txt"),
}
IGNORED = ("__pycache__", ".pyc", "daemon")


def version():
    with open(os.path.join(PLUGIN_SRC, NAME, "backend.py"), encoding="utf-8") as f:
        return re.search(r'__version__ = "([^"]+)"', f.read()).group(1)


def plugin_files():
    """[(archive_path, source_path)] for every file of the plugin."""
    out = [(f"{NAME}.desktop", os.path.join(PLUGIN_SRC, f"{NAME}.desktop"))]
    base = os.path.join(PLUGIN_SRC, NAME)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED)
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, PLUGIN_SRC).replace(os.sep, "/")
            out.append((rel, src))
    for name, src in DAEMON_SOURCES.items():
        out.append((f"{NAME}/daemon/{name}", src))
    return out


def build_zip(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{NAME}-{version()}.zip")
    files = plugin_files()
    dirs = set()
    for arc, _ in files:
        parts = arc.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]) + "/")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Krita's importer looks for an explicit "kritaautoselect/" entry.
        for d in sorted(dirs):
            zf.writestr(zipfile.ZipInfo(d), "")
        for arc, src in files:
            zf.write(src, arc)
    return path


def _backend():
    """The plugin's backend.py (stdlib only), loaded by path: importing the
    kritaautoselect package would import Krita."""
    mod = sys.modules.get("_autoselect_backend")
    if mod is None:
        path = os.path.join(PLUGIN_SRC, NAME, "backend.py")
        spec = importlib.util.spec_from_file_location("_autoselect_backend",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


def default_pykrita(platform=None, environ=None):
    """Krita's pykrita dir for the user. platform/environ default to this
    machine; passing them lets tests resolve Windows/macOS paths on Linux."""
    bk = _backend()
    plat = bk.platform_id(platform)
    environ = os.environ if environ is None else environ
    home = bk._home(environ, plat)
    if plat == "windows":
        base = environ.get("APPDATA") or bk._join(plat, home, "AppData",
                                                  "Roaming")
        return bk._join(plat, base, "krita", "pykrita")
    if plat == "macos":
        return bk._join(plat, home, "Library", "Application Support",
                        "krita", "pykrita")
    base = environ.get("XDG_DATA_HOME") or bk._join(plat, home, ".local",
                                                    "share")
    return bk._join(plat, base, "krita", "pykrita")


def default_kritarc(platform=None, environ=None):
    """Krita's settings file (where [python] enable_<plugin>=true lives)."""
    bk = _backend()
    plat = bk.platform_id(platform)
    environ = os.environ if environ is None else environ
    home = bk._home(environ, plat)
    if plat == "windows":
        base = environ.get("LOCALAPPDATA") or bk._join(plat, home, "AppData",
                                                       "Local")
        return bk._join(plat, base, "kritarc")
    if plat == "macos":
        return bk._join(plat, home, "Library", "Preferences", "kritarc")
    base = environ.get("XDG_CONFIG_HOME") or bk._join(plat, home, ".config")
    return bk._join(plat, base, "kritarc")


def install(pykrita):
    dest = os.path.join(pykrita, NAME)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for arc, src in plugin_files():
        target = os.path.join(pykrita, *arc.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
    return dest


def uninstall(pykrita):
    """Remove the plugin (folder + .desktop) from pykrita. Returns the
    paths removed. The backend (venv, weights) is left alone."""
    removed = []
    dest = os.path.join(pykrita, NAME)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
        removed.append(dest)
    desktop = os.path.join(pykrita, f"{NAME}.desktop")
    if os.path.isfile(desktop):
        os.remove(desktop)
        removed.append(desktop)
    return removed


# ----------------------------------------------------------------------------
# Component contract for the ecosystem installer (autopainter/installer)
# ----------------------------------------------------------------------------

COMPONENT = "krita-autoselect"
KRITARC_SECTION = "python"
KRITARC_KEY = f"enable_{NAME}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _action(kind, src, dst, op="install", **detail):
    return {"kind": kind, "op": op, "src": src, "dst": dst, "detail": detail}


def _detect(env, plat, bk):
    """Fills what env did not give: system python, GPU, Mac CPU. Only probes
    this machine when the plan is for this machine's platform."""
    environ = env.get("environ")
    environ = os.environ if environ is None else environ
    here = bk.platform_id(None) == plat
    out = {}
    if "system_python" in env:
        out["system_python"], out["python_problem"] = env["system_python"], ""
    elif here:
        out["system_python"], out["python_problem"] = \
            bk.pick_system_python(environ, plat)
    else:
        out["system_python"], out["python_problem"] = None, ""
    if "gpu" in env:
        out["gpu"] = env["gpu"]
    else:
        out["gpu"] = bk.detect_nvidia_gpu(environ, plat) if here else None
    if "mac_cpu" in env:
        out["mac_cpu"] = env["mac_cpu"]
    else:
        out["mac_cpu"] = (bk.macos_cpu() if here and plat == "macos"
                          else bk.MAC_ARM64 if plat == "macos" else None)
    return out


def install_component(plan_only=True, uninstall=False, env=None):
    """Contract of §8: the actions that install (or uninstall) AI Select.

    env (all optional): {"platform": "linux|windows|macos",
    "environ": {...}, "pykrita": dir, "kritarc": file, "data_dir": dir,
    "system_python": [argv] | None, "gpu": {"name", "compute_cap"} | None,
    "mac_cpu": "arm64|rosetta|x86_64", "weights": dir}. What is missing is
    resolved for this machine (or left as a placeholder for another OS).

    Action = {"kind": copy|write|kritarc|venv|pip, "op": install|remove,
    "src", "dst", "detail": {...}}. With plan_only=False the actions are
    also executed here (only what this script can do by itself: copy,
    kritarc and, on uninstall, the removals; venv, pip and the marker
    write, which goes after verifying, are left to the installer). Raises bk.PlanError when the
    machine cannot run the backend (Mac Intel).
    """
    env = dict(env or {})
    bk = _backend()
    plat = bk.platform_id(env.get("platform"))
    environ = env.get("environ")
    environ = os.environ if environ is None else environ
    pykrita = env.get("pykrita") or default_pykrita(plat, environ)
    kritarc = env.get("kritarc") or default_kritarc(plat, environ)
    data_dir = env.get("data_dir") or bk.default_data_dir(plat, environ)
    venv_dir = bk._join(plat, data_dir, ".venv")
    py = bk.venv_python(venv_dir, plat)

    def dst_of(arc):
        return bk._join(plat, pykrita, *arc.split("/"))

    if uninstall:
        actions = [_action("copy", None, dst_of(arc), op="remove")
                   for arc, _ in plugin_files()]
        actions.append(_action("copy", None, dst_of(NAME), op="remove",
                               tree=True))
        actions.append(_action("kritarc", None, kritarc, op="remove",
                               section=KRITARC_SECTION, key=KRITARC_KEY))
        actions.append(_action("write", None, bk.install_marker(data_dir, plat),
                               op="remove"))
        # The venv goes (it is ours and weighs GBs); weights and models in
        # data_dir/models stay, as §8 asks.
        actions.append(_action("venv", None, venv_dir, op="remove",
                               keep=[bk.default_weights_dir(data_dir, plat)]))
        if not plan_only:
            _execute(actions)
        return actions

    found = _detect(env, plat, bk)
    if plat == "macos" and found["mac_cpu"] == bk.MAC_INTEL:
        raise bk.PlanError(bk.MAC_INTEL_PROBLEM)

    actions = [_action("copy", src, dst_of(arc), sha256=_sha256(src))
               for arc, src in plugin_files()]
    actions.append(_action("kritarc", None, kritarc, section=KRITARC_SECTION,
                           key=KRITARC_KEY, value="true"))
    actions.append(_action("venv", None, venv_dir,
                           python=found["system_python"],
                           python_problem=found["python_problem"] or None,
                           prefer_uv=True, uv_python=bk.PYTHON_FOR_UV))
    index_url, label = bk.torch_variant(plat, found["gpu"])
    torch_args = ["torch", "torchvision"]
    if index_url:
        torch_args += ["--index-url", index_url]
    actions.append(_action("pip", None, venv_dir, python=py, args=torch_args,
                           label=f"PyTorch — {label}"))
    requirements = DAEMON_SOURCES["requirements.txt"]
    actions.append(_action("pip", requirements, venv_dir, python=py,
                           args=["-r", requirements],
                           sha256=_sha256(requirements),
                           label="daemon dependencies"))
    verify = bk.VERIFY_CODE
    if found["gpu"] and plat != "macos":
        verify += bk.VERIFY_CUDA_CODE
    # The plugin reads this marker to call the install complete.
    actions.append(_action("write", None, bk.install_marker(data_dir, plat),
                           content="ok\n", verify_python=py,
                           verify_code=verify))
    if not plan_only:
        _execute(actions)
    return actions


def plan_notes(env=None):
    """Prerequisites this component checks but does not install."""
    env = dict(env or {})
    bk = _backend()
    plat = bk.platform_id(env.get("platform"))
    environ = env.get("environ")
    environ = os.environ if environ is None else environ
    data_dir = env.get("data_dir") or bk.default_data_dir(plat, environ)
    setting = env.get("weights") or environ.get("AUTOSELECT_WEIGHTS_PATH", "")
    weights = bk.resolve_weights(setting, data_dir, plat)
    target = setting or bk.default_weights_dir(data_dir, plat)
    notes = []
    if not weights or not bk.weights_complete(weights):
        notes.append({
            "kind": "weights", "ok": False, "dst": target,
            "detail": "Faltan los pesos de SAM 3 (gated en Hugging Face: "
                      "aceptá la licencia en https://huggingface.co/"
                      "facebook/sam3 y pasá HF_TOKEN).",
            "argv": [bk.venv_python(bk._join(plat, data_dir, ".venv"), plat),
                     DAEMON_SOURCES["download_weights.py"], target]})
    else:
        notes.append({"kind": "weights", "ok": True, "dst": weights})
    return notes


def _kritarc_set(path, section, key, value):
    """Set (value=str) or remove (value=None) key in [section] of an INI
    file, touching nothing else (Krita writes its own keys and order)."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    header = f"[{section}]"
    out, in_sec, done, seen_sec = [], False, False, False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_sec and not done and value is not None:
                out.append(f"{key}={value}")
                done = True
            in_sec = stripped == header
            seen_sec = seen_sec or in_sec
        elif in_sec and stripped.split("=", 1)[0].strip() == key:
            if value is not None and not done:
                out.append(f"{key}={value}")
            done = True
            continue
        out.append(line)
    if value is not None and not done:
        if not seen_sec:
            if out and out[-1].strip():
                out.append("")
            out.append(header)
        out.append(f"{key}={value}")
    new = "\n".join(out) + "\n" if out else ""
    if lines == out and os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    return True


def _execute(actions):
    """Run the actions this script can run by itself (see install_component).
    Idempotent: a copy whose destination already has the same hash is
    skipped. Returns how many actions changed something."""
    changed = 0
    for a in actions:
        kind, op, dst, d = a["kind"], a["op"], a["dst"], a["detail"]
        if kind == "copy" and op == "install":
            if os.path.isfile(dst) and _sha256(dst) == d["sha256"]:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(a["src"], dst)
            changed += 1
        elif kind == "copy" and op == "remove":
            if d.get("tree") and os.path.isdir(dst):
                shutil.rmtree(dst)
                changed += 1
            elif os.path.isfile(dst):
                os.remove(dst)
                changed += 1
        elif kind == "kritarc":
            value = d.get("value") if op == "install" else None
            changed += _kritarc_set(dst, d["section"], d["key"], value)
        elif kind == "write" and op == "remove" and os.path.isfile(dst):
            os.remove(dst)
            changed += 1
        elif kind == "venv" and op == "remove" and os.path.isdir(dst):
            shutil.rmtree(dst)
            changed += 1
        # install venv/pip/write(marker): the ecosystem installer runs them
    return changed


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the AI Select Krita plugin")
    p.add_argument("--out", default=os.path.join(ROOT, "dist"),
                   help="directory for the zip (default: dist/)")
    p.add_argument("--install", nargs="?", const=default_pykrita(),
                   metavar="PYKRITA_DIR",
                   help="copy the plugin into Krita's pykrita directory")
    p.add_argument("--plan-json", action="store_true",
                   help="print the install (or, with --uninstall, the "
                        "uninstall) plan as JSON and change nothing")
    p.add_argument("--uninstall", nargs="?", const=default_pykrita(),
                   metavar="PYKRITA_DIR",
                   help="remove the plugin from pykrita (backend venv and "
                        "weights are kept)")
    p.add_argument("--platform", choices=("linux", "windows", "macos"),
                   help="with --plan-json: plan for another OS")
    args = p.parse_args(argv)
    if args.plan_json:
        env = {"platform": args.platform} if args.platform else {}
        if args.uninstall and args.uninstall != default_pykrita():
            env["pykrita"] = args.uninstall
        if args.install and args.install != default_pykrita():
            env["pykrita"] = args.install
        try:
            actions = install_component(plan_only=True,
                                        uninstall=bool(args.uninstall),
                                        env=env)
        except _backend().PlanError as e:
            print(json.dumps({"component": COMPONENT, "error": str(e)}))
            return 2
        out = {"component": COMPONENT, "version": version(),
               "uninstall": bool(args.uninstall), "actions": actions}
        if not args.uninstall:
            out["notes"] = plan_notes(env)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    if args.uninstall:
        removed = uninstall(args.uninstall)
        print("removed:", ", ".join(removed) if removed else "nothing")
    elif args.install:
        print("installed:", install(args.install))
    else:
        print("built:", build_zip(args.out))


if __name__ == "__main__":
    sys.exit(main())
