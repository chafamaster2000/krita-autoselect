"""Build the self-contained Krita plugin (AI Select) from this repo.

The plugin carries the daemon (server.py, engine.py, download_weights.py,
requirements.txt) in kritaautoselect/daemon/. Those files are copied from
the repo root at build time — the root stays the only source of truth, and
the copies never live in git.

Usage:
  python scripts/build_plugin.py                 # -> dist/kritaautoselect-<ver>.zip
  python scripts/build_plugin.py --install       # copy into Krita's pykrita dir
  python scripts/build_plugin.py --install DIR   # ... or into DIR

The zip is what Krita's Tools > Scripts > "Import Python Plugin from File..."
expects: kritaautoselect.desktop + kritaautoselect/ (with directory entries).
"""
import argparse
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


def default_pykrita():
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("APPDATA", ""), "krita", "pykrita")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/krita/pykrita")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "krita", "pykrita")


def install(pykrita):
    dest = os.path.join(pykrita, NAME)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for arc, src in plugin_files():
        target = os.path.join(pykrita, *arc.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the AI Select Krita plugin")
    p.add_argument("--out", default=os.path.join(ROOT, "dist"),
                   help="directory for the zip (default: dist/)")
    p.add_argument("--install", nargs="?", const=default_pykrita(),
                   metavar="PYKRITA_DIR",
                   help="copy the plugin into Krita's pykrita directory")
    args = p.parse_args(argv)
    if args.install:
        print("installed:", install(args.install))
    else:
        print("built:", build_zip(args.out))


if __name__ == "__main__":
    main()
