"""Weights downloader that tolerates broken proxies.

Some AV/proxy setups mangle the ETag header (it arrives masked as "****…"),
which breaks huggingface_hub's cache filenames on Windows (WinError 123).
This script bypasses that: it lists the repo files, downloads each over
plain authenticated HTTPS with Range-based resume, and never uses the ETag.

Safe to re-run: complete files (size checked against the repo listing) are
skipped and interrupted ones resume from their `.part`.

Usage:
  python scripts/download_weights.py [target_dir] [--only FILE ...]

Then point the daemon at it:
  AUTOSELECT_WEIGHTS_PATH=<target_dir>
"""
import argparse
import os
import sys
import time
from urllib.parse import urljoin

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO = os.environ.get("AUTOSELECT_MODEL", "facebook/sam3")
CHUNK = 1024 * 1024
MAX_ATTEMPTS = 60
PROGRESS_EVERY_S = 5.0
# Final .part -> file rename: on Windows an antivirus or the search indexer
# briefly holds a freshly written file open and the rename fails with
# PermissionError (WinError 5/32). Retry for ~10 s instead of failing the
# install at the very end of a 3.4 GB download.
REPLACE_ATTEMPTS = 20
REPLACE_DELAY_S = 0.5
# Original-format checkpoint, redundant with model.safetensors (~3.4 GB saved).
SKIP_FILES = {"sam3.pt"}
# Written once every file is in place; the Krita plugin only treats the
# folder as complete with it (same name as backend.WEIGHTS_DONE_MARKER).
DONE_MARKER = ".autoselect-complete"
# Retrying can't fix these: no access (gated license / token) or no file.
FATAL_STATUS = (401, 403, 404)

GATED_HELP = (
    "facebook/sam3 is gated on Hugging Face: (1) accept the license at "
    "https://huggingface.co/facebook/sam3, (2) authenticate (HF_TOKEN or "
    "`hf auth login`), (3) fine-grained tokens need 'Access public gated "
    "repositories' enabled at https://huggingface.co/settings/tokens."
)


def _resolve_cdn_url(client, url, headers):
    """Follow the huggingface.co redirect by hand: auth goes to the hub only,
    never to the CDN (a presigned CDN URL rejects foreign Authorization)."""
    r = client.get(url, headers=headers, follow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308):
        return urljoin(url, r.headers["Location"])  # may be relative
    r.raise_for_status()
    return url  # served directly (small files)


def _fmt(n):
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MB"


def _replace(src, dst):
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            time.sleep(REPLACE_DELAY_S)


def download_file(client, url, headers, dest, expected_size=None):
    tmp = dest + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected_size is not None and offset > expected_size:
            os.remove(tmp)  # corrupt leftover: start over
            offset = 0
        if expected_size is not None and offset == expected_size:
            break
        try:
            final_url = _resolve_cdn_url(client, url, headers)
            req_headers = {}
            if final_url == url:
                req_headers.update(headers)
            if offset:
                req_headers["Range"] = f"bytes={offset}-"
                print(f"  resuming at {_fmt(offset)}", flush=True)
            with client.stream("GET", final_url, headers=req_headers,
                               follow_redirects=True) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                resumed = bool(offset) and r.status_code == 206
                done = offset if resumed else 0
                last = time.monotonic()
                with open(tmp, "ab" if resumed else "wb") as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last >= PROGRESS_EVERY_S:
                            last = time.monotonic()
                            pct = (f" ({100 * done / expected_size:.0f}%)"
                                   if expected_size else "")
                            total = f"/{_fmt(expected_size)}" if expected_size else ""
                            print(f"  {_fmt(done)}{total}{pct}", flush=True)
            break
        except (httpx.HTTPError, OSError) as e:
            if isinstance(e, httpx.HTTPStatusError) and \
                    e.response.status_code in FATAL_STATUS:
                raise
            print(f"  retry {attempt}/{MAX_ATTEMPTS} at "
                  f"{os.path.getsize(tmp) if os.path.exists(tmp) else 0} "
                  f"bytes: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
    else:
        raise RuntimeError(f"Gave up downloading {url}")
    if expected_size is not None and os.path.getsize(tmp) != expected_size:
        raise RuntimeError(f"{dest}: got {os.path.getsize(tmp)} bytes, "
                           f"expected {expected_size}")
    _replace(tmp, dest)


def list_files(repo):
    """[(name, size | None)] — sizes let re-runs detect truncated files."""
    api = HfApi()
    try:
        return [(f.path, getattr(f, "size", None))
                for f in api.list_repo_tree(repo, recursive=True)
                if type(f).__name__ != "RepoFolder"]
    except AttributeError:  # very old huggingface_hub
        return [(name, None) for name in api.list_repo_files(repo)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", nargs="?", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "sam3"))
    parser.add_argument("--only", action="append", default=[],
                        help="download just this repo file (repeatable)")
    args = parser.parse_args(argv)

    target = args.target
    os.makedirs(target, exist_ok=True)
    marker = os.path.join(target, DONE_MARKER)
    headers = build_hf_headers()  # picks up the stored login / HF_TOKEN
    try:
        files = list_files(REPO)
    except Exception as e:
        print(f"ERROR: cannot list {REPO}: {type(e).__name__}: {e}", flush=True)
        print(GATED_HELP, flush=True)
        return 1
    if args.only:
        files = [(n, s) for n, s in files if n in args.only]
    print(f"{REPO}: {len(files)} files -> {target}", flush=True)
    with httpx.Client(timeout=60) as client:
        for name, size in files:
            if name in SKIP_FILES and name not in args.only:
                print(f"- {name} (skipped)", flush=True)
                continue
            dest = os.path.join(target, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and (size is None
                                         or os.path.getsize(dest) == size):
                print(f"- {name} (already there)", flush=True)
                continue
            print(f"- {name}" + (f" ({_fmt(size)})" if size else ""),
                  flush=True)
            try:
                download_file(client, hf_hub_url(REPO, name), headers, dest,
                              size)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    print(f"ERROR: {e}\n{GATED_HELP}", flush=True)
                    return 1
                if e.response.status_code == 404:
                    print(f"ERROR: {name} not found in {REPO}: {e}",
                          flush=True)
                    return 1
                raise
    if not args.only:
        with open(marker, "w") as f:
            f.write(REPO + "\n")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
