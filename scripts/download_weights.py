"""Weights downloader that tolerates broken proxies.

Some AV/proxy setups mangle the ETag header (it arrives masked as "****…"),
which breaks huggingface_hub's cache filenames on Windows (WinError 123).
This script bypasses that: it lists the repo files, downloads each over
plain authenticated HTTPS with Range-based resume, and never uses the ETag.

Usage:
  python scripts/download_weights.py [target_dir]

Then point the daemon at it:
  AUTOSELECT_WEIGHTS_PATH=<target_dir>
"""
import os
import sys
import time

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO = os.environ.get("AUTOSELECT_MODEL", "facebook/sam3")
CHUNK = 1024 * 1024
MAX_ATTEMPTS = 60


def _resolve_cdn_url(client, url, headers):
    """Follow the huggingface.co redirect by hand: auth goes to the hub only,
    never to the CDN (a presigned CDN URL rejects foreign Authorization)."""
    r = client.get(url, headers=headers, follow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308):
        return r.headers["Location"]
    r.raise_for_status()
    return url  # served directly (small files)


def download_file(client, url, headers, dest):
    tmp = dest + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            final_url = _resolve_cdn_url(client, url, headers)
            req_headers = {}
            if final_url == url:
                req_headers.update(headers)
            if offset:
                req_headers["Range"] = f"bytes={offset}-"
            with client.stream("GET", final_url, headers=req_headers,
                               follow_redirects=True) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if offset and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
            break
        except (httpx.HTTPError, OSError) as e:
            print(f"  retry {attempt}/{MAX_ATTEMPTS} at "
                  f"{os.path.getsize(tmp) if os.path.exists(tmp) else 0} "
                  f"bytes: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
    else:
        raise RuntimeError(f"Gave up downloading {url}")
    os.replace(tmp, dest)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "sam3")
    os.makedirs(target, exist_ok=True)
    headers = build_hf_headers()  # picks up the stored login / HF_TOKEN
    files = HfApi().list_repo_files(REPO)
    print(f"{REPO}: {len(files)} files -> {target}", flush=True)
    with httpx.Client(timeout=60) as client:
        for name in files:
            dest = os.path.join(target, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                print(f"- {name} (already there)", flush=True)
                continue
            print(f"- {name}", flush=True)
            download_file(client, hf_hub_url(REPO, name), headers, dest)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
