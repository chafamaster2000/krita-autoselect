# krita-autoselect

Local **SAM 3** segmentation daemon. Send it an image plus a prompt — a text
concept ("the red car"), click points, a bounding box, or a combination — and
it returns a canvas-sized grayscale mask (PNG) with per-instance metadata.

Built as the segmentation backend for [`kri select sam`](https://github.com/chafamaster2000/krita-ai-cli)
(AI-assisted selections in Krita), but Krita-independent: it segments any
PNG/JPEG over plain local HTTP.

```
kri select sam "the red car"          Krita canvas → daemon → Krita selection
curl -X POST :5679/segment -d ...     any image → mask
```

## Krita plugin: AI Select (standalone)

`krita-plugin/kritaautoselect` is a Krita docker that needs nothing else: no
`kri`, no terminal, no other plugin. Type what to select ("the red car") or
click an object; the plugin finds or starts this daemon by itself, the way
Krita AI Diffusion manages its local ComfyUI.

**Install the plugin** (Krita 5.2+; the code also runs on Krita 6 / PyQt6):

```bash
python scripts/build_plugin.py            # -> dist/kritaautoselect-<version>.zip
```

In Krita: *Tools → Scripts → Import Python Plugin from File…* → pick the zip,
enable it, restart Krita, then *Settings → Dockers → AI Select*. (For
development, `python scripts/build_plugin.py --install` copies it straight
into your `pykrita` folder.) The zip carries `server.py`, `engine.py`,
`download_weights.py` and `requirements.txt` from this repo's root, copied
at build time; the root files remain the only source.

**Set up the backend** (gear icon in the docker):

- *Already have an install?* Point **Instalación** at it. A folder with a
  `.venv` (`bin/python` or `Scripts\python.exe`) works, and
  `models/sam3` inside it is used as the weights automatically. This repo
  folder itself qualifies.
- *Starting fresh?* Click **Instalar backend**. It creates a venv in the
  per-user data folder (`~/.local/share/krita-autoselect`,
  `%LOCALAPPDATA%\krita-autoselect`, `~/Library/Application Support/krita-autoselect`)
  with `uv` if available, otherwise the system `python3 -m venv`. It then
  installs torch (CUDA 12.8 wheels with an NVIDIA GPU, 12.6 for pre-Volta,
  CPU otherwise) and the requirements, and downloads the gated weights (see
  below) using the **Token HF** field or your `hf auth login`. Progress
  shows in the docker; **Cancelar** stops it, and running it again resumes.
  If you already have the weights, set **Pesos** to their folder and the
  download is skipped.

**Use it:** type a prompt and press *Seleccionar* (or Shift+Enter). If a
healthy daemon already answers on the configured URL it is reused and never
stopped by the plugin. Otherwise the plugin starts its own, which:

- logs to `<data folder>/logs/daemon.log` (**Ver log** in the docker);
- is restarted if it crashes (torch can die natively under memory pressure),
  but not after a startup failure or 3 crashes in 10 minutes;
- stops when Krita quits, and also when Krita dies hard (it watches its
  stdin pipe);
- can start with Krita (**Iniciar el backend al abrir Krita**).

The header shows the backend state: not installed, stopped, starting,
loading the model, ready, or error with the reason. The other settings are
the idle seconds before VRAM is released and the ComfyUI URL asked to free
VRAM (blank = auto: AI Diffusion's configured server, then ports 8188 and
8001). Settings live in `kritarc`, under `[kritaautoselect]`.

## Design

- **Lazy VRAM**: the model loads on first request and unloads after an idle
  timeout (default 5 min, configurable, `0` = stay resident). Before loading,
  if VRAM is scarce it asks ComfyUI (`/free`) to drop its cached models —
  this daemon is designed to *coexist* with image generation on one GPU.
- **One checkpoint, two heads**: a text prompt routes to SAM 3's *concept*
  head (PCS — every instance matching "the red car"); a clicks/box-only
  prompt routes to the *tracker* head (PVS, SAM2-style — the exact object
  under your click). With text, points/box act as concept refiners instead.
- **stdlib HTTP** on `127.0.0.1` only, browser-origin requests rejected,
  optional shared token. The heavy deps (torch, transformers) are only
  imported when a segmentation actually runs.

## Install (manual)

The Krita plugin can do all of this for you (see above). By hand:

```bash
python -m venv .venv
# Windows: .venv/Scripts/pip …   Linux/macOS: .venv/bin/pip …
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

### Weights (gated)

`facebook/sam3` is gated on Hugging Face:

1. Accept the license at <https://huggingface.co/facebook/sam3>.
2. Authenticate on this machine: `hf auth login` (or set `HF_TOKEN`).
3. Fine-grained token? Enable **"Access public gated repositories"** in its
   settings at <https://huggingface.co/settings/tokens> — without it,
   downloads fail with 403 even after accepting the license.

First segmentation downloads the weights (~3.4 GB) to the HF cache. If your
network mangles ETags (some AV/proxies do — symptoms: `WinError 123` with a
`****` filename), use the bundled resilient downloader instead:

```bash
.venv/bin/python scripts/download_weights.py   # -> models/sam3 (resumable)
AUTOSELECT_WEIGHTS_PATH=models/sam3 .venv/bin/python server.py
```

Fully offline alternative: set `AUTOSELECT_WEIGHTS_PATH` to a local
directory containing the model files.

## Run

```bash
.venv/bin/python server.py
# [autoselect] listening on http://127.0.0.1:5679 (model loads on first request)
```

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOSELECT_PORT` | `5679` | HTTP port |
| `AUTOSELECT_MODEL` | `facebook/sam3` | Hugging Face model id |
| `AUTOSELECT_WEIGHTS_PATH` | — | local model dir (skips HF entirely) |
| `AUTOSELECT_DEVICE` | auto | `cuda` / `cpu` |
| `AUTOSELECT_DTYPE` | auto | `bfloat16` (cuda) / `float16` / `float32` |
| `AUTOSELECT_UNLOAD_AFTER_S` | `300` | idle seconds before freeing VRAM; `0` = resident |
| `AUTOSELECT_COMFYUI_URL` | `http://127.0.0.1:8001` | ComfyUI to ask for VRAM; empty disables |
| `AUTOSELECT_TOKEN` | — | require `X-Autoselect-Token` header |
| `AUTOSELECT_PARENT_WATCH` | — | `stdin`: exit when stdin closes (set by the Krita plugin) |
| `HF_TOKEN` | — | standard HF auth for the gated weights |

## API

### `GET /health`

```json
{"status": "ok", "service": "krita-autoselect", "model": "facebook/sam3",
 "loaded": false, "loading": false, "device": null, "unload_after_s": 300.0}
```

### `POST /segment`

```json
{
  "image_b64": "<base64 PNG/JPEG>",
  "text": "the red car",
  "points": [[400, 300]],
  "point_labels": [1],
  "box": [100, 100, 300, 200],
  "threshold": 0.5,
  "mask_threshold": 0.5,
  "combine": "union",
  "list_only": false
}
```

- At least one prompt (`text`, `points`, `box`) is required.
- With `text`: concept mode — all matching instances; `points`/`box` refine
  the concept (1 = include, 0 = exclude).
- Without `text`: object mode — `points` are clicks on ONE object
  (`point_labels` 0 = "not this part"), `box` surrounds the object; returns
  that exact object's mask.
- `box` is `[x, y, width, height]` in pixels.
- `combine`: `"union"` merges every instance above `threshold` into one mask;
  an integer selects a single instance (0 = best score).
- `list_only: true` returns metadata only (no mask) — inspect, then decide.

Response:

```json
{"status": "ok", "width": 1024, "height": 768, "count": 2,
 "instances": [{"index": 0, "score": 0.97, "box": [412.0, 288.5, 590.2, 401.0]},
               {"index": 1, "score": 0.81, "box": [102.0, 95.3, 260.8, 240.1]}],
 "mask_b64": "<base64 grayscale PNG, canvas-sized>"}
```

Errors come as `{"error": "..."}` with status 400 (bad request) or 500
(model/weights failures — the message tells you what to fix).

## Tests

Contract tests and the plugin's backend logic run with stdlib only (no
Krita, no torch, no GPU, no weights):

```bash
python -m unittest discover -s tests -v
```

`tests/test_download_resume.py` also needs `httpx` + `huggingface_hub` and
is skipped without them (run it with the backend venv's Python).
