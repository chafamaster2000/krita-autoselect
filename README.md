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

## Design

- **Lazy VRAM**: the model loads on first request and unloads after an idle
  timeout (default 5 min, configurable, `0` = stay resident). Before loading,
  if VRAM is scarce it asks ComfyUI (`/free`) to drop its cached models —
  this daemon is designed to *coexist* with image generation on one GPU.
- **One model**: SAM 3 does both concept prompts (text) and visual prompts.
  Click points are converted to tiny positive/negative exemplar boxes.
- **stdlib HTTP** on `127.0.0.1` only, browser-origin requests rejected,
  optional shared token. The heavy deps (torch, transformers) are only
  imported when a segmentation actually runs.

## Install

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt
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
.venv/Scripts/python scripts/download_weights.py   # -> models/sam3
AUTOSELECT_WEIGHTS_PATH=models/sam3 .venv/Scripts/python server.py
```

Fully offline alternative: set `AUTOSELECT_WEIGHTS_PATH` to a local
directory containing the model files.

## Run

```bash
.venv/Scripts/python server.py
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
| `HF_TOKEN` | — | standard HF auth for the gated weights |

## API

### `GET /health`

```json
{"status": "ok", "service": "krita-autoselect", "model": "facebook/sam3",
 "loaded": false, "device": null, "unload_after_s": 300.0}
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

- At least one prompt (`text`, `points`, `box`) is required; they compose.
- `points` are clicks; `point_labels` 1 = include (default), 0 = exclude.
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

Contract tests run with stdlib only (no torch, no GPU, no weights):

```bash
python -m unittest discover -s tests -v
```
