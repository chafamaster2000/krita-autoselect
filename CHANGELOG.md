# Changelog

Las versiones siguen [semver](https://semver.org/lang/es/). La versión vive en
`__version__` de `krita-plugin/kritaautoselect/backend.py`; de ahí salen el
nombre del zip (`dist/kritaautoselect-<versión>.zip`), el `version` del plan
de instalación (`build_plugin.py --plan-json`) y el tag anotado `vX.Y.Z`.
La API HTTP del daemon tiene su propio número (`api_version` en
`GET /health`).

## [Unreleased]

### Arreglado

- **Mac Intel ya no corta la instalación con `--skip-heavy`.**
  `scripts/build_plugin.py --plan-json --skip-heavy` (o
  `env["skip_heavy"]` en `install_component`) planea sólo el plugin (copias y
  `kritarc`, sin venv, torch, dependencias del daemon ni marcador) en
  cualquier plataforma. En un Mac Intel, en vez de negarse, avisa en `notes`
  (`kind: "mac_intel"`) que SAM 3 no corre ahí. Sin `--skip-heavy` el plan en
  Mac Intel sigue fallando, pero el mensaje sugiere `--skip-heavy`.

## [0.2.0] - 2026-11-11

Primera versión con tag. Junta las ramas `standalone-plugin`, `os-*`
(`os-merged`) y `segment-many`.

### Agregado

- **`POST /segment_many`** (`api_version` 2): muchos prompts sobre la misma
  imagen en un solo pedido, con caché de features de visión por imagen y los
  prompts en lotes. `GET /health` anuncia `capabilities: ["segment",
  "segment_many"]` para que los clientes detecten un daemon viejo sin
  esperar un 404.
- **AI Select autónomo:** el plugin ya no necesita nada más (ni kri ni una
  terminal). Encuentra o arranca el daemon, lo supervisa, y desde el engranaje
  instala el venv con torch y baja los pesos gated de SAM 3, como hace
  AI Diffusion con su ComfyUI. `kritaautoselect.get_manager()` deja que otros
  plugins usen el mismo daemon.
- **Contrato de componente** para el instalador del ecosistema:
  `scripts/build_plugin.py --plan-json [--platform …] [--uninstall]` y
  `install_component(plan_only, uninstall, env)`.
- `scripts/build_plugin.py` arma el zip instalable desde Krita
  (*Import Python Plugin from File…*) o lo copia a pykrita con `--install`.
- Compatibilidad Krita 5 (PyQt5) y Krita 6 (PyQt6) con el mismo código.

### Arreglado

- **Linux:** sin uv (Python del sistema con `ensurepip`), sin GPU NVIDIA
  (CPU con aviso), Krita en Flatpak y dos Kritas abiertos a la vez.
- **Windows:** directorio de trabajo, puerto, `nvidia-smi` y token de
  Hugging Face; el home sale de `USERPROFILE`.
- **macOS:** MPS en Apple Silicon, aviso claro en Mac Intel (no hay PyTorch
  nuevo para x86_64) y el entorno mínimo que da Finder al abrir Krita.

## Antes de 0.2.0 (sin tag)

- 2026-07: daemon SAM 3 (`/segment`) con prompt de texto, puntos y caja; modo
  objeto (click o caja, estilo SAM 2) y arrastre tipo marquesina; descarga
  de pesos resumible; docker AI Select en Krita con selecciones nativas
  (hormigas visibles); `run_daemon.ps1`, supervisor con reinicio automático
  en Windows.
