"""Lógica pura del backend de AI Select: sin Krita, sin Qt, solo stdlib.

Todo lo que el plugin decide sobre el daemon SAM 3 vive acá para poder
testearlo sin Krita ni torch:

- rutas por sistema operativo (carpeta de datos, python del venv, pesos);
- cómo encontrar una instalación existente (repo de desarrollo o la que
  instaló el propio plugin);
- reusar un daemon que ya responde vs. arrancar uno propio;
- política de reinicio tras un crash (sin loops);
- el plan de instalación (uv o python -m venv, torch CUDA o CPU, pesos).

El manejo de procesos e hilos está en manager.py; la UI en __init__.py.
"""
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys

__version__ = "0.2.0"

SERVICE_NAME = "krita-autoselect"
APP_DIR_NAME = "krita-autoselect"
DEFAULT_PORT = 5679
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"
DEFAULT_UNLOAD_AFTER_S = 300
MODEL_ID = "facebook/sam3"
# Puertos donde suele estar ComfyUI: el default de ComfyUI y el del server
# gestionado de Krita AI Diffusion en la máquina donde nació este repo.
COMFYUI_FALLBACK_URLS = ("http://127.0.0.1:8188", "http://127.0.0.1:8001")
# Archivos del daemon que el zip del plugin lleva adentro (carpeta daemon/).
DAEMON_FILES = ("server.py", "engine.py", "download_weights.py",
                "requirements.txt")
# Lo mínimo que tiene que haber en una carpeta de pesos formato HF.
WEIGHT_MARKERS = ("config.json", "model.safetensors")
VENV_DIR_NAMES = (".venv", "venv")
PYTHON_FOR_UV = "3.12"
MIN_SYSTEM_PYTHON = (3, 10)
# Sin uv, en Linux/macOS: el python3 del sistema y, si no sirve (viejo o sin
# ensurepip), versiones explícitas que suelen convivir (deadsnakes, etc.).
POSIX_PYTHON_NAMES = ("python3", "python3.12", "python3.13", "python3.11",
                      "python3.14", "python3.10")
# Debian/Ubuntu separan ensurepip en python3-venv: sin él, `-m venv` falla
# y deja un venv a medias (bin/python sin pip).
PYTHON_PROBE_CODE = (
    "import sys\n"
    "try:\n"
    "    import ensurepip, venv\n"
    "    ok = 1\n"
    "except ImportError:\n"
    "    ok = 0\n"
    "print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], ok))\n"
)
VENV_HINT = ("en Debian/Ubuntu instalá python3-venv (sudo apt install "
             "python3-venv), o instalá uv (https://docs.astral.sh/uv/)")

TORCH_INDEX = "https://download.pytorch.org/whl/"
# cu128 ya no trae kernels para Maxwell/Pascal (compute capability < 7.0);
# para esas placas, cu126 (mismo criterio que Krita AI Diffusion).
TORCH_CUDA_DEFAULT = "cu128"
TORCH_CUDA_LEGACY = "cu126"

VERIFY_CODE = (
    "import torch, transformers\n"
    "from transformers import Sam3Model\n"
    "cuda = torch.cuda.is_available()\n"
    "print('torch', torch.__version__, '| transformers', "
    "transformers.__version__, '| cuda', cuda, "
    "torch.cuda.get_device_name(0) if cuda else '')\n"
)


# ----------------------------------------------------------------------------
# Plataforma y rutas
# ----------------------------------------------------------------------------

def platform_id(platform=None):
    """'windows' | 'macos' | 'linux' (cualquier otro POSIX cuenta como linux)."""
    platform = platform or sys.platform
    if platform in ("windows", "macos", "linux"):
        return platform
    if platform.startswith("win"):
        return "windows"
    if platform == "darwin":
        return "macos"
    return "linux"


def _home(environ):
    return (environ.get("HOME") or environ.get("USERPROFILE")
            or os.path.expanduser("~"))


def _join(plat, *parts):
    """Join con el separador del SO destino (para testear Windows en Linux)."""
    sep = "\\" if plat == "windows" else "/"
    out = parts[0].rstrip("\\/") if len(parts) > 1 else parts[0]
    for part in parts[1:]:
        out += sep + part.strip("\\/")
    return out


def default_data_dir(platform=None, environ=None):
    """Carpeta por usuario donde el plugin instala el backend y deja logs."""
    plat = platform_id(platform)
    environ = os.environ if environ is None else environ
    home = _home(environ)
    if plat == "windows":
        base = environ.get("LOCALAPPDATA") or _join(plat, home, "AppData", "Local")
    elif plat == "macos":
        base = _join(plat, home, "Library", "Application Support")
    else:
        base = environ.get("XDG_DATA_HOME") or _join(plat, home, ".local", "share")
    return _join(plat, base, APP_DIR_NAME)


def log_dir(data_dir, platform=None):
    return _join(platform_id(platform), data_dir, "logs")


def venv_python(venv_dir, platform=None):
    """Intérprete dentro de un venv: Scripts/python.exe vs bin/python."""
    plat = platform_id(platform)
    if plat == "windows":
        return _join(plat, venv_dir, "Scripts", "python.exe")
    return _join(plat, venv_dir, "bin", "python")


def find_venv_python(install_dir, platform=None, isfile=os.path.isfile):
    """Python del venv de una instalación (.venv del repo o venv), o None."""
    if not install_dir:
        return None
    plat = platform_id(platform)
    for name in VENV_DIR_NAMES:
        candidate = venv_python(_join(plat, install_dir, name), plat)
        if isfile(candidate):
            return candidate
    return None


def weights_complete(path, isfile=os.path.isfile):
    return bool(path) and all(
        isfile(os.path.join(path, name)) for name in WEIGHT_MARKERS)


def default_weights_dir(install_dir, platform=None):
    return _join(platform_id(platform), install_dir, "models", "sam3")


def resolve_weights(setting, install_dir, platform=None, isfile=os.path.isfile):
    """Carpeta de pesos a pasarle al daemon (AUTOSELECT_WEIGHTS_PATH).

    El setting manda si apunta a algo; si no, <install>/models/sam3 si está
    completa; si no, None (el daemon baja de Hugging Face a su caché)."""
    if setting:
        return setting
    if install_dir:
        candidate = default_weights_dir(install_dir, platform)
        if weights_complete(candidate, isfile):
            return candidate
    return None


def find_daemon_file(name, bundled_dir, install_dir, isfile=os.path.isfile):
    """Un archivo del daemon: primero la copia que trae el plugin (misma
    versión que la UI), después la instalación (repo de desarrollo)."""
    candidates = []
    if bundled_dir:
        candidates.append(os.path.join(bundled_dir, name))
    if install_dir:
        candidates.append(os.path.join(install_dir, name))
        candidates.append(os.path.join(install_dir, "scripts", name))
    for candidate in candidates:
        if isfile(candidate):
            return candidate
    return None


def venv_broken(install_dir, platform=None, isfile=os.path.isfile):
    """¿Hay un venv (pyvenv.cfg) cuyo intérprete ya no existe? Pasa en Linux
    cuando el venv apuntaba a /usr/bin/python3.X y la distro actualizó Python."""
    if not install_dir or find_venv_python(install_dir, platform, isfile):
        return False
    plat = platform_id(platform)
    return any(isfile(_join(plat, install_dir, name, "pyvenv.cfg"))
               for name in VENV_DIR_NAMES)


class Install:
    """Instalación resuelta: qué python corre qué server con qué pesos."""

    def __init__(self, install_dir, python, server_script, weights,
                 venv_broken=False):
        self.install_dir = install_dir
        self.python = python
        self.server_script = server_script
        self.weights = weights
        self.venv_broken = venv_broken

    @property
    def ok(self):
        return bool(self.python and self.server_script)

    def problem(self):
        """Motivo legible (UI en castellano) si no está lista, o None."""
        if not self.python and self.venv_broken:
            return (f"El entorno Python de {self.install_dir} está roto: su "
                    "intérprete ya no existe (¿se actualizó el Python del "
                    "sistema?). «Instalar backend» lo recrea.")
        if not self.python:
            return (f"No hay entorno Python en {self.install_dir} "
                    "(falta .venv). Usá «Instalar backend» o apuntá a una "
                    "instalación existente.")
        if not self.server_script:
            return "No encuentro server.py del backend (reinstalá el plugin)."
        return None

    def key(self):
        return (self.python, self.server_script, self.weights)


def resolve_install(install_dir, weights_setting, bundled_dir, platform=None,
                    isfile=os.path.isfile):
    return Install(
        install_dir=install_dir,
        python=find_venv_python(install_dir, platform, isfile),
        server_script=find_daemon_file("server.py", bundled_dir, install_dir,
                                       isfile),
        weights=resolve_weights(weights_setting, install_dir, platform, isfile),
        venv_broken=venv_broken(install_dir, platform, isfile),
    )


# ----------------------------------------------------------------------------
# URL y health
# ----------------------------------------------------------------------------

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


def parse_url(url):
    """(host, port) de 'http://host:port[/...]'. Port por defecto 80."""
    rest = (url or "").strip().split("://", 1)[-1]
    hostport = rest.partition("/")[0]
    if hostport.startswith("["):  # IPv6 literal
        host, _, port = hostport[1:].partition("]")
        port = port.lstrip(":")
    else:
        host, _, port = hostport.partition(":")
    try:
        port = int(port) if port else 80
    except ValueError:
        raise ValueError(f"Puerto inválido en la URL: {url!r}")
    return host or "127.0.0.1", port


def is_local_host(host):
    return host in LOCAL_HOSTS


class HealthProbe:
    """Resultado de GET /health. kind: 'autoselect' | 'other' | 'down'."""

    def __init__(self, kind, data=None, error=""):
        self.kind = kind
        self.data = data or {}
        self.error = error

    @property
    def loaded(self):
        return bool(self.data.get("loaded") or self.data.get("tracker_loaded"))

    @property
    def loading(self):
        return bool(self.data.get("loading"))

    def __repr__(self):
        return f"HealthProbe({self.kind!r}, {self.data!r}, {self.error!r})"


def classify_health(status, body):
    """Respuesta HTTP de /health → HealthProbe. Cualquier otra cosa que
    conteste en el puerto es 'other' (puerto ocupado por otro programa)."""
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return HealthProbe("other", error=f"HTTP {status}, no es JSON")
    if isinstance(data, dict) and data.get("service") == SERVICE_NAME:
        return HealthProbe("autoselect", data)
    return HealthProbe("other", error=f"HTTP {status}: {str(body)[:80]}")


def probe_health(url, timeout=2.0):
    try:
        host, port = parse_url(url)
    except ValueError as e:
        return HealthProbe("down", error=str(e))
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read(65536)
            return classify_health(resp.status, body)
        finally:
            conn.close()
    except (OSError, http.client.HTTPException) as e:
        return HealthProbe("down", error=f"{type(e).__name__}: {e}")


def request_error(exc, timeout):
    """Error de red de un pedido al daemon → dict para el docker. "_conn"
    (reintentar tras el reinicio) solo si la conexión se cayó: un timeout es
    el daemon todavía calculando (sin GPU, SAM 3 en CPU tarda minutos) y
    reintentar encolaría el mismo trabajo otra vez."""
    if isinstance(exc, socket.timeout):
        return {"error": f"El backend no respondió en {timeout:.0f} s (sin "
                         "GPU, SAM 3 en CPU puede tardar minutos). Probá de "
                         "nuevo cuando termine."}
    return {"error": f"Sin conexión con el server ({exc})", "_conn": True}


# Acciones de decide_action().
REUSE = "reuse"            # ya hay un daemon sano en la URL: usarlo
START = "start"            # arrancar uno propio
WAIT = "wait"              # el propio está arrancando: esperar
PORT_BUSY = "port_busy"    # otro programa ocupa el puerto
REMOTE_DOWN = "remote_down"  # URL no local que no contesta: no se puede arrancar
NOT_INSTALLED = "not_installed"


def decide_action(health_kind, url_is_local, installed, owned_alive):
    """Reusar vs arrancar. Un daemon que contesta siempre se reusa (sea
    propio o ajeno); solo se arranca uno si nadie contesta, la URL es local
    y hay una instalación."""
    if health_kind == "autoselect":
        return REUSE
    if health_kind == "other":
        return PORT_BUSY
    if owned_alive:
        return WAIT
    if not url_is_local:
        return REMOTE_DOWN
    if not installed:
        return NOT_INSTALLED
    return START


# ----------------------------------------------------------------------------
# Política de reinicio
# ----------------------------------------------------------------------------

# Salidas "pedidas": no son crashes (exit limpio, SIGTERM/SIGINT de alguien).
_INTENTIONAL_EXIT = {0}
for _sig in ("SIGTERM", "SIGINT"):
    if hasattr(signal, _sig) and os.name != "nt":
        _INTENTIONAL_EXIT.add(-getattr(signal, _sig))


class RestartPolicy:
    """Decide si relanzar el daemon cuando muere.

    torch puede morir con un crash nativo bajo presión de memoria (ver
    run_daemon.ps1): eso se reinicia. Lo que NO se reinicia: una salida
    pedida, un daemon que murió antes de contestar /health (error de
    arranque: reintentar no lo arregla) y más de `max_crashes` crashes en
    `window_s` segundos (loop)."""

    def __init__(self, max_crashes=3, window_s=600.0, base_delay_s=2.0):
        self.max_crashes = max_crashes
        self.window_s = window_s
        self.base_delay_s = base_delay_s
        self.crashes = []

    def on_exit(self, returncode, was_healthy, now, stop_requested=False):
        """→ (restart: bool, delay_s: float, reason: str)."""
        if stop_requested:
            return False, 0.0, "stopped"
        if returncode in _INTENTIONAL_EXIT:
            return False, 0.0, "exited"
        if not was_healthy:
            return False, 0.0, "startup_failed"
        self.crashes = [t for t in self.crashes if now - t < self.window_s]
        self.crashes.append(now)
        if len(self.crashes) > self.max_crashes:
            return False, 0.0, "crash_loop"
        return True, self.base_delay_s * len(self.crashes), "crash"


def describe_exit(returncode):
    if returncode is None:
        return "sigue corriendo"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except (ValueError, AttributeError):
            name = f"señal {-returncode}"
        return f"terminado por {name}"
    if returncode > 0xFFFF:  # NTSTATUS de Windows (p.ej. 0xC0000005)
        return f"código 0x{returncode:08X}"
    return f"código {returncode}"


# ----------------------------------------------------------------------------
# Entorno de los procesos hijos
# ----------------------------------------------------------------------------

_PATH_LIST_VARS = ("PATH", "LD_LIBRARY_PATH", "QT_PLUGIN_PATH",
                   "PYTHONPATH", "XDG_DATA_DIRS")
# Variables que Krita (o el AppImage) setea para SU Python embebido y que
# romperían el Python del venv.
_DROP_VARS = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
              "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONPLATLIBDIR",
              "LD_PRELOAD", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV")


def krita_prefixes(environ=None, executable=None):
    """Carpetas propias de Krita cuyas entradas en PATH/LD_LIBRARY_PATH no
    deben heredar los hijos (AppImage montado; carpeta de instalación)."""
    environ = os.environ if environ is None else environ
    executable = sys.executable if executable is None else executable
    prefixes = []
    if environ.get("APPDIR"):
        prefixes.append(environ["APPDIR"])
    base = os.path.basename(executable or "").lower()
    if base.startswith("krita"):
        prefixes.append(os.path.dirname(os.path.dirname(executable)))
    return [p.rstrip("\\/") for p in prefixes if p]


def _under(path, prefixes):
    norm = os.path.normcase(path)
    return any(norm == os.path.normcase(p)
               or norm.startswith(os.path.normcase(p) + os.sep)
               or norm.startswith(os.path.normcase(p) + "/")
               for p in prefixes)


def sanitize_env(environ, prefixes=()):
    """Copia del entorno sin el Python de Krita ni rutas del AppImage."""
    env = {k: v for k, v in environ.items() if k not in _DROP_VARS}
    for var in _PATH_LIST_VARS:
        if var not in env:
            continue
        parts = [p for p in env[var].split(os.pathsep)
                 if p and not _under(p, prefixes)]
        if parts:
            env[var] = os.pathsep.join(parts)
        else:
            del env[var]
    return env


def daemon_env(base_env, port, weights, unload_after_s, comfyui_url,
               hf_token=""):
    env = dict(base_env)
    env.update({
        "AUTOSELECT_PORT": str(port),
        "AUTOSELECT_UNLOAD_AFTER_S": str(unload_after_s),
        "AUTOSELECT_COMFYUI_URL": comfyui_url or "",
        # El daemon se muere solo si el plugin (su stdin) desaparece.
        "AUTOSELECT_PARENT_WATCH": "stdin",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    if weights:
        env["AUTOSELECT_WEIGHTS_PATH"] = weights
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


# ----------------------------------------------------------------------------
# ComfyUI (liberar VRAM antes de cargar SAM 3)
# ----------------------------------------------------------------------------

def ai_diffusion_settings_path(platform=None, environ=None):
    """settings.json de Krita AI Diffusion (para leer su URL de ComfyUI)."""
    plat = platform_id(platform)
    environ = os.environ if environ is None else environ
    home = _home(environ)
    if plat == "windows":
        base = environ.get("APPDATA") or _join(plat, home, "AppData", "Roaming")
        return _join(plat, base, "krita", "ai_diffusion", "settings.json")
    if plat == "macos":
        return _join(plat, home, "Library", "Application Support", "krita",
                     "ai_diffusion", "settings.json")
    base = environ.get("XDG_DATA_HOME") or _join(plat, home, ".local", "share")
    return _join(plat, base, "krita", "ai_diffusion", "settings.json")


def read_ai_diffusion_url(path):
    """URL de ComfyUI configurada en AI Diffusion, o None."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    url = data.get("server_url") if isinstance(data, dict) else None
    if not url or not isinstance(url, str):
        return None
    return normalize_url(url)


def normalize_url(url):
    url = url.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


def comfyui_candidates(setting, ai_diffusion_url=None):
    """URLs a probar, en orden. Un setting explícito se usa tal cual."""
    if setting:
        return [normalize_url(setting)]
    out = []
    for url in ((ai_diffusion_url,) + COMFYUI_FALLBACK_URLS):
        if url and url not in out:
            out.append(url)
    return out


def probe_comfyui(url, timeout=1.0):
    """¿Hay un ComfyUI en esa URL? (GET /system_stats, solo lectura)."""
    try:
        host, port = parse_url(url)
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/system_stats")
            resp = conn.getresponse()
            data = json.loads(resp.read(65536))
            return resp.status == 200 and isinstance(data, dict) and \
                "system" in data
        finally:
            conn.close()
    except Exception:
        return False


def pick_comfyui_url(candidates, probe=probe_comfyui):
    """Primer candidato que contesta como ComfyUI. Si ninguno contesta
    (ComfyUI apagado o reiniciándose) se queda con el primero: el pedido de
    /free es best-effort y falla sin consecuencias."""
    for url in candidates:
        if probe(url):
            return url
    return candidates[0] if candidates else ""


# ----------------------------------------------------------------------------
# Hugging Face
# ----------------------------------------------------------------------------

def hf_token_paths(environ=None):
    """Dónde deja el token `hf auth login`, en el orden de huggingface_hub
    (HF_TOKEN_PATH, HF_HOME, XDG_CACHE_HOME), y al final siempre
    ~/.cache/huggingface/token: en Krita Flatpak XDG_CACHE_HOME apunta al
    sandbox (~/.var/app/org.kde.krita/cache) pero el login se hizo afuera."""
    environ = os.environ if environ is None else environ
    home = _home(environ)

    def expand(path):  # huggingface_hub hace expanduser
        return home + path[1:] if path == "~" or path.startswith("~/") \
            else path

    paths = []
    if environ.get("HF_TOKEN_PATH"):
        paths.append(expand(environ["HF_TOKEN_PATH"]))
    if environ.get("HF_HOME"):
        paths.append(os.path.join(expand(environ["HF_HOME"]), "token"))
    if environ.get("XDG_CACHE_HOME"):
        paths.append(os.path.join(expand(environ["XDG_CACHE_HOME"]),
                                  "huggingface", "token"))
    default = os.path.join(home, ".cache", "huggingface", "token")
    if default not in paths:
        paths.append(default)
    return paths


def find_hf_token(setting, environ=None, read=None):
    """(token, origen) — setting del plugin, variable de entorno o el login
    de `hf auth login`. (None, None) si no hay ninguno."""
    environ = os.environ if environ is None else environ
    if setting and setting.strip():
        return setting.strip(), "configuración del plugin"
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if environ.get(var, "").strip():
            return environ[var].strip(), f"variable {var}"

    def _read(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    read = read or _read
    for path in hf_token_paths(environ):
        try:
            token = read(path).strip()
        except OSError:
            continue
        if token:
            return token, f"login de Hugging Face ({path})"
    return None, None


# ----------------------------------------------------------------------------
# Instalador
# ----------------------------------------------------------------------------

class PlanError(RuntimeError):
    """El plan de instalación no se puede armar (mensaje para la UI)."""


class Step:
    """Un paso del instalador: un comando, o un paso salteado (resume).
    hint: qué hacer si falla (se suma al mensaje de error)."""

    def __init__(self, key, title, argv=None, env=None, skip_reason=None,
                 hint=None):
        self.key = key
        self.title = title
        self.argv = argv
        self.env = env or {}
        self.skip_reason = skip_reason
        self.hint = hint

    @property
    def skipped(self):
        return self.skip_reason is not None

    def __repr__(self):
        return f"Step({self.key!r}, argv={self.argv!r}, skip={self.skip_reason!r})"


def _which(name, environ, prefixes, which):
    found = which(name, path=environ.get("PATH"))
    if found and not _under(found, prefixes):
        return found
    return None


def find_uv(environ=None, platform=None, which=shutil.which,
            isfile=os.path.isfile, prefixes=()):
    """uv en PATH o en sus lugares de instalación habituales (Krita lanzado
    desde el menú no siempre tiene ~/.local/bin en el PATH)."""
    environ = os.environ if environ is None else environ
    plat = platform_id(platform)
    exe = "uv.exe" if plat == "windows" else "uv"
    found = _which(exe, environ, prefixes, which)
    if found:
        return found
    home = _home(environ)
    candidates = [_join(plat, home, ".local", "bin", exe),
                  _join(plat, home, ".cargo", "bin", exe)]
    if plat == "windows" and environ.get("LOCALAPPDATA"):
        candidates.append(_join(plat, environ["LOCALAPPDATA"], "Programs",
                                "uv", exe))
    if plat == "macos":
        candidates += ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
    for candidate in candidates:
        if isfile(candidate) and not _under(candidate, prefixes):
            return candidate
    return None


def find_system_python(environ=None, platform=None, which=shutil.which,
                       isfile=os.path.isfile, prefixes=()):
    """argv de un Python del sistema para `-m venv` (nunca el de Krita)."""
    environ = os.environ if environ is None else environ
    plat = platform_id(platform)
    if plat == "windows":
        launcher = _which("py.exe", environ, prefixes, which)
        if launcher:
            return [launcher, "-3"]
        for name in ("python.exe", "python3.exe"):
            found = _which(name, environ, prefixes, which)
            # El python.exe de WindowsApps es un atajo a la Store, no Python.
            if found and "windowsapps" not in found.lower():
                return [found]
        return None
    found = _which("python3", environ, prefixes, which)
    if found:
        return [found]
    for candidate in ("/usr/bin/python3", "/usr/local/bin/python3",
                      "/opt/homebrew/bin/python3"):
        if isfile(candidate) and not _under(candidate, prefixes):
            return [candidate]
    return None


def check_venv_python(argv, run=subprocess.run, env=None):
    """¿Sirve este Python para `-m venv` + pip? → (versión | None, problema).
    problema es "" si sirve: ≥ MIN_SYSTEM_PYTHON y con ensurepip. env: el
    entorno ya saneado (sin el PYTHONPATH de Krita), como el del instalador."""
    exe = argv[0]
    try:
        proc = run(list(argv) + ["-c", PYTHON_PROBE_CODE], capture_output=True,
                   text=True, timeout=30, env=env)
        ver, ok = proc.stdout.split()[:2]
        version = tuple(int(x) for x in ver.split("."))
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{exe}: no se pudo ejecutar ({e})"
    except (AttributeError, ValueError):
        return None, f"{exe}: no responde como Python"
    name = "%d.%d" % version
    if version < MIN_SYSTEM_PYTHON:
        return version, (f"{exe} es Python {name}; hace falta "
                         f"{'%d.%d' % MIN_SYSTEM_PYTHON} o más nuevo")
    if ok != "1":
        return version, (f"{exe} (Python {name}) no trae ensurepip: en "
                         f"Debian/Ubuntu falta el paquete python{name}-venv")
    return version, ""


def pick_system_python(environ=None, platform=None, which=shutil.which,
                       isfile=os.path.isfile, prefixes=(), run=subprocess.run):
    """(argv | None, problema): el primer Python del sistema que puede crear
    un venv con pip. En Linux/macOS prueba python3 y python3.X (lo corre:
    versión y ensurepip). En Windows es find_system_python, sin probar."""
    environ = os.environ if environ is None else environ
    plat = platform_id(platform)
    if plat == "windows":
        return find_system_python(environ, plat, which, isfile, prefixes), ""
    dirs = ["/usr/bin", "/usr/local/bin"]
    if plat == "macos":
        dirs.append("/opt/homebrew/bin")
    candidates = []
    for name in POSIX_PYTHON_NAMES:
        found = _which(name, environ, prefixes, which)
        candidates += [found] if found else []
        candidates += [d + "/" + name for d in dirs
                       if isfile(d + "/" + name)
                       and not _under(d + "/" + name, prefixes)]
    seen, problems = set(), []
    for path in candidates:
        real = os.path.realpath(path)  # python3 -> python3.12: probar 1 vez
        if real in seen:
            continue
        seen.add(real)
        _, problem = check_venv_python([path], run, dict(environ))
        if not problem:
            return [path], ""
        problems.append(problem)
    return None, "; ".join(problems)


def parse_nvidia_smi(output):
    """Salida de `nvidia-smi --query-gpu=name,compute_cap --format=csv,
    noheader` → {"name", "compute_cap": (major, minor) | None} de la 1ª GPU."""
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, cap = line.partition(",")
        compute_cap = None
        try:
            major, _, minor = cap.strip().partition(".")
            compute_cap = (int(major), int(minor or 0))
        except ValueError:
            pass
        return {"name": name.strip(), "compute_cap": compute_cap}
    return None


def detect_nvidia_gpu(environ=None, platform=None, which=shutil.which,
                      run=subprocess.run, prefixes=()):
    """Info de la GPU NVIDIA, o None si no hay (o no hay driver)."""
    environ = os.environ if environ is None else environ
    plat = platform_id(platform)
    exe = "nvidia-smi.exe" if plat == "windows" else "nvidia-smi"
    smi = _which(exe, environ, prefixes, which)
    if not smi and plat == "windows":
        candidate = os.path.join(environ.get("SystemRoot", r"C:\Windows"),
                                 "System32", exe)
        smi = candidate if os.path.isfile(candidate) else None
    if not smi:
        return None
    flags = 0x08000000 if plat == "windows" else 0  # CREATE_NO_WINDOW
    for query in ("name,compute_cap", "name"):
        try:
            proc = run([smi, f"--query-gpu={query}", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=15,
                       creationflags=flags)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode == 0:
            return parse_nvidia_smi(proc.stdout)
    return None


def torch_variant(platform=None, gpu=None):
    """(index_url | None, etiqueta) para instalar torch."""
    plat = platform_id(platform)
    if plat == "macos":
        return None, "PyPI (Metal/CPU)"
    if gpu:
        cap = gpu.get("compute_cap")
        tag = TORCH_CUDA_LEGACY if cap and cap < (7, 0) else TORCH_CUDA_DEFAULT
        cuda = tag[2:4] + "." + tag[4:]
        return TORCH_INDEX + tag, f"CUDA {cuda} ({gpu.get('name') or 'NVIDIA'})"
    return TORCH_INDEX + "cpu", "CPU (sin GPU NVIDIA)"


def is_flatpak(environ=None, isfile=os.path.isfile):
    """¿Krita corre dentro de Flatpak (org.kde.krita)?"""
    environ = os.environ if environ is None else environ
    return bool(environ.get("FLATPAK_ID")) or isfile("/.flatpak-info")


FLATPAK_NOTE = (
    "el backend se instala y corre DENTRO del sandbox de Flatpak. Ahí no se "
    "ve la GPU NVIDIA (ni nvidia-smi ni /dev/nvidia*; Krita Flatpak no puede "
    "lanzar procesos fuera con flatpak-spawn --host), así que va PyTorch CPU "
    "(lento). Para usar CUDA: instalá y corré el daemon fuera de Flatpak "
    "(README, sección Linux) y el plugin lo reusa por la URL.")
AMD_NOTE = (
    "hay una GPU AMD (/dev/kfd) pero el instalador solo arma CUDA o CPU: va "
    "PyTorch CPU. Para ROCm, reinstalá torch en el venv con el índice ROCm "
    "de pytorch.org (pip install torch torchvision --index-url "
    "https://download.pytorch.org/whl/rocmX.Y); el daemon lo usa igual.")


def install_notes(platform=None, environ=None, gpu=None,
                  exists=os.path.exists):
    """[(título, texto)]: avisos del entorno para el log del instalador."""
    if platform_id(platform) != "linux":
        return []
    notes = []
    if is_flatpak(environ, exists):
        notes.append(("Krita Flatpak", FLATPAK_NOTE))
    if not gpu and exists("/dev/kfd"):
        notes.append(("GPU AMD", AMD_NOTE))
    return notes


def build_install_plan(install_dir, bundled_dir, platform=None, uv=None,
                       system_python=None, gpu=None, weights_setting="",
                       hf_token=None, hf_token_from_setting=False,
                       isfile=os.path.isfile, venv_exists=None,
                       python_problem=""):
    """Lista de Step para instalar (o retomar) el backend en install_dir.

    Cada paso es idempotente: el venv existente se saltea, uv/pip no
    reinstalan lo que ya está y el descargador de pesos retoma archivos a
    medias. Por eso re-ejecutar el instalador es seguro.

    system_python solo hace falta sin uv y sin venv usable; python_problem
    explica por qué no se encontró uno (pick_system_python). El token HF
    encontrado siempre se le pasa al descargador (hf_token_from_setting
    quedó sin efecto): en Flatpak el hijo no ve el login de afuera."""
    plat = platform_id(platform)
    if not install_dir:
        raise PlanError("Falta la carpeta de instalación.")
    requirements = find_daemon_file("requirements.txt", bundled_dir,
                                    install_dir, isfile)
    downloader = find_daemon_file("download_weights.py", bundled_dir,
                                  install_dir, isfile)
    if not requirements or not downloader:
        raise PlanError("El plugin no trae los archivos del backend "
                        "(carpeta daemon/). Reinstalalo desde el zip.")
    venv_dir = _join(plat, install_dir, ".venv")
    existing = find_venv_python(install_dir, plat, isfile)
    py = existing or venv_python(venv_dir, plat)
    steps = []
    if not uv and not system_python and not existing:
        why = f" ({python_problem})" if python_problem else ""
        raise PlanError("No encontré uv ni un Python del sistema que pueda "
                        f"crear el entorno{why}. Instalá uv (recomendado, "
                        "trae su propio Python: https://docs.astral.sh/uv/) o "
                        "Python 3.10+ con venv (Debian/Ubuntu: sudo apt "
                        "install python3-venv) y volvé a intentar.")

    # 1. Entorno Python
    if existing:
        steps.append(Step("venv", "Entorno Python",
                          skip_reason=f"ya existe ({existing})"))
        # Sin uv hace falta pip adentro: un venv de uv no lo trae, y uno de
        # `python3 -m venv` que falló por falta de ensurepip quedó sin él.
        bin_dir = existing.rsplit("\\" if plat == "windows" else "/", 1)[0]
        pip_path = _join(plat, bin_dir,
                         "pip.exe" if plat == "windows" else "pip")
        if not uv and not isfile(pip_path):
            steps.append(Step("pip", "Instalar pip en el entorno (ensurepip)",
                              [py, "-m", "ensurepip", "--upgrade",
                               "--default-pip"], hint=VENV_HINT))
    elif uv:
        steps.append(Step("venv", f"Crear entorno Python {PYTHON_FOR_UV} (uv)",
                          [uv, "venv", "--python", PYTHON_FOR_UV,
                           "--allow-existing", venv_dir]))
    elif venv_broken(install_dir, plat, isfile):
        # `-m venv` sobre un venv cuyo python desapareció no lo arregla
        # (deja bin/python colgando): hay que recrearlo.
        steps.append(Step("venv", "Recrear entorno Python roto "
                          "(python -m venv --clear)",
                          list(system_python) + ["-m", "venv", "--clear",
                                                 venv_dir], hint=VENV_HINT))
    else:
        steps.append(Step("venv", "Crear entorno Python (python -m venv)",
                          list(system_python) + ["-m", "venv", venv_dir],
                          hint=VENV_HINT))

    def pip(*args):
        if uv:
            return [uv, "pip", "install", "--python", py, *args]
        return [py, "-m", "pip", "install", "--disable-pip-version-check",
                *args]

    # 2. torch (CUDA si hay GPU NVIDIA)
    index_url, label = torch_variant(plat, gpu)
    torch_args = ["torch", "torchvision"]
    if index_url:
        torch_args += ["--index-url", index_url]
    steps.append(Step("torch", f"Instalar PyTorch — {label}", pip(*torch_args)))

    # 3. resto de dependencias
    steps.append(Step("deps", "Instalar dependencias (transformers, …)",
                      pip("-r", requirements)))

    # 4. pesos (gated): se saltean si ya hay una carpeta completa
    configured = resolve_weights(weights_setting, install_dir, plat, isfile)
    target = weights_setting or default_weights_dir(install_dir, plat)
    if configured and weights_complete(configured, isfile):
        steps.append(Step("weights", "Pesos de SAM 3",
                          skip_reason=f"ya están en {configured}"))
    else:
        if not hf_token:
            raise PlanError(
                "Para bajar los pesos de SAM 3 hace falta un token de Hugging "
                "Face: aceptá la licencia en https://huggingface.co/"
                "facebook/sam3 y pegá un token en la configuración (o "
                "corré `hf auth login`). Si ya tenés los pesos, poné su "
                "carpeta en «Pesos».")
        env = {"HF_TOKEN": hf_token}
        steps.append(Step("weights", "Descargar pesos de SAM 3 (~3.4 GB, "
                          "se puede retomar)",
                          [py, "-u", downloader, target], env=env))

    # 5. verificación rápida (importa torch + Sam3Model)
    steps.append(Step("verify", "Verificar instalación",
                      [py, "-c", VERIFY_CODE]))
    return steps


# ----------------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------------

def rotate_log(path, max_bytes=5 * 1024 * 1024):
    """Si el log pasa de max_bytes, lo mueve a <path>.1 (una sola copia)."""
    try:
        if os.path.getsize(path) > max_bytes:
            os.replace(path, path + ".1")
    except OSError:
        pass


def tail_file(path, max_lines=40, max_bytes=64 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-max_lines:])


def last_error_line(text):
    """La línea más útil de un log para mostrar en el estado del docker."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for line in reversed(lines):
        low = line.lower()
        if ("error" in low or "exception" in low or "address already" in low
                or "no module named" in low):
            return line[:240]
    return lines[-1][:240] if lines else ""
