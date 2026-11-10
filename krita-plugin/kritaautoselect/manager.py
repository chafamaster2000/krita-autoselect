"""Ciclo de vida del daemon SAM 3 y del instalador, sin Krita ni Qt.

DaemonManager hace con el daemon lo que Krita AI Diffusion hace con su
ComfyUI "gestionado":

- si ya hay un daemon sano en la URL configurada, lo reusa (y nunca lo
  para: no es suyo);
- si no, lo arranca como proceso hijo con el python del venv, con la salida
  a un archivo de log;
- lo supervisa: si muere por un crash lo relanza, con límite (RestartPolicy);
- al cerrar Krita para el que arrancó. Además el daemon lee su stdin: si
  Krita muere de golpe, el pipe se cierra y el daemon sale solo.

Ningún método bloquea al llamador: todo lo lento corre en hilos y avisa
por callbacks (que la UI convierte en señales Qt).
"""
import os
import subprocess
import threading
import time

try:
    from . import backend as bk
except ImportError:  # tests: el directorio del plugin está en sys.path
    import backend as bk

STARTUP_TIMEOUT_S = 180.0
HEALTH_POLL_S = 0.5
IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


class Config:
    """Foto de la configuración (se arma en el hilo de UI, se lee en hilos)."""

    def __init__(self, url=bk.DEFAULT_URL, install_dir="", weights_setting="",
                 hf_token="", unload_after_s=bk.DEFAULT_UNLOAD_AFTER_S,
                 comfyui_url="", comfyui_free=True, bundled_dir="",
                 data_dir=None):
        self.url = url or bk.DEFAULT_URL
        self.data_dir = data_dir or bk.default_data_dir()
        self.install_dir = install_dir or self.data_dir
        self.weights_setting = weights_setting
        self.hf_token = hf_token
        self.unload_after_s = unload_after_s
        self.comfyui_url = comfyui_url
        self.comfyui_free = comfyui_free
        self.bundled_dir = bundled_dir

    @property
    def log_dir(self):
        return bk.log_dir(self.data_dir)

    @property
    def daemon_log(self):
        return os.path.join(self.log_dir, "daemon.log")

    @property
    def install_log(self):
        return os.path.join(self.log_dir, "install.log")

    def resolve(self):
        return bk.resolve_install(self.install_dir, self.weights_setting,
                                  self.bundled_dir)

    def launch_key(self):
        """Lo que, si cambia, obliga a relanzar el daemon propio."""
        return (self.url, self.install_dir, self.weights_setting,
                self.hf_token, str(self.unload_after_s), self.comfyui_url,
                self.comfyui_free)


class _Session:
    """Un daemon propio (con sus reinicios) bajo un hilo supervisor."""

    def __init__(self, cfg, install):
        self.cfg = cfg
        self.install = install
        self.proc = None
        self.stop_event = threading.Event()
        self.policy = bk.RestartPolicy()
        self.thread = None
        self.stop_reason = None


class DaemonManager:
    # Estados (snapshot["state"]):
    CHECKING = "checking"
    NOT_INSTALLED = "not_installed"
    STOPPED = "stopped"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"

    def __init__(self, cfg=None, popen=subprocess.Popen,
                 probe=bk.probe_health, comfy_probe=bk.probe_comfyui,
                 base_env=None, startup_timeout_s=STARTUP_TIMEOUT_S):
        self._cfg = cfg or Config()
        self._popen = popen
        self._probe = probe
        self._comfy_probe = comfy_probe
        self._base_env = base_env
        self._startup_timeout_s = startup_timeout_s
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._listeners = []
        self._session = None
        self._busy = False
        self._state = {"state": self.CHECKING, "detail": "", "owned": False,
                       "loaded": False, "pid": None,
                       "url": self._cfg.url}

    # ----- config / listeners -----

    @property
    def config(self):
        return self._cfg

    def configure(self, cfg):
        """Nueva config. Si cambió algo que afecta al daemon propio que está
        corriendo, se lo para: se relanza con la config nueva al próximo uso."""
        with self._lock:
            old, self._cfg = self._cfg, cfg
            session = self._session
            changed = old.launch_key() != cfg.launch_key()
        if session and changed:
            self._stop_session(session, "Configuración cambiada: el backend "
                               "se relanza al próximo uso")
        self.refresh()

    def add_listener(self, fn):
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn):
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _set(self, **changes):
        with self._lock:
            self._state.update(changes)
            self._state["url"] = self._cfg.url
            snap = dict(self._state)
            listeners = list(self._listeners)
            self._cond.notify_all()
        for fn in listeners:
            try:
                fn(snap)
            except Exception:
                # Listener muerto (p.ej. docker destruido): se descarta.
                self.remove_listener(fn)

    def set_busy(self, busy):
        """La UI avisa que hay un pedido en vuelo (para 'Cargando modelo')."""
        with self._lock:
            self._busy = bool(busy)

    @property
    def owned_alive(self):
        with self._lock:
            s = self._session
            return bool(s and s.thread and s.thread.is_alive()
                        and not s.stop_event.is_set())

    # ----- estado sin arrancar nada -----

    def refresh(self):
        threading.Thread(target=self._refresh_work, daemon=True).start()

    def _refresh_work(self):
        cfg = self._cfg
        probe = self._probe(cfg.url)
        if probe.kind == "autoselect":
            self._set_ready(probe)
            return
        with self._lock:
            state = self._state["state"]
        if self.owned_alive and state in (self.STARTING, self.ERROR):
            return  # el supervisor es la fuente de verdad mientras arranca
        if probe.kind == "other":
            self._set(state=self.ERROR, owned=False, loaded=False, pid=None,
                      detail=self._port_busy_msg(cfg))
            return
        if self.owned_alive:
            return
        if state == self.ERROR and self._state.get("sticky"):
            return  # error de arranque/crash loop: queda hasta otro intento
        install = cfg.resolve()
        host, _ = bk.parse_url(cfg.url)
        if not bk.is_local_host(host):
            self._set(state=self.ERROR, owned=False, loaded=False, pid=None,
                      detail=f"No responde el server remoto {cfg.url}")
        elif not install.ok:
            self._set(state=self.NOT_INSTALLED, owned=False, loaded=False,
                      pid=None, detail=install.problem())
        else:
            self._set(state=self.STOPPED, owned=False, loaded=False, pid=None,
                      detail="Se inicia solo al seleccionar")

    def _set_ready(self, probe):
        with self._lock:
            owned = self.owned_alive
            busy = self._busy
            pid = self._session.proc.pid if owned and self._session.proc \
                else None
        loading = probe.loading or (busy and not probe.loaded)
        self._set(state=self.LOADING if loading else self.READY,
                  owned=owned, loaded=probe.loaded, pid=pid, sticky=False,
                  device=probe.data.get("device"),
                  detail="" if owned else "daemon existente (no lo inició "
                  "el plugin)")

    @staticmethod
    def _port_busy_msg(cfg):
        _, port = bk.parse_url(cfg.url)
        return (f"El puerto {port} está ocupado por otro programa (no es "
                "krita-autoselect). Cambiá la URL en la configuración.")

    # ----- arrancar / esperar -----

    def ensure_ready(self, callback=None, timeout_s=None):
        """En un hilo: reusar o arrancar el daemon y esperar a que conteste.
        callback(ok: bool, message: str) se llama desde ese hilo."""
        def work():
            ok, msg = self._ensure_ready_blocking(timeout_s)
            if callback:
                try:
                    callback(ok, msg)
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def start(self, callback=None):
        """Botón «Iniciar»: lo mismo que ensure_ready, sin pedido detrás."""
        with self._lock:
            if self._state.get("sticky"):
                self._state["sticky"] = False
        self.ensure_ready(callback)

    def _ensure_ready_blocking(self, timeout_s=None):
        cfg = self._cfg
        probe = self._probe(cfg.url)
        install = cfg.resolve()
        host, _ = bk.parse_url(cfg.url)
        action = bk.decide_action(probe.kind, bk.is_local_host(host),
                                  install.ok, self.owned_alive)
        if action == bk.REUSE:
            self._set_ready(probe)
            return True, ""
        if action == bk.PORT_BUSY:
            msg = self._port_busy_msg(cfg)
            self._set(state=self.ERROR, detail=msg, owned=False, pid=None)
            return False, msg
        if action == bk.REMOTE_DOWN:
            msg = f"No responde el server remoto {cfg.url}"
            self._set(state=self.ERROR, detail=msg, owned=False, pid=None)
            return False, msg
        if action == bk.NOT_INSTALLED:
            msg = install.problem()
            self._set(state=self.NOT_INSTALLED, detail=msg, owned=False,
                      pid=None)
            return False, msg
        if action == bk.START:
            self._start_session(cfg, install)
        return self._wait_ready(timeout_s or self._startup_timeout_s + 5)

    def _wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                state = self._state["state"]
                if state in (self.READY, self.LOADING):
                    return True, ""
                if state in (self.ERROR, self.STOPPED, self.NOT_INSTALLED) \
                        and not self.owned_alive:
                    return False, self._state.get("detail") or state
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, "El backend no respondió a tiempo"
                self._cond.wait(min(left, 1.0))

    def _start_session(self, cfg, install):
        with self._lock:
            if self.owned_alive:
                return
            session = _Session(cfg, install)
            self._session = session
            session.thread = threading.Thread(
                target=self._supervise, args=(session,), daemon=True,
                name="autoselect-supervisor")
            self._state["sticky"] = False
        self._set(state=self.STARTING, owned=True, loaded=False, pid=None,
                  detail="Iniciando backend…")
        session.thread.start()

    def _comfyui_url(self, cfg):
        if not cfg.comfyui_free:
            return ""
        aid = bk.read_ai_diffusion_url(bk.ai_diffusion_settings_path())
        return bk.pick_comfyui_url(
            bk.comfyui_candidates(cfg.comfyui_url, aid), self._comfy_probe)

    def _spawn(self, session):
        cfg, install = session.cfg, session.install
        os.makedirs(cfg.log_dir, exist_ok=True)
        bk.rotate_log(cfg.daemon_log)
        _, port = bk.parse_url(cfg.url)
        base = self._base_env if self._base_env is not None else \
            bk.sanitize_env(os.environ, bk.krita_prefixes())
        # El token se pasa siempre: en Krita Flatpak el daemon no ve el login
        # de `hf auth login` hecho fuera del sandbox (XDG_CACHE_HOME).
        token, _ = bk.find_hf_token(cfg.hf_token, base)
        env = bk.daemon_env(base, port, install.weights, cfg.unload_after_s,
                            self._comfyui_url(cfg), token or "")
        log = open(cfg.daemon_log, "ab")
        try:
            header = (f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"AI Select inicia el daemon: {install.python} -u "
                      f"{install.server_script} (puerto {port}, pesos "
                      f"{install.weights or 'Hugging Face'}, ComfyUI "
                      f"{env['AUTOSELECT_COMFYUI_URL'] or 'off'}) =====\n")
            log.write(header.encode("utf-8"))
            log.flush()
            kwargs = {}
            if IS_WINDOWS:
                kwargs["creationflags"] = CREATE_NO_WINDOW
            return self._popen(
                [install.python, "-u", install.server_script],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                env=env, cwd=os.path.dirname(install.server_script),
                **kwargs)
        finally:
            log.close()  # el hijo tiene su propio descriptor

    def _supervise(self, session):
        cfg = session.cfg
        while not session.stop_event.is_set():
            try:
                proc = self._spawn(session)
            except Exception as e:
                session.stop_event.set()
                self._forget(session)
                self._set(state=self.ERROR, owned=False, pid=None, sticky=True,
                          detail=f"No pude iniciar el backend: {e}")
                return
            session.proc = proc
            self._set(state=self.STARTING, owned=True, pid=proc.pid,
                      detail="Iniciando backend…")
            healthy = self._wait_healthy(session, proc)
            if session.stop_event.is_set():
                self._terminate(proc)
                break
            if healthy:
                rc = proc.wait()
            else:
                rc = proc.poll()
                if rc is None:  # no contestó a tiempo: matar
                    self._terminate(proc)
                    rc = proc.poll()
            if session.stop_event.is_set():
                break
            # ¿Contesta otro daemon en la URL? Pasa cuando dos Kritas lo
            # arrancan a la vez: el nuestro no pudo tomar el puerto (código 3)
            # o murió y el puerto ya es de otro. Se reusa ese, sin relanzar.
            other = self._probe(cfg.url, timeout=1.0)
            if other.kind == "autoselect":
                session.stop_event.set()
                self._forget(session)
                self._append_log(cfg, f"[AI Select] el daemon propio salió "
                                      f"({bk.describe_exit(rc)}) y otro ya "
                                      f"atiende {cfg.url}: se reusa ese\n")
                self._set_ready(other)
                return
            restart, delay, reason = session.policy.on_exit(
                rc, healthy, time.monotonic())
            tail = bk.tail_file(cfg.daemon_log, 60)
            why = bk.last_error_line(tail)
            if not restart:
                msgs = {
                    "exited": "El backend se cerró ({exit}).",
                    "startup_failed": "El backend no arrancó ({exit}): {why}",
                    "crash_loop": "El backend se cayó varias veces seguidas "
                                  "({exit}); no lo relanzo más. Revisá el log.",
                }
                if not healthy and rc is None:
                    msg = "El backend no respondió a tiempo. Revisá el log."
                else:
                    msg = msgs.get(reason, "El backend terminó ({exit})")
                msg = msg.format(exit=bk.describe_exit(rc), why=why)
                session.stop_event.set()
                self._forget(session)
                if reason == "exited":
                    self._set(state=self.STOPPED, owned=False, pid=None,
                              loaded=False, detail=msg)
                else:
                    self._set(state=self.ERROR, owned=False, pid=None,
                              loaded=False, sticky=True, detail=msg)
                return
            self._set(state=self.STARTING, owned=True, pid=None, loaded=False,
                      detail=f"El backend se cayó ({bk.describe_exit(rc)}); "
                             f"reiniciando en {delay:.0f}s…")
            self._append_log(cfg, f"[AI Select] el daemon murió "
                                  f"({bk.describe_exit(rc)}); reinicio en "
                                  f"{delay:.0f}s\n")
            if session.stop_event.wait(delay):
                break
        # parado a pedido
        self._forget(session)
        self._set(state=self.STOPPED, owned=False, pid=None, loaded=False,
                  detail=session.stop_reason or "Backend detenido")

    def _forget(self, session):
        with self._lock:
            if self._session is session:
                self._session = None

    def _wait_healthy(self, session, proc):
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            if session.stop_event.is_set() or proc.poll() is not None:
                return False
            probe = self._probe(session.cfg.url, timeout=1.0)
            if probe.kind == "autoselect":
                self._set_ready(probe)
                return True
            session.stop_event.wait(HEALTH_POLL_S)
        return False

    @staticmethod
    def _append_log(cfg, text):
        try:
            with open(cfg.daemon_log, "ab") as f:
                f.write(text.encode("utf-8"))
        except OSError:
            pass

    # ----- parar -----

    @staticmethod
    def _terminate(proc, wait_s=3.0):
        """stdin cerrado + SIGTERM (TerminateProcess en Windows). Con
        wait_s=0 no espera: el hilo supervisor recoge el proceso."""
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # el daemon sale solo al ver EOF
        except OSError:
            pass
        try:
            proc.terminate()
            if wait_s > 0:
                proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass

    def _stop_session(self, session, reason="Backend detenido"):
        session.stop_reason = reason
        session.stop_event.set()
        self._terminate(session.proc, wait_s=0.0)

    def stop(self):
        """Botón «Parar»: para el daemon propio (en un hilo). Un daemon que
        no arrancó el plugin no se toca. → True si había uno propio."""
        with self._lock:
            session = self._session
        if not session:
            return False
        threading.Thread(target=self._stop_session, args=(session,),
                         daemon=True).start()
        return True

    def shutdown(self, wait_s=3.0):
        """Al cerrar Krita: parar el daemon propio y esperar (poco) a que
        salga. SIGTERM lo mata en el acto, así que la espera es corta."""
        with self._lock:
            session = self._session
        if not session:
            return
        session.stop_reason = "Krita se cerró"
        session.stop_event.set()
        self._terminate(session.proc, wait_s=wait_s)


# ----------------------------------------------------------------------------
# Instalador
# ----------------------------------------------------------------------------

class InstallRunner:
    """Corre un plan de instalación (lista de backend.Step) en un hilo.

    on_line(text) recibe cada línea de salida; on_step(index, step) al
    empezar cada paso; on_done(ok, message) al terminar. Todo desde el hilo
    del instalador. cancel() mata el paso en curso (y sus hijos)."""

    def __init__(self, on_line=None, on_step=None, on_done=None,
                 popen=subprocess.Popen, base_env=None):
        self._on_line = on_line or (lambda text: None)
        self._on_step = on_step or (lambda i, step: None)
        self._on_done = on_done or (lambda ok, msg: None)
        self._popen = popen
        self._base_env = base_env
        self._cancel = threading.Event()
        self._proc = None
        self._thread = None
        self._lock = threading.Lock()

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self, plan_builder, log_path=None):
        """plan_builder() corre en el hilo (detectar GPU/uv lanza procesos)
        y devuelve la lista de Step, o levanta backend.PlanError."""
        if self.running:
            return False
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, args=(plan_builder, log_path), daemon=True,
            name="autoselect-installer")
        self._thread.start()
        return True

    def cancel(self):
        self._cancel.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def _emit(self, log, text):
        if log:
            try:
                log.write(text + "\n")
                log.flush()
            except (OSError, ValueError):
                pass
        try:
            self._on_line(text)
        except Exception:
            pass

    def _run(self, plan_builder, log_path):
        log = None
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                log = open(log_path, "a", encoding="utf-8")
            except OSError:
                log = None
        ok, msg = False, ""
        try:
            self._emit(log, f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            "instalación del backend =====")
            try:
                steps = plan_builder()
            except bk.PlanError as e:
                ok, msg = False, str(e)
                self._emit(log, "✗ " + msg)
                return
            total = len(steps)
            for i, step in enumerate(steps):
                if self._cancel.is_set():
                    ok, msg = False, "cancelado"
                    return
                try:
                    self._on_step(i, step)
                except Exception:
                    pass
                prefix = f"[{i + 1}/{total}] {step.title}"
                if step.skipped:
                    self._emit(log, f"{prefix}: ✓ {step.skip_reason}")
                    continue
                self._emit(log, prefix)
                self._emit(log, "$ " + " ".join(_quote(a) for a in step.argv))
                rc = self._run_step(step, log)
                if self._cancel.is_set():
                    ok, msg = False, "cancelado"
                    self._emit(log, "✗ cancelado (se puede retomar)")
                    return
                if rc != 0:
                    ok, msg = False, (f"falló «{step.title}» "
                                      f"({bk.describe_exit(rc)})")
                    if getattr(step, "hint", None):
                        msg += f" — {step.hint}"
                    self._emit(log, "✗ " + msg)
                    return
            ok, msg = True, "Backend instalado"
            self._emit(log, "✓ " + msg)
        except Exception as e:  # nunca dejar el hilo sin avisar
            ok, msg = False, f"error inesperado: {e}"
            self._emit(log, "✗ " + msg)
        finally:
            if log:
                log.close()
            try:
                self._on_done(ok, msg)
            except Exception:
                pass

    def _run_step(self, step, log):
        base = self._base_env if self._base_env is not None else \
            bk.sanitize_env(os.environ, bk.krita_prefixes())
        env = dict(base)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    "UV_NO_PROGRESS": "1"})
        env.update(step.env)
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW | \
                CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # para matar el grupo entero
        try:
            proc = self._popen(step.argv, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env,
                               bufsize=1, text=True, encoding="utf-8",
                               errors="replace", **kwargs)
        except OSError as e:
            self._emit(log, f"no pude ejecutar {step.argv[0]}: {e}")
            return 127
        with self._lock:
            self._proc = proc
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._emit(log, "  " + line)
            return proc.wait()
        finally:
            with self._lock:
                self._proc = None


def _quote(arg):
    arg = str(arg)
    if "\n" in arg:
        return "'<código>'"
    return f'"{arg}"' if " " in arg else arg


def _kill_tree(proc):
    """Mata un paso del instalador con sus hijos (uv/pip lanzan procesos)."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass
