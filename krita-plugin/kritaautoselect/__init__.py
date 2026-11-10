"""AI Select — selección por segmentación (SAM 3) dentro de Krita.

Docker estilo krita-ai-diffusion que habla con el daemon krita-autoselect
(HTTP local, puerto 5679). El plugin es autónomo: trae el código del daemon
(carpeta daemon/), lo instala si hace falta (venv + torch + pesos) y lo
arranca/supervisa/para solo — como el "managed server" de AI Diffusion.
Ver backend.py (lógica pura) y manager.py (procesos e hilos).

Dos formas de seleccionar:

- Prompt de texto ("the red car") → todas las instancias del concepto.
- Modo click (botón lazo) → click sobre el canvas selecciona el objeto
  exacto bajo el cursor (Ctrl+click = excluir esa parte).

La red corre en un thread (nunca en el hilo de UI de Krita); el resultado
vuelve por señal Qt y la selección se aplica en el main thread.
"""
import sys

from krita import (
    DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Extension, Krita,
    Selection,
)

# Krita 5 usa PyQt5 (Qt5); Krita 6 usa PyQt6 (Qt6). El módulo `krita` ya
# importó el binding correcto antes que nosotros (y en Krita 5.3+/6 bloquea
# importar el otro), así que seguimos al que esté cargado. Mismo patrón que
# los plugins oficiales de Krita 6 (try PyQt6 / except PyQt5).
_USE_QT6 = "PyQt6.QtCore" in sys.modules or "PyQt5.QtCore" not in sys.modules
if _USE_QT6:
    try:
        from PyQt6.QtCore import (
            QBuffer, QByteArray, QEvent, QIODevice, QObject, QPoint, QPointF,
            QRect, QSize, Qt, QTimer, QUrl, pyqtSignal,
        )
        from PyQt6.QtGui import QColor, QDesktopServices, QImage, QPalette
        from PyQt6.QtWidgets import (
            QAbstractScrollArea, QApplication, QCheckBox, QComboBox, QDialog,
            QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
            QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
            QRubberBand, QSpinBox, QToolButton, QVBoxLayout, QWidget,
        )
        # Qt6 movió QOpenGLWidget de QtWidgets a su propio módulo.
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:
        _USE_QT6 = False
if not _USE_QT6:
    from PyQt5.QtCore import (  # noqa: F811
        QBuffer, QByteArray, QEvent, QIODevice, QObject, QPoint, QPointF,
        QRect, QSize, Qt, QTimer, QUrl, pyqtSignal,
    )
    from PyQt5.QtGui import (  # noqa: F811
        QColor, QDesktopServices, QImage, QPalette,
    )
    from PyQt5.QtWidgets import (  # noqa: F811
        QAbstractScrollArea, QApplication, QCheckBox, QComboBox, QDialog,
        QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QOpenGLWidget, QPlainTextEdit,
        QPushButton, QRubberBand, QSpinBox, QToolButton, QVBoxLayout, QWidget,
    )

import atexit
import base64
import http.client
import json
import os
import threading

from . import backend as bk
from .manager import Config, DaemonManager, InstallRunner

SETTINGS_GROUP = "kritaautoselect"
DEFAULT_URL = bk.DEFAULT_URL
# Copia del daemon que trae el plugin (la genera scripts/build_plugin.py).
BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "daemon")
MODES = [("Reemplazar", "replace"), ("Agregar", "add"),
         ("Restar", "subtract"), ("Intersecar", "intersect")]
# Herramienta inofensiva mientras el modo click está armado: aunque algún
# evento se escape del filtro, una selección de contorno sin arrastre no
# pinta ni destruye nada.
SAFE_TOOL = "KisToolSelectOutline"

# Dockers vivos (para la acción de atajo de teclado).
_dockers = []

# Clases C++ del canvas de Krita (OpenGL y fallback QPainter).
CANVAS_CLASSES = ("KisOpenGLCanvas2", "KisQPainterCanvas")


def _event_pos(event):
    """Posición local del mouse como QPoint (Qt6: position(); Qt5: pos())."""
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


def _event_posf(event):
    """Posición local del mouse como QPointF (subpíxel en Qt6)."""
    if hasattr(event, "position"):
        return QPointF(event.position())
    return QPointF(event.pos())


def _find_canvas_widgets(qwin):
    """Widgets del canvas en la ventana. En ambas versiones el canvas OpenGL
    es un QOpenGLWidget (Qt6: QtOpenGLWidgets); si no hay ninguno, se busca
    por nombre de clase C++ (cubre el canvas QPainter)."""
    if qwin is None:
        return []
    targets = qwin.findChildren(QOpenGLWidget)
    if not targets:
        targets = [w for w in qwin.findChildren(QWidget)
                   if w.metaObject().className() in CANVAS_CLASSES]
    return targets


def _palette_colors():
    """Colores según el tema de Krita (patrón de ai_diffusion/ui/theme.py)."""
    is_dark = QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
    if is_dark:
        return {"green": "#3b3", "yellow": "#cc3", "red": "#c33",
                "grey": "#888"}
    return {"green": "#292", "yellow": "#883", "red": "#911", "grey": "#666"}


def _read_setting(key, default=""):
    return Krita.instance().readSetting(SETTINGS_GROUP, key, default)


def _write_setting(key, value):
    Krita.instance().writeSetting(SETTINGS_GROUP, key, str(value))


def _read_bool(key, default):
    value = _read_setting(key, "true" if default else "false")
    return str(value).lower() in ("true", "1", "yes")


def _read_int(key, default):
    try:
        return int(float(_read_setting(key, str(default))))
    except (TypeError, ValueError):
        return default


def _config_from_settings():
    """Foto de la config (hilo de UI) para el manager, que la lee en hilos."""
    install_dir = _read_setting("install_dir", "")
    if not install_dir:  # setting viejo: carpeta del repo
        install_dir = _read_setting("repo_path", "")
    return Config(
        url=_read_setting("server_url", DEFAULT_URL).strip() or DEFAULT_URL,
        install_dir=install_dir.strip(),
        weights_setting=_read_setting("weights_path", "").strip(),
        hf_token=_read_setting("hf_token", "").strip(),
        unload_after_s=_read_int("unload_after_s", bk.DEFAULT_UNLOAD_AFTER_S),
        comfyui_url=_read_setting("comfyui_url", "").strip(),
        comfyui_free=_read_bool("comfyui_free", True),
        bundled_dir=BUNDLED_DIR,
    )


class _Hub(QObject):
    """Señales compartidas por todos los dockers (una por ventana). Los
    hilos del manager/instalador emiten acá; Qt las entrega en el hilo de UI."""

    state = pyqtSignal(dict)
    install_line = pyqtSignal(str)
    install_step = pyqtSignal(str)
    install_done = pyqtSignal(bool, str)


_hub = None
_manager = None
_runner = None


def _get_hub():
    global _hub
    if _hub is None:
        _hub = _Hub()
    return _hub


def _get_manager():
    global _manager
    if _manager is None:
        hub = _get_hub()
        _manager = DaemonManager(_config_from_settings())
        _manager.add_listener(hub.state.emit)
    return _manager


def _get_runner():
    global _runner
    if _runner is None:
        hub = _get_hub()
        _runner = InstallRunner(
            on_line=hub.install_line.emit,
            on_step=lambda i, step: hub.install_step.emit(step.title),
            on_done=hub.install_done.emit)
    return _runner


def _shutdown():
    """Al cerrar Krita: parar el daemon que arrancó el plugin y cualquier
    instalación en curso. Un daemon ajeno no se toca."""
    try:
        if _runner is not None and _runner.running:
            _runner.cancel()
        if _manager is not None:
            _manager.shutdown()
    except Exception:
        pass


def _build_plan(cfg):
    """Corre en el hilo del instalador (detectar GPU lanza nvidia-smi)."""
    prefixes = bk.krita_prefixes()
    env = bk.sanitize_env(os.environ, prefixes)
    uv = bk.find_uv(env, prefixes=prefixes)
    # Sin uv: un python3 que pueda crear el venv (≥3.10 y con ensurepip).
    system_python, python_problem = (None, "") if uv else \
        bk.pick_system_python(env, prefixes=prefixes)
    gpu = bk.detect_nvidia_gpu(env, prefixes=prefixes)
    token, source = bk.find_hf_token(cfg.hf_token)
    steps = bk.build_install_plan(
        cfg.install_dir, BUNDLED_DIR, uv=uv, system_python=system_python,
        gpu=gpu, weights_setting=cfg.weights_setting, hf_token=token,
        hf_token_from_setting=bool(cfg.hf_token),
        python_problem=python_problem)
    tool = ("uv " + uv if uv else
            "python " + " ".join(system_python) if system_python else
            "venv existente")
    info = (f"carpeta {cfg.install_dir} · {tool} · "
            f"GPU {gpu['name'] if gpu else 'ninguna NVIDIA'} · "
            f"token HF: {source or 'no hace falta'}")
    notes = [bk.Step("note", title, skip_reason=text)
             for title, text in bk.install_notes(gpu=gpu)]
    return [bk.Step("info", "Detectado", skip_reason=info)] + notes + steps


class DaemonClient(QObject):
    """HTTP hacia el daemon en un thread; resultados por señal Qt (queued)."""

    finished = pyqtSignal(dict)   # respuesta de /segment (o {"error": ...})
    ready = pyqtSignal(bool, str)  # backend listo (o no) para un pedido

    @property
    def base(self):
        return _read_setting("server_url", DEFAULT_URL)

    def _request(self, method, path, payload, timeout, signal):
        def work():
            try:
                rest = self.base.split("://", 1)[-1]
                hostport = rest.partition("/")[0]
                host, _, port = hostport.partition(":")
                conn = http.client.HTTPConnection(
                    host, int(port or 80), timeout=timeout)
                body = json.dumps(payload).encode() if payload else None
                conn.request(method, path, body,
                             {"Content-Type": "application/json"})
                resp = conn.getresponse()
                raw = resp.read().decode()
                conn.close()
                try:
                    data = json.loads(raw)
                except ValueError:
                    data = {"error": f"Respuesta inválida del server: "
                                     f"{raw[:120]}"}
            except (OSError, http.client.HTTPException) as e:
                # Conexión caída (p.ej. el daemon murió en medio del pedido):
                # el docker reintenta una vez tras el reinicio. Un timeout no.
                data = bk.request_error(e, timeout)
            except Exception as e:
                data = {"error": f"Error hablando con el server ({e})"}
            signal.emit(data if isinstance(data, dict) else {"error": "?"})

        threading.Thread(target=work, daemon=True).start()

    def segment(self, payload, timeout=300):
        self._request("POST", "/segment", payload, timeout, self.finished)


class CanvasClickFilter(QObject):
    """Modo click: captura el click sobre el canvas y lo convierte a
    coordenadas de imagen (zoom/pan/rotación resueltos por Krita 5.2+
    con flakeToImageTransform).

    Traga el press Y el release del botón izquierdo mientras está armado:
    entregar solo la mitad del par a la herramienta activa de Krita deja su
    máquina de estados de input colgada."""

    # x, y, kind: "new" = objeto nuevo, "pos" = refinar (Shift),
    # "neg" = excluir parte (Ctrl)
    clicked = pyqtSignal(int, int, str)
    # x, y, w, h en coordenadas de documento (arrastre tipo marquesina)
    dragged = pyqtSignal(int, int, int, int)

    DRAG_THRESHOLD_PX = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        # Instalado sobre QApplication (corre ANTES que el KisInputManager de
        # Krita, que también filtra a nivel app); solo actúa sobre eventos
        # cuyo receptor es el canvas.
        self.canvas_widgets = []
        self._press_pos = None
        self._press_widget = None
        self._rubber = None

    def _doc_point(self, event):
        window = Krita.instance().activeWindow()
        view = window.activeView() if window else None
        doc = Krita.instance().activeDocument()
        if view is None or doc is None or \
                not hasattr(view, "flakeToImageTransform"):
            return None, None
        point = view.flakeToImageTransform().map(_event_posf(event))
        return (int(point.x()), int(point.y())), doc

    def eventFilter(self, obj, event):
        try:
            etype = event.type()
            if obj not in self.canvas_widgets:
                return False
            if etype == QEvent.Type.MouseMove:
                if self._press_pos is None:
                    return False  # hover normal: no tocar
                if self._rubber is None and \
                        (_event_pos(event) - self._press_pos).manhattanLength() > \
                        self.DRAG_THRESHOLD_PX:
                    self._rubber = QRubberBand(QRubberBand.Shape.Rectangle,
                                               self._press_widget)
                if self._rubber is not None:
                    self._rubber.setGeometry(
                        QRect(self._press_pos, _event_pos(event)).normalized())
                    self._rubber.show()
                return True
            if etype not in (QEvent.Type.MouseButtonPress,
                             QEvent.Type.MouseButtonRelease,
                             QEvent.Type.MouseButtonDblClick):
                return False
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            if etype == QEvent.Type.MouseButtonDblClick:
                return True
            if etype == QEvent.Type.MouseButtonPress:
                self._press_pos = QPoint(_event_pos(event))
                self._press_widget = obj
                return True

            # MouseButtonRelease: ¿fue click o arrastre?
            press_pos = self._press_pos
            was_drag = self._rubber is not None
            if self._rubber is not None:
                self._rubber.hide()
                self._rubber.deleteLater()
                self._rubber = None
            self._press_pos = None
            self._press_widget = None
            if press_pos is None:
                return True

            (x2, y2), doc = self._doc_point(event)
            if doc is None:
                return True

            if was_drag:
                view = Krita.instance().activeWindow().activeView()
                p1 = view.flakeToImageTransform().map(QPointF(press_pos))
                x1, y1 = int(p1.x()), int(p1.y())
                bx, by = min(x1, x2), min(y1, y2)
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                bx, by = max(0, bx), max(0, by)
                bw = min(bw, doc.width() - bx)
                bh = min(bh, doc.height() - by)
                if bw > 4 and bh > 4:
                    self.dragged.emit(bx, by, bw, bh)
                return True

            if 0 <= x2 < doc.width() and 0 <= y2 < doc.height():
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    kind = "neg"
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    kind = "pos"
                else:
                    kind = "new"
                self.clicked.emit(x2, y2, kind)
            return True
        except Exception:
            return False  # jamás dejar que una excepción suba al event loop


class AutoSelectDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Select")
        self._colors = _palette_colors()
        self._client = DaemonClient(self)
        self._client.finished.connect(self._on_segment_result,
                                      Qt.ConnectionType.QueuedConnection)
        self._client.ready.connect(self._on_backend_ready,
                                   Qt.ConnectionType.QueuedConnection)
        self._click_filter = CanvasClickFilter(self)
        self._click_filter.clicked.connect(self._on_canvas_click,
                                           Qt.ConnectionType.QueuedConnection)
        self._click_filter.dragged.connect(self._on_canvas_drag,
                                           Qt.ConnectionType.QueuedConnection)
        self._filtered_widgets = []
        self._busy = False
        self._pending = None      # payload esperando que el backend esté listo
        self._retry_left = 0
        self._installing = False
        self._log_dialog = None
        self._previous_tool = None
        # Puntos acumulados del modo click (refinamiento estilo SAM).
        self._click_points = []
        self._click_labels = []
        self._build_ui()
        self._load_settings()
        _dockers.append(self)
        hub = _get_hub()
        hub.state.connect(self._on_backend_state)
        hub.install_line.connect(self._on_install_line)
        hub.install_step.connect(self._on_install_step)
        hub.install_done.connect(self._on_install_done)
        self._installing = _get_runner().running
        self._update_install_buttons()
        self._on_backend_state(_get_manager().snapshot())
        # Refresco periódico del estado (rápido mientras hay un pedido, para
        # mostrar "Cargando modelo"); el health corre en un hilo.
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_backend)
        self._poll.start(1000)
        self._poll_ticks = 0
        QTimer.singleShot(500, _get_manager().refresh)

    # ----- UI -----

    def _build_ui(self):
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header: estado del backend + engranaje (muestra/oculta la config)
        header = QHBoxLayout()
        self._backend_label = QLabel("Backend: comprobando…")
        self._backend_label.setWordWrap(True)
        header.addWidget(self._backend_label, 1)
        gear = QToolButton()
        gear.setIcon(Krita.instance().icon("configure"))
        gear.setAutoRaise(True)
        gear.setToolTip("Configurar / instalar el backend")
        gear.clicked.connect(self._toggle_config)
        header.addWidget(gear)
        layout.addLayout(header)

        # Prompt (Shift+Enter o botón = seleccionar)
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText(
            'Qué seleccionar, ej: "the red car"  (Shift+Enter)')
        self._prompt.setFixedHeight(52)
        self._prompt.setTabChangesFocus(True)
        self._prompt.setFrameStyle(QFrame.Shape.NoFrame)
        self._prompt.installEventFilter(self)
        layout.addWidget(self._prompt)

        # Fila: modo de combinación + feather
        opts = QHBoxLayout()
        self._mode = QComboBox()
        for label, value in MODES:
            self._mode.addItem(label, value)
        opts.addWidget(self._mode, 1)
        opts.addWidget(QLabel("Feather:"))
        self._feather = QSpinBox()
        self._feather.setRange(0, 200)
        self._feather.setSuffix(" px")
        self._feather.setValue(0)
        opts.addWidget(self._feather)
        layout.addLayout(opts)

        self._goto_check = QCheckBox("Ir a la selección (zoom)")
        self._goto_check.setChecked(True)
        layout.addWidget(self._goto_check)

        # Fila: botón Seleccionar + toggle de modo click (lazo)
        actions = QHBoxLayout()
        self._select_btn = QPushButton("Seleccionar")
        self._select_btn.setMinimumHeight(32)
        self._select_btn.clicked.connect(self._segment_from_prompt)
        actions.addWidget(self._select_btn, 1)
        self._click_btn = QToolButton()
        self._click_btn.setIcon(
            Krita.instance().icon("tool_outline_selection"))
        self._click_btn.setCheckable(True)
        self._click_btn.setToolTip(
            "Modo click: tocá un objeto en el canvas para seleccionarlo.\n"
            "Arrastrá un recuadro: el objeto de adentro (con prompt escrito: "
            "las instancias del prompt que caigan adentro).\n"
            "Shift+click: sumar un punto de refinamiento al mismo objeto.\n"
            "Ctrl+click: excluir esa parte del objeto.")
        self._click_btn.setFixedSize(34, 32)
        self._click_btn.toggled.connect(self._toggle_click_mode)
        actions.addWidget(self._click_btn)
        layout.addLayout(actions)

        # Resultado de la última operación
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._config = self._build_config()
        self._config.setVisible(False)
        layout.addWidget(self._config)

        layout.addStretch()
        self.setWidget(root)

    def _build_config(self):
        """Config del backend (oculta por defecto, estilo "Connection")."""
        box = QGroupBox("Backend SAM 3")
        cfg = QVBoxLayout(box)
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        def path_row(edit, title):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            browse = QToolButton()
            browse.setText("…")
            browse.clicked.connect(lambda: self._browse_dir(edit, title))
            row.addWidget(browse)
            return row

        self._url_edit = QLineEdit(DEFAULT_URL)
        self._url_edit.setToolTip(
            "Si ya hay un daemon contestando en esta URL se usa ese; si no, "
            "el plugin arranca el suyo en ese puerto.")
        self._url_edit.editingFinished.connect(self._save_settings)
        form.addRow("URL:", self._url_edit)

        self._install_edit = QLineEdit()
        self._install_edit.setPlaceholderText(bk.default_data_dir())
        self._install_edit.setToolTip(
            "Carpeta con el .venv del backend. Puede ser una instalación "
            "existente (p.ej. el repo krita-autoselect) o vacía = la carpeta "
            "por defecto donde instala «Instalar backend».")
        self._install_edit.editingFinished.connect(self._save_settings)
        form.addRow("Instalación:", path_row(self._install_edit,
                                             "Carpeta de instalación"))

        self._weights_edit = QLineEdit()
        self._weights_edit.setPlaceholderText(
            "auto: <instalación>/models/sam3 o Hugging Face")
        self._weights_edit.setToolTip(
            "Carpeta con los pesos de SAM 3 en formato Hugging Face "
            "(config.json + model.safetensors).")
        self._weights_edit.editingFinished.connect(self._save_settings)
        form.addRow("Pesos:", path_row(self._weights_edit,
                                       "Carpeta de pesos SAM 3"))

        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("vacío = login de `hf auth login`")
        self._token_edit.setToolTip(
            "Token de Hugging Face para bajar facebook/sam3 (repo con "
            "licencia: aceptala en huggingface.co/facebook/sam3). Se guarda "
            "en la configuración de Krita (kritarc).")
        self._token_edit.editingFinished.connect(self._save_settings)
        form.addRow("Token HF:", self._token_edit)

        self._unload_spin = QSpinBox()
        self._unload_spin.setRange(0, 24 * 3600)
        self._unload_spin.setSingleStep(60)
        self._unload_spin.setSuffix(" s")
        self._unload_spin.setSpecialValueText("nunca")
        self._unload_spin.setToolTip(
            "Segundos sin uso hasta liberar la VRAM (0 = modelo siempre "
            "cargado).")
        self._unload_spin.editingFinished.connect(self._save_settings)
        form.addRow("Liberar VRAM tras:", self._unload_spin)

        comfy_row = QHBoxLayout()
        self._comfy_check = QCheckBox()
        self._comfy_check.setToolTip(
            "Antes de cargar SAM 3, si falta VRAM, pedirle a ComfyUI que "
            "descargue sus modelos.")
        self._comfy_check.toggled.connect(self._save_settings)
        comfy_row.addWidget(self._comfy_check)
        self._comfy_edit = QLineEdit()
        self._comfy_edit.setPlaceholderText("auto (AI Diffusion / 8188 / 8001)")
        self._comfy_edit.editingFinished.connect(self._save_settings)
        comfy_row.addWidget(self._comfy_edit, 1)
        form.addRow("ComfyUI:", comfy_row)

        self._autostart_check = QCheckBox("Iniciar el backend al abrir Krita")
        self._autostart_check.toggled.connect(self._save_settings)
        form.addRow("", self._autostart_check)
        cfg.addLayout(form)

        srv_row = QHBoxLayout()
        for text, slot in (("Probar", lambda: _get_manager().refresh()),
                           ("Iniciar", self._start_server),
                           ("Parar", self._stop_server),
                           ("Ver log", self._show_log)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            srv_row.addWidget(btn)
        cfg.addLayout(srv_row)

        inst_row = QHBoxLayout()
        self._install_btn = QPushButton("Instalar backend")
        self._install_btn.setToolTip(
            "Crea el entorno Python, instala torch (CUDA si hay GPU NVIDIA) "
            "y las dependencias, y baja los pesos de SAM 3 (~3.4 GB). Se "
            "puede cancelar y volver a correr: retoma donde quedó.")
        self._install_btn.clicked.connect(self._install_backend)
        inst_row.addWidget(self._install_btn, 1)
        self._cancel_btn = QPushButton("Cancelar")
        self._cancel_btn.clicked.connect(lambda: _get_runner().cancel())
        inst_row.addWidget(self._cancel_btn)
        cfg.addLayout(inst_row)

        self._install_log = QPlainTextEdit()
        self._install_log.setReadOnly(True)
        self._install_log.setMaximumBlockCount(3000)
        self._install_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._install_log.setMinimumHeight(140)
        self._install_log.setVisible(False)
        cfg.addWidget(self._install_log)

        hint = QLabel(
            '<a href="https://github.com/chafamaster2000/krita-autoselect">'
            "Ayuda (README)</a>")
        hint.setOpenExternalLinks(True)
        cfg.addWidget(hint)
        return box

    def _browse_dir(self, edit, title):
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            edit.setText(path)
            self._save_settings()

    def canvasChanged(self, canvas):
        # Si cambia la vista con el modo click armado, re-enganchar el filtro.
        try:
            if self._click_btn.isChecked():
                self._detach_click_filter()
                self._attach_click_filter()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            if obj is self._prompt and event.type() == QEvent.Type.KeyPress:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._segment_from_prompt()
                    return True
        except Exception:
            pass
        return False

    # ----- settings -----

    def _load_settings(self):
        cfg = _config_from_settings()
        self._url_edit.setText(cfg.url)
        self._install_edit.setText(
            _read_setting("install_dir", "") or _read_setting("repo_path", ""))
        self._weights_edit.setText(cfg.weights_setting)
        self._token_edit.setText(cfg.hf_token)
        self._unload_spin.setValue(int(cfg.unload_after_s))
        self._comfy_check.setChecked(cfg.comfyui_free)
        self._comfy_edit.setText(cfg.comfyui_url)
        self._autostart_check.setChecked(_read_bool("autostart", False))

    def _save_settings(self, *_):
        _write_setting("server_url", self._url_edit.text().strip()
                       or DEFAULT_URL)
        _write_setting("install_dir", self._install_edit.text().strip())
        _write_setting("weights_path", self._weights_edit.text().strip())
        _write_setting("hf_token", self._token_edit.text().strip())
        _write_setting("unload_after_s", self._unload_spin.value())
        _write_setting("comfyui_free",
                       "true" if self._comfy_check.isChecked() else "false")
        _write_setting("comfyui_url", self._comfy_edit.text().strip())
        _write_setting("autostart",
                       "true" if self._autostart_check.isChecked() else "false")
        new = _config_from_settings()
        if new.launch_key() != _get_manager().config.launch_key():
            _get_manager().configure(new)

    def _toggle_config(self):
        self._config.setVisible(not self._config.isVisible())

    # ----- estado -----

    def _set_status(self, text, color="grey"):
        self._status.setText(
            f'<span style="color:{self._colors[color]};">{text}</span>')

    def _set_backend_text(self, text, color, tooltip=""):
        self._backend_label.setText(
            f'<span style="color:{self._colors[color]};">{text}</span>')
        self._backend_label.setToolTip(tooltip)

    def _poll_backend(self):
        # 1 s mientras hay un pedido en vuelo; 10 s en reposo.
        self._poll_ticks += 1
        if self._busy or self._poll_ticks % 10 == 0:
            _get_manager().refresh()

    def _on_backend_state(self, snap):
        if self._installing:
            return  # el instalador escribe su propio estado
        state = snap.get("state")
        detail = snap.get("detail") or ""
        tip = detail
        if snap.get("pid"):
            tip = (tip + "\n" if tip else "") + f"PID {snap['pid']}"
        if state == DaemonManager.READY:
            text = ("Listo — modelo cargado" if snap.get("loaded")
                    else "Listo — modelo en frío (carga al seleccionar)")
            if not snap.get("owned"):
                text += " · daemon existente"
            self._set_backend_text(text, "green", tip)
        elif state == DaemonManager.LOADING:
            self._set_backend_text(
                "Cargando modelo SAM 3… (la primera vez tarda)", "yellow", tip)
        elif state == DaemonManager.STARTING:
            self._set_backend_text(detail or "Iniciando backend…", "yellow",
                                   tip)
        elif state == DaemonManager.STOPPED:
            self._set_backend_text(f"Backend detenido — {detail}", "grey", tip)
        elif state == DaemonManager.NOT_INSTALLED:
            self._set_backend_text(
                "Backend no instalado — engranaje → «Instalar backend» o "
                "apuntá a una instalación", "red", tip)
        elif state == DaemonManager.ERROR:
            self._set_backend_text(f"Error del backend: {detail}", "red", tip)
        else:
            self._set_backend_text("Backend: comprobando…", "grey", tip)

    # ----- server local -----

    def _start_server(self):
        _get_manager().start()

    def _stop_server(self):
        if not _get_manager().stop():
            self._set_status("No hay un backend iniciado por el plugin (un "
                             "daemon existente no se toca)", "grey")

    def _show_log(self):
        if self._log_dialog is None:
            self._log_dialog = LogDialog(self)
        self._log_dialog.show_log(_get_manager().config)

    # ----- instalador -----

    def _update_install_buttons(self):
        self._install_btn.setEnabled(not self._installing)
        self._cancel_btn.setEnabled(self._installing)

    def _install_backend(self):
        runner = _get_runner()
        if runner.running:
            return
        self._save_settings()
        cfg = _config_from_settings()
        _get_manager().stop()  # no instalar con el daemon propio corriendo
        self._install_log.clear()
        self._install_log.setVisible(True)
        self._config.setVisible(True)
        if runner.start(lambda: _build_plan(cfg), cfg.install_log):
            for docker in _live_dockers():
                docker._installing = True
                docker._update_install_buttons()
                docker._set_backend_text("Instalando backend…", "yellow")

    def _on_install_line(self, text):
        self._install_log.setVisible(True)
        self._install_log.appendPlainText(text)

    def _on_install_step(self, title):
        self._installing = True
        self._update_install_buttons()
        self._set_backend_text(f"Instalando backend: {title}…", "yellow")

    def _on_install_done(self, ok, message):
        self._installing = False
        self._update_install_buttons()
        if ok:
            self._set_status("Backend instalado ✓", "green")
        elif message == "cancelado":
            self._set_status("Instalación cancelada — «Instalar backend» "
                             "retoma donde quedó", "yellow")
        else:
            self._set_status(f"La instalación falló: {message}", "red")
        _get_manager().configure(_config_from_settings())

    # ----- segmentación -----

    def _grab_canvas_b64(self):
        doc = Krita.instance().activeDocument()
        if doc is None:
            return None, None
        w, h = doc.width(), doc.height()
        image = doc.projection(0, 0, w, h)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        return base64.b64encode(bytes(ba.data())).decode("ascii"), doc

    def _segment_from_prompt(self):
        text = self._prompt.toPlainText().strip()
        if not text:
            self._set_status("Escribí qué seleccionar", "yellow")
            return
        self._click_points = []  # el prompt arranca una selección nueva
        self._click_labels = []
        self._segment({"text": text})

    def _on_canvas_click(self, x, y, kind):
        """Click = objeto nuevo; Shift+click suma un punto de refinamiento;
        Ctrl+click excluye una parte. Los refinamientos re-segmentan el MISMO
        objeto con todos los puntos acumulados."""
        if kind == "new" or not self._click_points:
            if kind == "neg":
                self._set_status(
                    "Primero un click normal sobre el objeto; Ctrl+click "
                    "después, para excluir partes", "yellow")
                return
            self._click_points = [[x, y]]
            self._click_labels = [1]
        else:
            self._click_points.append([x, y])
            self._click_labels.append(0 if kind == "neg" else 1)
        self._segment({"points": list(self._click_points),
                       "point_labels": list(self._click_labels)})

    def _on_canvas_drag(self, x, y, w, h):
        """Arrastre tipo marquesina. Sin prompt: el objeto dentro del
        recuadro. Con prompt: instancias del concepto que caen adentro
        ("encontrá la mano acá adentro")."""
        self._click_points = []
        self._click_labels = []
        text = self._prompt.toPlainText().strip()
        if text:
            self._segment({"text": text, "within": [x, y, w, h]})
        else:
            self._segment({"box": [x, y, w, h]})

    def _segment(self, prompt_payload):
        if self._busy:
            return
        if self._installing:
            self._set_status("Esperá a que termine la instalación", "yellow")
            return
        image_b64, doc = self._grab_canvas_b64()
        if image_b64 is None:
            self._set_status("No hay documento abierto", "red")
            return
        self._set_busy(True)
        payload = {"image_b64": image_b64, "combine": "union"}
        payload.update(prompt_payload)
        self._pending = payload
        self._retry_left = 1
        self._request_backend()

    def _set_busy(self, busy):
        self._busy = busy
        _get_manager().set_busy(busy)
        self._select_btn.setEnabled(not busy)

    def _request_backend(self):
        """Primero el backend (reusar o arrancar, en un hilo); el pedido
        sale en _on_backend_ready."""
        self._set_status("Preparando el backend…", "yellow")
        client = self._client
        _get_manager().ensure_ready(
            lambda ok, msg: _safe_emit(client.ready, ok, msg))

    def _on_backend_ready(self, ok, message):
        if self._pending is None:
            return
        if not ok:
            self._pending = None
            self._set_busy(False)
            self._set_status(message or "El backend no está disponible",
                             "red")
            return
        self._set_status("Segmentando… (la primera vez carga el modelo)",
                         "yellow")
        self._client.segment(self._pending)

    def _on_segment_result(self, data):
        if data.get("_conn") and self._retry_left > 0 and self._pending:
            # El daemon se cayó en medio del pedido (crash nativo de torch):
            # el supervisor lo relanza; reintentar una vez.
            self._retry_left -= 1
            self._set_status("El backend se cayó; reintentando…", "yellow")
            QTimer.singleShot(1500, self._request_backend)
            return
        self._pending = None
        self._set_busy(False)
        _get_manager().refresh()
        if data.get("error"):
            error = data["error"]
            if "out of memory" in error.lower():
                error = ("Sin VRAM libre para SAM 3: otro programa (ComfyUI, "
                         "un juego…) ocupa la GPU y no la liberó. Probá de "
                         "nuevo cuando termine. Detalle en «Ver log».")
                self._status.setToolTip(data["error"])
            self._set_status(error, "red")
            return
        if not data.get("count") or not data.get("mask_b64"):
            self._set_status("No encontré nada con ese prompt", "yellow")
            return
        try:
            self._apply_mask(data["mask_b64"])
        except Exception as e:
            self._set_status(f"Error aplicando la selección: {e}", "red")
            return
        best = data["instances"][0].get("score", 0)
        plural = "instancia" if data["count"] == 1 else "instancias"
        if best < 0.5 and self._click_points:
            self._set_status(
                f"Score bajo ({best}) — click ambiguo. Shift+click suma "
                "otro punto del objeto; Ctrl+click excluye partes.", "yellow")
        else:
            self._set_status(
                f"Seleccioné {data['count']} {plural} (score {best})", "green")
        if self._goto_check.isChecked():
            self._zoom_to_selection()

    # Cada modo del combo mapea a la variante nativa de "Select Opaque".
    SELECT_OPAQUE_ACTIONS = {
        "replace": "selectopaque",
        "add": "selectopaque_add",
        "subtract": "selectopaque_subtract",
        "intersect": "selectopaque_intersect",
    }

    def _apply_mask(self, mask_b64):
        """Máscara PNG → selección vía el "Seleccionar opaco" nativo de Krita
        sobre una capa temporal. Es el único camino que genera el contorno
        (línea punteada): las selecciones armadas con setPixelData nunca
        muestran marching ants. El feather se aplica como blur de la máscara
        ANTES, así la selección suave conserva su contorno."""
        doc = Krita.instance().activeDocument()
        if doc is None:
            raise RuntimeError("no hay documento")
        img = QImage()
        img.loadFromData(base64.b64decode(mask_b64))
        w, h = doc.width(), doc.height()
        if img.width() != w or img.height() != h:
            img = img.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        img = img.convertToFormat(QImage.Format.Format_Grayscale8)

        feather = self._feather.value()
        if feather > 0:
            # Blur barato sin dependencias: downscale + upscale suavizado.
            factor = max(1, int(feather))
            small_w = max(1, w // (factor + 1))
            small_h = max(1, h // (factor + 1))
            img = (img
                   .scaled(small_w, small_h, Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
                   .scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))

        mode = self._mode.currentData()
        action_name = self.SELECT_OPAQUE_ACTIONS.get(mode, "selectopaque")
        if mode in ("subtract", "intersect") and doc.selection() is None:
            raise RuntimeError("no hay selección previa para combinar")

        bpl = img.bytesPerLine()
        raw = img.constBits().asstring(bpl * h)
        tight = raw if bpl == w else b"".join(
            raw[i * bpl:i * bpl + w] for i in range(h))
        argb = bytearray(w * h * 4)
        argb[0::4] = tight
        argb[1::4] = tight
        argb[2::4] = tight
        argb[3::4] = tight
        prev = doc.activeNode()
        tmp = doc.createNode("__autoselect_tmp", "paintlayer")
        doc.rootNode().addChildNode(tmp, None)
        try:
            tmp.setPixelData(bytes(argb), 0, 0, w, h)
            doc.setActiveNode(tmp)
            doc.waitForDone()
            action = Krita.instance().action(action_name)
            if action is None:
                raise RuntimeError(f"acción no encontrada: {action_name}")
            action.trigger()
            # NO processEvents acá: puede re-entrar otros handlers con la
            # capa temporal viva (mutación concurrente → crash en libkis).
            doc.waitForDone()
        finally:
            if prev is not None:
                doc.setActiveNode(prev)
            tmp.remove()
        doc.refreshProjection()

    # ----- feedback visual: "mostrame lo que seleccionaste" -----

    def _zoom_to_selection(self):
        """Encajar la selección en ~40% del viewport y centrarla. El factor
        de zoom se calcula contra la escala real actual (transformada del
        canvas), así no depende del DPI del documento."""
        try:
            doc = Krita.instance().activeDocument()
            sel = doc.selection() if doc else None
            window = Krita.instance().activeWindow()
            view = window.activeView() if window else None
            if sel is None or view is None or \
                    not hasattr(view, "flakeToImageTransform"):
                return
            gl_widgets = _find_canvas_widgets(window.qwindow())
            if not gl_widgets:
                return
            gl = gl_widgets[0]

            inverse, ok = view.flakeToImageTransform().inverted()
            if not ok:
                return
            current_scale = abs(inverse.m11()) or 1e-6
            margin = 2.5
            desired_scale = min(
                gl.width() / max(1.0, sel.width() * margin),
                gl.height() / max(1.0, sel.height() * margin))
            desired_scale = max(0.05, min(6.0, desired_scale))
            canvas = view.canvas()
            canvas.setZoomLevel(
                canvas.zoomLevel() * desired_scale / current_scale)
            QApplication.processEvents()

            inverse, ok = view.flakeToImageTransform().inverted()
            if not ok:
                return
            center_doc = QPointF(sel.x() + sel.width() / 2.0,
                                 sel.y() + sel.height() / 2.0)
            target = inverse.map(center_doc)
            scroller = gl.parentWidget()
            while scroller is not None and \
                    not isinstance(scroller, QAbstractScrollArea):
                scroller = scroller.parentWidget()
            if scroller is not None:
                center = gl.rect().center()
                hbar = scroller.horizontalScrollBar()
                vbar = scroller.verticalScrollBar()
                hbar.setValue(hbar.value() + int(target.x() - center.x()))
                vbar.setValue(vbar.value() + int(target.y() - center.y()))
        except Exception:
            pass

    # ----- modo click -----

    def _toggle_click_mode(self, enabled):
        if enabled:
            count = self._attach_click_filter()
            if count == 0:
                self._set_status("No encontré el canvas para el modo click",
                                 "red")
                self._click_btn.setChecked(False)
                return
            self._set_status(
                "Modo click: tocá el objeto o arrastrá un recuadro. "
                "Shift+click refina, Ctrl+click excluye.", "green")
        else:
            self._detach_click_filter()
            self._click_points = []
            self._click_labels = []
            self._set_status("Modo click desactivado", "grey")

    def _attach_click_filter(self):
        window = Krita.instance().activeWindow()
        if window is None:
            return 0
        targets = _find_canvas_widgets(window.qwindow())
        if not targets:
            return 0
        self._click_filter.canvas_widgets = targets
        # A nivel aplicación: el KisInputManager de Krita también filtra ahí,
        # y el último filtro instalado corre primero — así el click nos llega
        # antes de que la herramienta activa lo procese.
        QApplication.instance().installEventFilter(self._click_filter)
        self._filtered_widgets = targets
        self._switch_to_safe_tool()
        return len(targets)

    def _detach_click_filter(self):
        try:
            QApplication.instance().removeEventFilter(self._click_filter)
        except RuntimeError:
            pass
        self._click_filter.canvas_widgets = []
        self._filtered_widgets = []
        self._restore_tool()

    def _switch_to_safe_tool(self):
        """Como una tool de verdad: mientras el modo click está armado, la
        herramienta activa pasa a ser la selección de contorno (el pincel no
        puede pintar). Krita no expone cuál era la herramienta anterior, así
        que al salir se vuelve siempre al pincel."""
        try:
            safe = Krita.instance().action(SAFE_TOOL)
            if safe:
                safe.trigger()
        except Exception:
            pass

    def _restore_tool(self):
        try:
            brush = Krita.instance().action("KritaShape/KisToolBrush")
            if brush:
                brush.trigger()
        except Exception:
            pass


class AutoSelectExtension(Extension):
    """Registra la acción de atajo: alternar el modo click sin ir al panel
    (Herramientas → Scripts → "AI Select: modo click"; atajo asignable en
    Configurar Krita → Atajos de teclado)."""

    def setup(self):
        _get_hub()
        manager = _get_manager()
        try:
            notifier = Krita.instance().notifier()
            notifier.setActive(True)
            notifier.applicationClosing.connect(_shutdown)
        except Exception:
            pass
        atexit.register(_shutdown)
        if _read_bool("autostart", False):
            QTimer.singleShot(2000, manager.ensure_ready)

    def createActions(self, window):
        action = window.createAction(
            "kritaautoselect_toggle_click", "AI Select: modo click",
            "tools/scripts")
        action.triggered.connect(_toggle_click_mode_action)


def _safe_emit(signal, *args):
    try:
        signal.emit(*args)
    except RuntimeError:
        pass  # el docker se destruyó mientras tanto


def _live_dockers():
    alive = []
    for docker in _dockers:
        try:
            docker.objectName()
            alive.append(docker)
        except RuntimeError:
            continue
    return alive


class LogDialog(QDialog):
    """Visor del log del daemon y del instalador (se refresca solo)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Select — log del backend")
        self.resize(820, 480)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self._which = QComboBox()
        self._which.addItem("Daemon", "daemon")
        self._which.addItem("Instalación", "install")
        self._which.currentIndexChanged.connect(self.reload)
        top.addWidget(self._which)
        self._path_label = QLabel()
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self._path_label, 1)
        folder = QPushButton("Abrir carpeta")
        folder.clicked.connect(self._open_folder)
        top.addWidget(folder)
        layout.addLayout(top)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._text)
        self._cfg = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.reload)

    def show_log(self, cfg):
        self._cfg = cfg
        self.reload()
        self.show()
        self.raise_()
        self._timer.start(2000)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _path(self):
        if self._cfg is None:
            return ""
        if self._which.currentData() == "install":
            return self._cfg.install_log
        return self._cfg.daemon_log

    def reload(self, *_):
        path = self._path()
        self._path_label.setText(path)
        text = bk.tail_file(path, max_lines=400, max_bytes=256 * 1024)
        if text != self._text.toPlainText():
            self._text.setPlainText(text or "(vacío)")
            bar = self._text.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _open_folder(self):
        if self._cfg is not None:
            os.makedirs(self._cfg.log_dir, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._cfg.log_dir))


def _toggle_click_mode_action():
    for docker in _dockers:
        try:
            docker._click_btn.toggle()
            return
        except RuntimeError:
            continue  # docker destruido


Krita.instance().addExtension(AutoSelectExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("aiSelect", DockWidgetFactoryBase.DockPosition.DockRight,
                      AutoSelectDocker)
)
