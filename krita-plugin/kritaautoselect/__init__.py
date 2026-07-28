"""AI Select — selección por segmentación (SAM 3) dentro de Krita.

Docker estilo krita-ai-diffusion que habla con el daemon krita-autoselect
(HTTP local, puerto 5679). Dos formas de seleccionar:

- Prompt de texto ("the red car") → todas las instancias del concepto.
- Modo click (botón lazo) → click sobre el canvas selecciona el objeto
  exacto bajo el cursor (Ctrl+click = excluir esa parte).

La red corre en un thread (nunca en el hilo de UI de Krita); el resultado
vuelve por señal Qt y la selección se aplica en el main thread.
"""
from krita import (
    DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita, Selection,
)
from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPalette
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QOpenGLWidget, QPlainTextEdit, QPushButton, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
import base64
import http.client
import json
import os
import subprocess
import threading

SETTINGS_GROUP = "kritaautoselect"
DEFAULT_URL = "http://127.0.0.1:5679"
MODES = [("Reemplazar", "replace"), ("Agregar", "add"),
         ("Restar", "subtract"), ("Intersecar", "intersect")]


def _palette_colors():
    """Colores según el tema de Krita (patrón de ai_diffusion/ui/theme.py)."""
    is_dark = QApplication.palette().color(QPalette.Window).lightness() < 128
    if is_dark:
        return {"green": "#3b3", "yellow": "#cc3", "red": "#c33",
                "grey": "#888"}
    return {"green": "#292", "yellow": "#883", "red": "#911", "grey": "#666"}


def _read_setting(key, default=""):
    return Krita.instance().readSetting(SETTINGS_GROUP, key, default)


def _write_setting(key, value):
    Krita.instance().writeSetting(SETTINGS_GROUP, key, str(value))


class DaemonClient(QObject):
    """HTTP hacia el daemon en un thread; resultados por señal Qt (queued)."""

    finished = pyqtSignal(dict)   # respuesta de /segment (o {"error": ...})
    health = pyqtSignal(dict)     # respuesta de /health

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
                data = json.loads(resp.read().decode())
                conn.close()
            except Exception as e:
                data = {"error": f"Sin conexión con el server ({e})"}
            signal.emit(data if isinstance(data, dict) else {"error": "?"})

        threading.Thread(target=work, daemon=True).start()

    def segment(self, payload, timeout=300):
        self._request("POST", "/segment", payload, timeout, self.finished)

    def check_health(self, timeout=3):
        self._request("GET", "/health", None, timeout, self.health)


class CanvasClickFilter(QObject):
    """Modo click: captura el click sobre el canvas y lo convierte a
    coordenadas de imagen (zoom/pan/rotación resueltos por Krita 5.2+
    con flakeToImageTransform).

    Traga el press Y el release del botón izquierdo mientras está armado:
    entregar solo la mitad del par a la herramienta activa de Krita deja su
    máquina de estados de input colgada."""

    clicked = pyqtSignal(int, int, bool)  # x, y, es_negativo

    def eventFilter(self, obj, event):
        try:
            etype = event.type()
            if etype not in (QEvent.MouseButtonPress,
                             QEvent.MouseButtonRelease,
                             QEvent.MouseButtonDblClick):
                return False
            if event.button() != Qt.LeftButton:
                return False
            if etype != QEvent.MouseButtonPress:
                return True  # release/dblclick del par: solo tragarlo
            window = Krita.instance().activeWindow()
            view = window.activeView() if window else None
            doc = Krita.instance().activeDocument()
            if view is None or doc is None:
                return False
            if not hasattr(view, "flakeToImageTransform"):
                return False  # Krita < 5.2: sin mapeo confiable
            transform = view.flakeToImageTransform()
            point = transform.map(QPointF(event.pos()))
            x, y = int(point.x()), int(point.y())
            if 0 <= x < doc.width() and 0 <= y < doc.height():
                negative = bool(event.modifiers() & Qt.ControlModifier)
                self.clicked.emit(x, y, negative)
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
                                      Qt.QueuedConnection)
        self._client.health.connect(self._on_health, Qt.QueuedConnection)
        self._click_filter = CanvasClickFilter(self)
        self._click_filter.clicked.connect(self._on_canvas_click,
                                           Qt.QueuedConnection)
        self._filtered_widgets = []
        self._server_process = None
        self._busy = False
        self._build_ui()
        self._load_settings()
        QTimer.singleShot(1500, self._client.check_health)

    # ----- UI -----

    def _build_ui(self):
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header: estado + engranaje (muestra/oculta la config del server)
        header = QHBoxLayout()
        self._status = QLabel("...")
        self._status.setWordWrap(True)
        header.addWidget(self._status, 1)
        gear = QToolButton()
        gear.setIcon(Krita.instance().icon("configure"))
        gear.setAutoRaise(True)
        gear.setToolTip("Configurar server")
        gear.clicked.connect(self._toggle_config)
        header.addWidget(gear)
        layout.addLayout(header)

        # Prompt (Shift+Enter o botón = seleccionar)
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText(
            'Qué seleccionar, ej: "the red car"  (Shift+Enter)')
        self._prompt.setFixedHeight(52)
        self._prompt.setTabChangesFocus(True)
        self._prompt.setFrameStyle(QFrame.NoFrame)
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
            "Modo click: tocá un objeto en el canvas para seleccionarlo\n"
            "(Ctrl+click = excluir esa parte)")
        self._click_btn.setFixedSize(34, 32)
        self._click_btn.toggled.connect(self._toggle_click_mode)
        actions.addWidget(self._click_btn)
        layout.addLayout(actions)

        # Config del server (oculta por defecto, estilo "Connection")
        self._config = QGroupBox("Server de segmentación")
        cfg = QVBoxLayout(self._config)
        cfg.addWidget(QLabel("URL (existente o gestionado):"))
        row = QHBoxLayout()
        self._url_edit = QLineEdit(DEFAULT_URL)
        self._url_edit.editingFinished.connect(self._save_settings)
        row.addWidget(self._url_edit, 1)
        test = QPushButton("Probar")
        test.clicked.connect(self._client.check_health)
        row.addWidget(test)
        cfg.addLayout(row)
        cfg.addWidget(QLabel("Carpeta del repo krita-autoselect (local):"))
        self._repo_edit = QLineEdit()
        self._repo_edit.setPlaceholderText("D:\\krita-autoselect")
        self._repo_edit.editingFinished.connect(self._save_settings)
        cfg.addWidget(self._repo_edit)
        srv_row = QHBoxLayout()
        self._start_btn = QPushButton("Iniciar server")
        self._start_btn.clicked.connect(self._start_server)
        srv_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Parar")
        self._stop_btn.clicked.connect(self._stop_server)
        srv_row.addWidget(self._stop_btn)
        cfg.addLayout(srv_row)
        hint = QLabel(
            '<a href="https://github.com/chafamaster2000/krita-autoselect">'
            "Instalación del server (README)</a>")
        hint.setOpenExternalLinks(True)
        cfg.addWidget(hint)
        self._config.setVisible(False)
        layout.addWidget(self._config)

        layout.addStretch()
        self.setWidget(root)

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
            if obj is self._prompt and event.type() == QEvent.KeyPress:
                if event.key() in (Qt.Key_Return, Qt.Key_Enter) and \
                        event.modifiers() & Qt.ShiftModifier:
                    self._segment_from_prompt()
                    return True
        except Exception:
            pass
        return False

    # ----- settings -----

    def _load_settings(self):
        self._url_edit.setText(_read_setting("server_url", DEFAULT_URL))
        self._repo_edit.setText(_read_setting("repo_path", ""))

    def _save_settings(self):
        _write_setting("server_url", self._url_edit.text().strip())
        _write_setting("repo_path", self._repo_edit.text().strip())

    def _toggle_config(self):
        self._config.setVisible(not self._config.isVisible())

    # ----- estado -----

    def _set_status(self, text, color="grey"):
        self._status.setText(
            f'<span style="color:{self._colors[color]};">{text}</span>')

    def _on_health(self, data):
        if data.get("status") == "ok":
            loaded = data.get("loaded") or data.get("tracker_loaded")
            detail = "modelo cargado" if loaded else "modelo en frío"
            self._set_status(f"Conectado — {detail}", "green")
        else:
            self._set_status(
                "Server no conectado (engranaje → Iniciar server)", "yellow")

    # ----- server local -----

    def _repo_path(self):
        return self._repo_edit.text().strip() or "D:\\krita-autoselect"

    def _start_server(self):
        if self._server_process and self._server_process.poll() is None:
            self._set_status("El server ya está corriendo", "green")
            return
        repo = self._repo_path()
        python = os.path.join(repo, ".venv", "Scripts", "python.exe")
        server = os.path.join(repo, "server.py")
        if not os.path.exists(python) or not os.path.exists(server):
            self._set_status(
                f"No encuentro el server en {repo} (ver README)", "red")
            return
        env = dict(os.environ)
        weights = os.path.join(repo, "models", "sam3")
        if os.path.isdir(weights):
            env["AUTOSELECT_WEIGHTS_PATH"] = weights
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self._server_process = subprocess.Popen(
            [python, "-u", server], env=env, creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._set_status("Iniciando server...", "yellow")
        QTimer.singleShot(2500, self._client.check_health)

    def _stop_server(self):
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            self._set_status("Server detenido", "grey")
        else:
            self._set_status("No hay server iniciado desde acá", "grey")

    # ----- segmentación -----

    def _grab_canvas_b64(self):
        doc = Krita.instance().activeDocument()
        if doc is None:
            return None, None
        w, h = doc.width(), doc.height()
        image = doc.projection(0, 0, w, h)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        return base64.b64encode(bytes(ba)).decode("ascii"), doc

    def _segment_from_prompt(self):
        text = self._prompt.toPlainText().strip()
        if not text:
            self._set_status("Escribí qué seleccionar", "yellow")
            return
        self._segment({"text": text})

    def _on_canvas_click(self, x, y, negative):
        self._segment({"points": [[x, y]],
                       "point_labels": [0 if negative else 1]})

    def _segment(self, prompt_payload):
        if self._busy:
            return
        image_b64, doc = self._grab_canvas_b64()
        if image_b64 is None:
            self._set_status("No hay documento abierto", "red")
            return
        self._busy = True
        self._select_btn.setEnabled(False)
        self._set_status("Segmentando... (la primera vez carga el modelo)",
                         "yellow")
        payload = {"image_b64": image_b64, "combine": "union"}
        payload.update(prompt_payload)
        self._client.segment(payload)

    def _on_segment_result(self, data):
        self._busy = False
        self._select_btn.setEnabled(True)
        if data.get("error"):
            self._set_status(data["error"], "red")
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
        self._set_status(
            f"Seleccioné {data['count']} {plural} (score {best})", "green")

    def _apply_mask(self, mask_b64):
        """Máscara PNG → Selection, con el modo del combo y feather opcional.
        Mismo mecanismo que la acción select_from_mask de kritamcp."""
        doc = Krita.instance().activeDocument()
        if doc is None:
            raise RuntimeError("no hay documento")
        img = QImage()
        img.loadFromData(base64.b64decode(mask_b64))
        w, h = doc.width(), doc.height()
        if img.width() != w or img.height() != h:
            img = img.scaled(w, h, Qt.IgnoreAspectRatio,
                             Qt.SmoothTransformation)
        img = img.convertToFormat(QImage.Format_Grayscale8)
        bpl = img.bytesPerLine()
        raw = img.constBits().asstring(bpl * h)
        data = raw if bpl == w else b"".join(
            raw[i * bpl:i * bpl + w] for i in range(h))
        sel = Selection()
        sel.setPixelData(data, 0, 0, w, h)

        mode = self._mode.currentData()
        current = doc.selection()
        if mode == "replace" or current is None:
            target = sel
        elif mode == "add":
            current.add(sel)
            target = current
        elif mode == "subtract":
            current.subtract(sel)
            target = current
        else:
            current.intersect(sel)
            target = current
        feather = self._feather.value()
        if feather > 0:
            target.feather(feather)
        doc.setSelection(target)

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
                "Modo click: tocá el objeto (Ctrl+click excluye). "
                "Volvé a apretar el botón para salir.", "green")
        else:
            self._detach_click_filter()
            self._set_status("Modo click desactivado", "grey")

    def _attach_click_filter(self):
        window = Krita.instance().activeWindow()
        if window is None:
            return 0
        qwin = window.qwindow()
        targets = qwin.findChildren(QOpenGLWidget)
        if not targets:
            targets = [w for w in qwin.findChildren(QWidget)
                       if w.metaObject().className()
                       in ("KisOpenGLCanvas2", "KisQPainterCanvas")]
        for widget in targets:
            widget.installEventFilter(self._click_filter)
            self._filtered_widgets.append(widget)
        return len(self._filtered_widgets)

    def _detach_click_filter(self):
        for widget in self._filtered_widgets:
            try:
                widget.removeEventFilter(self._click_filter)
            except RuntimeError:
                pass  # widget destruido
        self._filtered_widgets = []


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("aiSelect", DockWidgetFactoryBase.DockRight,
                      AutoSelectDocker)
)
