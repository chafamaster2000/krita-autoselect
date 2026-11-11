"""Windows behaviour of the plugin and the daemon, simulated on any OS.

No Windows, Krita or torch needed: the Windows code paths run by patching
the hooks the modules already expose (backend's `platform` arguments,
manager.IS_WINDOWS, server.IS_WINDOWS), by swapping backend's `os` for one
with ntpath semantics (';' PATH, case-insensitive backslash paths), and by
faking Popen / subprocess.run / shutil.which.
"""
import errno
import ntpath
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "krita-plugin", "kritaautoselect"))
import backend as bk  # noqa: E402
import build_plugin  # noqa: E402
import engine  # noqa: E402
import manager as mg  # noqa: E402
import server  # noqa: E402

try:
    import httpx
    import download_weights as dw
except ImportError:  # pragma: no cover
    dw = None

# A user folder with a space and a non-ASCII character, and HOME pointing
# somewhere else (what Git Bash / MSYS2 / Cygwin leave behind).
USERPROFILE = "C:\\Users\\José Pérez"
WIN_ENV = {
    "HOME": "/c/Users/jose",
    "USERPROFILE": USERPROFILE,
    "LOCALAPPDATA": USERPROFILE + "\\AppData\\Local",
    "APPDATA": USERPROFILE + "\\AppData\\Roaming",
    "SYSTEMROOT": "C:\\WINDOWS",
}
KRITA_EXE = "C:\\Program Files\\Krita (x64)\\bin\\krita.exe"


def files(*paths):
    present = set(paths)
    return lambda p: p in present


def windows_os():
    """backend's `os` with Windows path semantics (the real os otherwise)."""
    fake = types.SimpleNamespace(
        **{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    fake.path, fake.sep, fake.altsep = ntpath, "\\", "/"
    fake.pathsep, fake.name = ";", "nt"
    return mock.patch.object(bk, "os", fake)


class WindowsPathsTest(unittest.TestCase):
    def test_user_folders_follow_userprofile_not_home(self):
        """huggingface_hub and uv's installer resolve ~ via USERPROFILE
        (os.path.expanduser ignores HOME on Windows since Python 3.8)."""
        token = USERPROFILE + "\\.cache\\huggingface\\token"

        def read(path):
            if path == token:
                return "hf_from_login\r\n"
            raise OSError(path)

        with windows_os():
            self.assertEqual(bk.hf_token_paths(WIN_ENV, "win32"), [token])
            self.assertEqual(bk.find_hf_token("", WIN_ENV, read, "win32"),
                             ("hf_from_login",
                              f"login de Hugging Face ({token})"))
        self.assertEqual(
            bk.find_uv(WIN_ENV, "win32", which=lambda *a, **k: None,
                       isfile=files(USERPROFILE + "\\.local\\bin\\uv.exe")),
            USERPROFILE + "\\.local\\bin\\uv.exe")
        env = {k: v for k, v in WIN_ENV.items() if k != "LOCALAPPDATA"}
        self.assertEqual(bk.default_data_dir("win32", env),
                         USERPROFILE + "\\AppData\\Local\\krita-autoselect")
        # Linux/macOS keep preferring HOME.
        self.assertEqual(bk.hf_token_paths({"HOME": "/h", "USERPROFILE": "x"},
                                           "linux"),
                         ["/h/.cache/huggingface/token"])

    def test_hf_home_and_appdata_paths(self):
        env = dict(WIN_ENV, HF_HOME="D:\\hf\\")
        with windows_os():
            self.assertEqual(bk.hf_token_paths(env, "win32")[0],
                             "D:\\hf\\token")
        self.assertEqual(
            bk.ai_diffusion_settings_path("win32", WIN_ENV),
            USERPROFILE + "\\AppData\\Roaming\\krita\\ai_diffusion\\"
                          "settings.json")
        # Big, machine-specific data (venv + 3.4 GB of weights) must not
        # roam: LOCALAPPDATA, not APPDATA.
        self.assertEqual(bk.default_data_dir("win32", WIN_ENV),
                         USERPROFILE + "\\AppData\\Local\\krita-autoselect")

    def test_install_plan_survives_spaces_and_accents(self):
        data = USERPROFILE + "\\AppData\\Local\\krita-autoselect"
        bundle = (USERPROFILE + "\\AppData\\Roaming\\krita\\pykrita\\"
                  "kritaautoselect\\daemon")
        uv = USERPROFILE + "\\.local\\bin\\uv.exe"
        with windows_os():
            steps = {s.key: s for s in bk.build_install_plan(
                data, bundle, platform="win32", uv=uv,
                gpu={"name": "RTX 3090", "compute_cap": (8, 6)},
                hf_token="tok", isfile=files(
                    bundle + "\\requirements.txt",
                    bundle + "\\download_weights.py"))}
        py = data + "\\.venv\\Scripts\\python.exe"
        self.assertEqual(steps["venv"].argv, [uv, "venv", "--python", "3.12",
                                              "--allow-existing",
                                              data + "\\.venv"])
        self.assertIn(py, steps["torch"].argv)
        self.assertEqual(steps["torch"].argv[-1],
                         "https://download.pytorch.org/whl/cu128")
        self.assertEqual(steps["weights"].argv,
                         [py, "-u", bundle + "\\download_weights.py",
                          data + "\\models\\sam3"])
        # Popen hands CreateProcessW one command line: every path with a
        # space comes out quoted as a single argument.
        line = subprocess.list2cmdline(steps["weights"].argv)
        self.assertTrue(line.startswith(f'"{py}" -u "{bundle}'), line)

    def test_run_daemon_ps1_and_plugin_agree_on_the_repo_layout(self):
        """The plugin pointed at the dev repo runs what run_daemon.ps1 runs."""
        with open(os.path.join(ROOT, "run_daemon.ps1"), encoding="utf-8") as f:
            joins = dict(re.findall(r'\$(\w+) = Join-Path \$repo "([^"]+)"',
                                    f.read()))
        repo = "D:\\src\\krita-autoselect"
        isfile = files(repo + "\\" + joins["python"],
                       repo + "\\" + joins["server"],
                       *(repo + "\\" + joins["weights"] + "\\" + name
                         for name in bk.WEIGHT_FILES))
        with windows_os():
            inst = bk.resolve_install(repo, "", "", "win32", isfile)
        self.assertTrue(inst.ok)
        self.assertEqual(inst.python, repo + "\\" + joins["python"])
        self.assertEqual(inst.server_script, repo + "\\" + joins["server"])
        self.assertEqual(inst.weights, repo + "\\" + joins["weights"])


class WindowsEnvTest(unittest.TestCase):
    def test_child_env_drops_krita_python_and_bin_case_insensitively(self):
        with windows_os():
            prefixes = bk.krita_prefixes({}, KRITA_EXE)
            self.assertEqual(prefixes, ["C:\\Program Files\\Krita (x64)"])
            env = bk.sanitize_env(dict(
                WIN_ENV,
                PATH="c:\\program files\\krita (x64)\\bin\\;C:\\WINDOWS\\"
                     "system32;" + USERPROFILE + "\\.local\\bin",
                PYTHONHOME="C:\\Program Files\\Krita (x64)\\python",
                PYTHONPATH="C:\\Program Files\\Krita (x64)\\lib\\"
                           "krita-python-libs"), prefixes)
        self.assertEqual(env["PATH"], "C:\\WINDOWS\\system32;" + USERPROFILE
                         + "\\.local\\bin")
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        # Python on Windows can't even seed its RNG / start Winsock without
        # SYSTEMROOT: it must survive.
        self.assertEqual(env["SYSTEMROOT"], "C:\\WINDOWS")

    def test_tools_bundled_with_krita_or_store_stubs_are_skipped(self):
        found = {
            "py.exe": None,
            "python.exe": "C:\\Program Files\\Krita (x64)\\python\\python.exe",
            "uv.exe": "C:\\Program Files\\Krita (x64)\\bin\\uv.exe",
        }
        prefixes = ["C:\\Program Files\\Krita (x64)"]
        with windows_os():
            self.assertIsNone(bk.find_system_python(
                WIN_ENV, "win32", lambda n, path=None: found.get(n),
                files(), prefixes))
            self.assertIsNone(bk.find_uv(
                WIN_ENV, "win32", lambda n, path=None: found.get(n),
                files(), prefixes))
            self.assertEqual(bk.find_system_python(
                WIN_ENV, "win32",
                lambda n, path=None: {"py.exe": "C:\\WINDOWS\\py.exe"}.get(n),
                files(), prefixes), ["C:\\WINDOWS\\py.exe", "-3"])

    def test_system_python_version_is_checked(self):
        """py -3 resolving to an old Python (3.8) must not build the venv."""
        which = lambda n, path=None: {"py.exe": "C:\\WINDOWS\\py.exe"}.get(n)
        calls = []

        def run(answer):
            def _run(argv, **kw):
                calls.append(argv[:2])
                return types.SimpleNamespace(returncode=0, stdout=answer)
            return _run

        with windows_os():
            argv, problem = bk.pick_system_python(
                WIN_ENV, "win32", which, files(), run=run("3.8 1\n"))
            self.assertIsNone(argv)
            self.assertIn("3.10", problem)
            self.assertEqual(calls, [["C:\\WINDOWS\\py.exe", "-3"]])
            argv, problem = bk.pick_system_python(
                WIN_ENV, "win32", which, files(), run=run("3.12 1\n"))
            self.assertEqual((argv, problem), (["C:\\WINDOWS\\py.exe", "-3"], ""))
            self.assertEqual(bk.pick_system_python(
                WIN_ENV, "win32", lambda n, path=None: None, files(),
                run=run("")), (None, ""))

    def test_nvidia_smi_fallback_uses_uppercased_systemroot(self):
        """os.environ copies have uppercase keys on Windows: 'SYSTEMROOT'."""
        calls = []

        def run(argv, **kw):
            calls.append((argv, kw))
            return subprocess.CompletedProcess(
                argv, 0, "NVIDIA GeForce GTX 1080, 6.1\n", "")

        smi = "D:\\WINDOWS\\System32\\nvidia-smi.exe"
        with windows_os():
            gpu = bk.detect_nvidia_gpu(
                {"SYSTEMROOT": "D:\\WINDOWS", "PATH": ""}, "win32",
                which=lambda *a, **k: None, run=run, isfile=files(smi))
        self.assertEqual(gpu, {"name": "NVIDIA GeForce GTX 1080",
                               "compute_cap": (6, 1)})
        self.assertEqual(calls[0][0][0], smi)
        self.assertEqual(calls[0][1]["creationflags"], mg.CREATE_NO_WINDOW)
        self.assertTrue(bk.torch_variant("win32", gpu)[0].endswith("/cu126"))


class WindowsDaemonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="autoselect-win-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_daemon_cwd_does_not_lock_the_plugin_folder(self):
        """Krita's 'Import Python Plugin from File' does rmtree(pykrita/
        kritaautoselect) then copytree. On Windows a folder that is some
        process's cwd can't be removed, so a daemon running with its cwd in
        daemon/ made the update fail halfway, leaving the plugin without its
        .py files."""
        pykrita = os.path.join(self.tmp, "pykrita")
        build_plugin.install(pykrita)
        plugin = os.path.join(pykrita, "kritaautoselect")
        bundle = os.path.join(plugin, "daemon")
        cfg = mg.Config(url="http://127.0.0.1:5679", bundled_dir=bundle,
                        install_dir=os.path.join(self.tmp, "install"),
                        comfyui_free=False,
                        data_dir=os.path.join(self.tmp, "data"))
        install = bk.Install(cfg.install_dir, "C:\\venv\\Scripts\\python.exe",
                             os.path.join(bundle, "server.py"), None)
        calls = []

        def popen(argv, **kw):
            calls.append((argv, kw))
            return types.SimpleNamespace(pid=4242)

        m = mg.DaemonManager(cfg, popen=popen, base_env={"SYSTEMROOT": "C:\\"})
        with mock.patch.object(mg, "IS_WINDOWS", True):
            m._spawn(mg._Session(cfg, install))
        argv, kw = calls[0]
        self.assertEqual(argv[1:], ["-u", os.path.join(bundle, "server.py")])
        self.assertEqual(kw["creationflags"], mg.CREATE_NO_WINDOW)
        self.assertIs(kw["stdin"], subprocess.PIPE)  # parent watch
        self.assertEqual(kw["env"]["AUTOSELECT_PARENT_WATCH"], "stdin")
        self.assertEqual(kw["env"]["PYTHONIOENCODING"], "utf-8")
        cwd = os.path.abspath(kw["cwd"])
        self.assertTrue(os.path.isdir(cwd))

        # Krita's importer, with Windows' rule for a running process's cwd.
        def windows_rmtree(path):
            for root, _dirs, names in os.walk(path, topdown=False):
                for name in names:
                    os.remove(os.path.join(root, name))
                if os.path.abspath(root) == cwd:
                    raise PermissionError(errno.EACCES, "[WinError 32] The "
                                          "process cannot access the file "
                                          "because it is being used by "
                                          "another process", root)
                os.rmdir(root)

        windows_rmtree(plugin)
        fresh = os.path.join(self.tmp, "fresh")
        build_plugin.install(fresh)
        shutil.copytree(os.path.join(fresh, "kritaautoselect"), plugin)
        self.assertTrue(os.path.isfile(os.path.join(plugin, "__init__.py")))

    def test_stop_without_sigterm(self):
        """Windows has no SIGTERM: stop = close stdin (the daemon exits on
        EOF), TerminateProcess, and kill if it is still there."""
        order = []

        class Proc:
            stdin = types.SimpleNamespace(close=lambda: order.append("eof"))

            def poll(self):
                return None

            def terminate(self):
                order.append("terminate")

            def kill(self):
                order.append("kill")

            def wait(self, timeout=None):
                order.append("wait")
                if order.count("wait") == 1:
                    raise subprocess.TimeoutExpired("python.exe", timeout)
                return 1

        mg.DaemonManager._terminate(Proc(), wait_s=0.01)
        self.assertEqual(order, ["eof", "terminate", "wait", "kill", "wait"])

    def test_windows_crash_codes_restart(self):
        p = bk.RestartPolicy()
        # torch dying natively: access violation / fail-fast (NTSTATUS).
        self.assertEqual(p.on_exit(0xC0000005, True, 0)[2], "crash")
        self.assertEqual(p.on_exit(0xC0000409, True, 1)[2], "crash")
        self.assertEqual(bk.describe_exit(0xC0000409), "código 0xC0000409")
        # our own TerminateProcess (exit code 1) is never a crash
        self.assertEqual(p.on_exit(1, True, 2, stop_requested=True)[2],
                         "stopped")

    def test_log_with_crlf_and_spanish_windows_errors(self):
        log = os.path.join(self.tmp, "daemon.log")
        with open(log, "wb") as f:
            f.write("===== AI Select inicia el daemon =====\n".encode("utf-8"))
            f.write("[autoselect] error: cannot listen on 127.0.0.1:5679: "
                    "[WinError 10048] Solo se permite un uso de cada "
                    "dirección de socket\r\n".encode("utf-8"))
        tail = bk.tail_file(log)
        self.assertNotIn("\r", tail)
        self.assertIn("dirección", bk.last_error_line(tail))


class WindowsInstallerTest(unittest.TestCase):
    def test_steps_run_hidden_in_their_own_group(self):
        calls = []

        def popen(argv, **kw):
            calls.append(kw)
            return types.SimpleNamespace(stdout=iter(["Instalando…\n"]),
                                         wait=lambda: 0, pid=7)

        lines = []
        r = mg.InstallRunner(on_line=lines.append, popen=popen,
                             base_env={"SYSTEMROOT": "C:\\WINDOWS"})
        with mock.patch.object(mg, "IS_WINDOWS", True):
            rc = r._run_step(bk.Step("x", "x", ["uv.exe", "pip"]), None)
        self.assertEqual(rc, 0)
        kw = calls[0]
        self.assertEqual(kw["creationflags"],
                         mg.CREATE_NO_WINDOW | mg.CREATE_NEW_PROCESS_GROUP)
        self.assertNotIn("start_new_session", kw)  # POSIX-only
        self.assertEqual((kw["encoding"], kw["errors"]), ("utf-8", "replace"))
        self.assertEqual(kw["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(lines, ["  Instalando…"])

    def test_cancel_kills_the_whole_tree_with_taskkill(self):
        with mock.patch.object(mg, "IS_WINDOWS", True), \
                mock.patch("subprocess.run") as run:
            mg._kill_tree(types.SimpleNamespace(pid=1234))
        argv = run.call_args[0][0]
        self.assertEqual(argv, ["taskkill", "/F", "/T", "/PID", "1234"])
        self.assertEqual(run.call_args[1]["creationflags"],
                         mg.CREATE_NO_WINDOW)


class FakeEngine:
    def info(self):
        return {"model": "fake", "loaded": False}


class WindowsServerTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(socket, "SO_REUSEADDR"), "no SO_REUSEADDR")
    def test_no_reuseaddr_on_windows(self):
        """On Windows SO_REUSEADDR lets a second socket bind a port that is
        still listening: two daemons would share 5679 with no 'address in
        use' (the exit-3 path) and requests would land on either."""
        with mock.patch.object(server, "IS_WINDOWS", True):
            srv = server.make_server(FakeEngine(), port=0)
        self.addCleanup(srv.server_close)
        self.assertEqual(srv.socket.getsockopt(socket.SOL_SOCKET,
                                               socket.SO_REUSEADDR), 0)
        with mock.patch.object(server, "IS_WINDOWS", True):
            with self.assertRaises(OSError):
                server.make_server(FakeEngine(), port=srv.server_address[1])
        # POSIX keeps it (restart through TIME_WAIT).
        with mock.patch.object(server, "IS_WINDOWS", False):
            posix = server.make_server(FakeEngine(), port=0)
        self.addCleanup(posix.server_close)
        self.assertNotEqual(posix.socket.getsockopt(socket.SOL_SOCKET,
                                                    socket.SO_REUSEADDR), 0)

    def test_nvidia_smi_does_not_inherit_the_parent_watch_pipe(self):
        """The daemon's stdin is the pipe a watcher thread is blocked
        reading. On Windows that pending ReadFile locks the pipe for any
        child that inherits it (its CRT calls GetFileType on startup), so
        nvidia-smi hung until the 10 s timeout on every model load."""
        done = subprocess.CompletedProcess([], 0, "20480\n", "")
        with mock.patch("shutil.which", return_value="nvidia-smi.exe"), \
                mock.patch("subprocess.run", return_value=done) as run:
            free = engine.Sam3Engine._free_vram_nvidia_smi()
        self.assertEqual(free, 20480 * 1024 ** 2)
        self.assertIs(run.call_args[1]["stdin"], subprocess.DEVNULL)


PAYLOAD = bytes(range(256)) * 64


class _Files(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)


@unittest.skipIf(dw is None, "needs httpx + huggingface_hub (backend venv)")
class WindowsDownloadTest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Files)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/f"
        self.dest = os.path.join(tempfile.mkdtemp(), "model.safetensors")

    def locked(self, times):
        """os.replace failing like Windows while an antivirus scans the
        freshly closed .part ([WinError 32])."""
        real, state = os.replace, {"left": times}

        def replace(src, dst):
            if state["left"]:
                state["left"] -= 1
                raise PermissionError(errno.EACCES, "[WinError 32] The "
                                      "process cannot access the file", src)
            return real(src, dst)
        return replace

    def test_final_rename_waits_out_a_transient_lock(self):
        with mock.patch.object(dw, "REPLACE_DELAY_S", 0), \
                mock.patch.object(dw.os, "replace", self.locked(3)), \
                httpx.Client() as client:
            dw.download_file(client, self.url, {}, self.dest, len(PAYLOAD))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_permanent_lock_still_fails_and_keeps_the_part(self):
        with mock.patch.object(dw, "REPLACE_DELAY_S", 0), \
                mock.patch.object(dw, "REPLACE_ATTEMPTS", 3), \
                mock.patch.object(dw.os, "replace", self.locked(99)), \
                httpx.Client() as client:
            with self.assertRaises(PermissionError):
                dw.download_file(client, self.url, {}, self.dest,
                                 len(PAYLOAD))
        # the next run resumes from the complete .part
        self.assertEqual(os.path.getsize(self.dest + ".part"), len(PAYLOAD))


if __name__ == "__main__":
    unittest.main()
