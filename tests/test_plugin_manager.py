"""DaemonManager / InstallRunner with real processes, stdlib only.

The managed "daemon" is the real server.py run by this test's Python: its
/health answers without torch (the model only loads on /segment), so the
whole start → reuse → crash/restart → stop lifecycle runs in a second or two.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "krita-plugin", "kritaautoselect"))
import backend as bk  # noqa: E402
import manager as mg  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="autoselect-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # A fake install whose "venv python" is this interpreter.
        venv_bin = os.path.join(self.tmp, "install", ".venv",
                                "Scripts" if os.name == "nt" else "bin")
        os.makedirs(venv_bin)
        python = os.path.join(venv_bin, "python.exe" if os.name == "nt"
                              else "python")
        try:
            os.symlink(sys.executable, python)
        except (OSError, NotImplementedError):
            shutil.copy(sys.executable, python)
        self.bundle = os.path.join(self.tmp, "daemon")
        os.makedirs(self.bundle)
        for name in ("server.py", "engine.py"):
            shutil.copy(os.path.join(ROOT, name), self.bundle)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"

    def make(self, url=None, **kw):
        cfg = mg.Config(url=url or self.url,
                        install_dir=os.path.join(self.tmp, "install"),
                        bundled_dir=self.bundle, comfyui_free=False,
                        data_dir=os.path.join(self.tmp, "data"))
        m = mg.DaemonManager(cfg, base_env=dict(os.environ),
                             startup_timeout_s=kw.pop("startup", 20), **kw)
        self.addCleanup(m.shutdown)
        self.states = []
        m.add_listener(lambda s: self.states.append(s["state"]))
        return m

    def ensure(self, m, timeout=30):
        box = {}
        done = threading.Event()
        m.ensure_ready(lambda ok, msg: (box.update(ok=ok, msg=msg), done.set()))
        self.assertTrue(done.wait(timeout), "ensure_ready never called back")
        return box["ok"], box["msg"]

    def test_starts_reuses_and_stops_its_own_daemon(self):
        m = self.make()
        ok, msg = self.ensure(m)
        self.assertTrue(ok, msg)
        snap = m.snapshot()
        self.assertEqual(snap["state"], m.READY)
        self.assertTrue(snap["owned"])
        pid = snap["pid"]
        self.assertIn(m.STARTING, self.states)
        # log file gets the daemon's output
        self.assertTrue(wait_for(lambda: "listening" in bk.tail_file(
            m.config.daemon_log)))
        # second request: same process
        ok, _ = self.ensure(m)
        self.assertTrue(ok)
        self.assertEqual(m.snapshot()["pid"], pid)
        m.shutdown()
        self.assertTrue(wait_for(lambda: bk.probe_health(self.url, 0.5).kind
                                 == "down"))
        self.assertTrue(wait_for(lambda: m.snapshot()["state"] == m.STOPPED))

    def test_reuses_existing_daemon_and_never_stops_it(self):
        env = dict(os.environ, AUTOSELECT_PORT=str(self.port))
        ext = subprocess.Popen([sys.executable, os.path.join(ROOT, "server.py")],
                               env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        self.addCleanup(ext.wait)
        self.addCleanup(ext.terminate)
        self.assertTrue(wait_for(lambda: bk.probe_health(self.url, 0.5).kind
                                 == "autoselect"))
        m = self.make()
        ok, _ = self.ensure(m)
        self.assertTrue(ok)
        self.assertFalse(m.snapshot()["owned"])
        self.assertFalse(m.stop())  # nothing of ours to stop
        m.shutdown()
        time.sleep(0.3)
        self.assertIsNone(ext.poll())

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_crash_is_restarted(self):
        m = self.make()
        self.assertTrue(self.ensure(m)[0])
        pid = m.snapshot()["pid"]
        os.kill(pid, signal.SIGKILL)  # native-crash stand-in
        self.assertTrue(wait_for(lambda: m.snapshot()["state"] == m.READY
                                 and m.snapshot()["pid"] not in (None, pid),
                                 timeout=20))
        self.assertIn("murió", bk.tail_file(m.config.daemon_log))

    def test_startup_failure_is_an_error_without_restart_loop(self):
        with open(os.path.join(self.bundle, "server.py"), "w") as f:
            f.write("raise SystemExit('boom: missing dependency')\n")
        m = self.make()
        ok, msg = self.ensure(m)
        self.assertFalse(ok)
        self.assertIn("boom", msg)
        self.assertEqual(m.snapshot()["state"], m.ERROR)
        self.assertFalse(m.owned_alive)

    def test_port_taken_by_other_program(self):
        class Other(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<html>not us</html>")

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", self.port), Other)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        m = self.make()
        ok, msg = self.ensure(m)
        self.assertFalse(ok)
        self.assertIn(str(self.port), msg)
        self.assertFalse(m.owned_alive)

    def test_not_installed(self):
        m = self.make()
        m.configure(mg.Config(url=self.url, install_dir=os.path.join(
            self.tmp, "empty"), bundled_dir=self.bundle, comfyui_free=False,
            data_dir=os.path.join(self.tmp, "data")))
        ok, msg = self.ensure(m)
        self.assertFalse(ok)
        self.assertEqual(m.snapshot()["state"], m.NOT_INSTALLED)

    @unittest.skipIf(os.name == "nt", "POSIX pipe semantics")
    def test_daemon_exits_when_parent_stdin_closes(self):
        """Krita killed hard: its end of the stdin pipe closes → daemon exits."""
        env = dict(os.environ, AUTOSELECT_PORT=str(self.port),
                   AUTOSELECT_PARENT_WATCH="stdin")
        proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "server.py")],
                                env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL)
        self.assertTrue(wait_for(lambda: bk.probe_health(self.url, 0.5).kind
                                 == "autoselect"))
        proc.stdin.close()
        self.assertEqual(proc.wait(timeout=10), 0)


class InstallRunnerTest(unittest.TestCase):
    def run_plan(self, steps, cancel_after=None):
        lines, done = [], threading.Event()
        result = {}
        r = mg.InstallRunner(on_line=lines.append,
                             on_done=lambda ok, msg: (result.update(
                                 ok=ok, msg=msg), done.set()),
                             base_env=dict(os.environ))
        log = os.path.join(tempfile.mkdtemp(), "install.log")
        r.start(lambda: steps, log)
        if cancel_after:
            time.sleep(cancel_after)
            r.cancel()
        self.assertTrue(done.wait(20))
        return result, lines, log

    def test_runs_steps_and_logs(self):
        steps = [bk.Step("a", "skip me", skip_reason="ya está"),
                 bk.Step("b", "hello", [sys.executable, "-c", "print('hola')"])]
        result, lines, log = self.run_plan(steps)
        self.assertTrue(result["ok"])
        self.assertTrue(any("hola" in l for l in lines))
        self.assertTrue(any("ya está" in l for l in lines))
        with open(log, encoding="utf-8") as f:
            self.assertIn("hola", f.read())

    def test_failure_stops_the_plan(self):
        steps = [bk.Step("a", "fails", [sys.executable, "-c", "raise SystemExit(2)"]),
                 bk.Step("b", "never", [sys.executable, "-c", "print('nope')"])]
        result, lines, _ = self.run_plan(steps)
        self.assertFalse(result["ok"])
        self.assertIn("fails", result["msg"])
        self.assertFalse(any("nope" in l for l in lines))

    def test_cancel_kills_running_step(self):
        steps = [bk.Step("a", "slow", [sys.executable, "-c",
                                        "import time; print('go', flush=True);"
                                        " time.sleep(60)"])]
        t0 = time.monotonic()
        result, _, _ = self.run_plan(steps, cancel_after=1.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["msg"], "cancelado")
        self.assertLess(time.monotonic() - t0, 10)

    def test_plan_error_is_reported(self):
        def boom():
            raise bk.PlanError("falta token")
        done, result = threading.Event(), {}
        r = mg.InstallRunner(on_done=lambda ok, msg: (result.update(ok=ok, msg=msg),
                                                      done.set()))
        r.start(boom)
        self.assertTrue(done.wait(5))
        self.assertEqual(result, {"ok": False, "msg": "falta token"})


if __name__ == "__main__":
    unittest.main()
