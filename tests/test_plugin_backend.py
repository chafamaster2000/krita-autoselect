"""Pure logic of the Krita plugin's backend management (backend.py).

Stdlib only: no Krita, no Qt, no torch. backend.py is loaded straight from
the plugin directory (the package __init__ imports krita, so it can't be
imported as a package here).
"""
import os
import sys
import unittest

PLUGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "krita-plugin", "kritaautoselect")
sys.path.insert(0, PLUGIN_DIR)
import backend as bk  # noqa: E402


def files(*paths):
    present = set(paths)
    return lambda p: p in present


class PathsTest(unittest.TestCase):
    def test_platform_id(self):
        self.assertEqual(bk.platform_id("win32"), "windows")
        self.assertEqual(bk.platform_id("darwin"), "macos")
        self.assertEqual(bk.platform_id("linux"), "linux")

    def test_default_data_dir_per_os(self):
        self.assertEqual(
            bk.default_data_dir("linux", {"HOME": "/home/u"}),
            "/home/u/.local/share/krita-autoselect")
        self.assertEqual(
            bk.default_data_dir("linux", {"HOME": "/home/u",
                                          "XDG_DATA_HOME": "/data"}),
            "/data/krita-autoselect")
        self.assertEqual(
            bk.default_data_dir("win32", {"USERPROFILE": "C:\\Users\\u",
                                          "LOCALAPPDATA":
                                          "C:\\Users\\u\\AppData\\Local"}),
            "C:\\Users\\u\\AppData\\Local\\krita-autoselect")
        self.assertEqual(
            bk.default_data_dir("darwin", {"HOME": "/Users/u"}),
            "/Users/u/Library/Application Support/krita-autoselect")

    def test_venv_python_per_os(self):
        self.assertEqual(bk.venv_python("/x/.venv", "linux"),
                         "/x/.venv/bin/python")
        self.assertEqual(bk.venv_python("D:\\x\\.venv", "win32"),
                         "D:\\x\\.venv\\Scripts\\python.exe")

    def test_find_venv_python_prefers_dot_venv_then_venv(self):
        self.assertEqual(
            bk.find_venv_python("/r", "linux", files("/r/venv/bin/python")),
            "/r/venv/bin/python")
        self.assertEqual(
            bk.find_venv_python("/r", "linux", files(
                "/r/.venv/bin/python", "/r/venv/bin/python")),
            "/r/.venv/bin/python")
        self.assertEqual(
            bk.find_venv_python("D:\\r", "win32",
                                files("D:\\r\\.venv\\Scripts\\python.exe")),
            "D:\\r\\.venv\\Scripts\\python.exe")
        self.assertIsNone(bk.find_venv_python("/r", "linux", files()))
        self.assertIsNone(bk.find_venv_python("", "linux", files()))

    def test_existing_repo_install_resolves(self):
        """The dev repo (.venv + models/sam3) works with no reinstall."""
        isfile = files("/repo/.venv/bin/python", "/repo/server.py",
                       "/repo/models/sam3/config.json",
                       "/repo/models/sam3/model.safetensors")
        inst = bk.resolve_install("/repo", "", "/plugin/daemon", "linux", isfile)
        self.assertTrue(inst.ok)
        self.assertEqual(inst.python, "/repo/.venv/bin/python")
        self.assertEqual(inst.server_script, "/repo/server.py")
        self.assertEqual(inst.weights, "/repo/models/sam3")

    def test_bundled_server_wins_over_install_copy(self):
        isfile = files("/repo/.venv/bin/python", "/repo/server.py",
                       "/plugin/daemon/server.py")
        inst = bk.resolve_install("/repo", "", "/plugin/daemon", "linux", isfile)
        self.assertEqual(inst.server_script, "/plugin/daemon/server.py")
        self.assertIsNone(inst.weights)  # -> Hugging Face cache

    def test_weights_setting_wins(self):
        self.assertEqual(bk.resolve_weights("/w", "/repo", "linux", files()),
                         "/w")

    def test_missing_venv_is_a_readable_problem(self):
        inst = bk.resolve_install("/nowhere", "", "/plugin/daemon", "linux",
                                  files("/plugin/daemon/server.py"))
        self.assertFalse(inst.ok)
        self.assertIn(".venv", inst.problem())


class DecisionTest(unittest.TestCase):
    def test_reuse_any_healthy_daemon(self):
        for owned in (False, True):
            self.assertEqual(bk.decide_action("autoselect", True, True, owned),
                             bk.REUSE)
        # even without a local install, or with a remote URL
        self.assertEqual(bk.decide_action("autoselect", False, False, False),
                         bk.REUSE)

    def test_start_only_when_nobody_answers_locally_and_installed(self):
        self.assertEqual(bk.decide_action("down", True, True, False), bk.START)
        self.assertEqual(bk.decide_action("down", True, True, True), bk.WAIT)
        self.assertEqual(bk.decide_action("down", True, False, False),
                         bk.NOT_INSTALLED)
        self.assertEqual(bk.decide_action("down", False, True, False),
                         bk.REMOTE_DOWN)

    def test_port_taken_by_another_program(self):
        self.assertEqual(bk.decide_action("other", True, True, False),
                         bk.PORT_BUSY)

    def test_classify_health(self):
        ok = bk.classify_health(200, b'{"status": "ok", "service": '
                                     b'"krita-autoselect", "loaded": true}')
        self.assertEqual(ok.kind, "autoselect")
        self.assertTrue(ok.loaded)
        self.assertFalse(ok.loading)
        self.assertEqual(bk.classify_health(200, b"<html>").kind, "other")
        self.assertEqual(bk.classify_health(404, b'{"detail": "x"}').kind,
                         "other")

    def test_parse_url(self):
        self.assertEqual(bk.parse_url("http://127.0.0.1:5679"),
                         ("127.0.0.1", 5679))
        self.assertEqual(bk.parse_url("localhost:9000/x"), ("localhost", 9000))
        self.assertEqual(bk.parse_url("http://[::1]:5"), ("::1", 5))
        self.assertTrue(bk.is_local_host("localhost"))
        self.assertFalse(bk.is_local_host("192.168.0.2"))

    def test_probe_health_down(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens there now
        self.assertEqual(bk.probe_health(f"http://127.0.0.1:{port}", 1).kind,
                         "down")


class RestartPolicyTest(unittest.TestCase):
    def test_crash_restarts_with_backoff(self):
        p = bk.RestartPolicy(max_crashes=3, window_s=600, base_delay_s=2)
        self.assertEqual(p.on_exit(-11, True, 100), (True, 2, "crash"))
        self.assertEqual(p.on_exit(-6, True, 110), (True, 4, "crash"))

    def test_crash_loop_stops(self):
        p = bk.RestartPolicy(max_crashes=3, window_s=600)
        for t in (0, 10, 20):
            self.assertTrue(p.on_exit(-9, True, t)[0])
        self.assertEqual(p.on_exit(-9, True, 30)[2], "crash_loop")

    def test_old_crashes_expire(self):
        p = bk.RestartPolicy(max_crashes=1, window_s=60)
        self.assertTrue(p.on_exit(-11, True, 0)[0])
        self.assertTrue(p.on_exit(-11, True, 1000)[0])

    def test_no_restart_for_startup_failure_or_intentional_exit(self):
        p = bk.RestartPolicy()
        self.assertEqual(p.on_exit(1, False, 0)[2], "startup_failed")
        self.assertEqual(p.on_exit(0, True, 0)[2], "exited")
        self.assertEqual(p.on_exit(-11, True, 0, stop_requested=True)[2],
                         "stopped")
        if os.name != "nt":
            self.assertEqual(p.on_exit(-15, True, 0)[2], "exited")

    def test_describe_exit(self):
        self.assertIn("SIGKILL", bk.describe_exit(-9))
        self.assertEqual(bk.describe_exit(3), "código 3")
        self.assertIn("C0000005", bk.describe_exit(0xC0000005))


class EnvTest(unittest.TestCase):
    def test_sanitize_drops_krita_python_and_appimage_paths(self):
        env = {"PYTHONHOME": "/tmp/.mount_krita/usr",
               "PYTHONPATH": "/tmp/.mount_krita/usr/lib/python",
               "PATH": "/tmp/.mount_krita/usr/bin:/usr/bin:/home/u/.local/bin",
               "LD_LIBRARY_PATH": "/tmp/.mount_krita/usr/lib",
               "HOME": "/home/u"}
        out = bk.sanitize_env(env, ["/tmp/.mount_krita"])
        self.assertNotIn("PYTHONHOME", out)
        self.assertNotIn("PYTHONPATH", out)
        self.assertNotIn("LD_LIBRARY_PATH", out)
        self.assertEqual(out["PATH"], "/usr/bin:/home/u/.local/bin")
        self.assertEqual(out["HOME"], "/home/u")

    def test_krita_prefixes(self):
        self.assertEqual(bk.krita_prefixes({"APPDIR": "/tmp/.mount_x/"},
                                           "/usr/bin/python3"),
                         ["/tmp/.mount_x"])
        self.assertEqual(bk.krita_prefixes({}, "/opt/krita/bin/krita"),
                         ["/opt/krita"])

    def test_daemon_env(self):
        env = bk.daemon_env({"HOME": "/h"}, 5679, "/w", 300,
                            "http://127.0.0.1:8188", "tok")
        self.assertEqual(env["AUTOSELECT_PORT"], "5679")
        self.assertEqual(env["AUTOSELECT_WEIGHTS_PATH"], "/w")
        self.assertEqual(env["AUTOSELECT_COMFYUI_URL"], "http://127.0.0.1:8188")
        self.assertEqual(env["AUTOSELECT_PARENT_WATCH"], "stdin")
        self.assertEqual(env["HF_TOKEN"], "tok")
        bare = bk.daemon_env({}, 1, None, 0, "", "")
        self.assertNotIn("AUTOSELECT_WEIGHTS_PATH", bare)
        self.assertNotIn("HF_TOKEN", bare)


class ComfyAndTokenTest(unittest.TestCase):
    def test_comfyui_candidates_and_pick(self):
        self.assertEqual(bk.comfyui_candidates("127.0.0.1:9000"),
                         ["http://127.0.0.1:9000"])
        cands = bk.comfyui_candidates("", "http://127.0.0.1:8188")
        self.assertEqual(cands, ["http://127.0.0.1:8188",
                                 "http://127.0.0.1:8001"])
        self.assertEqual(
            bk.pick_comfyui_url(cands, lambda u: u.endswith("8001")),
            "http://127.0.0.1:8001")
        # nobody answers (ComfyUI restarting): keep the first, /free is
        # best-effort
        self.assertEqual(bk.pick_comfyui_url(cands, lambda u: False),
                         "http://127.0.0.1:8188")

    def test_ai_diffusion_settings_path(self):
        self.assertEqual(
            bk.ai_diffusion_settings_path("linux", {"HOME": "/h"}),
            "/h/.local/share/krita/ai_diffusion/settings.json")
        self.assertTrue(bk.ai_diffusion_settings_path(
            "win32", {"APPDATA": "C:\\A"}).endswith(
                "krita\\ai_diffusion\\settings.json"))

    def test_find_hf_token_order(self):
        cache = {"/h/.cache/huggingface/token": "cached\n"}

        def read(path):
            if path in cache:
                return cache[path]
            raise OSError(path)

        env = {"HOME": "/h"}
        self.assertEqual(bk.find_hf_token(" set ", env, read)[0], "set")
        self.assertEqual(bk.find_hf_token("", dict(env, HF_TOKEN="e"), read)[0],
                         "e")
        self.assertEqual(bk.find_hf_token("", env, read)[0], "cached")
        self.assertEqual(bk.find_hf_token("", {"HOME": "/x"}, read),
                         (None, None))


class InstallPlanTest(unittest.TestCase):
    BUNDLE = staticmethod(files("/p/daemon/requirements.txt",
                                "/p/daemon/download_weights.py"))

    def plan(self, **kw):
        args = dict(install_dir="/i", bundled_dir="/p/daemon", platform="linux",
                    uv="/u/uv", gpu={"name": "RTX 3090", "compute_cap": (8, 6)},
                    hf_token="tok", isfile=self.BUNDLE)
        args.update(kw)
        return {s.key: s for s in bk.build_install_plan(**args)}

    def test_uv_cuda_plan(self):
        p = self.plan()
        self.assertEqual(p["venv"].argv, ["/u/uv", "venv", "--python", "3.12",
                                          "--allow-existing", "/i/.venv"])
        self.assertEqual(p["torch"].argv, [
            "/u/uv", "pip", "install", "--python", "/i/.venv/bin/python",
            "torch", "torchvision", "--index-url",
            "https://download.pytorch.org/whl/cu128"])
        self.assertEqual(p["deps"].argv[-2:], ["-r", "/p/daemon/requirements.txt"])
        self.assertEqual(p["weights"].argv, [
            "/i/.venv/bin/python", "-u", "/p/daemon/download_weights.py",
            "/i/models/sam3"])
        self.assertEqual(p["weights"].env, {})  # token comes from HF cache
        self.assertEqual(p["verify"].argv[:2], ["/i/.venv/bin/python", "-c"])

    def test_pip_cpu_plan_without_uv(self):
        p = self.plan(uv=None, system_python=["/usr/bin/python3"], gpu=None)
        self.assertEqual(p["venv"].argv,
                         ["/usr/bin/python3", "-m", "venv", "/i/.venv"])
        self.assertEqual(p["torch"].argv[:4],
                         ["/i/.venv/bin/python", "-m", "pip", "install"])
        self.assertEqual(p["torch"].argv[-1],
                         "https://download.pytorch.org/whl/cpu")

    def test_windows_paths_and_legacy_gpu(self):
        isfile = files("C:\\p\\daemon/requirements.txt",
                       "C:\\p\\daemon/download_weights.py")
        p = self.plan(install_dir="C:\\i", bundled_dir="C:\\p\\daemon",
                      platform="win32", uv="C:\\uv.exe", isfile=isfile,
                      gpu={"name": "GTX 1080", "compute_cap": (6, 1)})
        self.assertIn("C:\\i\\.venv\\Scripts\\python.exe", p["torch"].argv)
        self.assertTrue(p["torch"].argv[-1].endswith("/cu126"))

    def test_macos_uses_pypi_torch(self):
        p = self.plan(platform="darwin", gpu=None)
        self.assertNotIn("--index-url", p["torch"].argv)

    def test_rerun_skips_existing_venv_and_weights(self):
        isfile = files("/p/daemon/requirements.txt",
                       "/p/daemon/download_weights.py", "/i/.venv/bin/python",
                       "/w/config.json", "/w/model.safetensors")
        p = self.plan(isfile=isfile, weights_setting="/w", hf_token=None)
        self.assertTrue(p["venv"].skipped)
        self.assertTrue(p["weights"].skipped)
        self.assertFalse(p["torch"].skipped)  # uv/pip are idempotent

    def test_token_from_setting_is_passed_to_downloader(self):
        p = self.plan(hf_token="hf_x", hf_token_from_setting=True)
        self.assertEqual(p["weights"].env, {"HF_TOKEN": "hf_x"})

    def test_missing_token_or_tools_is_a_plan_error(self):
        with self.assertRaises(bk.PlanError):
            self.plan(hf_token=None)
        with self.assertRaises(bk.PlanError):
            self.plan(uv=None, system_python=None)
        with self.assertRaises(bk.PlanError):
            self.plan(isfile=files())

    def test_find_uv_and_python_ignore_krita_bundle(self):
        def which(name, path=None):
            return {"uv": "/tmp/.mount_k/usr/bin/uv",
                    "python3": "/tmp/.mount_k/usr/bin/python3"}.get(name)
        isfile = files("/h/.local/bin/uv", "/usr/bin/python3")
        self.assertEqual(bk.find_uv({"HOME": "/h"}, "linux", which, isfile,
                                    ["/tmp/.mount_k"]), "/h/.local/bin/uv")
        self.assertEqual(bk.find_system_python({}, "linux", which, isfile,
                                               ["/tmp/.mount_k"]),
                         ["/usr/bin/python3"])

    def test_windows_python_skips_store_stub(self):
        def which(name, path=None):
            return {"python.exe": "C:\\Users\\u\\AppData\\Local\\Microsoft\\"
                                  "WindowsApps\\python.exe"}.get(name)
        self.assertIsNone(bk.find_system_python({}, "win32", which, files()))

    def test_parse_nvidia_smi_and_torch_variant(self):
        gpu = bk.parse_nvidia_smi("NVIDIA GeForce RTX 3090, 8.6\n")
        self.assertEqual(gpu, {"name": "NVIDIA GeForce RTX 3090",
                               "compute_cap": (8, 6)})
        self.assertEqual(bk.parse_nvidia_smi("Old GPU\n")["compute_cap"], None)
        self.assertIsNone(bk.parse_nvidia_smi(""))
        url, label = bk.torch_variant("linux", gpu)
        self.assertTrue(url.endswith("/cu128"))
        self.assertIn("12.8", label)


class LogTest(unittest.TestCase):
    def test_tail_and_last_error(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".log") as f:
            f.write("a\nb\nOSError: [Errno 98] Address already in use\nend\n")
        self.addCleanup(os.remove, f.name)
        self.assertEqual(bk.tail_file(f.name, 2).splitlines(),
                         ["OSError: [Errno 98] Address already in use", "end"])
        self.assertIn("Address already",
                      bk.last_error_line(bk.tail_file(f.name)))
        self.assertEqual(bk.tail_file("/no/such/file"), "")


if __name__ == "__main__":
    unittest.main()
