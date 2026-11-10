"""macOS behaviour, simulated on any OS (no Mac, no Krita, no torch).

sys.platform / platform.machine / env / shutil.which / subprocess are faked
so the macOS code paths of the plugin backend, the process manager, the
build script and the daemon's engine run here.
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "krita-plugin", "kritaautoselect"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import backend as bk  # noqa: E402
import build_plugin  # noqa: E402
import engine  # noqa: E402
import manager as mg  # noqa: E402

HOME = "/Users/u"
DATA = HOME + "/Library/Application Support/krita-autoselect"
# What launchd gives an app opened from the Finder / Dock.
FINDER_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
KRITA_APP = "/Applications/krita.app"
BUNDLE = ("/p/daemon/requirements.txt", "/p/daemon/download_weights.py")


def files(*paths):
    present = set(paths)
    return lambda p: p in present


def fake_which(*present):
    """shutil.which over a fake filesystem, honouring the PATH it gets."""
    present = set(present)

    def which(name, path=None):
        for d in (path or "").split(os.pathsep):
            if d and d.rstrip("/") + "/" + name in present:
                return d.rstrip("/") + "/" + name
        return None
    return which


def plan(**kw):
    args = dict(install_dir=DATA, bundled_dir="/p/daemon", platform="darwin",
                uv="/opt/homebrew/bin/uv", gpu=None, hf_token="tok",
                isfile=files(*BUNDLE), mac_cpu=bk.MAC_ARM64)
    args.update(kw)
    return {s.key: s for s in bk.build_install_plan(**args)}


class MacPathsTest(unittest.TestCase):
    def test_data_dir_logs_and_venv(self):
        env = {"HOME": HOME}
        self.assertEqual(bk.default_data_dir("darwin", env), DATA)
        self.assertEqual(bk.log_dir(DATA, "darwin"), DATA + "/logs")
        self.assertEqual(bk.venv_python(DATA + "/.venv", "darwin"),
                         DATA + "/.venv/bin/python")
        inst = bk.resolve_install(DATA, "", "/p/daemon", "darwin", files(
            DATA + "/.venv/bin/python", "/p/daemon/server.py",
            DATA + "/" + bk.INSTALL_DONE_MARKER,
            DATA + "/models/sam3/" + bk.WEIGHTS_DONE_MARKER))
        self.assertTrue(inst.ok)
        self.assertEqual(inst.weights, DATA + "/models/sam3")

    def test_install_dir_with_space_stays_one_argv_item(self):
        """'Application Support' has a space: argv lists, never a shell."""
        p = plan()
        self.assertEqual(p["venv"].argv[-1], DATA + "/.venv")
        self.assertIn(DATA + "/.venv/bin/python", p["torch"].argv)
        self.assertEqual(p["weights"].argv[-1], DATA + "/models/sam3")

    def test_ai_diffusion_settings_path(self):
        self.assertEqual(
            bk.ai_diffusion_settings_path("darwin", {"HOME": HOME}),
            HOME + "/Library/Application Support/krita/ai_diffusion/"
                   "settings.json")

    def test_hf_token_from_hf_login_cache(self):
        """huggingface_hub keeps ~/.cache/huggingface on macOS too (not
        ~/Library/Caches), and a Finder-launched Krita has no HF_TOKEN."""
        self.assertIn(HOME + "/.cache/huggingface/token",
                      bk.hf_token_paths({"HOME": HOME}))
        stored = {HOME + "/.cache/huggingface/token": "hf_mac\n"}

        def read(path):
            if path in stored:
                return stored[path]
            raise OSError(path)
        token, source = bk.find_hf_token("", {"HOME": HOME,
                                              "PATH": FINDER_PATH}, read)
        self.assertEqual(token, "hf_mac")
        self.assertIn(".cache/huggingface/token", source)

    def test_build_plugin_installs_into_macos_pykrita(self):
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.dict(os.environ, {"HOME": HOME}):
            self.assertEqual(build_plugin.default_pykrita(),
                             HOME + "/Library/Application Support/krita/"
                                    "pykrita")


class MacFinderEnvTest(unittest.TestCase):
    """Krita opened from the Finder: no Homebrew / ~/.local/bin in PATH."""

    def test_uv_found_outside_finder_path(self):
        env = {"HOME": HOME, "PATH": FINDER_PATH}
        which = fake_which("/opt/homebrew/bin/uv")  # not on PATH: not found
        self.assertEqual(bk.find_uv(env, "darwin", which,
                                    files("/opt/homebrew/bin/uv")),
                         "/opt/homebrew/bin/uv")
        self.assertEqual(bk.find_uv(env, "darwin", which, files(
            HOME + "/.local/bin/uv", "/opt/homebrew/bin/uv")),
            HOME + "/.local/bin/uv")
        self.assertEqual(bk.find_uv(env, "darwin", which,
                                    files("/usr/local/bin/uv")),
                         "/usr/local/bin/uv")

    def test_xcode_python3_stub_is_never_used(self):
        """/usr/bin/python3 pops the CLT install dialog (or is 3.9)."""
        env = {"HOME": HOME, "PATH": FINDER_PATH}
        which = fake_which("/usr/bin/python3")
        self.assertIsNone(bk.find_system_python(
            env, "darwin", which, files("/usr/bin/python3")))
        with self.assertRaises(bk.PlanError):
            plan(uv=None, system_python=bk.find_system_python(
                env, "darwin", which, files("/usr/bin/python3")))
        clt = fake_which("/Library/Developer/CommandLineTools/usr/bin/python3")
        self.assertIsNone(bk.find_system_python(
            {"PATH": "/Library/Developer/CommandLineTools/usr/bin"},
            "darwin", clt, files()))

    def test_homebrew_or_python_org_python_when_no_uv(self):
        env = {"HOME": HOME, "PATH": FINDER_PATH}
        which = fake_which("/usr/bin/python3")
        for path in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3",
                     "/Library/Frameworks/Python.framework/Versions/Current/"
                     "bin/python3", "/opt/local/bin/python3"):
            self.assertEqual(bk.find_system_python(
                env, "darwin", which, files("/usr/bin/python3", path)),
                [path])
        # Launched from a terminal: PATH has Homebrew first, use it.
        self.assertEqual(bk.find_system_python(
            {"PATH": "/opt/homebrew/bin:" + FINDER_PATH}, "darwin",
            fake_which("/opt/homebrew/bin/python3", "/usr/bin/python3"),
            files()), ["/opt/homebrew/bin/python3"])

    def test_pick_system_python_skips_stub_and_finds_framework(self):
        """The installer calls pick_system_python (it probes the version),
        not find_system_python: the stub must be skipped there too."""
        env = {"HOME": HOME, "PATH": FINDER_PATH}
        which = fake_which("/usr/bin/python3")  # the CLT stub, on PATH
        framework = ("/Library/Frameworks/Python.framework/Versions/Current/"
                     "bin/python3")
        probed = []

        def run(argv, **kw):
            probed.append(argv[0])
            return types.SimpleNamespace(returncode=0, stdout="3.12 1\n")

        argv, problem = bk.pick_system_python(
            env, "darwin", which, files("/usr/bin/python3", framework),
            run=run)
        self.assertEqual((argv, problem), ([framework], ""))
        self.assertNotIn("/usr/bin/python3", probed)  # never executed
        for path in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3",
                     "/opt/local/bin/python3"):
            probed.clear()
            argv, _ = bk.pick_system_python(
                env, "darwin", which, files("/usr/bin/python3", path),
                run=run)
            self.assertEqual(argv, [path])
            self.assertNotIn("/usr/bin/python3", probed)
        # Only the stub (and the CLT copy): nothing is probed at all.
        probed.clear()
        clt = "/Library/Developer/CommandLineTools/usr/bin"
        argv, _ = bk.pick_system_python(
            {"PATH": clt + ":" + FINDER_PATH}, "darwin",
            fake_which(clt + "/python3", "/usr/bin/python3"),
            files("/usr/bin/python3"), run=run)
        self.assertIsNone(argv)
        self.assertEqual(probed, [])

    def test_plan_from_finder_env_uses_pypi_torch(self):
        env = {"HOME": HOME, "PATH": FINDER_PATH}
        isfile = files(*BUNDLE, "/opt/homebrew/bin/python3", "/usr/bin/python3")
        which = fake_which("/usr/bin/python3")
        uv = bk.find_uv(env, "darwin", which, isfile)
        py = bk.find_system_python(env, "darwin", which, isfile)
        gpu = bk.detect_nvidia_gpu(env, "darwin", which)
        self.assertIsNone(uv)
        self.assertIsNone(gpu)  # no nvidia-smi on a Mac
        p = plan(uv=uv, system_python=py, gpu=gpu, isfile=isfile)
        self.assertEqual(p["venv"].argv, ["/opt/homebrew/bin/python3", "-m",
                                          "venv", DATA + "/.venv"])
        self.assertEqual(p["torch"].argv[-2:], ["torch", "torchvision"])
        self.assertNotIn("--index-url", p["torch"].argv)
        self.assertIn("PyPI", p["torch"].title)


class MacCpuTest(unittest.TestCase):
    def sysctl(self, stdout="", rc=0, exc=None):
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            if exc:
                raise exc
            return types.SimpleNamespace(returncode=rc, stdout=stdout)
        return run, calls

    def test_apple_silicon_native(self):
        run, calls = self.sysctl()
        self.assertEqual(bk.macos_cpu("arm64", run), bk.MAC_ARM64)
        self.assertEqual(calls, [])

    def test_rosetta_is_still_apple_silicon(self):
        run, calls = self.sysctl("1\n")
        self.assertEqual(bk.macos_cpu("x86_64", run), bk.MAC_ROSETTA)
        self.assertEqual(calls[0][0], "/usr/sbin/sysctl")  # not on PATH lookup

    def test_intel(self):
        for run, _ in (self.sysctl("0\n"), self.sysctl("", rc=1),
                       self.sysctl(exc=OSError("no sysctl"))):
            self.assertEqual(bk.macos_cpu("x86_64", run), bk.MAC_INTEL)

    def test_machine_comes_from_platform_module(self):
        run, _ = self.sysctl()
        with mock.patch("platform.machine", return_value="arm64"):
            self.assertEqual(bk.macos_cpu(run=run), bk.MAC_ARM64)

    def test_intel_mac_install_is_refused_clearly(self):
        with self.assertRaises(bk.PlanError) as cm:
            plan(mac_cpu=bk.MAC_INTEL)
        msg = str(cm.exception)
        self.assertIn("Intel", msg)
        self.assertIn("2.2", msg)
        self.assertIn("Apple Silicon", msg)

    def test_apple_silicon_and_rosetta_install(self):
        for cpu in (bk.MAC_ARM64, bk.MAC_ROSETTA):
            p = plan(mac_cpu=cpu)
            self.assertNotIn("--index-url", p["torch"].argv)
        # mac_cpu only matters on macOS
        plan(platform="linux", mac_cpu=bk.MAC_INTEL,
             gpu={"name": "RTX", "compute_cap": (8, 6)})


KRITA_ENV = {
    "PYTHONHOME": KRITA_APP + "/Contents/Frameworks/Python.framework/"
                              "Versions/3.10",
    "PYTHONPATH": KRITA_APP + "/Contents/Resources/krita/pykrita",
    "PYTHONEXECUTABLE": KRITA_APP + "/Contents/MacOS/krita",
    "__PYVENV_LAUNCHER__": KRITA_APP + "/Contents/MacOS/krita",
    "DYLD_LIBRARY_PATH": KRITA_APP + "/Contents/Frameworks",
    "DYLD_FRAMEWORK_PATH": KRITA_APP + "/Contents/Frameworks",
    "DYLD_FALLBACK_LIBRARY_PATH": KRITA_APP + "/Contents/lib",
    "DYLD_INSERT_LIBRARIES": KRITA_APP + "/Contents/Frameworks/x.dylib",
    "PATH": KRITA_APP + "/Contents/MacOS:" + FINDER_PATH,
    "HOME": HOME,
}


class MacKritaEnvTest(unittest.TestCase):
    def assert_clean(self, env):
        for var in env:
            self.assertFalse(var.startswith("DYLD_"), var)
        for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE",
                    "__PYVENV_LAUNCHER__"):
            self.assertNotIn(var, env)
        self.assertEqual(env["PATH"], FINDER_PATH)
        self.assertEqual(env["HOME"], HOME)

    def test_krita_prefixes_cover_the_app_bundle(self):
        self.assertEqual(bk.krita_prefixes(
            {}, KRITA_APP + "/Contents/MacOS/krita"), [KRITA_APP])
        # Krita's embedded Python as sys.executable
        self.assertEqual(bk.krita_prefixes(
            {}, "/Applications/Krita 5.2.app/Contents/Frameworks/"
                "Python.framework/Versions/3.10/bin/python3"),
            ["/Applications/Krita 5.2.app"])
        # a python.org framework Python is not Krita
        self.assertEqual(bk.krita_prefixes(
            {}, "/Library/Frameworks/Python.framework/Versions/3.12/"
                "Resources/Python.app/Contents/MacOS/Python"), [])

    def test_sanitize_env_strips_krita_python_and_dyld(self):
        prefixes = bk.krita_prefixes({}, KRITA_APP + "/Contents/MacOS/krita")
        env = bk.sanitize_env(KRITA_ENV, prefixes)
        self.assert_clean(env)
        # Krita's bundled python3 is not picked for `-m venv`.
        which = fake_which(KRITA_APP + "/Contents/MacOS/python3",
                           "/usr/bin/python3")
        self.assertIsNone(bk.find_system_python(
            KRITA_ENV, "darwin", which, files(), prefixes))

    def _patched_krita(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, KRITA_ENV))
        stack.enter_context(mock.patch.object(
            sys, "executable", KRITA_APP + "/Contents/MacOS/krita"))
        os.environ.pop("APPDIR", None)
        return stack

    def test_daemon_spawned_from_krita_app_gets_clean_env(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        calls = []

        def popen(argv, **kw):
            calls.append((argv, kw))
            return types.SimpleNamespace(pid=4242)

        cfg = mg.Config(url="http://127.0.0.1:5679", install_dir=DATA,
                        bundled_dir="/p/daemon", comfyui_free=False,
                        data_dir=tmp)
        install = bk.Install(DATA, DATA + "/.venv/bin/python",
                             "/p/daemon/server.py", None)
        with self._patched_krita():
            m = mg.DaemonManager(cfg, popen=popen)
            m._spawn(mg._Session(cfg, install))
        argv, kw = calls[0]
        self.assertEqual(argv, [DATA + "/.venv/bin/python", "-u",
                                "/p/daemon/server.py"])
        self.assert_clean(kw["env"])
        self.assertEqual(kw["env"]["AUTOSELECT_PARENT_WATCH"], "stdin")
        self.assertEqual(kw["stdin"], mg.subprocess.PIPE)  # EOF backstop
        self.assertNotIn("creationflags", kw)

    def test_installer_steps_get_clean_env_and_own_process_group(self):
        calls = []

        def popen(argv, **kw):
            calls.append((argv, kw))
            return types.SimpleNamespace(pid=1, stdout=iter(()),
                                         wait=lambda: 0)

        runner = mg.InstallRunner(popen=popen)
        with self._patched_krita():
            rc = runner._run_step(bk.Step("x", "x", ["/opt/homebrew/bin/uv"]),
                                  None)
        self.assertEqual(rc, 0)
        _, kw = calls[0]
        self.assert_clean(kw["env"])
        self.assertTrue(kw["start_new_session"])  # cancel kills uv + pip


def fake_ml(torch_version="2.9.0", torch_ok=True, cuda=False, mps=True):
    """Fake torch + transformers modules (sys.modules entries)."""
    loads = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source, dtype=None):
            obj = cls()
            obj.dtype = dtype
            loads.append(obj)
            return obj

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    torch = types.ModuleType("torch")
    torch.__version__ = torch_version
    torch.float32, torch.bfloat16, torch.float16 = "float32", "bfloat16", "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, empty_cache=mock.Mock(),
        get_device_name=lambda i: "GPU", mem_get_info=mock.Mock())
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.mps = types.SimpleNamespace(empty_cache=mock.Mock())

    tf = types.ModuleType("transformers")
    tf.__version__ = "5.22.2"
    for name in ("Sam3Model", "Sam3Processor", "Sam3TrackerModel",
                 "Sam3TrackerProcessor"):
        setattr(tf, name, type(name, (FakeModel,), {}))
    tfu = types.ModuleType("transformers.utils")
    tfu.is_torch_available = lambda: torch_ok
    tf.utils = tfu
    modules = {"torch": torch, "transformers": tf, "transformers.utils": tfu}
    return modules, torch, loads


class VerifyCodeTest(unittest.TestCase):
    def run_verify(self, **kw):
        modules, _, _ = fake_ml(**kw)
        out = io.StringIO()
        with mock.patch.dict(sys.modules, modules), \
                contextlib.redirect_stdout(out):
            exec(compile(bk.VERIFY_CODE, "<verify>", "exec"), {})
        return out.getvalue()

    def test_reports_mps(self):
        self.assertIn("| mps True", self.run_verify())

    def test_nvidia_gpu_without_cuda_torch_fails(self):
        """GPU detected but torch can't see CUDA (old driver, CPU wheel)."""
        p = bk.build_install_plan(
            "/i", "/p/daemon", "linux", uv="/u/uv", hf_token="t",
            gpu={"name": "RTX 3090", "compute_cap": (8, 6)},
            isfile=files("/p/daemon/requirements.txt",
                         "/p/daemon/download_weights.py"))
        verify = {s.key: s for s in p}["verify"]
        code = verify.argv[-1]
        self.assertIn(bk.VERIFY_CUDA_CODE, code)
        self.assertTrue(verify.hint)
        for cuda, fails in ((False, True), (True, False)):
            modules, torch, _ = fake_ml(cuda=cuda, mps=False)
            torch.version = types.SimpleNamespace(cuda="12.8")
            with mock.patch.dict(sys.modules, modules), \
                    contextlib.redirect_stdout(io.StringIO()):
                if fails:
                    with self.assertRaises(SystemExit) as cm:
                        exec(compile(code, "<verify>", "exec"), {})
                    self.assertIn("no ve CUDA", str(cm.exception.code))
                else:
                    exec(compile(code, "<verify>", "exec"), {})
        # Without a detected GPU (or on a Mac) CPU torch is expected.
        for plat, gpu in (("linux", None), ("darwin", None)):
            p = bk.build_install_plan(
                "/i", "/p/daemon", plat, uv="/u/uv", hf_token="t", gpu=gpu,
                isfile=files("/p/daemon/requirements.txt",
                             "/p/daemon/download_weights.py"),
                mac_cpu=bk.MAC_ARM64)
            verify = {s.key: s for s in p}["verify"]
            self.assertNotIn(bk.VERIFY_CUDA_CODE, verify.argv[-1])

    def test_fails_when_transformers_rejects_torch(self):
        """torch 2.2.2 (last Intel-Mac build): Sam3Model would import as a
        placeholder, so the bare import alone would pass verification."""
        with self.assertRaises(SystemExit) as cm:
            self.run_verify(torch_version="2.2.2", torch_ok=False)
        self.assertIn("2.2.2", str(cm.exception.code))


class EngineMacTest(unittest.TestCase):
    def make(self):
        eng = engine.Sam3Engine(comfyui_url="http://127.0.0.1:8188")
        eng._watchdog_started = True  # no background thread in tests
        return eng

    def test_mps_fallback_env_set_before_torch_import(self):
        spec = importlib.util.spec_from_file_location(
            "engine_as_macos", os.path.join(ROOT, "engine.py"))
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.dict(os.environ):
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # user's choice
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "0")

    def test_auto_device(self):
        _, torch, _ = fake_ml(mps=True)
        self.assertEqual(engine.Sam3Engine._auto_device(torch), "mps")
        _, torch, _ = fake_ml(cuda=True, mps=False)
        self.assertEqual(engine.Sam3Engine._auto_device(torch), "cuda")
        _, torch, _ = fake_ml(mps=False)
        self.assertEqual(engine.Sam3Engine._auto_device(torch), "cpu")
        del torch.backends.mps  # very old torch
        self.assertEqual(engine.Sam3Engine._auto_device(torch), "cpu")

    def test_loads_on_mps_in_fp32_without_nvidia_smi_or_comfyui(self):
        modules, torch, loads = fake_ml()
        eng = self.make()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch("shutil.which") as which, \
                mock.patch("subprocess.run") as run, \
                mock.patch("urllib.request.urlopen") as urlopen, \
                contextlib.redirect_stdout(io.StringIO()):
            eng._load_locked()
            eng._load_tracker_locked()
            self.assertEqual(eng.info()["device"], "mps")
            models = [m for m in loads if hasattr(m, "device")]
            self.assertEqual([(m.device, m.dtype) for m in models],
                             [("mps", "float32")] * 2)
            eng.unload()
        which.assert_not_called()
        run.assert_not_called()
        urlopen.assert_not_called()
        torch.mps.empty_cache.assert_called_once_with()
        torch.cuda.empty_cache.assert_not_called()

    def test_old_torch_on_intel_mac_is_a_clear_error(self):
        modules, _, loads = fake_ml(torch_version="2.2.2", torch_ok=False,
                                    mps=False)
        eng = self.make()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(sys, "platform", "darwin"), \
                mock.patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(engine.EngineError) as cm:
                eng._load_locked()
            with self.assertRaises(engine.EngineError):
                eng._load_tracker_locked()
        msg = str(cm.exception)
        self.assertIn("2.2.2", msg)
        self.assertIn("Intel", msg)
        self.assertNotIn("gated", msg.lower())  # not the HF-token hint
        self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
