"""The plugin zip is importable by Krita and carries the daemon unchanged."""
import contextlib
import io
import json
import os
import py_compile
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_plugin  # noqa: E402


class BuildPluginTest(unittest.TestCase):
    def test_zip_layout_matches_krita_importer(self):
        out = tempfile.mkdtemp()
        path = build_plugin.build_zip(out)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            # What Krita's plugin_importer checks: a .desktop naming the
            # library, an explicit "<lib>/" entry and "<lib>/__init__.py".
            self.assertIn("kritaautoselect.desktop", names)
            self.assertIn("kritaautoselect/", names)
            self.assertIn("kritaautoselect/__init__.py", names)
            self.assertIn("kritaautoselect/backend.py", names)
            self.assertIn("kritaautoselect/manager.py", names)
            self.assertFalse([n for n in names if "__pycache__" in n])
            for arc, src in build_plugin.DAEMON_SOURCES.items():
                with open(src, "rb") as f:
                    self.assertEqual(zf.read(f"kritaautoselect/daemon/{arc}"),
                                     f.read(), arc)

    def test_install_copies_tree(self):
        pykrita = tempfile.mkdtemp()
        build_plugin.install(pykrita)
        self.assertTrue(os.path.isfile(os.path.join(
            pykrita, "kritaautoselect.desktop")))
        self.assertTrue(os.path.isfile(os.path.join(
            pykrita, "kritaautoselect", "daemon", "server.py")))

    def test_built_plugin_compiles(self):
        pykrita = tempfile.mkdtemp()
        build_plugin.install(pykrita)
        for dirpath, _, files in os.walk(pykrita):
            for fn in files:
                if fn.endswith(".py"):
                    py_compile.compile(os.path.join(dirpath, fn), doraise=True)

    def test_plugin_exposes_public_get_manager(self):
        with open(os.path.join(build_plugin.PLUGIN_SRC, "kritaautoselect",
                               "__init__.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("\ndef get_manager():", src)
        self.assertIn("\n_get_manager = get_manager\n", src)


NVIDIA = {"name": "RTX 4090", "compute_cap": (8, 9)}


class ComponentPlanTest(unittest.TestCase):
    """install_component(): the contract the ecosystem installer consumes."""

    def plan(self, **env):
        env.setdefault("system_python", ["/usr/bin/python3"])
        env.setdefault("gpu", None)
        return build_plugin.install_component(plan_only=True, env=env)

    def by_kind(self, actions, kind):
        return [a for a in actions if a["kind"] == kind]

    def test_action_shape(self):
        actions = self.plan(platform="linux", environ={"HOME": "/h"})
        for a in actions:
            self.assertEqual(set(a), {"kind", "op", "src", "dst", "detail"})
            self.assertIn(a["kind"], ("copy", "symlink", "write", "kritarc",
                                      "venv", "pip", "mcp", "skill"))
            self.assertEqual(a["op"], "install")
        json.dumps(actions)  # must be JSON as is

    def test_linux_paths(self):
        actions = self.plan(platform="linux", environ={"HOME": "/h"})
        copies = self.by_kind(actions, "copy")
        self.assertEqual(len(copies), len(build_plugin.plugin_files()))
        dsts = {a["dst"] for a in copies}
        self.assertIn("/h/.local/share/krita/pykrita/kritaautoselect.desktop",
                      dsts)
        self.assertIn("/h/.local/share/krita/pykrita/kritaautoselect/daemon/"
                      "engine.py", dsts)
        for a in copies:
            self.assertEqual(len(a["detail"]["sha256"]), 64)
        rc, = self.by_kind(actions, "kritarc")
        self.assertEqual(rc["dst"], "/h/.config/kritarc")
        self.assertEqual(rc["detail"], {"section": "python",
                                        "key": "enable_kritaautoselect",
                                        "value": "true"})
        venv, = self.by_kind(actions, "venv")
        self.assertEqual(venv["dst"], "/h/.local/share/krita-autoselect/.venv")
        self.assertEqual(venv["detail"]["python"], ["/usr/bin/python3"])
        torch, deps = self.by_kind(actions, "pip")
        self.assertEqual(torch["detail"]["python"],
                         "/h/.local/share/krita-autoselect/.venv/bin/python")
        self.assertEqual(torch["detail"]["args"][-1],
                         "https://download.pytorch.org/whl/cpu")
        self.assertEqual(deps["detail"]["args"][0], "-r")
        marker, = self.by_kind(actions, "write")
        self.assertEqual(marker["dst"], "/h/.local/share/krita-autoselect/"
                                        ".autoselect-installed")

    def test_xdg_overrides(self):
        actions = self.plan(platform="linux", environ={
            "HOME": "/h", "XDG_DATA_HOME": "/d", "XDG_CONFIG_HOME": "/c"})
        self.assertIn("/d/krita/pykrita/kritaautoselect.desktop",
                      {a["dst"] for a in actions})
        self.assertEqual(self.by_kind(actions, "kritarc")[0]["dst"],
                         "/c/kritarc")
        self.assertEqual(self.by_kind(actions, "venv")[0]["dst"],
                         "/d/krita-autoselect/.venv")

    def test_windows_paths_follow_userprofile(self):
        environ = {"USERPROFILE": "C:\\Users\\Ana", "HOME": "/c/msys/home"}
        actions = self.plan(platform="windows", environ=environ, gpu=NVIDIA)
        dsts = {a["dst"] for a in actions}
        self.assertIn("C:\\Users\\Ana\\AppData\\Roaming\\krita\\pykrita\\"
                      "kritaautoselect\\daemon\\server.py", dsts)
        self.assertIn("C:\\Users\\Ana\\AppData\\Local\\kritarc", dsts)
        venv, = self.by_kind(actions, "venv")
        self.assertEqual(venv["dst"], "C:\\Users\\Ana\\AppData\\Local\\"
                                      "krita-autoselect\\.venv")
        torch = self.by_kind(actions, "pip")[0]
        self.assertTrue(torch["detail"]["python"].endswith(
            "\\.venv\\Scripts\\python.exe"))
        self.assertTrue(torch["detail"]["args"][-1].endswith("cu128"))

    def test_windows_appdata_wins(self):
        actions = self.plan(platform="windows", environ={
            "USERPROFILE": "C:\\U", "APPDATA": "D:\\Roam",
            "LOCALAPPDATA": "D:\\Loc"})
        dsts = {a["dst"] for a in actions}
        self.assertIn("D:\\Roam\\krita\\pykrita\\kritaautoselect.desktop", dsts)
        self.assertIn("D:\\Loc\\kritarc", dsts)

    def test_macos_paths_and_torch(self):
        actions = self.plan(platform="macos", environ={"HOME": "/Users/ana"},
                            mac_cpu="arm64")
        dsts = {a["dst"] for a in actions}
        self.assertIn("/Users/ana/Library/Application Support/krita/pykrita/"
                      "kritaautoselect.desktop", dsts)
        self.assertIn("/Users/ana/Library/Preferences/kritarc", dsts)
        torch = self.by_kind(actions, "pip")[0]
        self.assertEqual(torch["detail"]["args"], ["torch", "torchvision"])

    def test_mac_intel_is_refused(self):
        with self.assertRaises(build_plugin._backend().PlanError):
            self.plan(platform="macos", environ={"HOME": "/Users/a"},
                      mac_cpu="x86_64")

    def test_pre_volta_gpu_gets_cu126(self):
        actions = self.plan(platform="linux", environ={"HOME": "/h"},
                            gpu={"name": "GTX 1080", "compute_cap": (6, 1)})
        self.assertTrue(self.by_kind(actions, "pip")[0]["detail"]["args"][-1]
                        .endswith("cu126"))

    def test_uninstall_plan_keeps_weights(self):
        actions = build_plugin.install_component(
            plan_only=True, uninstall=True,
            env={"platform": "linux", "environ": {"HOME": "/h"}})
        self.assertTrue(all(a["op"] == "remove" for a in actions))
        venv, = self.by_kind(actions, "venv")
        self.assertEqual(venv["dst"], "/h/.local/share/krita-autoselect/.venv")
        self.assertEqual(venv["detail"]["keep"],
                         ["/h/.local/share/krita-autoselect/models/sam3"])
        self.assertFalse([a for a in actions if "models" in (a["dst"] or "")])

    def test_cli_plan_json(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = build_plugin.main(["--plan-json", "--platform", "windows"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["component"], "krita-autoselect")
        self.assertFalse(out["uninstall"])
        self.assertTrue(out["actions"])
        self.assertEqual(out["notes"][0]["kind"], "weights")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_plugin.main(["--plan-json", "--uninstall"])
        out = json.loads(buf.getvalue())
        self.assertTrue(out["uninstall"])
        self.assertNotIn("notes", out)


class ComponentExecuteTest(unittest.TestCase):
    """plan_only=False in a temporary tree: install, reinstall, uninstall."""

    def test_cycle(self):
        root = tempfile.mkdtemp()
        env = {"platform": "linux", "pykrita": os.path.join(root, "pykrita"),
               "kritarc": os.path.join(root, "kritarc"),
               "data_dir": os.path.join(root, "data"),
               "system_python": None, "gpu": None}
        with open(env["kritarc"], "w") as f:
            f.write("[General]\nfoo=1\n\n[python]\nenable_kritamcp=true\n")
        actions = build_plugin.install_component(plan_only=False, env=env)
        self.assertTrue(os.path.isfile(os.path.join(
            root, "pykrita", "kritaautoselect", "daemon", "server.py")))
        with open(env["kritarc"]) as f:
            rc = f.read()
        self.assertIn("[python]\nenable_kritamcp=true\n"
                      "enable_kritaautoselect=true\n", rc)
        self.assertIn("[General]\nfoo=1\n", rc)
        # reinstall: nothing changes
        self.assertEqual(build_plugin._execute(actions), 0)
        os.makedirs(os.path.join(root, "data", ".venv", "bin"))
        os.makedirs(os.path.join(root, "data", "models", "sam3"))
        build_plugin.install_component(plan_only=False, uninstall=True,
                                       env=env)
        self.assertEqual(os.listdir(env["pykrita"]), [])
        self.assertEqual(os.listdir(env["data_dir"]), ["models"])
        with open(env["kritarc"]) as f:
            rc = f.read()
        self.assertNotIn("kritaautoselect", rc)
        self.assertIn("enable_kritamcp=true", rc)
        self.assertEqual(build_plugin.install_component(
            plan_only=False, uninstall=True, env=env), build_plugin
            .install_component(plan_only=True, uninstall=True, env=env))

    def test_kritarc_created_when_missing(self):
        path = os.path.join(tempfile.mkdtemp(), "sub", "kritarc")
        self.assertTrue(build_plugin._kritarc_set(path, "python", "k", "true"))
        with open(path) as f:
            self.assertEqual(f.read(), "[python]\nk=true\n")
        self.assertFalse(build_plugin._kritarc_set(path, "python", "k",
                                                   "true"))

    def test_cli_uninstall(self):
        pykrita = tempfile.mkdtemp()
        build_plugin.install(pykrita)
        with contextlib.redirect_stdout(io.StringIO()):
            build_plugin.main(["--uninstall", pykrita])
        self.assertEqual(os.listdir(pykrita), [])


if __name__ == "__main__":
    unittest.main()
