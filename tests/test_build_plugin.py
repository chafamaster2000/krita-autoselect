"""The plugin zip is importable by Krita and carries the daemon unchanged."""
import os
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


if __name__ == "__main__":
    unittest.main()
