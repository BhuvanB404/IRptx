# t_cli.py -- cli01..: compile.py finds the core in three tiers and relays what it says.
#
# Each test builds a scratch copy of compile.py, fe/ and a src/ in a temp dir, so the real
# repo (and its .irptx-cache) is never touched. Needs the hackathon checkout for one real IR
# ($HK, default ../emmy-hackathon-2026); skipped without it.
# Run: .venv/bin/python -m unittest tests.t_cli
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HK = Path(os.environ.get("HK", ROOT.parent / "emmy-hackathon-2026"))
IR = HK / "problems" / "p00_scalar_add" / "k_add_pointwise.ir.json"
CORE_SH = """#!/bin/sh
head -n 1
echo "diag: from the fake core" >&2
exit %d
"""
MAIN_C = """#include <stdio.h>
#include "irptx.h"
int main(int argc, char **argv) {
    char b[64];
    (void)argc; (void)argv;
    if (fgets(b, sizeof b, stdin)) fputs(b, stdout);
    fputs("diag: from the cc-built core\\n", stderr);
    return 7;
}
"""


@unittest.skipUnless(IR.exists(), "hackathon checkout not found")
class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="irptx-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        shutil.copy(ROOT / "compile.py", self.tmp)
        shutil.copytree(ROOT / "fe", self.tmp / "fe", ignore=shutil.ignore_patterns("__pycache__"))
        (self.tmp / "src").mkdir()
        (self.tmp / "src" / "irptx.h").write_text('#define IRPTX_VER "9.9.9"\n')
        (self.tmp / "bin").mkdir()

    def core(self, code):
        exe = self.tmp / "bin" / "irptx"
        exe.write_text(CORE_SH % code)
        exe.chmod(0o755)

    def tiny_src(self):
        (self.tmp / "src" / "main.c").write_text(MAIN_C)

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, "-I", str(self.tmp / "compile.py"), *args],
                              capture_output=True, text=True, cwd="/", env={**os.environ, **(env or {})})

    def test_cli01_tier1_relays_stdout_stderr_and_exit(self):
        for code in (0, 1, 3):
            self.core(code)
            res = self.run_cli(str(IR))
            self.assertEqual((res.stdout, res.returncode), ("kir 1\n", code))
            self.assertEqual(res.stderr, "diag: from the fake core\n")

    def test_cli02_tier2_builds_once_atomically_and_rekeys(self):
        self.tiny_src()
        res = self.run_cli(str(IR))
        self.assertEqual((res.stdout, res.returncode), ("kir 1\n", 7), res.stderr)
        self.assertEqual(res.stderr, "diag: from the cc-built core\n")
        cache = self.tmp / ".irptx-cache"
        first = sorted(p.name for p in cache.iterdir())
        self.assertEqual(len(first), 1)
        self.assertFalse(first[0].endswith(".tmp"))
        mtime = (cache / first[0]).stat().st_mtime_ns
        self.run_cli(str(IR))
        self.assertEqual((cache / first[0]).stat().st_mtime_ns, mtime, "rebuilt without a change")
        (self.tmp / "src" / "main.c").write_text(MAIN_C + "/* changed */\n")
        self.run_cli(str(IR))
        self.assertEqual(len(list(cache.iterdir())), 2, "a changed src/ must get a new binary")
        self.assertFalse([p for p in cache.iterdir() if p.name.endswith(".tmp")])

    def test_cli03_tier3_is_e004_with_nothing_on_stdout(self):
        res = self.run_cli(str(IR))
        self.assertEqual((res.stdout, res.returncode), ("", 1))
        self.assertTrue(res.stderr.startswith("error[E004]: irptx: no usable core"), res.stderr)

    def test_cli04_unrunnable_bin_falls_through_to_cc(self):
        self.tiny_src()
        exe = self.tmp / "bin" / "irptx"
        exe.write_bytes(b"\x7fELF-not-really\n")
        exe.chmod(0o755)
        res = self.run_cli(str(IR))
        self.assertEqual((res.stdout, res.returncode), ("kir 1\n", 7), res.stderr)

    def test_cli05_timeout_is_e005(self):
        exe = self.tmp / "bin" / "irptx"
        exe.write_text("#!/bin/sh\nexec sleep 30\n")
        exe.chmod(0o755)
        spec = importlib.util.spec_from_file_location("cp_under_test", self.tmp / "compile.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.CORE_TIMEOUT = 1
        try:
            with self.assertRaises(mod.fe.FeError) as ctx:
                mod.cp_core((), b"kir 1\n")
        finally:
            sys.modules.pop("kirdump", None)
        self.assertEqual(ctx.exception.code, "E005")

    def test_cli06_version_from_core_or_header(self):
        res = self.run_cli("--version")
        self.assertEqual((res.stdout, res.returncode), ("irptx 9.9.9\n", 0), res.stderr)
        exe = self.tmp / "bin" / "irptx"
        exe.write_text("#!/bin/sh\necho irptx 1.2.3\n")
        exe.chmod(0o755)
        self.assertEqual(self.run_cli("--version").stdout, "irptx 1.2.3\n")

    def test_cli07_internal_error_is_e000_without_traceback(self):
        with open(self.tmp / "fe" / "kirdump.py", "a", encoding="utf-8") as fh:
            fh.write("\n\ndef fe_dump(*a, **k):\n    raise RuntimeError('boom')\n")
        for dbg, tb in (("0", False), ("1", True)):
            res = self.run_cli(str(IR), env={"IRPTX_DEBUG": dbg})
            self.assertEqual(res.stdout, "")
            self.assertEqual(res.returncode, 1)
            self.assertIn("error[E000]: irptx: internal error in the front end", res.stderr)
            self.assertEqual("Traceback" in res.stderr, tb)

    def test_cli08_bad_usage_and_missing_ir_refuse(self):
        for args, code in (([], "E001"), (["a", "b"], "E001"), (["--nope", "x"], "E001"),
                           (["/nonexistent.ir.json"], "E001")):
            res = self.run_cli(*args)
            self.assertEqual((res.stdout, res.returncode), ("", 1))
            self.assertIn(f"error[{code}]: irptx:", res.stderr)


if __name__ == "__main__":
    unittest.main()
