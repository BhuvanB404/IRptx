# t_fe.py -- fe01..: the front end dumps every open IR as well-formed KIR text.
#
# Skipped when the hackathon checkout ($HK, default ../emmy-hackathon-2026) is missing, so a
# machine without it stays green. Run: .venv/bin/python -m unittest tests.t_fe
import contextlib
import dataclasses
import io
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fe"))
HK = Path(os.environ.get("HK", ROOT.parent / "emmy-hackathon-2026"))
IRS = sorted((HK / "problems").glob("*/*.ir.json"))
HEXF = re.compile(r"(?<![\w.])-?0x[0-9a-f.]+p[+-]?\d+f")


def depth_ok(body):
    """True when parentheses and brackets balance outside strings (one S-expression)."""
    stack, instr = [], False
    for ch in body:
        if ch == '"':
            instr = not instr
        elif not instr and ch in "([":
            stack.append(ch)
        elif not instr and ch in ")]":
            if not stack or stack.pop() != {")": "(", "]": "["}[ch]:
                return False
    return not stack and not instr


def floats_in(obj):
    """Independent (recursive, test-only) collection of every float leaf."""
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [f for x in obj for f in floats_in(x)]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f for fl in dataclasses.fields(obj) for f in floats_in(getattr(obj, fl.name))]
    return []


@unittest.skipUnless(IRS, "hackathon checkout not found")
class FrontEnd(unittest.TestCase):
    def dump(self, path):
        import kirdump as fe
        with contextlib.redirect_stdout(io.StringIO()) as out:
            graph, kn = fe.fe_load(str(path))
            meta = fe.fe_meta(str(path), kn.op.name)
            bufs = fe.fe_bufs(graph, kn, meta)
            text = fe.fe_dump(kn.op.name, meta, bufs, kn.op.body)
        self.assertEqual(out.getvalue(), "", "import emmy / dump must not write stdout")
        return text, kn, meta

    def test_fe01_every_open_ir_dumps_well_formed(self):
        for path in IRS:
            with self.subTest(ir=path.parent.name):
                text, kn, meta = self.dump(path)
                lines = text.split("\n")
                self.assertTrue(text.isascii() and len(text) < 1 << 20)
                self.assertEqual(lines[0], "kir 1")
                self.assertEqual(lines[1], "kernel " + kn.op.name)
                self.assertEqual(sum(l.startswith("launch ") for l in lines), 1)
                args = [l.split()[2] for l in lines if l.startswith("arg ")]
                self.assertEqual(args, meta["arg_order"])
                zeros = [l.split()[1] for l in lines if l.startswith("zero ")]
                self.assertEqual(zeros, meta["zero_outputs"])
                self.assertEqual(lines.count("body"), 1)
                self.assertEqual(lines[-2], "end")
                body = "\n".join(lines[lines.index("body") + 1:-2])
                self.assertTrue(depth_ok(body))
                self.assertEqual(sorted(HEXF.findall(body)), sorted(
                    f.hex() + "f" for f in floats_in(kn.op.body)))

    def test_fe02_rank_zero_outputs_keep_rank_zero(self):
        for pid in ("r01_array_sum", "r03_mean"):
            path = next(p for p in IRS if p.parent.name == pid)
            text, _, _ = self.dump(path)
            outs = [l for l in text.split("\n") if l.startswith("arg ") and " out " in l]
            self.assertTrue(any(l.endswith(" f32 0") for l in outs), outs)

    def test_fe03_missing_meta_refuses_e002(self):
        import kirdump as fe
        with self.assertRaises(fe.FeError) as ctx:
            fe.fe_meta("/nonexistent/dir/k.ir.json", "k")
        self.assertEqual(ctx.exception.code, "E002")

    def test_fe04_floats_round_trip_and_bad_strings_refuse(self):
        import kirdump as fe
        for val in (0.1, -1e30, 1.0, 3.4028234663852886e38):
            self.assertEqual(float.fromhex(fe.fe_leaf(val)[:-1]), val)
        with self.assertRaises(fe.FeError):
            fe.fe_leaf('a"b')
        with self.assertRaises(fe.FeError) as ctx:
            fe.fe_dump("k", {"grid": [1] * 3, "block": [1] * 3, "smem_bytes": 0}, [], [chr(233)])
        self.assertEqual(ctx.exception.code, "E003")


if __name__ == "__main__":
    unittest.main()
