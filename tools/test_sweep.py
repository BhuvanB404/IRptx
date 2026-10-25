# test_sweep.py -- checks sw_parse against captured and derived submit.py output.
# Run: .venv/bin/python -m unittest discover -s tools -p 'test_*.py'
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep  # noqa: E402

PASS1 = ("[k_sum_1_reduce] 16 registers\n"
         "[k_sum_1_reduce] PASS  sum_1: max abs err 3.815e-06  (20.16 us)\n")


class ParseTest(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(sweep.sw_parse(PASS1, 0), ("pass", "", 20.16))

    def test_pass_sums_kernels(self):
        two = PASS1 + PASS1.replace("20.16", "4.0")
        self.assertEqual(sweep.sw_parse(two, 0)[2], 24.16)

    def test_wrong(self):
        t = ("[k] 12 registers\n[k] FAIL  y: max abs err 1.000e+00 (tolerance rtol=0.0001 "
             "atol=0.0001)\n    the parameters must be in meta.json's arg_order: x, y\n")
        st, why, us = sweep.sw_parse(t, 1)
        self.assertEqual((st, us), ("wrong", None))
        self.assertTrue(why.startswith("FAIL  y: max abs err"))

    def test_ptxas(self):
        t = "[k] ptxas rejected it:\nptxas fatal : bad thing\nmore\n"
        self.assertEqual(sweep.sw_parse(t, 1), ("ptxas", "ptxas fatal : bad thing", None))

    def test_launch(self):
        t = "[k] 8 registers\n[k] launch failed: illegal address\n"
        self.assertEqual(sweep.sw_parse(t, 1)[:2], ("run_error", "launch failed: illegal address"))

    def test_compiler_exit(self):
        t = "[k] your compiler failed (exit code 1):\nerror[E004]: irptx: no binary\n"
        self.assertEqual(sweep.sw_parse(t, 1)[:2], ("compile_error", "error[E004]: irptx: no binary"))

    def test_not_ptx_and_entry(self):
        self.assertEqual(sweep.sw_parse("[k] that is not PTX: it is empty\n", 1)[0], "compile_error")
        self.assertEqual(sweep.sw_parse("[k] compiled, but entry points are []\n", 1)[0], "compile_error")

    def test_runner_down_and_blank(self):
        self.assertEqual(sweep.sw_parse("cannot reach the runner at x\n", 1)[0], "run_error")
        self.assertEqual(sweep.sw_parse("", 1)[0], "run_error")

    def test_null_median(self):
        t = "[k] PASS  y: max abs err 0.0  (None us)\n[k] PASS  z: max abs err 0 (1.5e+01 us)\n"
        self.assertEqual(sweep.sw_parse(t, 0), ("pass", "", 15.0))

    def test_baseline(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            self.assertIsNotNone(sweep.sw_base(p)[1])          # missing
            p.write_text("{not json")
            self.assertIsNotNone(sweep.sw_base(p)[1])          # corrupt
            p.write_text("[]")
            self.assertIsNotNone(sweep.sw_base(p)[1])          # wrong shape
            p.write_text(json.dumps({"results": []}))
            self.assertEqual(sweep.sw_base(p), ({"results": []}, None))
            self.assertEqual(sweep.main(["--compare", "--out", str(Path(d) / "none.json")]), 2)

    def test_regress(self):
        old = {"results": [{"problem": "a", "status": "pass"}, {"problem": "b", "status": "wrong"}]}
        new = [{"problem": "a", "status": "wrong"}, {"problem": "b", "status": "pass"}]
        self.assertEqual(sweep.sw_regr(old, new), ["a"])


if __name__ == "__main__":
    unittest.main()
