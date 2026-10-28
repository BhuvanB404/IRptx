#!/usr/bin/env python3
"""rename.py -- the PTX must not depend on what things are called.

For every open-set problem: copy it under a new directory name, rename the kernel entry
and every buffer (IR graph, KernelOp body references, meta.json) to unrelated names of
different lengths, compile both with compile.py, map the new names in the renamed PTX
back to the old ones, and require the two PTX texts to be byte-identical. A difference
means some lowering decision keys on a name (or the directory), which the hidden set
would never trigger.

  python tests/rename.py [--compiler DIR] [--problems DIR] [--keep DIR]
Exit 1 on any difference or on a compile that succeeds for one side only.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HK = Path(os.environ.get("HK", ROOT.parent / "emmy-hackathon-2026"))
KW = ("input", "output", "src", "dst", "buffer", "src_buf", "dst_buffer", "src_buffer")


def new_name(i: int, kind: str) -> str:
    # Distinct lengths and no shared substrings with emmy's own naming.
    return f"qz{kind}{i}" + "w" * (i % 3)


def rename_ir(ir: dict, bmap: dict, entry_old: str, entry_new: str) -> dict:
    out = {"inputs": [bmap.get(n, n) for n in ir["inputs"]],
           "outputs": [bmap.get(n, n) for n in ir["outputs"]], "nodes": {}}
    kw = "|".join(KW)
    for key, node in ir["nodes"].items():
        node = json.loads(json.dumps(node))
        node["inputs"] = [bmap.get(n, n) for n in node.get("inputs", [])]
        for o in node.get("outputs", []):
            o["name"] = bmap.get(o["name"], o["name"])
        f = node.get("op_fields", {})
        if node["op"] == "KernelOp":
            assert f["name"] == entry_old
            f["name"] = entry_new
            f["zero_delegated"] = [bmap.get(n, n) for n in f.get("zero_delegated", [])]
            f["body"] = [re.sub(r"\b(%s)='([^']*)'" % kw,
                                lambda m: f"{m.group(1)}='{bmap.get(m.group(2), m.group(2))}'", b)
                         for b in f["body"]]
        out["nodes"][bmap.get(key, key)] = node
    return out


def rename_meta(meta: dict, bmap: dict, emap: dict, pid: str) -> dict:
    m = json.loads(json.dumps(meta))
    m["id"] = pid
    m["graph_inputs"] = [bmap.get(n, n) for n in m["graph_inputs"]]
    m["graph_outputs"] = [bmap.get(n, n) for n in m["graph_outputs"]]
    for k in m["kernels"]:
        k["entry"] = emap[k["entry"]]
        k["node"] = bmap.get(k["node"], k["node"])
        k["arg_order"] = [bmap.get(n, n) for n in k["arg_order"]]
        k["zero_outputs"] = [bmap.get(n, n) for n in k["zero_outputs"]]
        for a in k["args"]:
            a["name"] = bmap.get(a["name"], a["name"])
    return m


def compile_ir(compiler: Path, ir: Path):
    r = subprocess.run([sys.executable, str(compiler / "compile.py"), str(ir)],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr.strip()


def check(pdir: Path, tmp: Path, n: int, compiler: Path) -> list[str]:
    """Rename one problem into tmp and compare; returns a list of failure lines."""
    meta = json.loads((pdir / "meta.json").read_text())
    bufs = sorted({a["name"] for k in meta["kernels"] for a in k["args"]}
                  | set(meta["graph_inputs"]) | set(meta["graph_outputs"]))
    bmap = {b: new_name(i, "b") for i, b in enumerate(bufs)}
    emap = {k["entry"]: new_name(n * 10 + j, "k") for j, k in enumerate(meta["kernels"])}
    pid = f"zz{n}_renamed_problem"
    dst = tmp / pid
    dst.mkdir()
    (dst / "meta.json").write_text(json.dumps(rename_meta(meta, bmap, emap, pid), indent=1))
    errs = []
    for k in meta["kernels"]:
        ir = json.loads((pdir / f"{k['entry']}.ir.json").read_text())
        new_ir = dst / f"{emap[k['entry']]}.ir.json"
        new_ir.write_text(json.dumps(rename_ir(ir, bmap, k["entry"], emap[k["entry"]]), indent=2))
        rc0, ptx0, e0 = compile_ir(compiler, pdir / f"{k['entry']}.ir.json")
        rc1, ptx1, e1 = compile_ir(compiler, new_ir)
        if rc0 != rc1:
            errs.append(f"{k['entry']}: rc {rc0} vs renamed rc {rc1}: {e1 or e0}")
            continue
        if rc0 != 0:  # refused both times: fine, as long as it is the same refusal
            if e0.split(":")[0] != e1.split(":")[0]:
                errs.append(f"{k['entry']}: different refusals: {e0} / {e1}")
            continue
        back = ptx1
        for old, new in sorted({**bmap, **emap}.items(), key=lambda t: -len(t[1])):
            back = back.replace(new, old)
        if back != ptx0:
            d = list(difflib.unified_diff(ptx0.splitlines(), back.splitlines(), "orig", "renamed",
                                          n=0, lineterm=""))
            errs.append(f"{k['entry']}: PTX differs beyond names ({len(d)} diff lines)")
            errs += ["      " + l for l in d[2:8]]
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compiler", default=str(ROOT))
    ap.add_argument("--problems", default=str(HK / "problems"))
    ap.add_argument("--keep", help="keep the renamed problems in this directory")
    args = ap.parse_args()
    pdirs = sorted(p for p in Path(args.problems).iterdir() if (p / "meta.json").exists())
    if not pdirs:
        print(f"rename: skipped, no problems under {args.problems}")
        return 0
    tmp = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="rename."))
    tmp.mkdir(parents=True, exist_ok=True)
    bad = 0
    try:
        for n, p in enumerate(pdirs):
            errs = check(p, tmp, n, Path(args.compiler))
            bad += bool(errs)
            print(f"  {'DIFF' if errs else 'same':<5} {p.name}")
            for e in errs:
                print(f"        {e}")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"rename: {len(pdirs) - bad}/{len(pdirs)} identical up to names"
          + ("" if not bad else ", failed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
