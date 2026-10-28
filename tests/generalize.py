#!/usr/bin/env python3
"""generalize.py -- does the compiler work on kernels it has never seen?

Generates a held-out problem set (tests/heldout.py by default) with the
hackathon's own tools/generate.py, then for every kernel:

  1. compile.py <ir>         rc 1 with "error[E...]" is REFUSED, anything else non-zero is CRASH
  2. ptxas -arch=sm_89       rejection is PTXAS
  3. launch on the local GPU through libcuda (ctypes), under meta.json's geometry, with
     outputs pre-filled with NaN (unless zero_outputs), compared to expected.npz at the
     problem's rtol/atol: PASS or FAIL. With no GPU (or no libcuda) the result is ASM
     (compiled and assembled, not run).

Exit 1 when anything is FAIL, PTXAS or CRASH, or (with --baseline) when a problem that
passed in the baseline does not pass now. REFUSED alone does not fail the run unless
--strict; it is reported, because the hidden set scores a refusal as zero.

  python tests/generalize.py                      # build/generalize, tests/heldout.py
  python tests/generalize.py --set X.py --out D   # another expression file / output dir
  python tests/generalize.py --compiler DIR       # compile.py of another checkout
  python tests/generalize.py --json r.json --baseline old.json --time 20
  python tests/generalize.py --ref                # sanity: run emmy's .ref.cu via nvcc

--ref checks the held-out set itself (a reference that fails means the numpy ref in the
expression file is wrong); it never touches the compiler.
Knobs: HK (hackathon checkout), PTXAS, NVCC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HK = Path(os.environ.get("HK", ROOT.parent / "emmy-hackathon-2026"))
BAD = ("FAIL", "PTXAS", "CRASH")


def find_tool(name, env):
    if os.environ.get(env):
        return os.environ[env]
    for cand in (shutil.which(name), f"/opt/cuda/bin/{name}", f"/usr/local/cuda/bin/{name}"):
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


# ---- Generation ------------------------------------------------------------
def generate(exprs: Path, out: Path, regen: bool) -> None:
    """Run tools/generate.py into out/, unless out/ was built from these same bytes."""
    stamp = out / ".stamp"
    key = hashlib.sha256(exprs.read_bytes()).hexdigest()
    if not regen and stamp.exists() and stamp.read_text().strip() == key:
        return
    gen = HK / "tools" / "generate.py"
    if not gen.exists():
        sys.exit(f"generalize: no {gen}; set HK to the hackathon checkout")
    bindir = Path(sys.executable).parent
    emmy = bindir / "emmy" if (bindir / "emmy").exists() else shutil.which("emmy")
    if not emmy:
        sys.exit("generalize: no emmy CLI next to this python or on PATH")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    print(f"generalize: generating {exprs.name} into {out}", flush=True)
    r = subprocess.run([sys.executable, str(gen), "--emmy", str(emmy), "--out", str(out),
                        str(exprs)], env=env, text=True, capture_output=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print("generalize: some expressions did not trace (listed above); continuing")
    stamp.write_text(key + "\n")


# ---- Compile and assemble --------------------------------------------------
def compile_one(compiler: Path, ir: Path, ptx: Path):
    r = subprocess.run([sys.executable, str(compiler / "compile.py"), str(ir)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 1 and r.stderr.startswith("error[E"):
        return "REFUSED", r.stderr.strip().splitlines()[0][:160]
    if r.returncode != 0 or not r.stdout.strip():
        return "CRASH", f"rc {r.returncode}: {(r.stderr.strip().splitlines() or ['?'])[-1][:160]}"
    ptx.write_text(r.stdout)
    return None, ""


def assemble(ptxas, ptx: Path):
    if not ptxas:
        return None, ""
    r = subprocess.run([ptxas, "-arch=sm_89", "-o", os.devnull, str(ptx)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return "PTXAS", (r.stderr.strip().splitlines() or ["?"])[0][:160]
    return None, ""


def nvcc_ref(nvcc, cu: Path, ptx: Path):
    r = subprocess.run([nvcc, "-ptx", "-arch=sm_89", "-o", str(ptx), str(cu)],
                       capture_output=True, text=True, timeout=300)
    return (None, "") if r.returncode == 0 else ("CRASH", "nvcc: " + r.stderr[:160])


# ---- One problem -----------------------------------------------------------
def run_problem(pdir: Path, compiler: Path, ptxdir: Path, ptxas, use_gpu, iters, ref):
    meta = json.loads((pdir / "meta.json").read_text())
    ptxs = []
    for k in meta["kernels"]:
        ptx = ptxdir / f"{pdir.name}__{k['entry']}.ptx"
        if ref:
            st, why = nvcc_ref(find_tool("nvcc", "NVCC"), pdir / f"{k['entry']}.ref.cu", ptx)
        else:
            st, why = compile_one(compiler, pdir / f"{k['entry']}.ir.json", ptx)
        if not st:
            st, why = assemble(ptxas, ptx)
        if st:
            return dict(status=st, why=f"{k['entry']}: {why}")
        ptxs.append(str(ptx))
    if not use_gpu:
        return dict(status="ASM", why="no GPU, assembled only" if ptxas else "no GPU, no ptxas")
    # Each launch runs in its own process: a faulting kernel poisons the CUDA context.
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "gpurun.py"),
                        str(pdir), *ptxs, "--iters", str(iters)],
                       capture_output=True, text=True, timeout=300)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return dict(status="FAIL", why=f"runner rc {r.returncode}: {r.stderr.strip()[-160:]}")


def gpu_present() -> bool:
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "gpurun.py"),
                        "--probe"], capture_output=True, text=True)
    return r.returncode == 0


# ---- Main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default=str(ROOT / "tests" / "heldout.py"))
    ap.add_argument("--out", default=str(ROOT / "build" / "generalize"))
    ap.add_argument("--problems", help="use an existing problems dir as is (no generation)")
    ap.add_argument("--compiler", default=str(ROOT), help="checkout whose compile.py is used")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--strict", action="store_true", help="REFUSED also fails the run")
    ap.add_argument("--time", type=int, default=0, metavar="N", help="time N launches")
    ap.add_argument("--json", help="write per-problem results here")
    ap.add_argument("--baseline", help="results json to compare against")
    ap.add_argument("--only", help="comma-separated problem ids")
    args = ap.parse_args()

    if args.problems:
        pdirs_root = Path(args.problems)
    else:
        pdirs_root = Path(args.out) / "problems"
        generate(Path(args.set).resolve(), pdirs_root, args.regen)
    pdirs = sorted(p for p in pdirs_root.iterdir() if (p / "meta.json").exists())
    if args.only:
        pdirs = [p for p in pdirs if p.name in args.only.split(",")]
    ptxdir = Path(args.out) / ("ptx-ref" if args.ref else "ptx")
    ptxdir.mkdir(parents=True, exist_ok=True)
    ptxas = find_tool("ptxas", "PTXAS")
    use_gpu = not args.no_gpu and gpu_present()
    print(f"generalize: {len(pdirs)} problems, compiler {args.compiler}, "
          f"ptxas {'yes' if ptxas else 'MISSING'}, gpu {'yes' if use_gpu else 'no'}")

    results = {}
    for p in pdirs:
        res = run_problem(p, Path(args.compiler), ptxdir, ptxas, use_gpu, args.time, args.ref)
        results[p.name] = res
        us = f" {res['us']:.1f}us" if res.get("us") else ""
        print(f"  {res['status']:<8} {p.name:<24}{us} {res.get('why', '')}", flush=True)
    return report(results, args)


def report(results, args) -> int:
    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("generalize: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    bad = [n for n, r in results.items() if r["status"] in BAD
           or (args.strict and r["status"] == "REFUSED")]
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        for n, r in sorted(results.items()):
            was = base.get(n, {}).get("status")
            if was in ("PASS", "ASM") and r["status"] != was:
                print(f"  REGRESSED {n}: {was} -> {r['status']}")
                bad.append(n)
            elif was and was != r["status"]:
                print(f"  changed   {n}: {was} -> {r['status']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, sort_keys=True))
    print("generalize: " + ("failed: " + " ".join(sorted(set(bad))) if bad else "clean"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
