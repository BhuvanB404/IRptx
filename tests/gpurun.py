#!/usr/bin/env python3
"""gpurun.py -- launch one problem's PTX on the local GPU through libcuda (ctypes).

    python tests/gpurun.py <problem_dir> <kernel0.ptx> [<kernel1.ptx> ...] [--iters N]
    python tests/gpurun.py --probe        # exit 0 when a CUDA device is usable

One PTX file per kernel, in meta.json order; intermediates chain between kernels the way
the hackathon's submit.py chains them. Inputs come from the same seeds generate.py used;
outputs not listed in zero_outputs start as NaN, so an unwritten cell fails. The first
launch is the one checked against expected.npz; with --iters N, N more launches are timed
with CUDA events and the median per kernel is summed. Prints one JSON line:
{"status": "PASS"|"FAIL", "why": ..., "us": ...}. Only f32 buffers (all problems today).
"""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path

import numpy as np

cu = None


def ck(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what}: CUDA error {r} {s.value.decode() if s.value else ''}")


def init() -> bool:
    global cu
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    n = ctypes.c_int(0)
    if cu.cuInit(0) != 0 or cu.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value < 1:
        return False
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "device")
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "context")
    ck(cu.cuCtxSetCurrent(ctx), "set context")
    return True


def host_array(a, k, produced):
    n = math.prod(a["shape"])
    if "data" in a:
        return np.array(a["data"], dtype=np.float32).reshape(-1)
    if a.get("intermediate"):
        return produced[a["name"]].astype(np.float32).reshape(-1)
    if "seed" in a:
        rng = np.random.default_rng(a["seed"])
        return rng.standard_normal(a["shape"]).astype(np.float32).reshape(-1)
    if a.get("output"):
        fill = 0.0 if a["name"] in k["zero_outputs"] else np.nan
        return np.full(n, fill, np.float32)
    raise RuntimeError(f"arg {a['name']}: no data, seed or output flag")


def launch(fn, k, argv):
    g, b = k["grid"], k["block"]
    ck(cu.cuLaunchKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2], k["smem_bytes"],
                         None, argv, None), "launch")


def time_kernel(fn, k, argv, iters):
    e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
    ck(cu.cuEventCreate(ctypes.byref(e0), 0), "event")
    ck(cu.cuEventCreate(ctypes.byref(e1), 0), "event")
    times = []
    for _ in range(iters):
        ck(cu.cuEventRecord(e0, None), "record")
        launch(fn, k, argv)
        ck(cu.cuEventRecord(e1, None), "record")
        ck(cu.cuEventSynchronize(e1), "event sync")
        ms = ctypes.c_float()
        ck(cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1), "elapsed")
        times.append(ms.value * 1000.0)
    cu.cuEventDestroy_v2(e0)
    cu.cuEventDestroy_v2(e1)
    return float(np.median(times))


def run_kernel(k, ptx, produced, expected, meta, iters):
    """Launch one kernel; return (list of mismatch messages, median us or 0)."""
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    ck(cu.cuModuleLoadData(ctypes.byref(mod), Path(ptx).read_bytes() + b"\0"), "module load")
    ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, k["entry"].encode()), "get function")
    if k["smem_bytes"]:
        ck(cu.cuFuncSetAttribute(fn, 8, k["smem_bytes"]), "smem attribute")
    bufs = []
    for a in k["args"]:
        h = host_array(a, k, produced)
        d = ctypes.c_uint64()
        ck(cu.cuMemAlloc_v2(ctypes.byref(d), ctypes.c_size_t(max(4, h.size * 4))), "alloc")
        ck(cu.cuMemcpyHtoD_v2(d, ctypes.c_void_p(h.ctypes.data), ctypes.c_size_t(h.size * 4)), "h2d")
        bufs.append((a, h.size, d))
    argv = (ctypes.c_void_p * len(bufs))(
        *[ctypes.cast(ctypes.pointer(d), ctypes.c_void_p) for _, _, d in bufs])
    launch(fn, k, argv)
    ck(cu.cuCtxSynchronize(), "sync after launch")
    bad = []
    for a, n, d in bufs:
        if not a.get("output"):
            continue
        o = np.empty(n, np.float32)
        ck(cu.cuMemcpyDtoH_v2(ctypes.c_void_p(o.ctypes.data), d, ctypes.c_size_t(n * 4)), "d2h")
        produced[a["name"]] = o
        if a["name"] in expected:
            e = expected[a["name"]].reshape(-1).astype(np.float32)
            if not np.allclose(o, e, rtol=meta["rtol"], atol=meta["atol"]):
                err = float(np.nanmax(np.abs(o - e))) if not np.isnan(o).all() else float("nan")
                bad.append(f"{a['name']} maxerr={err:.3e} nan={int(np.isnan(o).sum())}")
    us = time_kernel(fn, k, argv, iters) if iters else 0.0
    for _, _, d in bufs:
        cu.cuMemFree_v2(d)
    cu.cuModuleUnload(mod)
    return bad, us


def main(argv) -> int:
    if argv == ["--probe"]:
        return 0 if init() else 1
    iters = 0
    if "--iters" in argv:
        i = argv.index("--iters")
        iters = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    pdir, ptxs = Path(argv[0]), argv[1:]
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if (pdir / "expected.npz").exists() else {}
    if not init():
        print(json.dumps(dict(status="FAIL", why="no CUDA device")))
        return 1
    produced, bad, total = {}, [], 0.0
    try:
        for k, ptx in zip(meta["kernels"], ptxs):
            b, us = run_kernel(k, ptx, produced, expected, meta, iters)
            bad += b
            total += us
    except RuntimeError as exc:
        print(json.dumps(dict(status="FAIL", why=str(exc)[:200])))
        return 1
    res = dict(status="FAIL" if bad else "PASS", why="; ".join(bad))
    if iters:
        res["us"] = round(total, 2)
    print(json.dumps(res))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
