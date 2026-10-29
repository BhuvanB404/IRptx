# compile.py -- entry point: load -> KIR text -> bin/irptx -> relay stdout/stderr/exit.
#
# Usage: python compile.py <kernel.ir.json> > out.ptx      (the evaluator's whole contract)
#        python compile.py --kir <kernel.ir.json>           (print the KIR text, no core)
#        python compile.py --version
# This file only plumbs: it makes no lowering decision (all of that is in src/*.c).
#
# Finding the core, in order (cp_core):
#   1. bin/irptx, if it is a file we may execute and the OS agrees to run it.
#   2. else a one-time `cc -std=c99 -O2` build of src/*.c into .irptx-cache/irptx-<key>, where
#      <key> is a hash of every src/*.c and src/*.h (name and bytes), so a changed source gets
#      a new binary and an unchanged one is reused. The build goes to a private temp name and
#      is renamed into place (atomic on POSIX), so concurrent runs never see half a binary.
#      Old irptx-<key> files are left alone (a concurrent run may still be using one).
#   3. else E004, naming what was tried.
# Names: cp_* are this file's helpers; fe_* live in fe/kirdump.py.
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# `python -I` / PYTHONSAFEPATH drop the script directory from sys.path, so add it ourselves.
for _d in (str(ROOT / "fe"), str(ROOT)):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import kirdump as fe  # noqa: E402

CORE_TIMEOUT = 20      # seconds for one core run
BUILD_TIMEOUT = 120    # seconds for the one-time cc build


# ---- Finding the core ----
def cp_key():
    """Hash of every C source and header (name and bytes): the cache key."""
    h = hashlib.sha256()
    for p in sorted([*(ROOT / "src").glob("*.c"), *(ROOT / "src").glob("*.h")]):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def cp_cache_dir():
    """.irptx-cache beside the repo, or a per-user temp dir when the repo is read-only."""
    for base in (ROOT / ".irptx-cache",
                 Path(tempfile.gettempdir()) / f"irptx-cache-{os.getuid()}"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            if os.access(base, os.W_OK):
                return base
        except OSError:
            continue
    raise fe.FeError("E004", "no writable cache directory for the one-time cc build")


def cp_build():
    """Return the path of the cached cc-built core, building it first if needed."""
    srcs = sorted((ROOT / "src").glob("*.c"))
    if not srcs:
        raise fe.FeError("E004", "no src/*.c to build")
    dest = cp_cache_dir() / f"irptx-{cp_key()}"
    if os.access(dest, os.X_OK):
        return dest
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        res = subprocess.run(["cc", "-std=c99", "-O2", "-o", str(tmp), *map(str, srcs)],
                             capture_output=True, text=True, timeout=BUILD_TIMEOUT, cwd=ROOT)
        if res.returncode != 0:
            lines = res.stderr.strip().splitlines() or ["(no output)"]
            first = next((l for l in lines if "error" in l or "undefined" in l), lines[0])[:200]
            raise fe.FeError("E004", f"cc could not build src/*.c: {first}")
        os.replace(tmp, dest)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise fe.FeError("E004", f"cc could not build src/*.c: {exc}") from exc
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def cp_exec(irptx_exe, args, kir):
    """Run one core binary. Returns CompletedProcess; OSError means it would not start."""
    try:
        return subprocess.run([str(irptx_exe), *args], input=kir, capture_output=True,
                              timeout=CORE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise fe.FeError("E005", f"the core ran past {CORE_TIMEOUT} s and was stopped") from exc


def cp_core(args=(), kir=b""):
    """Run the core by the three tiers; return CompletedProcess or raise FeError."""
    tried = []
    lib = ROOT / "bin" / "irptx"
    if lib.is_file() and os.access(lib, os.X_OK):
        try:
            return cp_exec(lib, args, kir)
        except OSError as exc:
            tried.append(f"bin/irptx would not run ({exc.strerror})")
    else:
        tried.append("bin/irptx is missing or not executable")
    try:
        return cp_exec(cp_build(), args, kir)
    except OSError as exc:
        tried.append(f"the cc-built core would not run ({exc.strerror})")
    except fe.FeError as err:
        if err.code != "E004":
            raise
        tried.append(err.what)
    raise fe.FeError("E004", "no usable core: " + "; ".join(tried))


# ---- Commands ----
def cp_version():
    """irptx X.Y.Z. Asks bin/irptx when it runs; else reads IRPTX_VER from src/irptx.h."""
    lib = ROOT / "bin" / "irptx"
    if lib.is_file() and os.access(lib, os.X_OK):
        try:
            res = cp_exec(lib, ["--version"], b"")
            if res.returncode == 0 and res.stdout.startswith(b"irptx "):
                sys.stdout.buffer.write(res.stdout)
                return 0
        except (OSError, fe.FeError):
            pass
    try:
        text = (ROOT / "src" / "irptx.h").read_text(encoding="ascii")
        ver = re.search(r'#define\s+IRPTX_VER\s+"([^"]+)"', text).group(1)
    except (OSError, AttributeError, ValueError) as exc:
        raise RuntimeError("cannot find IRPTX_VER in src/irptx.h") from exc
    print(f"irptx {ver}")
    return 0


def cp_kir(path):
    """Load the IR through Emmy and return its KIR text."""
    graph, node = fe.fe_load(path)
    meta = fe.fe_meta(path, node.op.name)
    bufs = fe.fe_bufs(graph, node, meta)
    return fe.fe_dump(node.op.name, meta, bufs, node.op.body)


def cp_run(path, want_kir):
    """Print KIR (want_kir) or run the core on it and relay stdout, stderr and exit code."""
    kir = cp_kir(path)
    if want_kir:
        sys.stdout.write(kir)
        return 0
    res = cp_core((), kir.encode("ascii"))
    if res.returncode < 0:
        raise fe.FeError("E005", f"the core died with signal {-res.returncode}")
    sys.stderr.buffer.write(res.stderr)
    sys.stderr.buffer.flush()
    sys.stdout.buffer.write(res.stdout)
    sys.stdout.buffer.flush()
    return res.returncode


def cp_main(argv):
    """Parse the command line, run, and return the process exit code (0 or 1, or the core's)."""
    try:
        if argv == ["--version"]:
            return cp_version()
        pos = [a for a in argv if not a.startswith("--")]
        flags = [a for a in argv if a.startswith("--")]
        if len(pos) != 1 or flags not in ([], ["--kir"]):
            raise fe.FeError("E001", "usage: compile.py [--kir | --version] <kernel.ir.json>")
        return cp_run(pos[0], bool(flags))
    except fe.FeError as err:
        fe.fe_report(err)
        return 1
    except Exception:  # anything else is our bug: one line, a traceback only on request
        print("error[E000]: irptx: internal error in the front end", file=sys.stderr)
        if os.environ.get("IRPTX_DEBUG") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(cp_main(sys.argv[1:]))
