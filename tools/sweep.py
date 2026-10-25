#!/usr/bin/env python3
# sweep.py -- send every problem through the hackathon runner and record the result.
#
# Runs `submit.py <problem> --compiler compile.py` once per problem directory,
# sequentially (the runner is shared, parallel launches only earn "busy"),
# reads the verdict out of submit.py's printed text, and writes
# results/sweep.json (contracts/sweep-results.md). --compare is Gate G3:
# exit 1 when anything that passed before no longer does.
import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 900                      # one problem, all kernels, runner queue included
CATS = {"p": 1, "r": 2, "m": 3, "f": 4, "a": 5}   # TASK.md: the id letter is the category

# ---- Parsing submit.py output ----
# Shapes come straight from submit.py's print/sys.exit calls. The first line
# that matches decides, because submit.py stops at the first failing kernel.
_PASS = re.compile(r"^\[[^\]]+\] PASS\s.*\((.*?) us\)")
_BAD = (                             # (prefix after "[entry] ", status)
    ("your compiler failed", "compile_error"),
    ("that is not PTX", "compile_error"),
    ("compiled, but entry points", "compile_error"),
    ("ptxas rejected it", "ptxas"),
    ("launch failed", "run_error"),
    ("FAIL", "wrong"),
    ("needs ", "run_error"),
    ("output ", "run_error"),
)


def sw_parse(text, rc):
    """Return (status, detail, median_us) for one submit.py run."""
    lines = text.splitlines()
    us, npass = 0.0, 0
    for i, ln in enumerate(lines):
        m = _PASS.match(ln)
        if m:
            npass += 1
            us += sw_num(m.group(1))
            continue
        tag = ln.split("] ", 1)[1] if ln.startswith("[") and "] " in ln else ""
        for pre, st in _BAD:
            if tag.startswith(pre):
                return st, sw_why(lines, i), None
    if lines and lines[0].startswith(("cannot reach", "unknown problem")):
        return "run_error", lines[0][:200], None
    if rc == 0 and npass:
        return "pass", "", round(us, 2)
    return "run_error", "no verdict in submit.py output (rc %d)" % rc, None


def sw_num(txt):
    """A median from submit.py's "(<x> us)"; '?' or 'None' (null median) count as 0."""
    try:
        return float(txt)
    except ValueError:
        return 0.0


def sw_why(lines, i):
    """First line of the failing stage's message: the line itself, or the one after a trailing colon."""
    ln = lines[i].split("] ", 1)[-1].strip()
    if ln.endswith(":"):
        for nxt in lines[i + 1:]:
            if nxt.strip():
                return nxt.strip()[:200]
    return ln[:200]


# ---- Running one problem ----

def sw_ctime(pdir):
    """Seconds spent in compile.py over the problem's kernels (the runner's time is excluded)."""
    t0 = time.monotonic()
    for ir in sorted(Path(pdir).glob("*.ir.json")):
        subprocess.run([sys.executable, "compile.py", str(ir)], cwd=ROOT,
                       capture_output=True, text=True, timeout=TIMEOUT_S)
    return round(time.monotonic() - t0, 2)


def sw_run(hk, pdir):
    """Run submit.py on one problem directory; returns the result record fields."""
    cmd = [sys.executable, str(hk / "submit.py"), str(pdir), "--compiler", "compile.py"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S)
        st, why, us = sw_parse(r.stdout + r.stderr, r.returncode)
    except subprocess.TimeoutExpired:
        st, why, us = "run_error", "timed out after %d s" % TIMEOUT_S, None
    try:
        cs = sw_ctime(pdir)
    except subprocess.TimeoutExpired:      # a hung compile.py must not abort the sweep
        st, why, us, cs = "compile_error", "compile.py timed out after %d s" % TIMEOUT_S, None, TIMEOUT_S
    return {"status": st, "detail": why, "median_us": us, "compile_s": cs}


def sw_list(hk, which):
    """(set, directory) pairs: open problems under the hackathon, extra ones under the repo."""
    out = []
    if which in ("open", "all"):
        out += [("open", d) for d in sorted((hk / "problems").glob("*")) if (d / "meta.json").exists()]
    if which in ("extra", "all"):
        out += [("extra", d) for d in sorted((ROOT / "extra_problems").glob("*")) if (d / "meta.json").exists()]
    return out


# ---- Output and comparison ----

def sw_sha():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def sw_summ(results):
    """category -> [passed, total], keys as strings to match the JSON contract."""
    s = {}
    for r in results:
        c = s.setdefault(str(r["category"]), [0, 0])
        c[1] += 1
        c[0] += r["status"] == "pass"
    return dict(sorted(s.items()))


def sw_regr(old, new):
    """Problems that passed in the baseline and do not pass now."""
    was = {r["problem"] for r in old.get("results", []) if r["status"] == "pass"}
    return sorted(r["problem"] for r in new if r["problem"] in was and r["status"] != "pass")


def sw_base(path):
    """Load the --compare baseline; (doc, None) or (None, why). A vacuous G3 pass is worse than none."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return None, "cannot use baseline %s: %s" % (path, e)
    if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
        return None, "baseline %s has no results list" % path
    return doc, None


def sw_args(argv):
    ap = argparse.ArgumentParser(description="Sweep every problem through the runner.")
    ap.add_argument("--hackathon", default=str(ROOT.parent / "emmy-hackathon-2026"))
    ap.add_argument("--problems", choices=("open", "extra", "all"), default="open")
    ap.add_argument("--only", default="", help="keep problems whose id contains this")
    ap.add_argument("--out", default=str(ROOT / "results" / "sweep.json"))
    ap.add_argument("--compare", action="store_true",
                    help="diff against the existing --out file, do not overwrite it")
    return ap.parse_args(argv)


def main(argv=None):
    a = sw_args(argv)
    hk, out = Path(a.hackathon).resolve(), Path(a.out)
    old = None
    if a.compare:                    # check the baseline first, before spending runner time
        old, why = sw_base(out)
        if why:
            print("sweep: " + why, file=sys.stderr)
            return 2
    todo = [(s, d) for s, d in sw_list(hk, a.problems) if a.only in d.name]
    if not todo:
        print("sweep: no problems matched", file=sys.stderr)
        return 2
    results = []
    for sset, d in todo:
        rec = {"problem": d.name, "category": CATS.get(d.name[0], 0), "set": sset, **sw_run(hk, d)}
        results.append(rec)
        print("%-8s %-24s %s" % (rec["status"].upper(), rec["problem"], rec["detail"]), flush=True)
    summ = sw_summ(results)
    for c, (p, t) in summ.items():
        print("category %s: %d/%d" % (c, p, t))
    if a.compare:
        bad = sw_regr(old, results)
        for p in bad:
            print("REGRESSION: %s passed before" % p)
        return 1 if bad else 0
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    doc = {"commit": sw_sha(), "timestamp": stamp, "summary": summ, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
