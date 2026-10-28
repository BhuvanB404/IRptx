#!/usr/bin/env bash
# hardcode.sh -- static check: nothing in src/ or fe/ may know the open set.
#
# Fails (exit 1) on any of these in src/*.c, src/*.h, fe/*.py or compile.py, comments
# INCLUDED (a problem named "as an example" in a comment should be reworded to describe
# the IR shape instead):
#   name   an open-set problem directory name (p00_scalar_add) or its short id (p00, f02),
#          a kernel entry name (k_softmax_05263d), the hash part of one (05263d), or a
#          distinctive buffer name (sum_1, mean_keepdim_count; names with no digit or
#          underscore, and x0/x1..., are ordinary words and are skipped)
#   size   an integer that the open set's shapes, element counts or launch geometry
#          produce AND that is distinctive: a power of two >= 8192, or a non-power-of-two
#          above 64 (decimal or hex). 32, 128, 256, 1024 and friends never fire.
#   const  a non-round float constant from an open-set ConstantOp (5.656854) or its
#          reciprocal.
# Names and numbers come from $HK/problems when the checkout exists, else from
# tests/rules.names and a built-in copy of the sizes. tests/hardcode.allow can permit a
# hit, one per line as "<path> <token> <text the line must contain>  # reason"; it is for
# fixed capacities and buffer sizes only, never for a lowering decision.
#
#   bash tests/hardcode.sh          Knobs: HK, PY, HARDCODE_ROOT (tree to scan)
set -u
export LC_ALL=C
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${HARDCODE_ROOT:-$(dirname "$SELF")}
HK=${HK:-$(dirname "$SELF")/../emmy-hackathon-2026}
PY=${PY:-$(dirname "$SELF")/.venv/bin/python}
[ -x "$PY" ] || PY=$(command -v python3)
[ -n "$PY" ] || { echo "hardcode: no python"; exit 1; }

exec "$PY" -I - "$ROOT" "$HK" "$SELF/rules.names" "$SELF/hardcode.allow" <<'EOF'
import glob, json, math, os, re, sys

root, hk, names_file, allow_file = sys.argv[1:5]

# Fallback sizes when there is no hackathon checkout: what the derivation below
# produced from the open set on 2026-10-28 (shapes, products, grid, block, smem).
FALLBACK_SIZES = {8192, 16384, 32768, 65536, 73728, 131072, 262144}
FALLBACK_CONSTS = {"5.656854"}


def pow2(n):
    return n > 0 and n & (n - 1) == 0


def distinctive(n):
    return (pow2(n) and n >= 8192) or (not pow2(n) and n > 64)


def from_checkout():
    names, sizes, consts = set(), set(), set()
    for d in sorted(glob.glob(os.path.join(hk, "problems", "*", "meta.json"))):
        pid = os.path.basename(os.path.dirname(d))
        names.add(pid)
        names.add(pid.split("_")[0])
        meta = json.load(open(d))
        for k in meta["kernels"]:
            names.add(k["entry"])
            m = re.search(r"_([0-9a-f]{6})$", k["entry"])
            if m:
                names.add(m.group(1))
            dims = k["grid"] + k["block"] + [k["smem_bytes"],
                                              math.prod(k["grid"]) * math.prod(k["block"])]
            for a in k["args"]:
                if re.search(r"[_0-9]", a["name"]) and not re.fullmatch(r"x[0-9]+", a["name"]):
                    names.add(a["name"])
                dims += list(a["shape"]) + [math.prod(a["shape"])]
                for v in a.get("data", []):
                    if float(v) != int(float(v)) and abs(v) > 1e-3:
                        consts.add(repr(float(v)))
                        consts.add(f"{1.0 / float(v):.6f}")
            sizes.update(n for n in dims if distinctive(int(n)))
    return names, sizes, consts


names, sizes, consts = from_checkout()
if not names:
    names = {l.strip() for l in open(names_file) if l.strip() and not l.startswith("#")}
    names |= {n.split("_")[0] for n in names if re.fullmatch(r"[pmrfa]\d\d_.*", n)}
    names |= {m.group(1) for n in names for m in [re.search(r"_([0-9a-f]{6})$", n)] if m}
    sizes, consts = FALLBACK_SIZES, FALLBACK_CONSTS

allow = []
if os.path.exists(allow_file):
    for l in open(allow_file):
        f = l.split("#", 1)[0].split()
        if len(f) == 3:
            allow.append(tuple(f))

pats = [("name", re.compile(r"(?<![A-Za-z0-9_%%])(%s)(?![A-Za-z0-9_])"
                            % "|".join(sorted(map(re.escape, names), key=len, reverse=True))))]
size_alt = [str(n) for n in sizes] + ["0x%x" % n for n in sizes]
pats.append(("size", re.compile(r"(?<![A-Za-z0-9_.])(%s)(?:[uUlL]*)(?![A-Za-z0-9_.])"
                                % "|".join(sorted(size_alt, key=len, reverse=True)), re.I)))
if consts:
    pats.append(("const", re.compile(r"(%s)" % "|".join(re.escape(c[:6]) for c in consts))))

files = sorted(glob.glob(os.path.join(root, "src", "*.[ch]"))
               + glob.glob(os.path.join(root, "fe", "*.py")))
if os.path.exists(os.path.join(root, "compile.py")):
    files.append(os.path.join(root, "compile.py"))

hits = 0
for f in files:
    rel = os.path.relpath(f, root)
    for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
        for kind, p in pats:
            for m in p.finditer(line):
                tok = m.group(1)
                if any(a == (rel, tok.lower(), a[2]) and a[2] in line for a in allow):
                    continue
                hits += 1
                print(f"  {kind:<5} {rel}:{i}: {tok}    | {line.strip()[:90]}")

print(f"[hardcode] {len(files)} files, {len(names)} names, {len(sizes)} sizes, "
      f"{len(consts)} constants from the open set")
print("hardcode: clean" if hits == 0 else f"hardcode: failed, {hits} hit(s)")
sys.exit(1 if hits else 0)
EOF
