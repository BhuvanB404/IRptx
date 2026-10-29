#!/usr/bin/env bash
# smoke.sh -- smk01..: every open IR compiles or refuses with an E-code.
#
# For each kernel IR under $HK/problems, run `compile.py <ir>`:
#   rc 0  stdout is one PTX module: version stamp comment first, then .version 8.0,
#         .target sm_89, .address_size 64, ASCII only, one .entry named as
#         the KIR says, .param count = KIR arg count, all .u64, stderr empty,
#         and ptxas accepts it for sm_89 and sm_120.
#   rc 1  stdout empty and stderr exactly one line "error[E0NN]: irptx: ..." (a refusal,
#         never a traceback). A reader code (E010-E015, E080-E082) is a failure: it means
#         the front end wrote KIR the core could not read.
#   else  failure (crash, signal, timeout).
# `compile.py --kir` must also exit 0 for every IR.
#
# Knobs: HK (hackathon checkout, default ../emmy-hackathon-2026), PY (default .venv/bin/python),
#        PTXAS (default: ptxas on PATH, /opt/cuda/bin/ptxas, or the venv wheel's).
# Exit 0 when nothing failed (or HK is absent), 1 otherwise.
set -u
export LC_ALL=C

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$SELF")
HK=${HK:-$ROOT/../emmy-hackathon-2026}
PY=${PY:-.venv/bin/python}
cd "$ROOT" || { echo "smk: no such tree: $ROOT"; exit 1; }

if ! ls "$HK"/problems/*/*.ir.json >/dev/null 2>&1; then
    echo "smk: skipped, no hackathon checkout"
    exit 0
fi

if [ -z "${PTXAS:-}" ]; then
    PTXAS=$(command -v ptxas || true)
    [ -n "$PTXAS" ] || { [ -x /opt/cuda/bin/ptxas ] && PTXAS=/opt/cuda/bin/ptxas; }
    if [ -z "$PTXAS" ]; then
        PTXAS=$(ls .venv/lib/python*/site-packages/nvidia/*/bin/ptxas 2>/dev/null | head -n 1)
    fi
fi
[ -n "$PTXAS" ] || echo "smk: note, no ptxas found, assembly checks skipped"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/smk.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT

n=0
fails=0
ok=0
refused=0
list=""

# check_ptx <ptx> <kir>: prints a failure reason and returns 1, or returns 0.
check_ptx() {
    local ptx=$1 kir=$2 want got nargs nparam stamp
    [ -s "$ptx" ] || { echo "empty stdout on exit 0"; return 1; }
    if LC_ALL=C grep -q "[^[:print:]"$'\t'"]" "$ptx"; then
        echo "non-ASCII byte"; return 1
    fi
    stamp=$(sed -n '1,3p' "$ptx" | grep -c '^// irptx [0-9]')
    [ "$stamp" -ge 1 ] || { echo "no version stamp in the first comment"; return 1; }
    got=$(grep -v '^[[:space:]]*//' "$ptx" | grep -v '^[[:space:]]*$' | sed -n '1,3p' | tr '\n' '|')
    [ "$got" = ".version 8.0|.target sm_89|.address_size 64|" ] \
        || { echo "header is '$got'"; return 1; }
    want=$(sed -n 's/^kernel //p' "$kir" | head -n 1)
    got=$(grep -c '^\(\.visible \)\{0,1\}\.entry ' "$ptx")
    [ "$got" -eq 1 ] || { echo "$got .entry lines"; return 1; }
    grep -q "^\\(\\.visible \\)\\?\\.entry $want\\b" "$ptx" || { echo "entry is not '$want'"; return 1; }
    nargs=$(grep -c '^arg ' "$kir")
    nparam=$(grep -c '^[[:space:]]*\.param ' "$ptx")
    [ "$nparam" -eq "$nargs" ] || { echo "$nparam params for $nargs args"; return 1; }
    if grep '^[[:space:]]*\.param ' "$ptx" | grep -qv '\.u64'; then
        echo "a param is not .u64"; return 1
    fi
    return 0
}

for ir in "$HK"/problems/*/*.ir.json; do
    prob=$(basename "$(dirname "$ir")")
    kern=$(basename "$ir" .ir.json)
    n=$((n + 1))
    id=$(printf 'smk %02d %s/%s' "$n" "$prob" "$kern")
    out=$TMP/out.ptx err=$TMP/err.txt kir=$TMP/kir.txt
    why=""
    code=""

    timeout 60 "$PY" compile.py --kir "$ir" >"$kir" 2>"$TMP/kerr.txt"
    krc=$?
    if [ "$krc" -ne 0 ]; then
        why="--kir exited $krc"
    else
        timeout 60 "$PY" compile.py "$ir" >"$out" 2>"$err"
        rc=$?
        nerr=$(wc -l <"$err")
        case $rc in
        0)
            if [ -s "$err" ]; then
                why="stderr not empty on exit 0"
            else
                why=$(check_ptx "$out" "$kir") || true
                if [ -z "$why" ] && [ -n "$PTXAS" ]; then
                    for arch in sm_89 sm_120; do
                        if ! "$PTXAS" -arch=$arch -o /dev/null "$out" >"$TMP/as.txt" 2>&1; then
                            why="ptxas $arch: $(head -n 1 "$TMP/as.txt")"
                            break
                        fi
                    done
                fi
                [ -n "$why" ] || code=ok
            fi ;;
        1)
            if [ -s "$out" ]; then
                why="stdout not empty on exit 1"
            elif [ "$nerr" -ne 1 ] || ! grep -q -E '^error\[E[0-9]{3}\]: irptx: ' "$err"; then
                why="stderr is not one error[E0NN] line: $(head -n 1 "$err" | cut -c 1-80)"
            else
                code=$(sed -n 's/^error\[\(E[0-9]*\)\].*/\1/p' "$err")
                case $code in
                E000|E004|E005)
                    why="crash or missing core, not a refusal ($code)" ;;
                E010|E011|E012|E013|E014|E015|E080|E081|E082)
                    why="the core could not read the KIR ($code)" ;;
                esac
            fi ;;
        *) why="exit $rc" ;;
        esac
    fi

    if [ -n "$why" ]; then
        echo "$id: FAIL $why"
        fails=$((fails + 1))
    elif [ "$code" = ok ]; then
        echo "$id: compiled"
        ok=$((ok + 1))
    else
        echo "$id: refused $code"
        refused=$((refused + 1))
    fi
done

note=""
[ -n "$PTXAS" ] || note=", assembly skipped"
echo "smk: $n kernels, $ok compiled, $refused refused, $fails failed$note"
[ "$fails" -eq 0 ]
