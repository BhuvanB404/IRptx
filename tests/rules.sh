#!/usr/bin/env bash
#
# Greps, not parsers. Reads baselines from tests/rules.tbl and fails (exit 1)
# when any count rises. Tolerates missing files and directories, and strips
# comments before matching code rules so a header like "/* x -- y */" or a
# remark about malloc never counts.
#
# Run from anywhere:  bash tests/rules.sh
# Knobs:  RULES_ROOT  tree to check (default: the repo this script lives in),
#         HK          hackathon checkout for deriving open-set names.
set -u
export LC_ALL=C

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${RULES_ROOT:-$(dirname "$SELF")}
TBL=$SELF/rules.tbl
NAMES=$SELF/rules.names
HK=${HK:-/home/bb/Projects/irptx/emmy-hackathon-2026}
cd "$ROOT" || { echo "rules: no such tree: $ROOT"; exit 1; }

# ---- File sets (missing ones simply come out empty) ----
CSRC=$(ls src/*.c src/*.h 2>/dev/null)
PYSRC=$(ls fe/*.py compile.py 2>/dev/null)
CODE="$CSRC $PYSRC"

# ---- Comment strippers: keep line numbers, drop comment text ----
# C: /* */ across lines and // to end of line; string and char literals are
# skipped so "/*" inside a string does not open a comment.
cstrip() {
    [ $# -gt 0 ] || return 0
    awk -v blank="${BLANK:-0}" '
    BEGIN { inc = 0 }
    {
        s = $0; out = ""; i = 1; n = length(s)
        while (i <= n) {
            c = substr(s, i, 1); d = substr(s, i, 2)
            if (inc) { if (d == "*/") { inc = 0; i += 2 } else i++; continue }
            if (d == "/*") { inc = 1; i += 2; out = out " "; continue }
            if (d == "//") break
            if (c == "\"" || c == "\047") {
                q = c; out = out c; i++
                while (i <= n) {
                    c = substr(s, i, 1); i++
                    if (c == "\\") { out = out (blank ? "xx" : c substr(s, i, 1)); i++ }
                    else if (c == q) { out = out c; break }
                    else out = out (blank ? "x" : c)
                }
                continue
            }
            out = out c; i++
        }
        print out
    }' "$@"
}
# Python: drop # comments outside string literals (single-line aware).
pystrip() {
    [ $# -gt 0 ] || return 0
    awk '
    {
        s = $0; out = ""; q = ""; n = length(s)
        for (i = 1; i <= n; i++) {
            c = substr(s, i, 1)
            if (q != "") { out = out c; if (c == "\\") { out = out substr(s, i+1, 1); i++ } else if (c == q) q = ""; continue }
            if (c == "\"" || c == "\047") { q = c; out = out c; continue }
            if (c == "#") break
            out = out c
        }
        print out
    }' "$@"
}
cstripped()  { cstrip $CSRC; }
# Same, with string and char literal contents blanked, for rules about code
# tokens (a "goto" inside a message is prose, not control flow).
cstripb()    { BLANK=1 cstrip $CSRC; }
pystripped() { pystrip $PYSRC; }
codestripped() { cstripped; pystripped; }
nlines() { grep -c . ; }

# ---- Rules: each prints a count ----
r_goto()  { cstripb | grep -oE '\b(goto|setjmp|longjmp)\b' | wc -l; }
r_alloc() { cstripb | grep -oE '\b(malloc|calloc|realloc|free)[[:space:]]*\(' | wc -l; }
# (*name)( covers declarations, casts and calls through a pointer; the bare
# "(*" before an identifier-free ")(" covers abstract declarators.
r_fnptr() { cstripb | grep -oE '\([[:space:]]*\*[[:space:]]*[A-Za-z_0-9]*[[:space:]]*\)[[:space:]]*\(' | wc -l; }

# A C function starts at a column-0 line with "(" that is not a prototype,
# macro or typedef, and ends at the next column-0 "}". Python: a def, ending
# at the next non-blank line indented no deeper than the def.
r_long() {
    local c p
    c=$(cstripb | awk '
        # A candidate start is buffered until a ";" (prototype or declaration,
        # single- or multi-line: dropped) or a "{" (a function: counting starts
        # at the candidate line).
        !f && !cand && /^[A-Za-z_][^;]*\(/ && !/^(typedef|#)/ { cand = 1; cn = 0 }
        cand { cn++
               if (/;/) cand = 0
               else if (/\{/) { f = !/\{.*\}[[:space:]]*$/; n = cn; cand = 0 }
               next_line = 1 }
        f && !next_line { n++ }
        { next_line = 0 }
        /^\}/ { if (f && n > 60) c++; f = 0; cand = 0 }
        END { print c + 0 }')
    p=0
    if [ -n "$PYSRC" ]; then
        p=$(for f in $PYSRC; do pystrip "$f" | awk '
            function close_def() { if (d >= 0 && last - start + 1 > 60) c++; d = -1 }
            BEGIN { d = -1 }
            /^[[:space:]]*$/ { next }
            { match($0, /^[ \t]*/); ind = RLENGTH
              if (d >= 0 && ind <= d) close_def()
              if ($0 ~ /^[ \t]*(async[ \t]+)?def[ \t]/ && d < 0) { d = ind; start = NR }
              last = NR }
            END { close_def(); print c + 0 }'; done | awk '{s += $1} END {print s + 0}')
    fi
    echo $((c + p))
}
# A snprintf is fine when its value flows somewhere: after =, (, &&, ||, a
# comparison, !, return or an arithmetic operator. Anything else, including
# a (void) cast, discards it.
r_snprintf() {
    cstripb | grep 'snprintf[[:space:]]*(' \
        | grep -cvE '(=|\(|&&|\|\||<|>|!|\+|-|\*|\?|:|,|return)[[:space:]]*snprintf'
}
r_dash() {
    local files
    files=$(ls -d src fe compile.py tests tools docs specs README.md CONTRIBUTING.md CHANGELOG.md 2>/dev/null)
    [ -n "$files" ] || { echo 0; return; }
    # shellcheck disable=SC2086
    grep -rIlP '\xe2\x80[\x93\x94]' $files 2>/dev/null | wc -l
}
r_ascii() {
    local files
    files=$(ls -d src fe compile.py tests 2>/dev/null)
    [ -n "$files" ] || { echo 0; return; }
    # shellcheck disable=SC2086
    grep -rIcP '[^\x00-\x7F]' $files 2>/dev/null | awk -F: '{s += $NF} END {print s + 0}'
}

# Open-set names: the union of rules.names and, when the hackathon checkout
# is present, every problem id, kernel entry and distinctive buffer name in
# $HK/problems/*/meta.json. "Distinctive" = has an underscore or digit, so
# generic words (x, add, softmax) never become patterns. Word-bounded, so
# k_matmul does not fire on k_matmul_tile.
openset_names() {
    {
        grep -vE '^[[:space:]]*(#|$)' "$NAMES" 2>/dev/null
        local m
        for m in "$HK"/problems/*/meta.json; do
            [ -f "$m" ] || continue
            grep -oE '"(id|entry|name)": *"[A-Za-z0-9_]+"' "$m" | sed -E 's/.*: *"(.*)"/\1/' \
                | grep -E '[_0-9]'
        done
    } | sort -u
}
r_openset() {
    local pat
    pat=$(openset_names | paste -sd'|')
    [ -n "$pat" ] || { echo 0; return; }
    codestripped | grep -oE "\\b($pat)\\b" | wc -l
}
# stdout: printf-family, puts, putchar, stdout, fd 1. fprintf(stderr, ...) is fine.
r_stdout() {
    local files f
    files=$(ls src/*.c src/*.h 2>/dev/null | grep -vE '^src/(emit|main)\.c$')
    [ -n "$files" ] || { echo 0; return; }
    for f in $files; do BLANK=1 cstrip "$f"; done \
        | grep -oE '\b(printf|vprintf|puts|putchar)[[:space:]]*\(|\bstdout\b|\bSTDOUT_FILENO\b|\bwrite[[:space:]]*\([[:space:]]*1[[:space:]]*,' \
        | wc -l
}
# Distinct E-codes used in src/ fe/ compile.py (comments stripped, err.c itself
# excluded) that never appear in src/err.c. With no err.c table yet, nothing
# is used, so the count is 0.
r_ecode() {
    local used table
    used=$( { [ -z "$CSRC" ] || for f in $CSRC; do [ "$f" = src/err.c ] || cstrip "$f"; done
              pystripped; } | grep -oE '\bE[0-9]{3}\b' | sort -u)
    table=$(cat src/err.c 2>/dev/null | grep -oE '\bE[0-9]{3}\b' | sort -u)
    [ -n "$used" ] || { echo 0; return; }
    comm -23 <(echo "$used") <(echo "$table") | grep -c .
}
# Python: any process spawning outside compile.py. In compile.py, each
# subprocess call must name irptx or cc within its next 3 lines, and
# shell=True is out. C: no system/popen/exec in src.
r_subproc() {
    local n=0 py f
    for f in $(ls fe/*.py 2>/dev/null); do
        n=$((n + $(pystrip "$f" | grep -cE '\b(subprocess|Popen)\b|\bos\.(system|popen|exec[lv]p?e?|spawn[lv]p?e?)\b|\bpty\.spawn\b')))
    done
    if [ -f compile.py ]; then
        n=$((n + $(pystrip compile.py | grep -cE 'shell[[:space:]]*=[[:space:]]*True|\bos\.(system|popen)\b')))
        n=$((n + $(pystrip compile.py | awk '
            { l[NR] = $0 }
            END { for (i = 1; i <= NR; i++)
                if (l[i] ~ /subprocess\.(run|Popen|call|check_call|check_output)|\<Popen\(/) {
                    # the whole call: join lines until the parentheses balance
                    w = l[i]; j = i
                    while (j < NR && (gsub(/\(/, "(", w) > gsub(/\)/, ")", w))) { j++; w = w " " l[j] }
                    if (w !~ /irptx|["\047]cc["\047]|\<CC\>|\<cc\>/) c++
                } print c + 0 }')))
    fi
    [ -z "$CSRC" ] || n=$((n + $(cstripb | grep -cE '\b(system|popen|execv|execl|execvp|execlp|execve|fork)[[:space:]]*\(')))
    echo "$n"
}
# Only when bin/irptx is committed. Times come from git (commit time), or the
# file mtime for anything modified or untracked, so a fresh clone (all mtimes
# equal to checkout time) is not stale by accident.
r_stale() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo 0; return; }
    git ls-files --error-unmatch bin/irptx >/dev/null 2>&1 || { echo 0; return; }
    local dirty f t bt n=0
    dirty=$(git status --porcelain -- bin src 2>/dev/null | cut -c4-)
    ts() {
        if printf '%s\n' "$dirty" | grep -qxF "$1"; then stat -c %Y "$1"
        else git log -1 --format=%ct -- "$1"; fi
    }
    bt=$(ts bin/irptx)
    for f in $(git ls-files src); do
        [ -f "$f" ] || continue
        t=$(ts "$f")
        [ "${t:-0}" -gt "${bt:-0}" ] && n=$((n + 1))
    done
    echo "$n"
}

# ---- Ratchet ----
echo "[rules] ratchet, from tests/rules.tbl"
bad=0
if [ ! -r "$TBL" ]; then
    echo "  ERROR  cannot read $TBL"
    echo "rules: failed"
    exit 1
fi
while read -r name limit label; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    if ! declare -F "r_$name" >/dev/null; then
        echo "  ERROR  $name: no such rule in rules.sh"; bad=1; continue
    fi
    now=$("r_$name" | tr -d ' ')
    case "$limit" in ''|*[!0-9]*) echo "  ERROR  $name: baseline '$limit' is not an integer"; bad=1; continue ;; esac
    case "$now" in ''|*[!0-9]*) echo "  ERROR  $name: rule printed '$now', not an integer"; bad=1; continue ;; esac
    if [ "$now" -gt "$limit" ]; then
        printf '  RISEN  %-8s %s: %s, baseline %s\n' "$name" "$label" "$now" "$limit"; bad=1
    elif [ "$now" -lt "$limit" ]; then
        printf '  fallen %-8s %s: %s, baseline %s, lower the baseline\n' "$name" "$label" "$now" "$limit"
    else
        printf '  held   %-8s %s: %s\n' "$name" "$label" "$now"
    fi
done < "$TBL"

if [ "$bad" -ne 0 ]; then
    echo "rules: failed"
    exit 1
fi
echo "rules: clean"
