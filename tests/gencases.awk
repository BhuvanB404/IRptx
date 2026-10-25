# gencases.awk -- turn the TH_REG lines of tests/t_*.c into build/tcases.h.
# Usage: awk -f tests/gencases.awk tests/t_*.c > build/tcases.h
# Exits 1 with a message when a line breaks the naming rules (tharns.h).
function die(msg) { printf "gencases: %s:%d: %s\n", FILENAME, FNR, msg > "/dev/stderr"; bad = 1; exit 1 }
/TH_REG\(/ && !/^TH_REG\(/ && !/^[ \t]*(\/\*|\*|#)/ { die("TH_REG is indented or not at column 0, so it would be dropped") }
/^TH_REG\(/ {
    line = $0
    if (!match(line, /^TH_REG\("[a-z]+", *[0-9]+, *".*", *[a-z]+[0-9]+\)$/)) die("TH_REG line is not on one line in the documented shape")
    s = line; sub(/^TH_REG\("/, "", s)
    fam = s; sub(/".*/, "", fam)
    s = substr(s, length(fam) + 2); sub(/^, */, "", s)
    num = s; sub(/,.*/, "", num); num += 0
    fn = line; sub(/\)$/, "", fn); sub(/.*, */, "", fn)
    base = FILENAME; sub(/.*\//, "", base)
    if (base != "t_" fam ".c") die("family " fam " does not belong in " base)
    if (fn != sprintf("%s%02d", fam, num)) die("function " fn " should be named " sprintf("%s%02d", fam, num))
    if (fn in seen) die("duplicate test " fn)
    seen[fn] = 1
    names[++n] = fn
}
END {
    if (bad) exit 1
    print "/* build/tcases.h -- generated from the TH_REG lines by tests/gencases.awk; do not edit */"
    for (i = 1; i <= n; i++) printf "void th_run_%s(const th_sel_t *s);\n", names[i]
    ng = 0
    for (i = 1; i <= n; i += 32) {
        printf "static void th_grp%d(const th_sel_t *s)\n{\n", ng
        for (j = i; j < i + 32 && j <= n; j++) printf "    th_run_%s(s);\n", names[j]
        print "}"
        ng++
    }
    print "static void th_all(const th_sel_t *s)\n{"
    for (g = 0; g < ng; g++) printf "    th_grp%d(s);\n", g
    print "    (void)s;\n}"
}
