/* err.c -- er_*: E-code table and er_fail.
 *
 * One list of every code the project can print, Python side included, so
 * there is one place to look a number up. tests/rules.sh reads this file and
 * fails when a code is used in src/, fe/ or compile.py without appearing
 * here, so a code goes in the table (keyed by its irptx.h macro), not in a comment.
 * Ranges are fixed in data-model.md ("E-codes"). */

#include <stdio.h>
#include <string.h>
#include "irptx.h"

/* ---- Table ---- */

typedef struct {
    int         code;               /* the E0NN macro from irptx.h */
    const char *meaning;            /* one line, for people reading the table */
} er_ent_t;

#define ER_WHAT_MAX 300u            /* longest `what` we print; the rest is cut */

static const er_ent_t ER_TAB[] = {
    /* E000: internal */
    { E000, "internal error, a bug in irptx rather than in the input" },
    /* E001-E009: front end (printed by compile.py, listed here for one table) */
    { E001, "IR file missing, unreadable or not valid JSON" },
    { E002, "meta.json missing or without the requested kernel entry" },
    { E003, "KIR dump not printable ASCII" },
    { E004, "core binary missing and the one-time cc build failed" },
    { E005, "core timed out (20 s) or could not be run" },
    { E009, "front end: other load failure (reserved)" },
    /* E010-E019: KIR reader */
    { E010, "KIR syntax error, truncated input, unknown field or unsupported format version" },
    { E011, "index out of range (node, item, field or string offset past its table)" },
    { E012, "kernel or buffer name is not a valid identifier" },
    { E013, "KIR string not printable ASCII, or over 63 characters" },
    { E014, "malformed number (integer or hex float)" },
    { E015, "required KIR field missing" },
    { E019, "KIR reader: other malformed input (reserved)" },
    /* E020-E059: isel */
    { E020, "unsupported node kind" },
    { E021, "unsupported op, or the body is not exactly one Tile" },
    { E022, "name not bound in any enclosing scope" },
    { E024, "dtype other than f32" },
    { E025, "launch uses y or z beyond 1" },
    { E026, "StridedLoop step that is not a constant above 0" },
    { E034, "warp shuffle whose length is not a power of two" },
    { E059, "isel: other unsupported construct (reserved)" },
    /* E060-E079: verify */
    { E060, "label undefined or defined twice" },
    { E061, "non-ASCII byte in the module" },
    { E062, "barrier or shuffle in a loop whose trip count may differ between threads of a block" },
    { E079, "verify: other failed check (reserved)" },
    /* E080-E099: limits */
    { E080, "input larger than IR_MAX_KIR" },
    { E081, "a fixed table is full (nodes, fields, args, registers)" },
    { E082, "nesting deeper than IR_MAX_DEP" },
    { E083, "more than IR_MAX_INS instructions" },
    { E084, "output larger than IR_MAX_OUT" },
    { E099, "limits: other (reserved)" }
};

#define ER_NTAB (sizeof ER_TAB / sizeof ER_TAB[0])

/* ---- Lookup ---- */

/* The table entry for `code`, or NULL when it is not listed. */
static const er_ent_t *er_find(int code)
{
    size_t i;

    for (i = 0; i < ER_NTAB; i++)
        if (ER_TAB[i].code == code) return &ER_TAB[i];
    return NULL;
}

/* The one-line meaning of a code, or NULL if unlisted. */
const char *er_text(int code)
{
    const er_ent_t *e = er_find(code);

    return e ? e->meaning : NULL;
}

/* ---- Failing ---- */

/* Print `error[E0NN]: irptx: cannot lower <what>` to stderr and return 1, so
 * a stage can `return er_fail(...)`. Nothing goes to stdout. `what` is NULL
 * safe, cut at ER_WHAT_MAX, and any byte outside printable ASCII becomes '?'
 * (PTX and the terminal both dislike the rest). A code that is not in the
 * table is our bug: it prints as E000 and says which code it was, rather
 * than inventing a number or staying silent. */
int er_fail(int code, const char *what)
{
    char buf[ER_WHAT_MAX + 1];
    size_t n = 0;
    const char *p = what ? what : "something";
    const int known = er_find(code) != NULL;

    while (*p && n < ER_WHAT_MAX) {
        unsigned char c = (unsigned char)*p++;
        buf[n++] = (c >= 0x20 && c < 0x7f) ? (char)c : '?';
    }
    buf[n] = '\0';
    if (known)
        fprintf(stderr, "error[E%03d]: irptx: cannot lower %s\n", code, buf);
    else
        fprintf(stderr, "error[E000]: irptx: cannot lower %s (unlisted error code %d)\n",
                buf, code);
    return 1;
}
