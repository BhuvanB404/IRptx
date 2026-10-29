/* kir.c -- kr_*: KIR text reader into fixed pools, explicit-stack parser.
 *
 * The text is a handful of header lines and one S-expression. The S-expression is read by one loop with an explicit stack
 * of open `(` and `[` frames: no recursion, depth capped at IR_MAX_DEP.
 *
 * A node's fields (and a list's items) must end up contiguous in their pool,
 * but a child opened halfway through its parent would interleave with it.
 * So the half-built children live on a scratch stack in the unused top end
 * of the same pool, growing downward, and are slid into the bottom end when
 * their `)` or `]` arrives. Nothing is allocated; the two ends meeting is
 * the pool being full (E081). */

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "irptx.h"
#include "kir.h"

/* ---- State ---- */

/* The one kernel. Big (about 16 MB of pools), so it is static. */
static struct { kir_t kir; } S;

/* A cursor over text that is not NUL terminated. */
typedef struct { const char *t; size_t n, p; } kr_cur_t;

/* One open `(` or `[`. */
typedef struct {
    uint8_t  node;                  /* 1 for `(`, 0 for `[` */
    uint32_t idx;                   /* node index (node frames) */
    uint32_t cnt;                   /* children pushed so far */
    uint32_t pname;                 /* name of the field being read (node frames) */
} kr_frm_t;

typedef struct {
    kr_cur_t   c;
    kr_frm_t   f[IR_MAX_DEP];
    uint32_t   sp;                  /* open frames */
    uint32_t   itop, ftop;          /* scratch stack tops, start at pool size */
    kir_val_t  root;
    int        done;
} kr_ps_t;

/* kr_slide copies elements through a small buffer; keep the types inside it. */
typedef char kr_sz_chk[(sizeof(kir_fld_t) <= 64 && sizeof(kir_val_t) <= 64) ? 1 : -1];

/* ---- Kinds ---- */

typedef struct { const char *name; uint16_t kind; } kr_kn_t;

/* Emmy class name to kind. The plain data nodes are here too. Anything not
 * listed (SelectBranch, say) is KR_OTHER and isel refuses it by name. */
static const kr_kn_t KR_KN[] = {
    { "Tile", KR_TILE }, { "Load", KR_LOAD }, { "Assign", KR_ASSIGN },
    { "Write", KR_WRITE }, { "Accum", KR_ACCUM }, { "Cond", KR_COND },
    { "StridedLoop", KR_STRIDEDLOOP }, { "Smem", KR_SMEM }, { "Sync", KR_SYNC },
    { "WarpShuffle", KR_WARPSHUFFLE }, { "TreeHalve", KR_TREEHALVE },
    { "Let", KR_LET }, { "Loop", KR_LOOP }, { "Init", KR_INIT },
    { "CpAsyncCopy", KR_CPASYNCCOPY }, { "CpAsyncCommit", KR_CPASYNCCOMMIT },
    { "CpAsyncWait", KR_CPASYNCWAIT },
    { "Pre", KR_PRE }, { "Carry", KR_CARRY }, { "Select", KR_SELECT },
    { "ZeroPrologue", KR_ZEROPROLOGUE }, { "TmaDescriptor", KR_TMADESCRIPTOR },
    { "TmaLoad", KR_TMALOAD }, { "MbarrierInit", KR_MBARRIERINIT },
    { "MbarrierArriveExpectTx", KR_MBARRIERARRIVEEXPECTTX },
    { "MbarrierArrive", KR_MBARRIERARRIVE }, { "MbarrierWait", KR_MBARRIERWAIT },
    { "SetMaxNReg", KR_SETMAXNREG }, { "FragmentApply", KR_FRAGMENTAPPLY },
    { "FragmentRowReduce", KR_FRAGMENTROWREDUCE }, { "RegFragment", KR_REGFRAGMENT },
    { "BlockScaleLoad", KR_BLOCKSCALELOAD }, { "LdmatrixLoad", KR_LDMATRIXLOAD },
    { "MmaSyncPtx", KR_MMASYNCPTX }, { "FragmentPromote", KR_FRAGMENTPROMOTE },
    { "FragmentRepack", KR_FRAGMENTREPACK }, { "WgmmaDescriptor", KR_WGMMADESCRIPTOR },
    { "WgmmaMma", KR_WGMMAMMA }, { "WgmmaFence", KR_WGMMAFENCE },
    { "WgmmaCommit", KR_WGMMACOMMIT }, { "WgmmaWait", KR_WGMMAWAIT },
    { "RegStore", KR_REGSTORE },
    { "Var", KR_VAR }, { "Literal", KR_LITERAL }, { "BinaryExpr", KR_BINARYEXPR },
    { "TernaryExpr", KR_TERNARYEXPR }, { "Builtin", KR_BUILTIN },
    { "FuncCallExpr", KR_FUNCCALLEXPR }, { "CastExpr", KR_CASTEXPR },
    { "FlatIndex", KR_FLATINDEX },
    { "Axis", KR_AXIS }, { "Dim", KR_DIM }, { "Window", KR_WINDOW },
    { "FragLayout", KR_FRAGLAYOUT }, { "KernelOp", KR_KERNELOP }
};

#define KR_NKN (sizeof KR_KN / sizeof KR_KN[0])

/* ---- Errors ---- */

/* One refusal with the byte offset, so a bad dump can be found in 1 MiB of
 * text. Returns 1 so callers write `return kr_bad(...)`. */
static int kr_bad(int code, const char *msg, size_t at)
{
    char b[200];
    int n = snprintf(b, sizeof b, "%s (at byte %lu)", msg, (unsigned long)at);

    return er_fail(code, (n < 0 || (size_t)n >= sizeof b) ? msg : b);
}

/* ---- Characters ---- */

static int kr_isws(int c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

/* A character that ends a bare token or a name. */
static int kr_isdl(int c)
{
    return kr_isws(c) || c == '(' || c == ')' || c == '[' || c == ']' || c == '"';
}

static int kr_isid0(int c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }

static int kr_isid(int c) { return kr_isid0(c) || (c >= '0' && c <= '9'); }

/* Skip blanks; returns 0 at the end of the text. */
static int kr_skip(kr_cur_t *c)
{
    while (c->p < c->n && kr_isws((unsigned char)c->t[c->p])) c->p++;
    return c->p < c->n;
}

/* ---- Strings ---- */

/* Copy `n` bytes into the string pool with a NUL and give back the offset.
 * Offset 0 is the empty string (nothing is stored there). 63 characters is
 * the longest string the contract allows (E013). */
static int kr_put(const char *s, size_t n, uint32_t *off)
{
    kir_t *K = &S.kir;

    if (n > IR_MAX_NAME) return er_fail(E013, "a KIR string over 63 characters");
    if ((size_t)K->nstr + n + 1u >= IR_MAX_STR)
        return er_fail(E081, "more KIR string bytes than the string pool holds");
    memcpy(K->str + K->nstr, s, n);
    K->str[K->nstr + n] = '\0';
    *off = K->nstr;
    K->nstr += (uint32_t)n + 1u;
    return 0;
}

/* ---- Numbers ---- */

/* Decimal integer with an optional leading minus, range checked against
 * int64 (E014). Python never writes a plus sign or a decimal point here. */
static int kr_int(const char *s, size_t n, int64_t *out)
{
    uint64_t v = 0, lim;
    size_t i = 0;
    int neg = (n > 0 && s[0] == '-');

    if (neg) i = 1;
    if (i >= n || n - i > 19) return er_fail(E014, "an integer that is empty or too long");
    lim = neg ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
    for (; i < n; i++) {
        unsigned d;
        if (s[i] < '0' || s[i] > '9') return er_fail(E014, "an integer with a non-digit in it");
        d = (unsigned)(s[i] - '0');
        if (v > (lim - d) / 10u) return er_fail(E014, "an integer outside the int64 range");
        v = v * 10u + d;
    }
    *out = neg ? (int64_t)(0u - v) : (int64_t)v;
    return 0;
}

/* Shape check for [-]0x<hex>[.<hex>]p[+-]<dec>f, so strtod is only handed
 * what float.hex() writes. strtod alone would also take "inf", "nan" and
 * decimal forms, and would stop quietly at trailing garbage. */
static int kr_hexok(const char *s, size_t n)
{
    size_t i = (n > 0 && s[0] == '-') ? 1 : 0;
    size_t h;

    if (n < i + 5 || s[n - 1] != 'f' || s[i] != '0' || s[i + 1] != 'x') return 0;
    n--;
    for (i += 2, h = 0; i < n && isxdigit((unsigned char)s[i]); i++) h++;
    if (i < n && s[i] == '.')
        for (i++; i < n && isxdigit((unsigned char)s[i]); i++) h++;
    if (h == 0 || i >= n || s[i] != 'p') return 0;
    i++;
    if (i < n && (s[i] == '+' || s[i] == '-')) i++;
    if (i >= n) return 0;
    for (; i < n; i++)
        if (s[i] < '0' || s[i] > '9') return 0;
    return 1;
}

/* Hex float token to f32 bits. The double is narrowed to float once and its
 * bit pattern is all that survives, as the f32 kernel would see it. One
 * that overflows float is refused rather than turned into an infinity. */
static int kr_flt(const char *s, size_t n, uint32_t *bits)
{
    char buf[IR_MAX_NAME + 1];
    char *end = NULL;
    double d;
    float f;

    if (n > IR_MAX_NAME || !kr_hexok(s, n)) return er_fail(E014, "a malformed hex float");
    memcpy(buf, s, n - 1u);
    buf[n - 1u] = '\0';
    d = strtod(buf, &end);
    if (end != buf + (n - 1u)) return er_fail(E014, "a hex float strtod would not take whole");
    /* range first: narrowing a double outside the float range is undefined
     * (C99 6.3.1.5), so the check cannot be made on the result */
    if (d > FLT_MAX || d < -FLT_MAX) return er_fail(E014, "a hex float outside the f32 range");
    f = (float)d;
    memcpy(bits, &f, sizeof *bits);
    return 0;
}

/* ---- Atoms ---- */

/* A quoted string: printable ASCII, no backslash, at most 63 characters, and
 * ended by a delimiter. A quote inside the text ends it early and leaves
 * junk behind it, which is caught as E013 too. */
static int kr_strv(kr_cur_t *c, kir_val_t *v)
{
    size_t b = ++c->p;
    uint32_t off;
    int rc;

    while (c->p < c->n && c->t[c->p] != '"') {
        unsigned char ch = (unsigned char)c->t[c->p];
        if (ch < 0x20 || ch >= 0x7f || ch == '\\')
            return kr_bad(E013, "a KIR string with an unprintable or backslash character", c->p);
        c->p++;
    }
    if (c->p >= c->n) return kr_bad(E010, "truncated KIR string", c->p);
    rc = kr_put(c->t + b, c->p - b, &off);
    if (rc) return rc;
    c->p++;
    if (c->p < c->n && !kr_isws((unsigned char)c->t[c->p]) &&
        c->t[c->p] != ')' && c->t[c->p] != ']')
        return kr_bad(E013, "a KIR string followed by junk (a quote inside it?)", c->p);
    v->tag = KV_STR;
    v->ref = off;
    return 0;
}

/* A bare token that names a number: decimal int or hex float. */
static int kr_num(const char *s, size_t n, kir_val_t *v)
{
    size_t i = (n > 0 && s[0] == '-') ? 1 : 0;

    if (n > i + 1 && s[i] == '0' && s[i + 1] == 'x') {
        v->tag = KV_FLT;
        return kr_flt(s, n, &v->fbits);
    }
    v->tag = KV_INT;
    return kr_int(s, n, &v->i);
}

/* A bare token: true, false, none, op:<name>, dt:<name> or a number. */
static int kr_atom(kr_cur_t *c, kir_val_t *v)
{
    size_t b = c->p, n;
    const char *s;

    while (c->p < c->n && !kr_isdl((unsigned char)c->t[c->p])) {
        unsigned char ch = (unsigned char)c->t[c->p];
        if (ch < 0x21 || ch >= 0x7f) return kr_bad(E010, "a non-ASCII byte in a KIR token", c->p);
        c->p++;
    }
    n = c->p - b;
    s = c->t + b;
    if (n == 0) return kr_bad(E010, "a bracket or quote where a value should be", b);
    if (n == 4 && !memcmp(s, "none", 4)) { v->tag = KV_NONE; return 0; }
    if (n == 4 && !memcmp(s, "true", 4)) { v->tag = KV_BOOL; v->i = 1; return 0; }
    if (n == 5 && !memcmp(s, "false", 5)) { v->tag = KV_BOOL; v->i = 0; return 0; }
    if (n > 3 && (!memcmp(s, "op:", 3) || !memcmp(s, "dt:", 3))) {
        v->tag = (s[0] == 'o') ? KV_OP : KV_DT;
        return kr_put(s + 3, n - 3u, &v->ref);
    }
    if ((s[0] >= '0' && s[0] <= '9') || s[0] == '-') return kr_num(s, n, v);
    return kr_bad(E010, "an unknown bare word in the KIR body", b);
}

/* ---- Pools ---- */

/* Slide the `m` newest scratch elements of a pool down to its bottom end and
 * return their first index (0 when m is 0). The scratch stack grows
 * downward, so they sit newest first: reverse them in place, then memmove
 * (the two ranges can overlap only when the pool is nearly full, and then
 * memmove is what keeps it right). `cnt` is the last used bottom index and
 * `top` the lowest used scratch index. */
static uint32_t kr_slide(void *pool, size_t sz, uint32_t *cnt, uint32_t *top, uint32_t m)
{
    char *b = (char *)pool;
    char tmp[64];
    uint32_t i, start = *cnt + 1u;

    for (i = 0; i < m / 2u; i++) {
        char *lo = b + (size_t)(*top + i) * sz;
        char *hi = b + (size_t)(*top + m - 1u - i) * sz;
        memcpy(tmp, lo, sz);
        memcpy(lo, hi, sz);
        memcpy(hi, tmp, sz);
    }
    if (m) memmove(b + (size_t)start * sz, b + (size_t)*top * sz, (size_t)m * sz);
    *cnt += m;
    *top += m;
    return m ? start : 0u;
}

/* Hand a finished value to the open frame, or make it the root. The scratch
 * slot must stay above the last final element, or the pool is full. */
static int kr_give(kr_ps_t *ps, const kir_val_t *v)
{
    kir_t *K = &S.kir;
    kr_frm_t *fr;

    if (ps->sp == 0) { ps->root = *v; ps->done = 1; return 0; }
    fr = &ps->f[ps->sp - 1u];
    if (fr->node) {
        if (ps->ftop < K->nfld + 2u) return er_fail(E081, "more node fields than the field pool holds");
        ps->ftop--;
        K->fld[ps->ftop].name = fr->pname;
        K->fld[ps->ftop].val = *v;
    } else {
        if (ps->itop < K->nitem + 2u) return er_fail(E081, "more list items than the item pool holds");
        ps->itop--;
        K->item[ps->itop] = *v;
    }
    fr->cnt++;
    return 0;
}

/* ---- Brackets ---- */

static uint16_t kr_kind(const char *s, size_t n)
{
    size_t i;

    for (i = 0; i < KR_NKN; i++)
        if (strlen(KR_KN[i].name) == n && !memcmp(KR_KN[i].name, s, n)) return KR_KN[i].kind;
    return KR_OTHER;
}

/* After the `(`: the kind name, a new node, a frame. */
static int kr_node(kr_ps_t *ps)
{
    kir_t *K = &S.kir;
    kr_cur_t *c = &ps->c;
    size_t b = c->p;
    uint32_t ni, off;
    int rc;

    if (c->p >= c->n || !kr_isid0((unsigned char)c->t[c->p]))
        return kr_bad(E010, "a '(' without a node kind after it", b);
    while (c->p < c->n && kr_isid((unsigned char)c->t[c->p])) c->p++;
    if (c->p < c->n && !kr_isws((unsigned char)c->t[c->p]) && c->t[c->p] != ')')
        return kr_bad(E010, "junk after a node kind", c->p);
    ni = IX_PNEW(K->nnode, IR_MAX_NODE - 1u);
    if (ni == 0) return er_fail(E081, "more KIR nodes than the node pool holds");
    rc = kr_put(c->t + b, c->p - b, &off);
    if (rc) return rc;
    K->node[ni].kind = kr_kind(c->t + b, c->p - b);
    K->node[ni].kname = off;
    K->node[ni].fld0 = 0;
    K->node[ni].nfld = 0;
    ps->f[ps->sp].idx = ni;
    return 0;
}

/* Open a `(` or `[`. Depth past IR_MAX_DEP refuses (E082). */
static int kr_open(kr_ps_t *ps, int node)
{
    kr_frm_t *fr;

    if (ps->sp >= IR_MAX_DEP) return er_fail(E082, "KIR nested deeper than 64 levels");
    fr = &ps->f[ps->sp];
    fr->node = (uint8_t)node;
    fr->idx = 0;
    fr->cnt = 0;
    fr->pname = 0;
    ps->c.p++;
    if (node) {
        int rc = kr_node(ps);
        if (rc) return rc;
    }
    ps->sp++;
    return 0;
}

/* A `)` or `]` arrived: finish the top frame and give its value upward. */
static int kr_shut(kr_ps_t *ps)
{
    kir_t *K = &S.kir;
    kr_frm_t *fr = &ps->f[ps->sp - 1u];
    kir_val_t v;

    memset(&v, 0, sizeof v);
    if (fr->node) {
        kir_node_t *nd = &K->node[fr->idx];
        nd->fld0 = kr_slide(K->fld, sizeof K->fld[0], &K->nfld, &ps->ftop, fr->cnt);
        nd->nfld = fr->cnt;
        v.tag = KV_NODE;
        v.ref = fr->idx;
    } else {
        v.ref = kr_slide(K->item, sizeof K->item[0], &K->nitem, &ps->itop, fr->cnt);
        v.n = fr->cnt;
        v.tag = KV_LIST;
    }
    ps->sp--;
    return kr_give(ps, &v);
}

/* Inside a node: read `name=` and leave the cursor on the value. Anything
 * else in field position is a malformed field token (E010). */
static int kr_fname(kr_ps_t *ps, kr_frm_t *fr)
{
    kr_cur_t *c = &ps->c;
    size_t b = c->p;

    if (!kr_isid0((unsigned char)c->t[c->p])) return kr_bad(E010, "a malformed field token (no name)", b);
    while (c->p < c->n && kr_isid((unsigned char)c->t[c->p])) c->p++;
    if (c->p >= c->n) return kr_bad(E010, "truncated KIR body", c->p);
    if (c->t[c->p] != '=') return kr_bad(E010, "a malformed field token (no '=')", c->p);
    if (kr_put(c->t + b, c->p - b, &fr->pname)) return 1;
    c->p++;
    if (c->p >= c->n) return kr_bad(E010, "truncated KIR body", c->p);
    if (kr_isws((unsigned char)c->t[c->p]) || c->t[c->p] == ')' || c->t[c->p] == ']')
        return kr_bad(E010, "a field with no value after '='", c->p);
    return 0;
}

/* ---- Body ---- */

/* One token's worth of progress: a closer, or a (field name and) value. */
static int kr_step(kr_ps_t *ps)
{
    kr_cur_t *c = &ps->c;
    kr_frm_t *fr = ps->sp ? &ps->f[ps->sp - 1u] : NULL;
    kir_val_t v;
    int ch, rc;

    if (!kr_skip(c)) return kr_bad(E010, "truncated KIR body", c->p);
    ch = (unsigned char)c->t[c->p];
    if (fr && (ch == ')' || ch == ']')) {
        if ((ch == ')') != (fr->node != 0)) return kr_bad(E010, "mismatched brackets", c->p);
        c->p++;
        return kr_shut(ps);
    }
    if (fr && fr->node && kr_fname(ps, fr)) return 1;
    ch = (unsigned char)c->t[c->p];
    if (ch == '(' || ch == '[') return kr_open(ps, ch == '(');
    memset(&v, 0, sizeof v);
    rc = (ch == '"') ? kr_strv(c, &v) : kr_atom(c, &v);
    return rc ? rc : kr_give(ps, &v);
}

/* The S-expression, then `end`, then nothing. Every step eats at least one
 * byte, so the guard is the text length. The root is the node itself or the
 * only node of a one-item list; anything else leaves root 0 and isel refuses
 * it (E021), because "exactly one Tile" is a lowering rule. */
static int kr_body(kr_cur_t *c)
{
    kir_t *K = &S.kir;
    kr_ps_t ps;
    IX_GUARD(g, c->n - c->p + 2u);

    memset(&ps, 0, sizeof ps);
    ps.c = *c;
    ps.itop = IR_MAX_ITEM;
    ps.ftop = IR_MAX_FLD;
    while (!ps.done && g--)
        if (kr_step(&ps)) return 1;
    *c = ps.c;
    if (!ps.done) return kr_bad(E010, "KIR body never closed", c->p);
    if (ps.root.tag == KV_NODE) K->root = ps.root.ref;
    else if (ps.root.tag == KV_LIST && ps.root.n == 1u && K->item[ps.root.ref].tag == KV_NODE)
        K->root = K->item[ps.root.ref].ref;
    if (!kr_skip(c) || c->n - c->p < 3u || memcmp(c->t + c->p, "end", 3))
        return kr_bad(E010, "KIR body not followed by 'end'", c->p);
    c->p += 3u;
    if (kr_skip(c)) return kr_bad(E010, "text after the KIR 'end' line", c->p);
    return 0;
}

/* ---- Header ---- */

/* Cut the next line out of `c` into `ln` (no newline). Returns 0 at the end. */
static int kr_ln(kr_cur_t *c, kr_cur_t *ln)
{
    size_t b = c->p;

    if (c->p >= c->n) return 0;
    while (c->p < c->n && c->t[c->p] != '\n') c->p++;
    ln->t = c->t;
    ln->p = b;
    ln->n = c->p;
    if (c->p < c->n) c->p++;
    return 1;
}

/* Next blank-separated token on the line; 0 at its end. */
static int kr_tk(kr_cur_t *ln, const char **s, size_t *n)
{
    while (ln->p < ln->n && kr_isws((unsigned char)ln->t[ln->p])) ln->p++;
    if (ln->p >= ln->n) return 0;
    *s = ln->t + ln->p;
    while (ln->p < ln->n && !kr_isws((unsigned char)ln->t[ln->p])) ln->p++;
    *n = (size_t)(ln->t + ln->p - *s);
    return 1;
}

/* Next token as an integer in [lo, hi] (E010 if the line ran out, E014 if the
 * number is bad or outside the range). */
static int kr_hint(kr_cur_t *ln, int64_t lo, int64_t hi, int32_t *out)
{
    const char *s;
    size_t n;
    int64_t v;

    if (!kr_tk(ln, &s, &n)) return kr_bad(E010, "a header line with a missing number", ln->p);
    if (kr_int(s, n, &v)) return 1;
    if (v < lo || v > hi) return kr_bad(E014, "a header number outside its range", ln->p);
    *out = (int32_t)v;
    return 0;
}

/* Next token as a name: [A-Za-z_][A-Za-z0-9_]*, at most 63 characters (E012).
 * Buffer and kernel names become comments only, never PTX identifiers, but
 * the contract still fixes their shape. Stored in the string pool. */
static int kr_hname(kr_cur_t *ln, uint32_t *off, char *copy)
{
    const char *s;
    size_t n, i;

    if (!kr_tk(ln, &s, &n)) return kr_bad(E010, "a header line with a missing name", ln->p);
    if (n > IR_MAX_NAME || !kr_isid0((unsigned char)s[0]))
        return kr_bad(E012, "a name that is not a valid identifier", ln->p);
    for (i = 1; i < n; i++)
        if (!kr_isid((unsigned char)s[i])) return kr_bad(E012, "a name that is not a valid identifier", ln->p);
    if (copy) { memcpy(copy, s, n); copy[n] = '\0'; }
    return kr_put(s, n, off);
}

/* No tokens may follow what a header line defines. */
static int kr_hend(kr_cur_t *ln)
{
    const char *s;
    size_t n;

    return kr_tk(ln, &s, &n) ? kr_bad(E010, "extra text at the end of a header line", ln->p) : 0;
}

/* kernel <name>: kept in K->name for the PTX header, and in the pool. */
static int kr_hkern(kr_cur_t *ln)
{
    uint32_t off;

    if (kr_hname(ln, &off, S.kir.name)) return 1;
    return kr_hend(ln);
}

static int kr_hlaunch(kr_cur_t *ln)
{
    kir_t *K = &S.kir;
    int i;

    for (i = 0; i < 3; i++)
        if (kr_hint(ln, 1, INT32_MAX, &K->grid[i])) return 1;
    for (i = 0; i < 3; i++)
        if (kr_hint(ln, 1, INT32_MAX, &K->block[i])) return 1;
    if (kr_hint(ln, 0, INT32_MAX, &K->smem)) return 1;
    return kr_hend(ln);
}

/* arg <i> <name> <in|const|out> <dtype> <rank> <dim0> ... */
static int kr_harg(kr_cur_t *ln)
{
    kir_t *K = &S.kir;
    kir_arg_t *a;
    const char *s;
    size_t n;
    int32_t idx, rank, d;

    if (K->narg >= IR_MAX_ARG) return er_fail(E081, "more kernel arguments than 32");
    a = &K->arg[K->narg];
    memset(a, 0, sizeof *a);
    if (kr_hint(ln, 0, INT32_MAX, &idx)) return 1;
    if ((uint32_t)idx != K->narg) return kr_bad(E010, "an arg line out of order", ln->p);
    if (kr_hname(ln, &a->name, NULL)) return 1;
    if (!kr_tk(ln, &s, &n)) return kr_bad(E010, "an arg line without a kind", ln->p);
    if (n == 2 && !memcmp(s, "in", 2)) a->kind = KA_IN;
    else if (n == 3 && !memcmp(s, "out", 3)) a->kind = KA_OUT;
    else if (n == 5 && !memcmp(s, "const", 5)) a->kind = KA_CONST;
    else return kr_bad(E010, "an arg kind other than in, const or out", ln->p);
    if (!kr_tk(ln, &s, &n)) return kr_bad(E010, "an arg line without a dtype", ln->p);
    a->dt = (n == 3 && !memcmp(s, "f32", 3)) ? KD_F32 : KD_OTHER;
    if (kr_hint(ln, 0, INT32_MAX, &rank)) return 1;
    if ((uint32_t)rank > IR_MAX_DIM) return er_fail(E081, "a buffer of rank over 8");
    a->rank = (uint8_t)rank;
    for (d = 0; d < rank; d++)
        if (kr_hint(ln, 1, INT32_MAX, &a->dim[d])) return 1;
    K->narg++;
    return kr_hend(ln);
}

static int kr_hzero(kr_cur_t *ln)
{
    kir_t *K = &S.kir;

    if (K->nzero >= IR_MAX_ARG) return er_fail(E081, "more zero lines than 32");
    if (kr_hname(ln, &K->zero[K->nzero], NULL)) return 1;
    K->nzero++;
    return kr_hend(ln);
}

/* One header line, keyed by its first word. `have` collects kernel (1) and
 * launch (2); body is only legal once both are in (E015). Returns 0 to go on,
 * 2 when `body` was read, 1 on error. */
static int kr_hline(kr_cur_t *ln, unsigned *have)
{
    const char *s;
    size_t n;

    if (!kr_tk(ln, &s, &n)) return 0;
    if (n == 6 && !memcmp(s, "kernel", 6) && !(*have & 1u)) {
        *have |= 1u;
        return kr_hkern(ln);
    }
    if (n == 6 && !memcmp(s, "launch", 6) && !(*have & 2u)) { *have |= 2u; return kr_hlaunch(ln); }
    if (n == 3 && !memcmp(s, "arg", 3)) return kr_harg(ln);
    if (n == 4 && !memcmp(s, "zero", 4)) return kr_hzero(ln);
    if (n == 4 && !memcmp(s, "body", 4)) {
        if (*have != 3u) return kr_bad(E015, "a KIR body before its kernel or launch line", ln->p);
        return kr_hend(ln) ? 1 : 2;
    }
    return kr_bad(E010, "an unknown or repeated KIR header line", ln->p);
}

/* ---- Loading ---- */

/* The format line: exactly `kir 1`. */
static int kr_ver(kr_cur_t *c)
{
    kr_cur_t ln;
    const char *s;
    size_t n;

    if (!kr_ln(c, &ln) || !kr_tk(&ln, &s, &n) || n != 3 || memcmp(s, "kir", 3))
        return kr_bad(E010, "KIR text without a 'kir 1' first line", 0);
    if (!kr_tk(&ln, &s, &n) || n != 1 || s[0] != '1')
        return kr_bad(E010, "a KIR format version other than 1", ln.p);
    return kr_hend(&ln);
}

int kr_load(const char *txt, size_t len)
{
    kir_t *K = &S.kir;
    kr_cur_t c, ln;
    unsigned have = 0;
    int rc = 0;
    IX_GUARD(g, len + 2u);

    if (!txt) return er_fail(E010, "no KIR text");
    if (len > IR_MAX_KIR) return er_fail(E080, "KIR text over 1 MiB");
    memset(K->name, 0, sizeof K->name);
    memset(K->grid, 0, sizeof K->grid);
    memset(K->block, 0, sizeof K->block);
    K->smem = 0;
    K->narg = K->nzero = K->root = 0;
    K->nnode = K->nfld = K->nitem = 0;
    K->nstr = 1u;                   /* offset 0 is the empty string */
    K->str[0] = '\0';
    memset(&K->node[0], 0, sizeof K->node[0]);
    memset(&K->fld[0], 0, sizeof K->fld[0]);
    memset(&K->item[0], 0, sizeof K->item[0]);
    c.t = txt; c.n = len; c.p = 0;
    if (kr_ver(&c)) return 1;
    while (rc == 0 && g--) {
        if (!kr_ln(&c, &ln)) return kr_bad(E010, "KIR text ends before its body", c.p);
        rc = kr_hline(&ln, &have);
    }
    if (rc != 2) return rc ? 1 : kr_bad(E010, "KIR header never reaches its body", c.p);
    return kr_body(&c);
}

/* ---- Lookup ---- */

const kir_t *kr_get(void)
{
    return &S.kir;
}

const char *kr_str(uint32_t off)
{
    return IX_CHK(off, S.kir.nstr) ? "" : S.kir.str + off;
}

const kir_node_t *kr_nod(const kir_val_t *v)
{
    if (!v || v->tag != KV_NODE || v->ref == 0 || v->ref > S.kir.nnode) return NULL;
    return &S.kir.node[v->ref];
}

const kir_val_t *kr_fld(const kir_node_t *nd, const char *name)
{
    const kir_t *K = &S.kir;
    uint32_t i;

    if (!nd || !name || nd->nfld == 0) return NULL;
    if (nd->fld0 == 0 || IX_CHK(nd->fld0 + nd->nfld - 1u, K->nfld + 1u)) return NULL;
    for (i = 0; i < nd->nfld; i++) {
        const kir_fld_t *f = &K->fld[nd->fld0 + i];
        if (!IX_CHK(f->name, K->nstr) && strcmp(K->str + f->name, name) == 0) return &f->val;
    }
    return NULL;
}

const kir_val_t *kr_kid(const kir_val_t *lst, uint32_t n)
{
    const kir_t *K = &S.kir;

    if (!lst || lst->tag != KV_LIST || n >= lst->n) return NULL;
    if (lst->ref == 0 || IX_CHK(lst->ref + n, K->nitem + 1u)) return NULL;
    return &K->item[lst->ref + n];
}
