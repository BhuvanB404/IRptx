/* t_kx.c -- expression folding, s32 ops, affine forms (kx01..kx13) and the
 * addressing arithmetic built on them: row-major flattening and the thread
 * decode (kx14..). Expressions are loaded through the real KIR reader as the
 * single index of a Load, so the tests see exactly what the selector sees.
 *
 * Device code is checked by running the emitted records in ti_run, a tiny
 * interpreter for the s32 integer ops the index math uses (mov tid/ctaid,
 * add, mul.lo, mad.lo, div, rem, shr, and, setp.ge and the guard branch). */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tharns.h"
#include "irptx.h"
#include "isel.h"
#include "kir.h"
#include "kexpr.h"

#define LIT(n) "(Literal value=" #n " dtype=\"int\")"
#define VAR(n) "(Var name=\"" n "\")"
#define BIN(o, l, r) "(BinaryExpr op=\"" o "\" left=" l " right=" r ")"
#define DIM(n) "(Dim expr=" LIT(n) " hint=none)"
#define LT3 BIN("<", VAR("a0"), LIT(3))
#define FLT(h) "(Literal value=" h " dtype=\"float\")"

static char kb[65536];              /* generated KIR text */
static char eb[512];                /* captured stderr */

/* The E-code of the single error line a failing call printed (rc must be 1),
 * 0 for rc 0 with a silent stderr, -1 for anything else. */
static int kx_code(int rc, size_t len)
{
    int i, nl = 0;

    if (rc == 0) return len == 0 ? 0 : -1;
    for (i = 0; eb[i]; i++) nl += eb[i] == '\n';
    if (rc != 1 || nl != 1 || strncmp(eb, "error[E", 7) != 0 || eb[10] != ']') return -1;
    return (eb[7] - '0') * 100 + (eb[8] - '0') * 10 + (eb[9] - '0');
}

/* Load a kernel with one rank-1 input and one Load whose index is `e`;
 * returns that index expression. The selection state is reset. */
static const kir_val_t *kx_load(const char *e)
{
    const kir_t *k;
    const kir_val_t *b;

    snprintf(kb, sizeof kb, "kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0\narg 0 x0 in f32 1 8\nbody\n"
        "[(Tile axes=[] body=[(Load names=[\"x\"] input=\"x0\" index=[%s] dtype=dt:f32)] "
        "block_threads=1 aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n", e);
    if (kr_load(kb, strlen(kb)) != 0) return NULL;
    is_reset();
    k = kr_get();
    b = kr_fld(&k->node[k->root], "body");
    return kr_fld(kr_nod(kr_kid(b, 0)), "index") ? kr_kid(kr_fld(kr_nod(kr_kid(b, 0)), "index"), 0) : NULL;
}

/* String pool offset of the text `s` in the loaded kernel, 0 if absent. */
static uint32_t kx_off(const char *s)
{
    const kir_t *k = kr_get();
    uint32_t i = 1u;

    while (i < k->nstr) {
        if (strcmp(k->str + i, s) == 0) return i;
        i += (uint32_t)strlen(k->str + i) + 1u;
    }
    return 0;
}

static int kx_fo(const char *e, int32_t *v, int *isc)
{
    const kir_val_t *x = kx_load(e);
    size_t len;
    int rc;

    if (!x) return -1;
    th_cap_on();
    rc = kx_fold(x, v, isc);
    len = th_cap_off(eb, sizeof eb);
    return kx_code(rc, len);
}

/* Fold to a constant: 1 and *v when it folded, else 0 (-1: refused or not a constant). */
static int kx_k(const char *e, int32_t *v)
{
    int isc = 0;

    return kx_fo(e, v, &isc) == 0 && isc ? 1 : 0;
}

/* Evaluate with a0 bound to %r1 when the kernel spells it. Records stay in the
 * selection state for the caller. */
static int kx_ev(const char *e, nv_opnd_t *o)
{
    const kir_val_t *x = kx_load(e);
    nv_opnd_t r;
    uint32_t a0;
    size_t len;
    int rc;

    if (!x) return -1;
    a0 = kx_off("a0");
    if (a0 && (rg_new(RF_R, &r) || is_push() || is_bset(a0, r))) return -1;
    th_cap_on();
    rc = kx_eval(x, o);
    len = th_cap_off(eb, sizeof eb);
    return kx_code(rc, len);
}

/* ---- Folding ---- */

static void kx01(void)
{
    int32_t v = 0;

    CHECK(kx_k(BIN("*", LIT(6), LIT(7)), &v) && v == 42);
    CHECK(kx_k(BIN("*", LIT(65536), LIT(65536)), &v) && v == 0);
    CHECK(kx_k(BIN("+", LIT(2147483647), LIT(1)), &v) && v == INT32_MIN);
    CHECK(kx_k(BIN("-", LIT(5), LIT(9)), &v) && v == -4);
    CHECK(kx_k(BIN("*", DIM(8), LIT(3)), &v) && v == 24);
    CHECK(kx_k(BIN("+", BIN("*", LIT(4), LIT(8)), LIT(1)), &v) && v == 33);
    PASS();
}
TH_REG("kx", 1, "folds literal arithmetic with s32 wraparound", kx01)

static void kx02(void)
{
    int32_t v = 0;

    CHECK(kx_k(BIN("/", LIT(7), LIT(2)), &v) && v == 3);
    CHECK(kx_k(BIN("//", LIT(7), LIT(2)), &v) && v == 3);
    CHECK(kx_k(BIN("/", LIT(-7), LIT(2)), &v) && v == -3);
    CHECK(kx_k(BIN("/", LIT(7), LIT(-2)), &v) && v == -3);
    CHECK(kx_k(BIN("/", LIT(-8), LIT(2)), &v) && v == -4);
    CHECK(kx_k(BIN("%", LIT(7), LIT(4)), &v) && v == 3);
    CHECK(kx_k(BIN("%", LIT(-7), LIT(2)), &v) && v == -1);
    CHECK(kx_k(BIN("%", LIT(7), LIT(-2)), &v) && v == 1);
    PASS();
}
TH_REG("kx", 2, "folds / and % truncating toward zero", kx02)

static void kx03(void)
{
    int32_t v = 0;

    CHECK(kx_k(BIN("<<", LIT(1), LIT(31)), &v) && v == INT32_MIN);
    CHECK(kx_k(BIN("<<", LIT(1), LIT(40)), &v) && v == 0);
    CHECK(kx_k(BIN("<<", LIT(1), LIT(-1)), &v) && v == 0);
    CHECK(kx_k(BIN(">>", LIT(-8), LIT(1)), &v) && v == -4);
    CHECK(kx_k(BIN(">>", LIT(-1), LIT(31)), &v) && v == -1);
    CHECK(kx_k(BIN(">>", LIT(-5), LIT(99)), &v) && v == -1);
    CHECK(kx_k(BIN(">>", LIT(5), LIT(99)), &v) && v == 0);
    CHECK(kx_k(BIN("^", LIT(6), LIT(3)), &v) && v == 5);
    CHECK(kx_k(BIN("&", LIT(6), LIT(3)), &v) && v == 2);
    CHECK(kx_k(BIN("|", LIT(6), LIT(3)), &v) && v == 7);
    PASS();
}
TH_REG("kx", 3, "folds shifts clamped like PTX and the bit operators", kx03)

static void kx04(void)
{
    int32_t v = 0;
    int isc = 1;

    CHECK(kx_k(BIN("<", LIT(1), LIT(2)), &v) && v == 1);
    CHECK(kx_k(BIN("==", LIT(1), LIT(2)), &v) && v == 0);
    CHECK(kx_k(BIN("&&", LIT(1), LIT(0)), &v) && v == 0);
    CHECK(kx_k(BIN("||", LIT(1), LIT(0)), &v) && v == 1);
    CHECK(kx_k(DIM(64), &v) && v == 64);
    CHECK(kx_k("(Dim expr=64 hint=none)", &v) && v == 64);
    CHECK(kx_k("(TernaryExpr cond=" LIT(1) " if_true=" LIT(5) " if_false=" LIT(9) ")", &v) && v == 5);
    CHECK(kx_fo(BIN("+", VAR("a0"), LIT(1)), &v, &isc) == 0 && isc == 0);
    PASS();
}
TH_REG("kx", 4, "folds comparisons, logic, Dim and ternary; a Var is not constant", kx04)

static void kx05(void)
{
    int32_t v = 0;
    int isc = 0;

    CHEQ(kx_fo(BIN("/", LIT(1), LIT(0)), &v, &isc), 59);
    CHEQ(kx_fo(BIN("%", LIT(1), LIT(0)), &v, &isc), 59);
    CHEQ(kx_fo(BIN("/", LIT(-2147483648), LIT(-1)), &v, &isc), 59);
    CHEQ(kx_fo(BIN("+", LIT(4294967296), LIT(1)), &v, &isc), 59);
    PASS();
}
TH_REG("kx", 5, "refuses /0, %0, INT_MIN/-1 and a literal outside s32 with E059", kx05)

/* ---- Lowering ---- */

static void kx06(void)
{
    nv_opnd_t o;
    uint32_t n;
    const nv_ins_t *p;

    CHEQ(kx_ev(VAR("a0"), &o), 0);
    CHECK(o.kind == OP_REG && o.rf == RF_R && o.v == 1u);
    is_recs(&n);
    CHEQ(n, 0u);
    CHEQ(kx_ev(BIN("+", BIN("*", VAR("a0"), LIT(2)), LIT(1)), &o), 0);
    CHECK(o.kind == OP_REG && o.rf == RF_R && o.v == 3u);
    p = is_recs(&n);
    CHEQ(n, 2u);
    CHECK(p[0].op == NV_MUL_LO_S32 && p[0].dst.v == 2u && p[0].src[0].v == 1u);
    CHECK(p[0].src[1].kind == OP_IMM && p[0].src[1].v == 2u);
    CHECK(p[1].op == NV_ADD_S32 && p[1].dst.v == 3u && p[1].src[0].v == 2u);
    CHECK(p[1].src[1].kind == OP_IMM && p[1].src[1].v == 1u);
    PASS();
}
TH_REG("kx", 6, "lowers (a0 * 2) + 1 to mul.lo then add, a Var to its binding", kx06)

static void kx07(void)
{
    static const struct { const char *op; int nv; int rf; } T[] = {
        {"+", NV_ADD_S32, RF_R}, {"-", NV_SUB_S32, RF_R}, {"*", NV_MUL_LO_S32, RF_R},
        {"/", NV_DIV_S32, RF_R}, {"//", NV_DIV_S32, RF_R}, {"%", NV_REM_S32, RF_R},
        {"<", NV_SETP_LT_S32, RF_P}, {"<=", NV_SETP_LE_S32, RF_P}, {">", NV_SETP_GT_S32, RF_P},
        {">=", NV_SETP_GE_S32, RF_P}, {"==", NV_SETP_EQ_S32, RF_P},
        {"&", NV_AND_B32, RF_R}, {"|", NV_OR_B32, RF_R}, {"^", NV_XOR_B32, RF_R},
        {"<<", NV_SHL_B32, RF_R}, {">>", NV_SHR_S32, RF_R}
    };
    char b[256];
    nv_opnd_t o;
    uint32_t n, i;
    const nv_ins_t *p;

    for (i = 0; i < sizeof T / sizeof T[0]; i++) {
        snprintf(b, sizeof b, "(BinaryExpr op=\"%s\" left=" VAR("a0") " right=" LIT(3) ")", T[i].op);
        CHEQ(kx_ev(b, &o), 0);
        p = is_recs(&n);
        CHEQ(n, 1u);
        CHECK(p[0].op == (nv_op_t)T[i].nv && o.rf == (rf_t)T[i].rf && p[0].dst.rf == (rf_t)T[i].rf);
        CHECK(p[0].src[0].v == 1u && p[0].src[1].kind == OP_IMM && p[0].src[1].v == 3u);
    }
    PASS();
}
TH_REG("kx", 7, "gives every binary operator its s32 or predicate record", kx07)

static void kx08(void)
{
    nv_opnd_t o;
    uint32_t n;
    const nv_ins_t *p;

    CHEQ(kx_ev(BIN("&&", BIN("<", VAR("a0"), LIT(3)), BIN("==", VAR("a0"), LIT(1))), &o), 0);
    p = is_recs(&n);
    CHECK(o.rf == RF_P && n == 3u && p[0].op == NV_SETP_LT_S32 && p[1].op == NV_SETP_EQ_S32);
    CHECK(p[2].op == NV_AND_PRED && p[2].src[0].rf == RF_P && p[2].src[1].rf == RF_P);
    CHEQ(kx_ev(BIN("||", BIN("<", VAR("a0"), LIT(3)), BIN("==", VAR("a0"), LIT(1))), &o), 0);
    p = is_recs(&n);
    CHECK(n == 3u && p[2].op == NV_OR_PRED);
    CHEQ(kx_ev(BIN("&&", VAR("a0"), LT3), &o), 0);
    p = is_recs(&n);
    CHECK(n == 3u && p[0].op == NV_SETP_LT_S32 && p[1].op == NV_SETP_NE_S32 && p[2].op == NV_AND_PRED);
    CHEQ(kx_ev(BIN("+", LT3, LIT(1)), &o), 0);
    p = is_recs(&n);
    CHECK(n == 3u && p[1].op == NV_SELP_S32 && p[2].op == NV_ADD_S32);
    CHEQ(kx_ev(BIN("&&", LIT(1), LT3), &o), 0);
    is_recs(&n);
    CHECK(o.rf == RF_P && n == 1u);
    PASS();
}
TH_REG("kx", 8, "lowers && and ||, coerces ints to predicates, folds a true side", kx08)

static void kx09(void)
{
    nv_opnd_t o;
    uint32_t n;
    const nv_ins_t *p;

    CHEQ(kx_ev("(TernaryExpr cond=" BIN("<", VAR("a0"), LIT(3)) " if_true=" VAR("a0") " if_false=" LIT(9) ")", &o), 0);
    p = is_recs(&n);
    CHECK(o.rf == RF_R && n == 2u && p[1].op == NV_SELP_S32 && p[1].src[0].rf == RF_R);
    CHECK(p[1].src[1].kind == OP_IMM && p[1].src[1].v == 9u && p[1].src[2].rf == RF_P);
    CHEQ(kx_ev("(TernaryExpr cond=" BIN("<", VAR("a0"), LIT(3))
               " if_true=" FLT("0x1.0000000000000p+0f") " if_false=" FLT("0x0.0p+0f") ")", &o), 0);
    p = is_recs(&n);
    CHECK(o.rf == RF_F && n == 2u && p[1].op == NV_SELP_F32 && p[1].src[0].kind == OP_FIMM && p[1].src[0].v == 0x3f800000u);
    CHEQ(kx_ev(BIN("+", BIN("*", LIT(4), LIT(8)), LIT(1)), &o), 0);
    is_recs(&n);
    CHECK(o.kind == OP_IMM && (int32_t)o.v == 33 && n == 0u);
    CHEQ(kx_ev(BIN("+", VAR("a0"), BIN("*", LIT(4), LIT(8))), &o), 0);
    p = is_recs(&n);
    CHECK(n == 1u && p[0].src[1].kind == OP_IMM && p[0].src[1].v == 32u);
    CHEQ(kx_ev(FLT("0x1.8000000000000p+0f"), &o), 0);
    CHECK(o.kind == OP_FIMM && o.v == 0x3fc00000u);
    PASS();
}
TH_REG("kx", 9, "lowers ternaries to selp, folds constants around a Var", kx09)

static void kx10(void)
{
    nv_opnd_t o;

    CHEQ(kx_ev(BIN("+", VAR("zz"), LIT(1)), &o), 22);
    CHEQ(kx_ev(BIN("**", LIT(1), LIT(2)), &o), 21);
    CHEQ(kx_ev(BIN("!=", LIT(1), LIT(2)), &o), 21);
    CHEQ(kx_ev("(FuncCallExpr name=\"f\" args=[])", &o), 20);
    CHEQ(kx_ev(BIN("+", LIT(1), FLT("0x1.0000000000000p+0f")), &o), 21);
    PASS();
}
TH_REG("kx", 10, "refuses an unbound name, unknown op and kind, and int + float", kx10)

/* A ternary chain down if_false holds two operands per level, so the operand
 * stack (IR_MAX_DEP) runs out between depth 20 and 54. The reader's own
 * nesting cap (64) is not reached: wrappers take 5. */
static void kx11(void)
{
    static char ch[8192];
    nv_opnd_t o;
    int d, deep;

    for (deep = 20; deep <= 54; deep += 34) {
        ch[0] = '\0';
        for (d = 0; d < deep; d++)
            strcat(ch, "(TernaryExpr cond=" LIT(1) " if_true=" LIT(2) " if_false=");
        strcat(ch, LIT(1));
        for (d = 0; d < deep; d++) strcat(ch, ")");
        CHEQ(kx_ev(ch, &o), deep == 20 ? 0 : 82);
    }
    PASS();
}
TH_REG("kx", 11, "evaluates a 20 deep ternary chain and refuses 54 deep with E082", kx11)

/* ---- Affine forms ---- */

static int kx_af(const char *e, kx_aff_t *a)
{
    const kir_val_t *x = kx_load(e);
    size_t len;
    int rc;

    if (!x) return -1;
    th_cap_on();
    rc = kx_aff(x, a);
    len = th_cap_off(eb, sizeof eb);
    return len == 0 ? rc : -1;      /* not affine must stay silent */
}

static void kx12(void)
{
    kx_aff_t a;

    CHEQ(kx_af(BIN("+", BIN("*", VAR("a0"), LIT(2)), LIT(1)), &a), 1);
    CHECK(a.nvar == 1u && a.coef[0] == 2 && a.cst == 1 && strcmp(kr_str(a.var[0]), "a0") == 0);
    CHEQ(kx_af(BIN("+", BIN("+", BIN("*", VAR("a0"), LIT(4)), BIN("*", VAR("a1"), LIT(128))), LIT(3)), &a), 1);
    CHECK(a.nvar == 2u && a.coef[0] == 4 && a.coef[1] == 128 && a.cst == 3);
    CHEQ(kx_af(BIN("*", LIT(4), VAR("a0")), &a), 1);
    CHECK(a.coef[0] == 4);
    CHEQ(kx_af(BIN("+", VAR("a0"), VAR("a0")), &a), 1);
    CHECK(a.nvar == 1u && a.coef[0] == 2);
    CHEQ(kx_af(BIN("-", VAR("a0"), VAR("a0")), &a), 1);
    CHECK(a.nvar == 0u && a.cst == 0);
    CHEQ(kx_af(BIN("*", BIN("+", VAR("a0"), LIT(1)), LIT(8)), &a), 1);
    CHECK(a.coef[0] == 8 && a.cst == 8);
    CHEQ(kx_af(LIT(7), &a), 1);
    CHECK(a.nvar == 0u && a.cst == 7);
    CHEQ(kx_af(DIM(64), &a), 1);
    CHECK(a.cst == 64);
    PASS();
}
TH_REG("kx", 12, "proves the p01 index (a0 * 2) + 1 and sums of products affine", kx12)

static void kx13(void)
{
    kx_aff_t a;

    CHEQ(kx_af(BIN("*", VAR("a0"), VAR("a1")), &a), 0);
    CHEQ(kx_af(BIN("/", VAR("a0"), LIT(4)), &a), 0);
    CHEQ(kx_af(BIN("%", VAR("a0"), LIT(4)), &a), 0);
    CHEQ(kx_af(BIN("+", BIN("*", BIN("/", VAR("a0"), LIT(8)), LIT(64)), BIN("%", VAR("a1"), LIT(8))), &a), 0);
    CHEQ(kx_af(BIN("<", VAR("a0"), LIT(4)), &a), 0);
    CHEQ(kx_af(BIN("+", BIN("+", BIN("+", BIN("+", VAR("a"), VAR("b")), VAR("c")), VAR("d")), VAR("e")), &a), 0);
    CHEQ(kx_af(BIN("*", LIT(2147483647), BIN("*", LIT(2147483647), VAR("a0"))), &a), 0);
    CHEQ(kx_af(BIN("**", VAR("a0"), LIT(2)), &a), 0);
    CHEQ(kx_af("(TernaryExpr cond=" LIT(1) " if_true=" VAR("a0") " if_false=" LIT(9) ")", &a), 0);
    PASS();
}
TH_REG("kx", 13, "calls / % comparisons, var * var, 5 vars and overflow not affine, silently", kx13)

/* ---- Running records ---- */

#define TI_R 512

/* One thread's registers: s32 values and predicates, numbered as the records name them. */
typedef struct { int32_t r[TI_R]; uint8_t p[TI_R]; } ti_t;

static int32_t ti_v(const ti_t *t, nv_opnd_t o)
{
    return o.kind == OP_IMM ? (int32_t)o.v : t->r[o.v];
}

/* Execute records [from, n) for thread (tid, cta). 0: ran to the end,
 * 1: left by a taken guard branch, -1: an op the interpreter does not know. */
static int ti_run(ti_t *t, const nv_ins_t *q, uint32_t from, uint32_t n, int32_t tid, int32_t cta)
{
    uint32_t i;

    for (i = from; i < n; i++) {
        uint32_t a = (uint32_t)ti_v(t, q[i].src[0]), b = (uint32_t)ti_v(t, q[i].src[1]);
        int32_t sa = (int32_t)a, sb = (int32_t)b;
        uint32_t d = q[i].dst.v;

        if (q[i].op == NV_BRA) {
            if (q[i].prd.kind == OP_REG && t->p[q[i].prd.v] != (uint8_t)q[i].pneg) return 1;
            continue;
        }
        if (q[i].op == NV_LABEL || q[i].op == NV_RET) continue;
        switch (q[i].op) {
        case NV_MOV_TID_X:   t->r[d] = tid; break;
        case NV_MOV_CTAID_X: t->r[d] = cta; break;
        case NV_ADD_S32:     t->r[d] = (int32_t)(a + b); break;
        case NV_SUB_S32:     t->r[d] = (int32_t)(a - b); break;
        case NV_MUL_LO_S32:  t->r[d] = (int32_t)(a * b); break;
        case NV_MAD_LO_S32:  t->r[d] = (int32_t)(a * b + (uint32_t)ti_v(t, q[i].src[2])); break;
        case NV_DIV_S32:     if (sb == 0) return -1; t->r[d] = sa / sb; break;
        case NV_REM_S32:     if (sb == 0) return -1; t->r[d] = sa % sb; break;
        case NV_SHR_S32:     t->r[d] = sb >= 31 ? (sa < 0 ? -1 : 0) : sa >> sb; break;
        case NV_AND_B32:     t->r[d] = (int32_t)(a & b); break;
        case NV_SETP_GE_S32: t->p[d] = sa >= sb; break;
        default: return -1;
        }
    }
    return 0;
}

/* ---- Row-major flattening ---- */

static uint32_t lcg_s = 24024u;     /* fixed seed: the same cases every run */

static uint32_t lcg(uint32_t m)
{
    lcg_s = lcg_s * 1664525u + 1013904223u;
    return (lcg_s >> 8) % m;
}

/* Load a kernel with one f32 input of extents dim[0..rank-1] and a Load
 * whose index list text is `idx`. Returns that list. */
static const kir_val_t *kx_lst(uint32_t rank, const int32_t *dim, const char *idx)
{
    const kir_t *k;
    const kir_node_t *ld;
    size_t n;
    uint32_t i;

    n = (size_t)snprintf(kb, sizeof kb, "kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0\narg 0 x0 in f32 %u", rank);
    for (i = 0; i < rank; i++) n += (size_t)snprintf(kb + n, sizeof kb - n, " %d", dim[i]);
    snprintf(kb + n, sizeof kb - n, "\nbody\n[(Tile axes=[] body=[(Load names=[\"x\"] input=\"x0\" index=[%s] "
        "dtype=dt:f32)] block_threads=1 aux_threads=0 raster_axes=none raster_group=none "
        "raster_orient=\"m\")]\nend\n", idx);
    if (kr_load(kb, strlen(kb)) != 0) return NULL;
    is_reset();
    k = kr_get();
    ld = kr_nod(kr_kid(kr_fld(&k->node[k->root], "body"), 0));
    return kr_fld(ld, "index");
}

/* One random case of `rank` axes: each axis is a constant, a Var or a Var
 * plus a constant, with values in range. Returns mm_gflat's result (the
 * E-code or 0) and the flat index it computed (running its records) in *got
 * against the one computed from the rule in *want. */
static int kx_flat(uint32_t rank, int sh, int32_t *got, int32_t *want)
{
    int32_t dim[IR_MAX_DIM], val[IR_MAX_DIM];
    char idx[1024], one[160];
    uint32_t i, n0, a0;
    size_t len, il = 0;
    ti_t t;
    nv_opnd_t o, reg;
    const kir_val_t *lst;
    int rc;

    idx[0] = '\0';
    *want = 0;
    for (i = 0; i < rank; i++) {
        uint32_t mode = lcg(3);

        dim[i] = (int32_t)lcg(9) + 1;
        val[i] = (int32_t)lcg((uint32_t)dim[i]);
        if (mode == 0) snprintf(one, sizeof one, "(Literal value=%d dtype=\"int\") ", val[i]);
        else if (mode == 1) snprintf(one, sizeof one, "(Var name=\"v%u\") ", i);
        else snprintf(one, sizeof one, "(BinaryExpr op=\"+\" left=(Var name=\"v%u\") right=(Literal value=0 dtype=\"int\")) ", i);
        il += (size_t)snprintf(idx + il, sizeof idx - il, "%s", one);
        *want = *want * dim[i] + val[i];
    }
    lst = kx_lst(rank, dim, idx);
    if (!lst) return -1;
    memset(&t, 0, sizeof t);
    if (mm_params(kr_get()) || is_push()) return -1;
    for (i = 0; i < rank; i++) {
        char nm[8];

        snprintf(nm, sizeof nm, "v%u", i);
        a0 = kx_off(nm);
        if (a0 && (rg_new(RF_R, &reg) || is_bset(a0, reg))) return -1;
        if (a0) t.r[reg.v] = val[i];
    }
    is_recs(&n0);
    th_cap_on();
    rc = sh ? mm_flat(kr_get()->arg[0].dim, rank, lst, &o) : mm_gflat(0, lst, &o);
    len = th_cap_off(eb, sizeof eb);
    rc = kx_code(rc, len);
    if (rc != 0) return rc;
    {
        uint32_t n1;
        const nv_ins_t *q = is_recs(&n1);

        if (ti_run(&t, q, n0, n1, 0, 0) != 0) return -1;
    }
    *got = ti_v(&t, o);
    return 0;
}

/* mm_gflat on a fixed index list text with nothing bound; the E-code or 0. */
static int kx_gf(uint32_t rank, const int32_t *dim, const char *idx, nv_opnd_t *o)
{
    const kir_val_t *lst = kx_lst(rank, dim, idx);
    size_t len;
    int rc;

    if (!lst || mm_params(kr_get())) return -1;
    th_cap_on();
    rc = mm_gflat(0, lst, o);
    len = th_cap_off(eb, sizeof eb);
    return kx_code(rc, len);
}

static void kx14(void)
{
    static const int32_t d[3] = {8, 4, 6};
    nv_opnd_t o, a;
    int32_t b = -1;
    uint32_t n0, n1;

    CHEQ(kx_gf(0, d, "", &o), 0);
    CHECK(o.kind == OP_IMM && o.v == 0u);
    is_recs(&n0);
    CHEQ(mm_gaddr(0, o, &a, &b), 0);
    CHECK(b == 0 && a.rf == RF_RD);
    is_recs(&n1);
    CHEQ(n1, n0);
    CHEQ(kx_gf(0, d, LIT(0), &o), 59);
    CHEQ(kx_gf(1, d, "", &o), 59);
    CHEQ(kx_gf(2, d, LIT(1), &o), 59);
    CHEQ(kx_gf(1, d, VAR("zzz"), &o), 22);
    CHEQ(kx_gf(1, d, FLT("0x1.0000000000000p+0f"), &o), 59);
    PASS();
}
TH_REG("kx", 14, "flattens rank 0 to the constant 0 and refuses a wrong index count", kx14)

static void kx15(void)
{
    static const int32_t d[3] = {2, 3, 5};
    nv_opnd_t o;
    uint32_t n;

    CHEQ(kx_gf(1, d, LIT(1), &o), 0);
    CHECK(o.kind == OP_IMM && (int32_t)o.v == 1);
    CHEQ(kx_gf(2, d, LIT(1) " " LIT(2), &o), 0);
    CHECK(o.kind == OP_IMM && (int32_t)o.v == 1 * 3 + 2);
    CHEQ(kx_gf(3, d, LIT(1) " " LIT(2) " " LIT(4), &o), 0);
    CHECK(o.kind == OP_IMM && (int32_t)o.v == (1 * 3 + 2) * 5 + 4);
    CHEQ(kx_gf(3, d, BIN("+", LIT(0), LIT(1)) " " LIT(2) " " LIT(4), &o), 0);
    CHECK(o.kind == OP_IMM && (int32_t)o.v == (1 * 3 + 2) * 5 + 4);
    is_recs(&n);
    CHEQ(n, 2u);                    /* the one parameter's prologue only: constants emit nothing */
    PASS();
}
TH_REG("kx", 15, "folds constant indices of rank 1 to 3 to an immediate, no code", kx15)

/* 40 random cases per rank against sum(idx[k] * prod(dim[k+1..])). */
static void kx_sweep(int sh)
{
    uint32_t rank;
    int c;

    for (rank = 0; rank <= 3; rank++) {
        for (c = 0; c < 40; c++) {
            int32_t got = -1, want = -2;

            CHEQ(kx_flat(rank, sh, &got, &want), 0);
            CHEQ(got, want);
        }
    }
}

static void kx16(void)
{
    kx_sweep(0);
    PASS();
}
TH_REG("kx", 16, "mm_gflat row-major matches the closed form for rank 0-3, random extents", kx16)

static void kx17(void)
{
    kx_sweep(1);
    PASS();
}
TH_REG("kx", 17, "mm_flat row-major matches the closed form on explicit extents", kx17)

/* ---- Thread decode ---- */

/* Load a kernel whose Tile has axes a0.. of the given extents, launched
 * grid x block. Returns the Tile node. */
static const kir_node_t *kx_tile(const int32_t *ext, uint32_t na, int32_t grid, int32_t block)
{
    size_t n;
    uint32_t i;
    const kir_t *k;

    n = (size_t)snprintf(kb, sizeof kb, "kir 1\nkernel k\nlaunch %d 1 1 %d 1 1 0\narg 0 x0 in f32 1 8\nbody\n"
        "[(Tile axes=[", grid, block);
    for (i = 0; i < na; i++)
        n += (size_t)snprintf(kb + n, sizeof kb - n, "(Axis name=\"a%u\" extent=(Dim expr=(Literal value=%d "
            "dtype=\"int\") hint=none) window=none) ", i, ext[i]);
    snprintf(kb + n, sizeof kb - n, "] body=[] block_threads=none aux_threads=0 raster_axes=none "
        "raster_group=none raster_orient=\"m\")]\nend\n");
    if (kr_load(kb, strlen(kb)) != 0) return NULL;
    is_reset();
    k = kr_get();
    return &k->node[k->root];
}

/* Run is_tile on the tile and execute every launched thread through the
 * records. Every gid in 0..N-1 must be reached exactly once, with each axis
 * value in range and the axes recomposing (row-major, last fastest) to the
 * gid; the surplus threads must leave by the guard. 0 when all that holds. */
static int kx_dec(const int32_t *ext, uint32_t na, int32_t grid, int32_t block)
{
    static uint8_t seen[4096];
    const kir_node_t *tile = kx_tile(ext, na, grid, block);
    const kir_val_t *body, *ax;
    const nv_ins_t *q;
    uint32_t n, nrec, i, ncell = 1;
    int32_t t, thr = grid * block;
    ti_t th;
    nv_opnd_t o;

    if (!tile || is_push() || is_tile(kr_get(), tile, &body) || is_tend()) return -1;
    for (i = 0; i < na; i++) ncell *= (uint32_t)ext[i];
    if (is_ncell() != ncell || ncell > sizeof seen) return -2;
    memset(seen, 0, sizeof seen);
    q = is_recs(&nrec);
    ax = kr_fld(tile, "axes");
    for (t = 0, n = 0; t < thr; t++) {
        int32_t gid = 0;
        int rc;

        memset(&th, 0, sizeof th);
        rc = ti_run(&th, q, 0, nrec, t % block, t / block);
        if (rc < 0) return -3;
        if (rc == 1) { if ((uint32_t)t < ncell) return -4; continue; }
        if ((uint32_t)t >= ncell) return -5;           /* a surplus thread got through */
        for (i = 0; i < na; i++) {
            int32_t v;

            if (is_find(kr_fld(kr_nod(kr_kid(ax, i)), "name")->ref, &o)) return -6;
            v = ti_v(&th, o);
            if (v < 0 || v >= ext[i]) return -7;
            gid = gid * ext[i] + v;
        }
        if (gid != t || seen[gid]) return -8;
        seen[gid] = 1;
        n++;
    }
    return n == ncell ? 0 : -9;
}

static void kx18(void)
{
    static const int32_t e1[1] = {37}, e2[2] = {6, 8}, e3[3] = {3, 1, 5};

    CHEQ(kx_dec(e1, 1, 1, 37), 0);
    CHEQ(kx_dec(e2, 2, 1, 48), 0);
    CHEQ(kx_dec(e3, 3, 1, 15), 0);
    CHEQ(kx_dec(e3, 3, 2, 8), 0);       /* grid > 1: gid = ctaid * block + tid, 1 surplus thread */
    CHEQ(kx_dec(e2, 2, 4, 16), 0);      /* 64 threads, 48 cells: 16 exit by the guard */
    PASS();
}
TH_REG("kx", 18, "decodes fixed shapes as a bijection over range(N), surplus threads exit", kx18)

/* Powers of two decode with shr and and, other extents with div and rem, the
 * first axis needs no remainder and an extent of 1 binds the immediate 0. */
static void kx19(void)
{
    static const int32_t e[3] = {3, 6, 4}, f[2] = {4, 1};
    const kir_node_t *tile = kx_tile(e, 3, 1, 72);
    const kir_val_t *body;
    const nv_ins_t *q;
    uint32_t n, i, nd = 0, ns = 0, nr = 0, na = 0;
    nv_opnd_t o;

    CHECK(tile && is_push() == 0 && is_tile(kr_get(), tile, &body) == 0);
    q = is_recs(&n);
    for (i = 0; i < n; i++) {
        nd += q[i].op == NV_DIV_S32;
        ns += q[i].op == NV_SHR_S32;
        nr += q[i].op == NV_REM_S32;
        na += q[i].op == NV_AND_B32;
    }
    CHECK(nd == 1u && ns == 1u && nr == 1u && na == 1u);
    tile = kx_tile(f, 2, 1, 4);
    CHECK(tile && is_push() == 0 && is_tile(kr_get(), tile, &body) == 0);
    CHEQ(is_find(kr_fld(kr_nod(kr_kid(kr_fld(tile, "axes"), 1)), "name")->ref, &o), 0);
    CHECK(o.kind == OP_IMM && o.v == 0u);
    PASS();
}
TH_REG("kx", 19, "decodes with shr/and for powers of two, div/rem otherwise, 0 for extent 1", kx19)

/* Random extents (1 to 4 axes, N up to 4096) decoded as a bijection over
 * range(N), with a launch of one block or several that leaves surplus
 * threads. */
static void kx20(void)
{
    int c;

    for (c = 0; c < 60; c++) {
        int32_t ext[4], cells = 1, blk, grid;
        uint32_t na = lcg(4) + 1u, i;

        for (i = 0; i < na; i++) {
            static const int32_t pick[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 32};

            ext[i] = pick[lcg(13)];
            cells *= ext[i];
        }
        if (cells > 4096) { c--; continue; }
        blk = c % 2 ? cells : (int32_t)lcg(64) + 1;
        grid = (cells + blk - 1) / blk;
        CHEQ(kx_dec(ext, na, grid, blk), 0);
    }
    PASS();
}
TH_REG("kx", 20, "decode is a bijection over range(N) for 60 random extent lists and launches", kx20)

/* ---- Vector alignment (T032) ---- */

/* mm_vec on an index list text against extents dim[0..rank-1]; -1 on a load error. */
static int kx_vec(uint32_t rank, const int32_t *dim, const char *idx, uint32_t w)
{
    const kir_val_t *lst = kx_lst(rank, dim, idx);

    return lst ? mm_vec(dim, rank, lst, w) : -1;
}

/* The task brief says kx20.. but kx14 to kx20 were taken by the decode and
 * mm_flat fixtures, so the alignment fixtures start at kx21. */
static void kx21(void)
{
    static const int32_t d[1] = {64};

    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(2)), LIT(0)), 2), 1);      /* p01 */
    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(4)), LIT(0)), 4), 1);
    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(4)), LIT(8)), 4), 1);      /* p02 style */
    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(4)), LIT(8)), 2), 1);      /* 4 and 8 divide by 2 */
    CHEQ(kx_vec(1, d, LIT(4), 4), 1);
    CHEQ(kx_vec(1, d, LIT(0), 2), 1);
    PASS();
}
TH_REG("kx", 21, "a vector access is proven for (a0 * 2) + 0, (a0 * 4) + k and constant multiples", kx21)

static void kx22(void)
{
    static const int32_t d[1] = {64};

    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(2)), LIT(1)), 2), 0);      /* odd constant */
    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(2)), LIT(0)), 4), 0);      /* coef 2, width 4 */
    CHEQ(kx_vec(1, d, VAR("a0"), 2), 0);
    CHEQ(kx_vec(1, d, BIN("+", BIN("*", VAR("a0"), LIT(4)), LIT(2)), 4), 0);
    CHEQ(kx_vec(1, d, LIT(3), 2), 0);
    CHEQ(kx_vec(1, d, VAR("a0"), 3), 0);                                          /* only 2 and 4 */
    CHEQ(kx_vec(1, d, VAR("a0"), 1), 0);
    PASS();
}
TH_REG("kx", 22, "a misaligned coefficient or constant, or a width other than 2 or 4, stays scalar", kx22)

static void kx23(void)
{
    static const int32_t d[1] = {64}, m[2] = {4, 6};

    CHEQ(kx_vec(1, d, BIN("/", VAR("a0"), LIT(4)), 4), 0);
    CHEQ(kx_vec(1, d, BIN("%", VAR("a0"), LIT(4)), 4), 0);
    CHEQ(kx_vec(1, d, BIN("*", VAR("a0"), VAR("a1")), 2), 0);
    /* Rank 2: strides {6, 1}. a0 * 6 is divisible by 2 but not by 4; a1 * 1 never. */
    CHEQ(kx_vec(2, m, VAR("a0") " " BIN("*", VAR("a1"), LIT(2)), 2), 1);
    CHEQ(kx_vec(2, m, VAR("a0") " " BIN("*", VAR("a1"), LIT(2)), 4), 0);
    CHEQ(kx_vec(2, m, VAR("a0") " " VAR("a1"), 2), 0);
    CHEQ(kx_vec(2, m, VAR("a0") " " LIT(1), 2), 0);
    PASS();
}
TH_REG("kx", 23, "a non-affine index stays scalar and the flat form multiplies the strides in", kx23)
