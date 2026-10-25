/* kexpr.c -- kx_*: Var, Literal, Dim, BinaryExpr, TernaryExpr; folding; affine proofs.
 *
 * One flattener turns an expression into postfix tokens (no recursion: a
 * frame stack of at most IR_MAX_DEP). kx_eval, kx_fold and kx_aff then run
 * the same tokens with their own operand stacks. Folding is not a separate
 * pass: kx_eval folds any operation whose operands are both immediates, so
 * kx_fold is "all leaves are literals, run it, expect an immediate". */

#include <string.h>
#include "irptx.h"
#include "isel.h"
#include "kir.h"
#include "kexpr.h"

/* ---- Tokens ---- */

#define KX_MAX_TOK 1024u            /* postfix tokens in one expression */
#define KX_LIM     2147483647        /* affine coefficient bound: s32 index math */

typedef enum { KT_VAR = 0, KT_INT, KT_FLT, KT_BIN, KT_SEL } kt_kind_t;

typedef enum {
    KO_ADD = 0, KO_SUB, KO_MUL, KO_DIV, KO_MOD,
    KO_LT, KO_LE, KO_GT, KO_GE, KO_EQ,
    KO_LAND, KO_LOR, KO_AND, KO_OR, KO_XOR, KO_SHL, KO_SHR,
    KO_N
} kx_op_t;

typedef struct {
    uint8_t  k;                     /* kt_kind_t */
    uint8_t  op;                    /* KT_BIN: kx_op_t */
    int32_t  v;                     /* KT_INT value */
    uint32_t nm;                    /* KT_VAR name offset; KT_FLT f32 bits */
} kx_tok_t;

/* One row per op string Emmy's BinaryExpr defines (ir/expr.py apply_binop).
 * "/" and "//" are both integer division. Emmy has no "!=". */
typedef struct {
    const char *s;
    kx_op_t     id;
    nv_op_t     nv;                 /* the instruction a non-constant operand pair emits */
} kx_row_t;

static const kx_row_t KX_TAB[] = {
    { "+",  KO_ADD,  NV_ADD_S32 },     { "-",  KO_SUB,  NV_SUB_S32 },
    { "*",  KO_MUL,  NV_MUL_LO_S32 },  { "/",  KO_DIV,  NV_DIV_S32 },
    { "//", KO_DIV,  NV_DIV_S32 },     { "%",  KO_MOD,  NV_REM_S32 },
    { "<",  KO_LT,   NV_SETP_LT_S32 }, { "<=", KO_LE,   NV_SETP_LE_S32 },
    { ">",  KO_GT,   NV_SETP_GT_S32 }, { ">=", KO_GE,   NV_SETP_GE_S32 },
    { "==", KO_EQ,   NV_SETP_EQ_S32 }, { "&&", KO_LAND, NV_AND_PRED },
    { "||", KO_LOR,  NV_OR_PRED },     { "&",  KO_AND,  NV_AND_B32 },
    { "|",  KO_OR,   NV_OR_B32 },      { "^",  KO_XOR,  NV_XOR_B32 },
    { "<<", KO_SHL,  NV_SHL_B32 },     { ">>", KO_SHR,  NV_SHR_S32 }
};

#define KX_NTAB (sizeof KX_TAB / sizeof KX_TAB[0])

/* Print one error when `loud`; kx_aff passes 0 because "not affine" is an
 * answer, not a failure. Always returns 1 so callers can `return`. */
static int kx_bad(int loud, int code, const char *pre, const char *nm)
{
    char msg[IR_MAX_NAME + 80];

    if (!loud) return 1;
    strcpy(msg, pre);
    if (nm) {
        strcat(msg, " '");
        strncat(msg, nm, IR_MAX_NAME);
        strcat(msg, "'");
    }
    return er_fail(code, msg);
}

static const kx_row_t *kx_row(const char *s)
{
    size_t i;

    for (i = 0; i < KX_NTAB; i++)
        if (strcmp(KX_TAB[i].s, s) == 0) return &KX_TAB[i];
    return NULL;
}

/* ---- Flattening: tree to postfix ---- */

typedef struct {
    const kir_node_t *nd;
    uint8_t           ent;          /* 1 once the node has been entered */
    uint8_t           st;           /* children visited so far */
    uint8_t           op;           /* BinaryExpr: kx_op_t, parsed on entry */
} kx_frm_t;

typedef struct {
    kx_tok_t tk[KX_MAX_TOK];
    uint32_t n;
    kx_frm_t fr[IR_MAX_DEP];
    uint32_t nfr;
    int      loud;
} kx_flat_t;

static int kx_emit(kx_flat_t *f, kt_kind_t k, kx_op_t op, int32_t v, uint32_t nm)
{
    kx_tok_t *t;

    if (f->n >= KX_MAX_TOK) return kx_bad(f->loud, E081, "an expression with more tokens than the limit", NULL);
    t = &f->tk[f->n++];
    t->k = (uint8_t)k;
    t->op = (uint8_t)op;
    t->v = v;
    t->nm = nm;
    return 0;
}

/* The IR names the f32 type "float"; an int stored in a float literal is
 * converted exactly the way a C cast would. */
static uint32_t kx_fbits(int64_t v)
{
    float x = (float)v;
    uint32_t b;

    memcpy(&b, &x, sizeof b);
    return b;
}

static int kx_lit(kx_flat_t *f, const kir_node_t *nd)
{
    const kir_val_t *v = kr_fld(nd, "value"), *d = kr_fld(nd, "dtype");
    int isf = d && d->tag == KV_STR && strcmp(kr_str(d->ref), "float") == 0;

    if (!v) return kx_bad(f->loud, E015, "a Literal with no value", NULL);
    if (v->tag == KV_FLT && !(d && d->tag == KV_STR && strcmp(kr_str(d->ref), "int") == 0))
        return kx_emit(f, KT_FLT, KO_ADD, 0, v->fbits);
    if (v->tag == KV_INT && isf) return kx_emit(f, KT_FLT, KO_ADD, 0, kx_fbits(v->i));
    if (v->tag != KV_INT && v->tag != KV_BOOL)
        return kx_bad(f->loud, E059, "a Literal whose value is not a number of its dtype", NULL);
    if (v->i > 2147483647 || v->i < -2147483647 - 1)
        return kx_bad(f->loud, E059, "an integer literal outside s32", NULL);
    return kx_emit(f, KT_INT, KO_ADD, (int32_t)v->i, 0);
}

/* Names of the child fields, in evaluation order, per node kind. */
static uint32_t kx_kids(kr_kind_t k, const char *const **nm)
{
    static const char *const BIN[] = { "left", "right" };
    static const char *const TER[] = { "cond", "if_true", "if_false" };
    static const char *const DIM[] = { "expr" };

    switch (k) {
    case KR_BINARYEXPR:  *nm = BIN; return 2;
    case KR_TERNARYEXPR: *nm = TER; return 3;
    case KR_DIM:         *nm = DIM; return 1;
    default:             *nm = NULL; return 0;
    }
}

/* Visit a child value: a node opens a frame, a bare number is a leaf. */
static int kx_desc(kx_flat_t *f, const kir_val_t *v)
{
    const kir_node_t *nd;
    kx_frm_t *fr;

    if (!v) return kx_bad(f->loud, E015, "an expression with a missing operand", NULL);
    if (v->tag == KV_INT || v->tag == KV_BOOL) {
        if (v->i > 2147483647 || v->i < -2147483647 - 1)
            return kx_bad(f->loud, E059, "an integer literal outside s32", NULL);
        return kx_emit(f, KT_INT, KO_ADD, (int32_t)v->i, 0);
    }
    if (v->tag == KV_FLT) return kx_emit(f, KT_FLT, KO_ADD, 0, v->fbits);
    nd = kr_nod(v);
    if (!nd) return kx_bad(f->loud, E020, "an operand that is not an expression node", NULL);
    if (f->nfr >= IR_MAX_DEP) return kx_bad(f->loud, E082, "an expression nested deeper than the limit", NULL);
    fr = &f->fr[f->nfr++];
    fr->nd = nd;
    fr->ent = 0;
    fr->st = 0;
    fr->op = 0;
    return 0;
}

/* Enter a frame's node: leaves emit and close, BinaryExpr parses its op. */
static int kx_enter(kx_flat_t *f, kx_frm_t *fr)
{
    const kir_val_t *o;
    const kx_row_t *r;

    switch ((kr_kind_t)fr->nd->kind) {
    case KR_VAR:
        o = kr_fld(fr->nd, "name");
        if (!o || o->tag != KV_STR) return kx_bad(f->loud, E015, "a Var with no name", NULL);
        f->nfr--;
        return kx_emit(f, KT_VAR, KO_ADD, 0, o->ref);
    case KR_LITERAL:
        f->nfr--;
        return kx_lit(f, fr->nd);
    case KR_BINARYEXPR:
        o = kr_fld(fr->nd, "op");
        if (!o || o->tag != KV_STR) return kx_bad(f->loud, E015, "a BinaryExpr with no op", NULL);
        r = kx_row(kr_str(o->ref));
        if (!r) return kx_bad(f->loud, E021, "expression op", kr_str(o->ref));
        fr->op = (uint8_t)r->id;
        return 0;
    case KR_TERNARYEXPR:
    case KR_DIM:
        return 0;
    default:
        return kx_bad(f->loud, E020, "expression kind", kr_str(fr->nd->kname));
    }
}

/* One step on the top frame: enter it, descend into its next child, or close
 * it with its postfix token. */
static int kx_step(kx_flat_t *f)
{
    kx_frm_t *fr = &f->fr[f->nfr - 1u];
    kr_kind_t kind = (kr_kind_t)fr->nd->kind;
    const char *const *nm;
    uint32_t nk = kx_kids(kind, &nm);

    if (!fr->ent) {
        fr->ent = 1;
        return kx_enter(f, fr);
    }
    if (fr->st < nk) {
        const kir_val_t *kid = kr_fld(fr->nd, nm[fr->st]);

        fr->st++;
        return kx_desc(f, kid);
    }
    f->nfr--;
    if (kind == KR_BINARYEXPR) return kx_emit(f, KT_BIN, (kx_op_t)fr->op, 0, 0);
    if (kind == KR_TERNARYEXPR) return kx_emit(f, KT_SEL, KO_ADD, 0, 0);
    return 0;
}

static int kx_flat(kx_flat_t *f, const kir_val_t *e, int loud)
{
    IX_GUARD(g, KX_MAX_TOK * 8u);

    f->n = 0;
    f->nfr = 0;
    f->loud = loud;
    if (kx_desc(f, e)) return 1;
    while (f->nfr > 0 && g--)
        if (kx_step(f)) return 1;
    if (f->nfr > 0) return kx_bad(loud, E082, "an expression that did not finish flattening", NULL);
    return 0;
}

/* ---- Operand typing ---- */

static int kx_isf(nv_opnd_t o)
{
    return o.kind == OP_FIMM || (o.kind == OP_REG && o.rf == RF_F);
}

static int kx_isp(nv_opnd_t o)
{
    return o.kind == OP_REG && o.rf == RF_P;
}

static int kx_isr(nv_opnd_t o)
{
    return o.kind == OP_REG && o.rf == RF_R;
}

/* Coerce to an s32 operand: a predicate becomes 1 or 0 with selp, as Python
 * would see a bool in arithmetic. Floats and 64-bit values are refused. */
static int kx_int(nv_opnd_t *o)
{
    nv_opnd_t d;

    if (o->kind == OP_IMM || kx_isr(*o)) return 0;
    if (kx_isf(*o)) return er_fail(E021, "a float operand in an integer expression");
    if (!kx_isp(*o)) return er_fail(E059, "an operand that is not an s32 value or a predicate");
    if (rg_new(RF_R, &d)) return 1;
    if (is_ins(NV_SELP_S32, d, is_oimm(1), is_oimm(0), *o)) return 1;
    *o = d;
    return 0;
}

/* Coerce to a predicate: an s32 register is tested against zero. An
 * immediate stays one; the caller folds it. */
static int kx_prd(nv_opnd_t *o)
{
    nv_opnd_t d;

    if (o->kind == OP_IMM || kx_isp(*o)) return 0;
    if (!kx_isr(*o)) return er_fail(E021, "a non-integer operand where a condition is needed");
    if (rg_new(RF_P, &d)) return 1;
    if (is_ins(NV_SETP_NE_S32, d, *o, is_oimm(0), is_onone())) return 1;
    *o = d;
    return 0;
}

/* ---- Constant arithmetic ---- */

/* s32 shift right that does not lean on implementation-defined behaviour. */
static int32_t kx_sar(int32_t a, uint32_t s)
{
    return a >= 0 ? (int32_t)((uint32_t)a >> s) : (int32_t)~(~(uint32_t)a >> s);
}

static int32_t kx_wrap(uint32_t u)
{
    return u > 2147483647u ? (int32_t)(-(int64_t)(4294967296u - u)) : (int32_t)u;
}

/* Divide or take the remainder with C/PTX truncation, as the reference CUDA
 * does (ruling: truncating everywhere). Refuses what PTX leaves undefined. */
static int kx_divm(kx_op_t op, int32_t a, int32_t b, int32_t *r)
{
    if (b == 0) return er_fail(E059, "a division or remainder by zero in a constant");
    if (a == -2147483647 - 1 && b == -1) return er_fail(E059, "INT_MIN divided by -1 in a constant");
    *r = (op == KO_DIV) ? a / b : a % b;
    return 0;
}

/* Fold `a op b` with the semantics of the instruction kx_bin would emit. */
static int kx_cfold(kx_op_t op, int32_t a, int32_t b, int32_t *r)
{
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;

    switch (op) {
    case KO_ADD: *r = kx_wrap(ua + ub); return 0;
    case KO_SUB: *r = kx_wrap(ua - ub); return 0;
    case KO_MUL: *r = kx_wrap((uint32_t)((uint64_t)ua * ub)); return 0;
    case KO_DIV: case KO_MOD: return kx_divm(op, a, b, r);
    case KO_LT:  *r = a < b; return 0;
    case KO_LE:  *r = a <= b; return 0;
    case KO_GT:  *r = a > b; return 0;
    case KO_GE:  *r = a >= b; return 0;
    case KO_EQ:  *r = a == b; return 0;
    case KO_AND: *r = kx_wrap(ua & ub); return 0;
    case KO_OR:  *r = kx_wrap(ua | ub); return 0;
    case KO_XOR: *r = kx_wrap(ua ^ ub); return 0;
    case KO_SHL: case KO_SHR:
        /* PTX: the count is unsigned and clamped at 32 */
        if (ub > 32u) ub = 32u;
        if (op == KO_SHL) *r = ub == 32u ? 0 : kx_wrap(ua << ub);
        else *r = ub == 32u ? (a < 0 ? -1 : 0) : kx_sar(a, ub);
        return 0;
    default: break;
    }
    return er_fail(E000, "an expression op with no folding rule");
}

/* ---- Lowering the operators ---- */

/* && and ||: both sides become predicates (Python bool()). A constant side
 * decides or vanishes; two constants fold. */
static int kx_logic(kx_op_t op, nv_opnd_t a, nv_opnd_t b, nv_opnd_t *out)
{
    nv_opnd_t d;
    int is_and = (op == KO_LAND);

    if (kx_prd(&a) || kx_prd(&b)) return 1;
    if (a.kind == OP_IMM && b.kind == OP_IMM) {
        *out = is_oimm(is_and ? (a.v != 0 && b.v != 0) : (a.v != 0 || b.v != 0));
        return 0;
    }
    if (a.kind == OP_IMM || b.kind == OP_IMM) {
        nv_opnd_t k = (a.kind == OP_IMM) ? a : b, o = (a.kind == OP_IMM) ? b : a;

        if ((k.v != 0) == is_and) *out = o;     /* true && x, false || x */
        else *out = is_oimm(is_and ? 0 : 1);    /* false && x, true || x */
        return 0;
    }
    if (rg_new(RF_P, &d)) return 1;
    if (is_ins(is_and ? NV_AND_PRED : NV_OR_PRED, d, a, b, is_onone())) return 1;
    *out = d;
    return 0;
}

static int kx_bin(kx_op_t op, nv_opnd_t a, nv_opnd_t b, nv_opnd_t *out)
{
    nv_opnd_t d;
    int32_t r = 0;
    size_t i = 0;
    rf_t rf = (op >= KO_LT && op <= KO_EQ) ? RF_P : RF_R;

    if (op == KO_LAND || op == KO_LOR) return kx_logic(op, a, b, out);
    if (kx_int(&a) || kx_int(&b)) return 1;
    if (a.kind == OP_IMM && b.kind == OP_IMM) {
        if (kx_cfold(op, (int32_t)a.v, (int32_t)b.v, &r)) return 1;
        *out = is_oimm(r);
        return 0;
    }
    while (i < KX_NTAB && KX_TAB[i].id != op) i++;
    if (i == KX_NTAB) return er_fail(E000, "an expression op with no instruction");
    if (rg_new(rf, &d)) return 1;
    if (is_ins(KX_TAB[i].nv, d, a, b, is_onone())) return 1;
    *out = d;
    return 0;
}

/* cond ? t : f. Both arms are already lowered. A constant condition picks an
 * arm; otherwise selp.f32 when both arms are floats, selp.s32 when both are
 * integers, and a mix is refused. */
static int kx_sel(nv_opnd_t c, nv_opnd_t t, nv_opnd_t f, nv_opnd_t *out)
{
    nv_opnd_t d;
    int flt = kx_isf(t);

    if (kx_isf(c)) return er_fail(E021, "a float used as a condition");
    if (kx_prd(&c)) return 1;
    if (c.kind == OP_IMM) {
        *out = c.v ? t : f;
        return 0;
    }
    if (flt != kx_isf(f)) return er_fail(E021, "a ternary mixing float and integer arms");
    if (!flt && (kx_int(&t) || kx_int(&f))) return 1;
    if (rg_new(flt ? RF_F : RF_R, &d)) return 1;
    if (is_ins(flt ? NV_SELP_F32 : NV_SELP_S32, d, t, f, c)) return 1;
    *out = d;
    return 0;
}

/* ---- Evaluation ---- */

static kx_flat_t KF;                /* shared scratch: 16 KB is not for the stack */

typedef struct {
    nv_opnd_t v[IR_MAX_DEP];
    uint32_t  n;
} kx_stk_t;

static int kx_push(kx_stk_t *s, nv_opnd_t o)
{
    if (s->n >= IR_MAX_DEP) return er_fail(E082, "an expression whose operand stack is deeper than the limit");
    s->v[s->n++] = o;
    return 0;
}

static int kx_pop(kx_stk_t *s, nv_opnd_t *o)
{
    if (s->n == 0) return er_fail(E000, "an expression with a missing operand on the stack");
    *o = s->v[--s->n];
    return 0;
}

/* A bound name must be something an s32 expression can use. */
static int kx_var(uint32_t nm, nv_opnd_t *o)
{
    if (is_find(nm, o)) return 1;
    if (o->kind == OP_IMM || o->kind == OP_FIMM) return 0;
    if (o->kind == OP_REG && o->rf != RF_RD) return 0;
    return er_fail(E059, "a name bound to a 64-bit or non-register value used in an expression");
}

static int kx_tok(kx_stk_t *s, const kx_tok_t *t)
{
    nv_opnd_t a, b, c, r;

    switch ((kt_kind_t)t->k) {
    case KT_VAR:
        if (kx_var(t->nm, &r)) return 1;
        return kx_push(s, r);
    case KT_INT: return kx_push(s, is_oimm(t->v));
    case KT_FLT: return kx_push(s, is_ofim(t->nm));
    case KT_BIN:
        if (kx_pop(s, &b) || kx_pop(s, &a)) return 1;
        if (kx_bin((kx_op_t)t->op, a, b, &r)) return 1;
        return kx_push(s, r);
    case KT_SEL:
        if (kx_pop(s, &b) || kx_pop(s, &a) || kx_pop(s, &c)) return 1;
        if (kx_sel(c, a, b, &r)) return 1;
        return kx_push(s, r);
    default: break;
    }
    return er_fail(E000, "an expression token of no known kind");
}

static int kx_run(const kx_flat_t *f, nv_opnd_t *out)
{
    kx_stk_t s;
    uint32_t i;

    s.n = 0;
    for (i = 0; i < f->n; i++)
        if (kx_tok(&s, &f->tk[i])) return 1;
    if (s.n != 1) return er_fail(E000, "an expression that left more than one value");
    *out = s.v[0];
    return 0;
}

/* Ternary arms and && / || sides are all lowered eagerly: PTX division does
 * not trap, so evaluating the dead side is harmless. */
int kx_eval(const kir_val_t *e, nv_opnd_t *out)
{
    if (kx_flat(&KF, e, 1)) return 1;
    return kx_run(&KF, out);
}

int kx_fold(const kir_val_t *e, int32_t *val, int *isc)
{
    nv_opnd_t r;
    uint32_t i;

    *isc = 0;
    if (kx_flat(&KF, e, 1)) return 1;
    for (i = 0; i < KF.n; i++)
        if (KF.tk[i].k == KT_VAR || KF.tk[i].k == KT_FLT) return 0;
    if (kx_run(&KF, &r)) return 1;
    if (r.kind != OP_IMM) return er_fail(E000, "a constant expression that did not fold");
    *val = (int32_t)r.v;
    *isc = 1;
    return 0;
}

/* ---- Affine indices ---- */

static int kx_alim(int64_t v)
{
    return v <= KX_LIM && v >= -(int64_t)KX_LIM;
}

/* a *= k, dropping variables whose coefficient became zero. */
static int kx_ascl(kx_aff_t *a, int64_t k)
{
    uint32_t i, m = 0;

    for (i = 0; i < a->nvar; i++) {
        a->coef[i] *= k;
        if (!kx_alim(a->coef[i])) return 0;
        if (a->coef[i] != 0) {
            a->var[m] = a->var[i];
            a->coef[m++] = a->coef[i];
        }
    }
    a->nvar = m;
    a->cst *= k;
    return kx_alim(a->cst);
}

/* a += sg * b, merging variables by name text. 0 when it stops being affine. */
static int kx_amrg(kx_aff_t *a, const kx_aff_t *b, int sg)
{
    uint32_t i, j;

    for (i = 0; i < b->nvar; i++) {
        for (j = 0; j < a->nvar; j++)
            if (strcmp(kr_str(a->var[j]), kr_str(b->var[i])) == 0) break;
        if (j == a->nvar) {
            if (a->nvar >= IR_MAX_AFF) return 0;
            a->var[a->nvar] = b->var[i];
            a->coef[a->nvar++] = 0;
        }
        a->coef[j] += sg * b->coef[i];
        if (!kx_alim(a->coef[j])) return 0;
    }
    a->cst += sg * b->cst;
    if (!kx_alim(a->cst)) return 0;
    return kx_ascl(a, 1);           /* drops variables that cancelled to zero */
}

/* Combine the top two entries with `op`; 0 when the result is not affine. */
static int kx_acomb(kx_op_t op, kx_aff_t *a, kx_aff_t *b)
{
    if (op == KO_ADD || op == KO_SUB) return kx_amrg(a, b, op == KO_ADD ? 1 : -1);
    if (op != KO_MUL) return 0;
    if (b->nvar == 0) return kx_ascl(a, b->cst);
    if (a->nvar != 0) return 0;
    if (!kx_ascl(b, a->cst)) return 0;
    *a = *b;
    return 1;
}

static int kx_atok(kx_aff_t *s, uint32_t *n, const kx_tok_t *t)
{
    kx_aff_t *top;

    if (t->k == KT_BIN) {
        if (*n < 2) return 0;
        (*n)--;
        return kx_acomb((kx_op_t)t->op, &s[*n - 1u], &s[*n]);
    }
    if (*n >= IR_MAX_DEP || (t->k != KT_VAR && t->k != KT_INT)) return 0;
    top = &s[(*n)++];
    memset(top, 0, sizeof *top);
    if (t->k == KT_INT) {
        top->cst = t->v;
        return 1;
    }
    top->var[0] = t->nm;
    top->coef[0] = 1;
    top->nvar = 1;
    return 1;
}

int kx_aff(const kir_val_t *e, kx_aff_t *out)
{
    kx_aff_t s[IR_MAX_DEP];
    uint32_t i, n = 0;

    if (kx_flat(&KF, e, 0)) return 0;
    for (i = 0; i < KF.n; i++)
        if (!kx_atok(s, &n, &KF.tk[i])) return 0;
    if (n != 1) return 0;
    *out = s[0];
    return 1;
}
