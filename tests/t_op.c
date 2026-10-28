/* t_op.c -- the op table (op01..): names, arity, identities and the records
 * op_emit appends. Record lists are checked by the real verifier on a kernel
 * loaded through the real reader. The ptxas check of a module holding every
 * op is a scratch step (see the task report), not part of the runner. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tharns.h"
#include "irptx.h"
#include "isel.h"
#include "kir.h"
#include "ops.h"

#define OP_KIR \
    "kir 1\nkernel k_empty\nlaunch 2 1 1 128 1 1 0\narg 0 x in f32 1 200\narg 1 y out f32 1 200\nbody\n" \
    "[(Tile axes=[(Axis name=\"a0\" extent=(Dim expr=(Literal value=200 dtype=\"int\") hint=none) window=none)] " \
    "body=[] block_threads=none aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n"

typedef struct {
    const char *name;
    opn_t       op;
    uint32_t    arity;
} op_exp_t;

/* Emmy's names and arities (ElementwiseImpl(name).arity). */
static const op_exp_t OP_EXP[] = {
    {"add", OPN_ADD, 2}, {"subtract", OPN_SUBTRACT, 2}, {"multiply", OPN_MULTIPLY, 2},
    {"divide", OPN_DIVIDE, 2}, {"true_divide", OPN_TRUE_DIVIDE, 2},
    {"maximum", OPN_MAXIMUM, 2}, {"minimum", OPN_MINIMUM, 2},
    {"relu", OPN_RELU, 1}, {"copy", OPN_COPY, 1}, {"negative", OPN_NEGATIVE, 1}, {"abs", OPN_ABS, 1},
    {"sqrt", OPN_SQRT, 1}, {"rsqrt", OPN_RSQRT, 1}, {"reciprocal", OPN_RECIPROCAL, 1},
    {"tanh", OPN_TANH, 1}, {"sin", OPN_SIN, 1}, {"cos", OPN_COS, 1}, {"log2", OPN_LOG2, 1},
    {"exp", OPN_EXP, 1}, {"exp_fast", OPN_EXP_FAST, 1}, {"log", OPN_LOG, 1},
    {"sigmoid", OPN_SIGMOID, 1}, {"silu", OPN_SILU, 1}, {"softplus", OPN_SOFTPLUS, 1},
    {"pow", OPN_POW, 2}
};
#define OP_NEXP (sizeof OP_EXP / sizeof OP_EXP[0])

static char eb[512];

static int op_fresh(void)
{
    if (kr_load(OP_KIR, strlen(OP_KIR)) != 0) return 1;
    is_reset();
    return 0;
}

/* Run the verifier with stderr captured; 0 when it accepts and says nothing. */
static int op_vf(void)
{
    int rc;
    size_t len;

    th_cap_on();
    rc = vf_run();
    len = th_cap_off(eb, sizeof eb);
    return rc == 0 && len == 0 ? 0 : 1;
}

/* Every listed name is found, maps to its own enum value, and nothing else is. */
static void op01(void)
{
    uint32_t i;

    for (i = 0; i < OP_NEXP; i++) CHEQ(op_find(OP_EXP[i].name), OP_EXP[i].op);
    PASS();
}
TH_REG("op", 1, "op_find maps every Emmy elementwise name to its op", op01)

/* Unknown names, `sub` (Emmy cannot emit it), near misses and NULL are not found. */
static void op02(void)
{
    CHEQ(op_find("sub"), OPN_NONE);
    CHEQ(op_find("exp_unknown"), OPN_NONE);
    CHEQ(op_find(""), OPN_NONE);
    CHEQ(op_find("Add"), OPN_NONE);
    CHEQ(op_find("add "), OPN_NONE);
    CHEQ(op_find("adds"), OPN_NONE);
    CHEQ(op_find(NULL), OPN_NONE);
    PASS();
}
TH_REG("op", 2, "op_find says not found for unknown, near-miss and NULL names", op02)

static void op03(void)
{
    uint32_t i;

    for (i = 0; i < OP_NEXP; i++) CHEQ(op_arity(OP_EXP[i].op), OP_EXP[i].arity);
    CHEQ(op_arity(OPN_NONE), 0u);
    CHEQ(op_arity(OPN_NCOUNT), 0u);
    CHEQ(op_arity((opn_t)200), 0u);
    PASS();
}
TH_REG("op", 3, "op_arity matches Emmy, 0 for none and out of range", op03)

/* Exact bits: add 0, multiply 1, maximum -1e30 (0xF149F2CA), minimum +1e30
 * (0x7149F2CA); every other op has none. No infinity anywhere. */
static void op04(void)
{
    uint32_t b = 0xDEADBEEFu, i;

    CHECK(op_ident(OPN_ADD, &b) == 1 && b == 0x00000000u);
    CHECK(op_ident(OPN_MULTIPLY, &b) == 1 && b == 0x3F800000u);
    CHECK(op_ident(OPN_MAXIMUM, &b) == 1 && b == 0xF149F2CAu);
    CHECK(op_ident(OPN_MINIMUM, &b) == 1 && b == 0x7149F2CAu);
    for (i = 0; i < OP_NEXP; i++) {
        opn_t o = OP_EXP[i].op;

        b = 0xDEADBEEFu;
        if (o == OPN_ADD || o == OPN_MULTIPLY || o == OPN_MAXIMUM || o == OPN_MINIMUM) continue;
        CHEQ(op_ident(o, &b), 0);
        CHEQ(b, 0xDEADBEEFu);       /* untouched when there is none */
    }
    CHEQ(op_ident(OPN_NONE, &b), 0);
    CHEQ(op_ident((opn_t)200, &b), 0);
    PASS();
}
TH_REG("op", 4, "identities have the exact f32 bits (-1e30 is 0xF149F2CA), only for add/multiply/maximum/minimum", op04)

/* The PTX opcode each op becomes, its source count and relu's zero immediate. */
static void op05(void)
{
    static const struct { opn_t op; nv_op_t nv; uint8_t nsrc; } want[] = {
        {OPN_ADD, NV_ADD_RN_F32, 2}, {OPN_SUBTRACT, NV_SUB_RN_F32, 2},
        {OPN_MULTIPLY, NV_MUL_RN_F32, 2}, {OPN_DIVIDE, NV_DIV_RN_F32, 2},
        {OPN_TRUE_DIVIDE, NV_DIV_RN_F32, 2}, {OPN_MAXIMUM, NV_MAX_F32, 2},
        {OPN_MINIMUM, NV_MIN_F32, 2}, {OPN_RELU, NV_MAX_F32, 2},
        {OPN_COPY, NV_MOV_F32, 1}, {OPN_NEGATIVE, NV_NEG_F32, 1}, {OPN_ABS, NV_ABS_F32, 1}
    };
    uint32_t i;

    for (i = 0; i < sizeof want / sizeof want[0]; i++) {
        nv_opnd_t d, a, b;
        const nv_ins_t *r;
        uint32_t n;

        CHEQ(op_fresh(), 0);
        CHEQ(rg_new(RF_F, &d), 0);
        CHEQ(rg_new(RF_F, &a), 0);
        CHEQ(rg_new(RF_F, &b), 0);
        CHEQ(op_emit(want[i].op, d, a, b), 0);
        r = is_recs(&n);
        CHEQ(n, 1u);
        CHEQ(r[0].op, want[i].nv);
        CHEQ(r[0].nsrc, want[i].nsrc);
        CHECK(r[0].dst.kind == OP_REG && r[0].dst.v == d.v);
        CHECK(r[0].src[0].kind == OP_REG && r[0].src[0].v == a.v);
        if (want[i].op == OPN_RELU)
            CHECK(r[0].src[1].kind == OP_FIMM && r[0].src[1].v == 0u);
        else if (want[i].nsrc == 2)
            CHECK(r[0].src[1].kind == OP_REG && r[0].src[1].v == b.v);
    }
    PASS();
}
TH_REG("op", 5, "op_emit appends one record per op, in the .rn / max / min / mov / neg / abs forms", op05)

/* A list with every op, inputs loaded from f32 immediates, passes the verifier. */
static void op06(void)
{
    uint32_t i;
    nv_opnd_t a, b;

    CHEQ(op_fresh(), 0);
    CHEQ(rg_new(RF_F, &a), 0);
    CHEQ(rg_new(RF_F, &b), 0);
    is_ins(NV_MOV_F32, a, is_ofim(0x3F800000u), is_onone(), is_onone());
    is_ins(NV_MOV_F32, b, is_ofim(0x40000000u), is_onone(), is_onone());
    for (i = 0; i < OP_NEXP; i++) {
        nv_opnd_t d;

        CHEQ(rg_new(RF_F, &d), 0);
        CHEQ(op_emit(OP_EXP[i].op, d, a, b), 0);
    }
    is_ins(NV_RET, is_onone(), is_onone(), is_onone(), is_onone());
    CHEQ(op_vf(), 0);
    PASS();
}
TH_REG("op", 6, "a record list holding every op is accepted by vf_run", op06)

/* An op outside the table is our bug: E000, one line, no record. */
static void op07(void)
{
    nv_opnd_t d;
    const nv_ins_t *r;
    uint32_t n;
    int rc;
    size_t len;

    CHEQ(op_fresh(), 0);
    CHEQ(rg_new(RF_F, &d), 0);
    th_cap_on();
    rc = op_emit(OPN_NONE, d, d, d);
    len = th_cap_off(eb, sizeof eb);
    CHEQ(rc, 1);
    CHECK(len > 0 && strncmp(eb, "error[E000]", 11) == 0);
    r = is_recs(&n);
    (void)r;
    CHEQ(n, 0u);
    PASS();
}
TH_REG("op", 7, "op_emit refuses OPN_NONE with E000 and appends nothing", op07)

/* How many records write the f32 register `d`, and whether the last one does. */
static uint32_t op_writes(nv_opnd_t d, int *last)
{
    uint32_t n, i, c = 0;
    const nv_ins_t *r = is_recs(&n);

    *last = 0;
    for (i = 0; i < n; i++) {
        if (r[i].dst.kind != OP_REG || r[i].dst.rf != RF_F || r[i].dst.v != d.v) continue;
        c++;
        *last = i + 1u == n;
    }
    return c;
}

/* How many records have opcode `op`. */
static uint32_t op_cnt(nv_op_t op)
{
    uint32_t n, i, c = 0;
    const nv_ins_t *r = is_recs(&n);

    for (i = 0; i < n; i++) c += r[i].op == op;
    return c;
}

/* op_emit of `op` with dst aliasing its first argument: 0 when dst is written
 * exactly once, by the last record, so the argument is never read stale. */
static int op_alias(opn_t op)
{
    nv_opnd_t d, b;
    int last = 0;

    if (op_fresh() || rg_new(RF_F, &d) || rg_new(RF_F, &b)) return 1;
    if (op_emit(op, d, d, b)) return 1;
    return op_writes(d, &last) == 1u && last ? 0 : 1;
}

/* A composite op is a sequence, and dst is written last so an accumulator's
 * home register can be both the destination and an argument. exp keeps
 * x * log2(e) in two parts (the second an fma by its low half) and calls
 * ex2.approx once, on the reduced argument. */
static void op08(void)
{
    uint32_t n, i, lo = 0;
    const nv_ins_t *r;

    CHEQ(op_alias(OPN_EXP), 0);
    r = is_recs(&n);
    for (i = 0; i < n; i++)
        lo += r[i].op == NV_FMA_RN_F32 && r[i].src[1].kind == OP_FIMM && r[i].src[1].v == 0x32A57060u;
    CHEQ(lo, 1u);
    CHEQ(op_cnt(NV_EX2_APPROX_F32), 1u);
    CHEQ(op_cnt(NV_CVT_RNI_F32_F32), 1u);
    PASS();
}
TH_REG("op", 8, "exp splits x * log2 e by an fma, ex2 sees the reduced argument once, dst written last", op08)

static void op09(void)
{
    uint32_t n;
    const nv_ins_t *r;

    CHEQ(op_alias(OPN_SIGMOID), 0);
    r = is_recs(&n);
    CHEQ(r[n - 1u].op, NV_DIV_RN_F32);
    CHECK(r[n - 1u].src[0].kind == OP_FIMM && r[n - 1u].src[0].v == 0x3F800000u);
    CHEQ(op_cnt(NV_EX2_APPROX_F32), 1u);
    CHEQ(op_alias(OPN_SILU), 0);
    r = is_recs(&n);
    CHEQ(r[n - 1u].op, NV_DIV_RN_F32);
    CHECK(r[n - 1u].src[0].kind == OP_REG && r[n - 1u].src[0].rf == RF_F);
    PASS();
}
TH_REG("op", 9, "sigmoid is 1 / (1 + exp(-x)) and silu x / (1 + exp(-x)) by div.rn, into dst last", op09)

/* pow is not ex2(y * lg2(x)): log2 |x| in two parts, one ex2 on the reduced
 * exponent, and the special cases settled by a final selp of 1.0. lg2.approx
 * appears only for the 0 / inf / NaN inputs. */
static void op10(void)
{
    uint32_t n;
    const nv_ins_t *r;

    CHEQ(op_alias(OPN_POW), 0);
    r = is_recs(&n);
    CHEQ(r[n - 1u].op, NV_SELP_F32);
    CHECK(r[n - 1u].src[0].kind == OP_FIMM && r[n - 1u].src[0].v == 0x3F800000u);
    CHEQ(op_cnt(NV_EX2_APPROX_F32), 1u);
    CHEQ(op_cnt(NV_LG2_APPROX_F32), 1u);
    CHEQ(op_cnt(NV_CVT_RZI_F32_F32), 1u);  /* the integer test on y */
    CHEQ(op_cnt(NV_DIV_RN_F32), 2u);       /* s = (m - 1) / (m + 1), in two parts */
    PASS();
}
TH_REG("op", 10, "pow takes log2 |x| in two parts, settles sign and special cases last", op10)

/* sin and cos: Cody-Waite in three fmas, then one guarded branch around the
 * Payne-Hanek path, whose product is six wide multiplies; no .approx. */
static void op11(void)
{
    uint32_t k;

    for (k = 0; k < 2u; k++) {
        CHEQ(op_alias(k ? OPN_COS : OPN_SIN), 0);
        CHEQ(op_cnt(NV_BRA), 1u);
        CHEQ(op_cnt(NV_LABEL), 1u);
        CHEQ(op_cnt(NV_MUL_WIDE_U32) + op_cnt(NV_MAD_WIDE_U32), 6u);
        CHEQ(op_cnt(NV_SHF_R_CLAMP_B32), 2u);
        CHEQ(op_cnt(NV_EX2_APPROX_F32), 0u);
        CHEQ(op_vf(), 0);                  /* labels and registers check out */
    }
    PASS();
}
TH_REG("op", 11, "sin and cos reduce by three fmas and branch around a Payne-Hanek path", op11)

/* tanh: one exp for the large side, a polynomial for the small one, a selp. */
static void op12(void)
{
    uint32_t n;
    const nv_ins_t *r;

    CHEQ(op_alias(OPN_TANH), 0);
    r = is_recs(&n);
    CHEQ(r[n - 1u].op, NV_SELP_F32);
    CHEQ(op_cnt(NV_EX2_APPROX_F32), 1u);
    CHEQ(op_cnt(NV_COPYSIGN_F32), 1u);
    CHEQ(op_alias(OPN_LOG), 0);
    CHEQ(op_cnt(NV_LG2_APPROX_F32), 1u);   /* only for 0, inf and NaN */
    CHEQ(op_alias(OPN_LOG2), 0);
    CHEQ(op_alias(OPN_SOFTPLUS), 0);
    PASS();
}
TH_REG("op", 12, "tanh, log, log2 and softplus write dst last; tanh picks exp or the polynomial", op12)
