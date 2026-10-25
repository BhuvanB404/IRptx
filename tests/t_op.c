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
    {"relu", OPN_RELU, 1}, {"copy", OPN_COPY, 1}, {"negative", OPN_NEGATIVE, 1}, {"abs", OPN_ABS, 1}
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
