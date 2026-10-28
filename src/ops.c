/* ops.c -- op_*: op enum, identities (our own table), op_emit switch. */

#include <string.h>
#include "irptx.h"
#include "isel.h"
#include "ops.h"
#include "fmath.h"

/* Exact f32 patterns. 1e30f is 0x7149F2CA; its negation flips the sign bit. */
#define OPF_ZERO   0x00000000u
#define OPF_ONE    0x3F800000u
#define OPF_P1E30  0x7149F2CAu
#define OPF_N1E30  0xF149F2CAu
/* ex2 and lg2 are defined in base 2, so the natural ones carry a factor:
 * log2(e) = 1.44269504f, ln(2) = 0.69314718f. */
#define OPF_LOG2E  0x3FB8AA3Bu
#define OPF_LN2    0x3F317218u

/* One row per opn_t, in enum order. `ident_ok` says whether `ident` counts. */
typedef struct {
    const char *name;
    uint8_t     arity;
    uint8_t     ident_ok;
    uint32_t    ident;
} op_row_t;

static const op_row_t OP_TAB[OPN_NCOUNT] = {
    [OPN_NONE]        = {"", 0, 0, 0u},
    [OPN_ADD]         = {"add", 2, 1, OPF_ZERO},
    [OPN_SUBTRACT]    = {"subtract", 2, 0, 0u},
    [OPN_MULTIPLY]    = {"multiply", 2, 1, OPF_ONE},
    [OPN_DIVIDE]      = {"divide", 2, 0, 0u},
    [OPN_TRUE_DIVIDE] = {"true_divide", 2, 0, 0u},
    [OPN_MAXIMUM]     = {"maximum", 2, 1, OPF_N1E30},
    [OPN_MINIMUM]     = {"minimum", 2, 1, OPF_P1E30},
    [OPN_RELU]        = {"relu", 1, 0, 0u},
    [OPN_COPY]        = {"copy", 1, 0, 0u},
    [OPN_NEGATIVE]    = {"negative", 1, 0, 0u},
    [OPN_ABS]         = {"abs", 1, 0, 0u},
    [OPN_SQRT]        = {"sqrt", 1, 0, 0u},
    [OPN_RSQRT]       = {"rsqrt", 1, 0, 0u},
    [OPN_RECIPROCAL]  = {"reciprocal", 1, 0, 0u},
    [OPN_TANH]        = {"tanh", 1, 0, 0u},
    [OPN_SIN]         = {"sin", 1, 0, 0u},
    [OPN_COS]         = {"cos", 1, 0, 0u},
    [OPN_LOG2]        = {"log2", 1, 0, 0u},
    [OPN_EXP]         = {"exp", 1, 0, 0u},
    [OPN_EXP_FAST]    = {"exp_fast", 1, 0, 0u},
    [OPN_LOG]         = {"log", 1, 0, 0u},
    [OPN_SIGMOID]     = {"sigmoid", 1, 0, 0u},
    [OPN_SILU]        = {"silu", 1, 0, 0u},
    [OPN_SOFTPLUS]    = {"softplus", 1, 0, 0u},
    [OPN_POW]         = {"pow", 2, 0, 0u},
};

opn_t op_find(const char *name)
{
    uint32_t i;

    if (name == NULL) return OPN_NONE;
    for (i = 1; i < OPN_NCOUNT; i++)
        if (strcmp(OP_TAB[i].name, name) == 0) return (opn_t)i;
    return OPN_NONE;
}

uint32_t op_arity(opn_t op)
{
    if ((unsigned)op >= OPN_NCOUNT) return 0;
    return OP_TAB[op].arity;
}

int op_ident(opn_t op, uint32_t *bits)
{
    if ((unsigned)op >= OPN_NCOUNT || !OP_TAB[op].ident_ok) return 0;
    *bits = OP_TAB[op].ident;
    return 1;
}

/* The ops that are a sequence rather than one record. exp_fast is the ex2
 * form Emmy renders as __expf. exp, log, log2, tanh, sin, cos and pow are
 * the accurate sequences in fmath.c, because Emmy's reference uses expf,
 * logf, tanhf, sinf, cosf and powf: the .approx forms miss those by far
 * more than 1e-6 (and lg2.approx near 1 is only good to 2^-22 absolute).
 * sigmoid, silu and softplus are built on that exp, softplus in the
 * overflow-free form max(x, 0) + ln(1 + exp(-|x|)) that Emmy decomposes
 * it into. */
static int op_seq(opn_t op, nv_opnd_t dst, nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t none = is_onone(), t, u;

    switch (op) {
    case OPN_EXP:       return fm_exp(dst, a);
    case OPN_EXP_FAST:
        if (rg_new(RF_F, &t)) return 1;
        if (is_ins(NV_MUL_RN_F32, t, a, is_ofim(OPF_LOG2E), none)) return 1;
        return is_ins(NV_EX2_APPROX_F32, dst, t, none, none);
    case OPN_LOG:       return fm_log(dst, a, 1);
    case OPN_LOG2:      return fm_log(dst, a, 0);
    case OPN_SQRT:      return is_ins(NV_SQRT_RN_F32, dst, a, none, none);
    case OPN_RSQRT:     return is_ins(NV_RSQRT_APPROX_F32, dst, a, none, none);  /* what rsqrtf is */
    case OPN_RECIPROCAL: return is_ins(NV_RCP_RN_F32, dst, a, none, none);
    case OPN_TANH:      return fm_tanh(dst, a);
    case OPN_SIN:       return fm_sin(dst, a, 0);
    case OPN_COS:       return fm_sin(dst, a, 1);
    case OPN_POW:       return fm_pow(dst, a, b);
    case OPN_SIGMOID:               /* 1 / (1 + exp(-x)) */
    case OPN_SILU:                  /* x / (1 + exp(-x)) */
        if (rg_new(RF_F, &t) || rg_new(RF_F, &u)) return 1;
        if (is_ins(NV_NEG_F32, t, a, none, none) || fm_exp(u, t)) return 1;
        if (is_ins(NV_ADD_RN_F32, u, u, is_ofim(OPF_ONE), none)) return 1;
        return is_ins(NV_DIV_RN_F32, dst, op == OPN_SILU ? a : is_ofim(OPF_ONE), u, none);
    case OPN_SOFTPLUS:
        if (rg_new(RF_F, &t) || rg_new(RF_F, &u)) return 1;
        if (is_ins(NV_ABS_F32, t, a, none, none) || is_ins(NV_NEG_F32, t, t, none, none)) return 1;
        if (fm_exp(u, t) || is_ins(NV_ADD_RN_F32, u, u, is_ofim(OPF_ONE), none)) return 1;
        if (fm_log(u, u, 1) || is_ins(NV_MAX_F32, t, a, is_ofim(OPF_ZERO), none)) return 1;
        return is_ins(NV_ADD_RN_F32, dst, t, u, none);
    default:
        return er_fail(E000, "op_emit asked for an op that is not in the table");
    }
}

int op_emit(opn_t op, nv_opnd_t dst, nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t none = is_onone();

    switch (op) {
    /* The .rn forms keep ptxas from fusing a mul and an add into an fma, which
     * would round once where Emmy's f32 reference rounds twice (R7). */
    case OPN_ADD:         /* dst = a + b */
        return is_ins(NV_ADD_RN_F32, dst, a, b, none);
    case OPN_SUBTRACT:    /* dst = a - b */
        return is_ins(NV_SUB_RN_F32, dst, a, b, none);
    case OPN_MULTIPLY:    /* dst = a * b */
        return is_ins(NV_MUL_RN_F32, dst, a, b, none);
    case OPN_DIVIDE:      /* dst = a / b, IEEE-rounded (not div.full) */
    case OPN_TRUE_DIVIDE: /* numpy true_divide is the same a / b on floats */
        return is_ins(NV_DIV_RN_F32, dst, a, b, none);
    case OPN_MAXIMUM:     /* dst = a if a >= b else b (NaN-ignoring per PTX max.f32) */
        return is_ins(NV_MAX_F32, dst, a, b, none);
    case OPN_MINIMUM:     /* dst = a if a <= b else b */
        return is_ins(NV_MIN_F32, dst, a, b, none);
    case OPN_RELU:        /* dst = max(a, 0) */
        return is_ins(NV_MAX_F32, dst, a, is_ofim(OPF_ZERO), none);
    case OPN_COPY:        /* dst = a */
        return is_ins(NV_MOV_F32, dst, a, none, none);
    case OPN_NEGATIVE:    /* dst = -a */
        return is_ins(NV_NEG_F32, dst, a, none, none);
    case OPN_ABS:         /* dst = |a| */
        return is_ins(NV_ABS_F32, dst, a, none, none);
    default:
        return op_seq(op, dst, a, b);
    }
}
