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
 * log2(e) = 1.44269504f, -log2(e) for a sigmoid's exp(-x), ln(2) = 0.69314718f. */
#define OPF_LOG2E  0x3FB8AA3Bu
#define OPF_NLOG2E 0xBFB8AA3Bu
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

/* The ops that are a sequence rather than one record. Each is written out in
 * plain maths, ex2/lg2 being the base-2 pair PTX has: exp(x) = ex2(x*log2e),
 * ln(x) = lg2(x)*ln2, sigmoid(x) = 1/(1+exp(-x)), softplus = ln(1+exp(x)),
 * siLU = x*sigmoid(x). Both ex2 and lg2 are .approx, which is what the
 * device's own ex2 does anyway. pow is fmath.c's powf. */
static int op_seq(opn_t op, nv_opnd_t dst, nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t none = is_onone(), t, u, v;

    switch (op) {
    case OPN_EXP:
    case OPN_EXP_FAST:              /* exp_fast is the ex2 form, not a slower one */
        if (rg_new(RF_F, &t)) return 1;
        if (is_ins(NV_MUL_RN_F32, t, a, is_ofim(OPF_LOG2E), none)) return 1;
        return is_ins(NV_EX2_APPROX_F32, dst, t, none, none);
    case OPN_LOG:
        if (rg_new(RF_F, &t)) return 1;
        if (is_ins(NV_LG2_APPROX_F32, t, a, none, none)) return 1;
        return is_ins(NV_MUL_RN_F32, dst, t, is_ofim(OPF_LN2), none);
    case OPN_LOG2:      return is_ins(NV_LG2_APPROX_F32, dst, a, none, none);
    case OPN_SQRT:      return is_ins(NV_SQRT_RN_F32, dst, a, none, none);
    case OPN_RSQRT:     return is_ins(NV_RSQRT_APPROX_F32, dst, a, none, none);
    case OPN_RECIPROCAL: return is_ins(NV_RCP_RN_F32, dst, a, none, none);
    case OPN_TANH:      return is_ins(NV_TANH_APPROX_F32, dst, a, none, none);
    case OPN_SIN:       return is_ins(NV_SIN_APPROX_F32, dst, a, none, none);
    case OPN_COS:       return is_ins(NV_COS_APPROX_F32, dst, a, none, none);
    case OPN_SIGMOID:
        if (rg_new(RF_F, &t) || rg_new(RF_F, &u)) return 1;
        if (is_ins(NV_MUL_RN_F32, t, a, is_ofim(OPF_NLOG2E), none)) return 1;
        if (is_ins(NV_EX2_APPROX_F32, u, t, none, none)) return 1;
        if (is_ins(NV_ADD_RN_F32, t, u, is_ofim(OPF_ONE), none)) return 1;
        return is_ins(NV_RCP_RN_F32, dst, t, none, none);
    case OPN_SILU:                  /* x * sigmoid(x), so the sigmoid needs its own temp */
        if (rg_new(RF_F, &t) || rg_new(RF_F, &u) || rg_new(RF_F, &v)) return 1;
        if (is_ins(NV_MUL_RN_F32, t, a, is_ofim(OPF_NLOG2E), none)) return 1;
        if (is_ins(NV_EX2_APPROX_F32, u, t, none, none)) return 1;
        if (is_ins(NV_ADD_RN_F32, u, u, is_ofim(OPF_ONE), none)) return 1;
        if (is_ins(NV_RCP_RN_F32, v, u, none, none)) return 1;
        return is_ins(NV_MUL_RN_F32, dst, a, v, none);
    case OPN_SOFTPLUS:
        if (rg_new(RF_F, &t) || rg_new(RF_F, &u) || rg_new(RF_F, &v)) return 1;
        if (is_ins(NV_MUL_RN_F32, t, a, is_ofim(OPF_LOG2E), none)) return 1;
        if (is_ins(NV_EX2_APPROX_F32, u, t, none, none)) return 1;
        if (is_ins(NV_ADD_RN_F32, v, u, is_ofim(OPF_ONE), none)) return 1;
        if (is_ins(NV_LG2_APPROX_F32, t, v, none, none)) return 1;
        return is_ins(NV_MUL_RN_F32, dst, t, is_ofim(OPF_LN2), none);
    case OPN_POW:       return fm_pow(dst, a, b);   /* powf, not ex2(y * lg2 x) */
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
