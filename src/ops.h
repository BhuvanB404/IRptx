/* ops.h -- the op table: which elementwise ops we lower, their arity, the
 * identity of the reducing ones, and the PTX they become.
 *
 * Names are Emmy's own ElementwiseImpl names (ir/elementwise.py, checked with
 * ElementwiseImpl(name)); the identities are copied by hand from its
 * _IDENTITY table so nothing about an op's meaning crosses the KIR boundary.
 * `sub` is in Emmy's cluster table but ElementwiseImpl("sub") raises, so Emmy
 * can never emit it and it is not here. */

#ifndef OPS_H
#define OPS_H

#include "irptx.h"

typedef enum {
    OPN_NONE = 0,                   /* not found; also the "no reducing op" value */
    /* one record each */
    OPN_ADD, OPN_SUBTRACT, OPN_MULTIPLY, OPN_DIVIDE, OPN_TRUE_DIVIDE,
    OPN_MAXIMUM, OPN_MINIMUM, OPN_RELU, OPN_COPY, OPN_NEGATIVE, OPN_ABS,
    OPN_SQRT, OPN_RSQRT, OPN_RECIPROCAL, OPN_TANH, OPN_SIN, OPN_COS, OPN_LOG2,
    /* a short sequence each (op_seq) */
    OPN_EXP, OPN_EXP_FAST, OPN_LOG, OPN_SIGMOID, OPN_SILU, OPN_SOFTPLUS, OPN_POW,
    OPN_NCOUNT                      /* count, keep last */
} opn_t;

/* The op called `name` (the text of a KIR op:<name>), or OPN_NONE. Prints
 * nothing: the caller raises E021 with its own context. */
opn_t op_find(const char *name);

/* Operands the op takes (1 or 2); 0 for OPN_NONE or an out-of-range value. */
uint32_t op_arity(opn_t op);

/* The f32 bit pattern of the op's reducing identity into *bits and return 1,
 * or return 0 when it has none. add 0, multiply 1, maximum -1e30, minimum
 * +1e30; never an infinity. */
int op_ident(opn_t op, uint32_t *bits);

/* Append the records for `dst = op(a)` or `dst = op(a, b)` (b is ignored by a
 * unary op). Most ops are one record; exp, log, sigmoid, silu, softplus and
 * pow are a short sequence into fresh temporaries, with dst written last so a
 * dst that aliases a or b (an accumulator's home register) is safe. 0, E083
 * when the record list is full, E000 for OPN_NONE. */
int op_emit(opn_t op, nv_opnd_t dst, nv_opnd_t a, nv_opnd_t b);

#endif /* OPS_H */
