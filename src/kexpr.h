/* kexpr.h -- index and predicate expressions: lowering, folding, affine proof.
 *
 * An expression arrives as a KIR value (a KV_NODE for Var, Literal, Dim,
 * BinaryExpr or TernaryExpr, or a bare KV_INT as in `Dim expr=64`). Nothing
 * here recurses: the tree is flattened to postfix tokens with an explicit
 * stack, then each entry point walks the tokens with another explicit stack,
 * both bounded by IR_MAX_DEP. */

#ifndef KEXPR_H
#define KEXPR_H

#include "irptx.h"

/* Lower `e` to PTX records and return where the value lives: an s32 register
 * (RF_R), a predicate (RF_P, for comparisons and && ||), an f32 register or
 * immediate, or an s32 immediate when every leaf was a literal. Names are
 * looked up with is_find. Returns 0, or 1 after printing one error:
 * E020 unknown node kind, E021 unknown op, E022 unbound name, E015 missing
 * field, E082 nesting past IR_MAX_DEP, E059 anything else unsupported. */
int kx_eval(const kir_val_t *e, nv_opnd_t *out);

/* Constant folding. Sets *isc to 1 and *val to the s32 result when every leaf
 * is an int literal (or a Dim of one), else *isc to 0. Arithmetic wraps like
 * PTX s32; / and % truncate toward zero (-7 / 2 == -3, -7 % 2 == -1) and a
 * shift count is unsigned and clamped at 32, exactly as the device does.
 * Division or remainder by zero and INT_MIN / -1 refuse with E059: 1 is
 * returned after printing. 0 otherwise. */
int kx_fold(const kir_val_t *e, int32_t *val, int *isc);

/* Affine form (research R6): sum(coef*var) + cst from Var, int Literal, Dim,
 * + and - and * by a constant. Returns 1 and fills *out when affine, 0 when
 * not (a / or %, a comparison, a ternary, a float, more than IR_MAX_AFF
 * variables, or anything malformed). Not affine is not an error: nothing is
 * printed. */
int kx_aff(const kir_val_t *e, kx_aff_t *out);

#endif /* KEXPR_H */
