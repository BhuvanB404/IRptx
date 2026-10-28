/* fmath.h -- fm_*: f32 maths that has to match the device library's
 * accuracy, not just its shape.
 *
 * Emmy's reference kernels call expf, logf, tanhf, sinf, cosf and powf, the
 * non-fast-math versions. The PTX .approx forms are much coarser than those
 * (tanh.approx is good to 2^-11, sin.approx loses accuracy as soon as the
 * argument leaves a few periods, and ex2(y*lg2(x)) is NaN for a negative x),
 * so each of these is a short record sequence instead: argument reduction
 * done carefully in f32, then a polynomial or an ex2.approx on a reduced
 * argument where that instruction is good to 2 ulp. Every helper writes `dst`
 * last, so `dst` may alias an argument (an accumulator's home register).
 * Each returns 0, or 1 after printing E081/E083 when a pool is full. */

#ifndef FMATH_H
#define FMATH_H

#include "irptx.h"

/* dst = e^x, within a few ulp over the whole f32 range (0 below, inf above). */
int fm_exp(nv_opnd_t dst, nv_opnd_t x);

/* dst = log2(x), or ln(x) when `natural` is set; NaN below 0, -inf at 0. */
int fm_log(nv_opnd_t dst, nv_opnd_t x, int natural);

/* dst = tanh(x). */
int fm_tanh(nv_opnd_t dst, nv_opnd_t x);

/* dst = sin(x), or cos(x) when `cosine` is set, for every f32 x. */
int fm_sin(nv_opnd_t dst, nv_opnd_t x, int cosine);

/* dst = x^y with powf's special cases (a negative base to an integer power,
 * pow(x, 0) = 1, pow(1, y) = 1, the zeros and infinities of C99 F.9.4.4). */
int fm_pow(nv_opnd_t dst, nv_opnd_t x, nv_opnd_t y);

#endif /* FMATH_H */
