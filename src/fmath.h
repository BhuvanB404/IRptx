/* fmath.h -- fm_*: f32 maths that has to match the device library's
 * accuracy, not just its shape.
 *
 * Emmy's reference kernels call powf, the non-fast-math version, and
 * ex2(y*lg2(x)) is NaN for a negative x and loses low bits whenever the
 * result is large, so pow is a record sequence instead: log2 |x| in two f32
 * parts, the exponent product kept in two parts, and ex2.approx only on the
 * reduced argument, where that instruction is good to 2 ulp. The helper
 * writes `dst` last, so `dst` may alias an argument (an accumulator's home
 * register). It returns 0, or 1 after printing E081/E083 when a pool is full. */

#ifndef FMATH_H
#define FMATH_H

#include "irptx.h"

/* dst = x^y with powf's special cases (a negative base to an integer power,
 * pow(x, 0) = 1, pow(1, y) = 1, the zeros and infinities of C99 F.9.4.4). */
int fm_pow(nv_opnd_t dst, nv_opnd_t x, nv_opnd_t y);

#endif /* FMATH_H */
