/* fmath.c -- fm_*: pow to the device library's accuracy.
 *
 * Constants are f32 bit patterns, each with the value it stands for. Split
 * constants (a _HI and a _LO) carry one value in two floats: hi is the value
 * rounded to f32, lo is the rest rounded to f32, so hi + lo is good to about
 * 2^-48 relative. Polynomial coefficients were fitted for minimum relative
 * error over the reduced interval named beside them, then rounded to f32;
 * the fit errors quoted are before that rounding. */

#include <stdint.h>
#include "irptx.h"
#include "isel.h"
#include "fmath.h"

#define FM_ONE     0x3F800000u      /* 1.0 */
#define FM_INF     0x7F800000u      /* +inf */
#define FM_QNAN    0x7FFFFFFFu      /* a quiet NaN */
#define FM_P252    0x437C0000u      /* 252.0: 2^252 overflows f32 whatever ex2 gives */
#define FM_N252    0xC37C0000u      /* -252.0: 2^-252 underflows to 0 */
#define FM_2P23    0x4B000000u      /* 2^23, lifts a subnormal into the normal range */
#define FM_N23     0xC1B80000u      /* -23.0, the exponent that lift adds */
#define FM_2P24    0x4B800000u      /* 2^24: every f32 at or above it is an even integer */
#define FM_MINN    0x00800000u      /* 2^-126, the smallest normal */
#define FM_K_HI    0x4038AA3Bu      /* 2/ln(2), the atanh series' factor for log2 */
#define FM_K_LO    0x33257060u

/* atanh(s) / s = 1 + s^2 * (1/3 + s^2/5 + ...): the series, which for
 * |s| <= 3 - 2*sqrt(2) leaves out less than 2^-34 of the total. */
static const uint32_t FM_AT[5] = { 0x3EAAAAABu, 0x3E4CCCCDu, 0x3E124925u, 0x3DE38E39u, 0x3DBA2E8Cu };
/* ---- Helpers ---- */

#define FN is_onone()
#define FF(b) is_ofim(b)
#define FI(v) is_oimm(v)

/* A fresh register of file `rf` in *d, then `op d, a, b, c`. */
static int fm_op(nv_op_t op, rf_t rf, nv_opnd_t *d, nv_opnd_t a, nv_opnd_t b, nv_opnd_t c)
{
    if (rg_new(rf, d)) return 1;
    return is_ins(op, *d, a, b, c);
}

#define FM1(op, rf, d, a)       fm_op(op, rf, d, a, FN, FN)
#define FM2(op, rf, d, a, b)    fm_op(op, rf, d, a, b, FN)
#define FM3(op, rf, d, a, b, c) fm_op(op, rf, d, a, b, c)

/* Horner: *d = c[n-1], then *d = *d * z + c[i] down to c[0]. */
static int fm_horn(nv_opnd_t *d, nv_opnd_t z, const uint32_t *c, uint32_t n)
{
    uint32_t i = n - 1u;

    if (FM3(NV_FMA_RN_F32, RF_F, d, z, FF(c[i]), FF(c[i - 1u]))) return 1;
    for (i = n - 2u; i > 0; i--)
        if (is_ins(NV_FMA_RN_F32, *d, z, *d, FF(c[i - 1u]))) return 1;
    return 0;
}

/* *s = 2^k as an f32, for an s32 k in -126..127: (k + 127) << 23. */
static int fm_scl(nv_opnd_t k, nv_opnd_t *s)
{
    nv_opnd_t b;

    if (FM2(NV_SHL_B32, RF_R, &b, k, FI(23))) return 1;
    if (is_ins(NV_ADD_S32, b, b, FI((int32_t)FM_ONE), FN)) return 1;
    return FM1(NV_MOV_B32, RF_F, s, b);
}

/* ---- Exponentials ---- */

/* dst = 2^(th + tl), where tl is the small part of the exponent. j is th to
 * the nearest integer, clamped so its scale stays a product of two normal
 * powers of two; ex2.approx then sees only f = (th - j) + tl, |f| <= 0.5 in
 * range, where it is good to 2 ulp. Out of range f is large and ex2 itself
 * returns 0 or inf. When th is not finite tl is NaN or junk and is dropped.
 * The split scale keeps a subnormal result to one rounding. */
static int fm_exp2(nv_opnd_t dst, nv_opnd_t th, nv_opnd_t tl)
{
    nv_opnd_t j, a, p, l, f, e, ji, j1, j2, s1, s2;

    if (FM1(NV_CVT_RNI_F32_F32, RF_F, &j, th)) return 1;
    if (is_ins(NV_MAX_F32, j, j, FF(FM_N252), FN) || is_ins(NV_MIN_F32, j, j, FF(FM_P252), FN))
        return 1;
    if (FM1(NV_ABS_F32, RF_F, &a, th) || FM2(NV_SETP_LT_F32, RF_P, &p, a, FF(FM_INF))) return 1;
    if (FM3(NV_SELP_F32, RF_F, &l, tl, FF(0), p)) return 1;
    if (FM2(NV_SUB_RN_F32, RF_F, &f, th, j) || is_ins(NV_ADD_RN_F32, f, f, l, FN)) return 1;
    if (FM1(NV_EX2_APPROX_F32, RF_F, &e, f)) return 1;
    if (FM1(NV_CVT_RZI_S32_F32, RF_R, &ji, j) || FM2(NV_SHR_S32, RF_R, &j1, ji, FI(1))) return 1;
    if (FM2(NV_SUB_S32, RF_R, &j2, ji, j1)) return 1;
    if (fm_scl(j1, &s1) || fm_scl(j2, &s2)) return 1;
    if (is_ins(NV_MUL_RN_F32, e, e, s1, FN)) return 1;
    return is_ins(NV_MUL_RN_F32, dst, e, s2, FN);
}

/* ---- Power ---- */

/* |x| = 2^ef * m with m in [sqrt(1/2), sqrt(2)): a subnormal is lifted by
 * 2^23 first, and the exponent is taken so that m lands either side of 1. */
static int fm_lgm(nv_opnd_t a, nv_opnd_t *ef, nv_opnd_t *m)
{
    nv_opnd_t ps, as, a2, ea, ia, e, mb;

    if (FM2(NV_SETP_LT_F32, RF_P, &ps, a, FF(FM_MINN))) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &as, a, FF(FM_2P23))) return 1;
    if (FM3(NV_SELP_F32, RF_F, &a2, as, a, ps) || FM3(NV_SELP_F32, RF_F, &ea, FF(FM_N23), FF(0), ps))
        return 1;
    if (FM1(NV_MOV_B32, RF_R, &ia, a2)) return 1;
    if (FM2(NV_ADD_S32, RF_R, &e, ia, FI(-0x3F3504F3)) || is_ins(NV_SHR_S32, e, e, FI(23), FN)) return 1;
    if (FM2(NV_SHL_B32, RF_R, &mb, e, FI(23)) || is_ins(NV_SUB_S32, mb, ia, mb, FN)) return 1;
    if (FM1(NV_MOV_B32, RF_F, m, mb)) return 1;
    if (FM1(NV_CVT_RN_F32_S32, RF_F, ef, e)) return 1;
    return is_ins(NV_ADD_RN_F32, *ef, *ef, ea, FN);
}

/* log2 |x| as lh + ll, good to about 2^-40 relative: ef + (2/ln 2) *
 * atanh(s) with s = (m - 1) / (m + 1) carried as sh + sl. m - 1 is exact,
 * m + 1 is g + gl, and the leading product K * s is kept in two parts; the
 * series tail is under 1% of the total, so f32 is enough for it. 0, inf and
 * NaN take lg2.approx's own -inf, inf and NaN. */
static int fm_lg2d(nv_opnd_t a, nv_opnd_t *lh, nv_opnd_t *ll)
{
    nv_opnd_t ef, m, f, g, gl, sh, ns, rr, sl, z, pp, t, ph, n, pl, d, pf, pi, lg;

    if (fm_lgm(a, &ef, &m)) return 1;
    if (FM2(NV_SUB_RN_F32, RF_F, &f, m, FF(FM_ONE)) || FM2(NV_ADD_RN_F32, RF_F, &g, m, FF(FM_ONE)))
        return 1;
    if (FM2(NV_SUB_RN_F32, RF_F, &gl, g, FF(FM_ONE)) || is_ins(NV_SUB_RN_F32, gl, m, gl, FN)) return 1;
    if (FM2(NV_DIV_RN_F32, RF_F, &sh, f, g) || FM1(NV_NEG_F32, RF_F, &ns, sh)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &rr, ns, g, f) || is_ins(NV_FMA_RN_F32, rr, ns, gl, rr)) return 1;
    if (FM2(NV_DIV_RN_F32, RF_F, &sl, rr, g)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &z, sh, sh) || fm_horn(&pp, z, FM_AT, 5u)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &t, z, sh) || is_ins(NV_MUL_RN_F32, t, t, pp, FN)) return 1;
    if (is_ins(NV_MUL_RN_F32, t, t, FF(FM_K_HI), FN)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &ph, sh, FF(FM_K_HI)) || FM1(NV_NEG_F32, RF_F, &n, ph)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &pl, sh, FF(FM_K_HI), n)) return 1;
    if (is_ins(NV_FMA_RN_F32, pl, sl, FF(FM_K_HI), pl)) return 1;
    if (is_ins(NV_FMA_RN_F32, pl, sh, FF(FM_K_LO), pl) || is_ins(NV_ADD_RN_F32, pl, pl, t, FN)) return 1;
    /* ef + ph by Fast2Sum: |ef| >= 1 > |ph| unless ef is 0, and then it is exact */
    if (FM2(NV_ADD_RN_F32, RF_F, lh, ef, ph) || FM2(NV_SUB_RN_F32, RF_F, &d, ef, *lh)) return 1;
    if (is_ins(NV_ADD_RN_F32, d, d, ph, FN) || is_ins(NV_ADD_RN_F32, d, d, pl, FN)) return 1;
    if (FM2(NV_SETP_GT_F32, RF_P, &pf, a, FF(0)) || FM2(NV_SETP_LT_F32, RF_P, &pi, a, FF(FM_INF)))
        return 1;
    if (is_ins(NV_AND_PRED, pf, pf, pi, FN) || FM1(NV_LG2_APPROX_F32, RF_F, &lg, a)) return 1;
    if (is_ins(NV_SELP_F32, *lh, *lh, lg, pf)) return 1;
    return FM3(NV_SELP_F32, RF_F, ll, d, FF(0), pf);
}

/* The sign and the special cases on top of |x|^y in res. y is an odd
 * integer when it truncates to itself, is under 2^24 and its low bit is
 * set; then a negative x (or -0) negates. A finite x below 0 to a power
 * that is not an integer is NaN. The answer is 1 for y = 0, for x = 1, and
 * for x = -1 to an infinite power, whatever NaN the rest made. */
static int fm_pfix(nv_opnd_t dst, nv_opnd_t res, nv_opnd_t x, nv_opnd_t a, nv_opnd_t y)
{
    nv_opnd_t yi, pint, ya, psm, yn, od, podd, ix, pn, nr, pxn, pf, p1, p2, p3, p4;

    if (FM1(NV_CVT_RZI_F32_F32, RF_F, &yi, y) || FM2(NV_SETP_EQ_F32, RF_P, &pint, yi, y)) return 1;
    if (FM1(NV_ABS_F32, RF_F, &ya, y) || FM2(NV_SETP_LT_F32, RF_P, &psm, ya, FF(FM_2P24))) return 1;
    if (FM1(NV_CVT_RZI_S32_F32, RF_R, &yn, y) || FM2(NV_AND_B32, RF_R, &od, yn, FI(1))) return 1;
    if (FM2(NV_SETP_NE_S32, RF_P, &podd, od, FI(0))) return 1;
    if (is_ins(NV_AND_PRED, podd, podd, pint, FN) || is_ins(NV_AND_PRED, podd, podd, psm, FN)) return 1;
    if (FM1(NV_MOV_B32, RF_R, &ix, x) || FM2(NV_SETP_LT_S32, RF_P, &pn, ix, FI(0))) return 1;
    if (is_ins(NV_AND_PRED, podd, podd, pn, FN) || FM1(NV_NEG_F32, RF_F, &nr, res)) return 1;
    if (is_ins(NV_SELP_F32, res, nr, res, podd)) return 1;
    if (FM2(NV_SETP_LT_F32, RF_P, &pxn, x, FF(0)) || FM2(NV_SETP_LT_F32, RF_P, &pf, a, FF(FM_INF)))
        return 1;
    if (is_ins(NV_AND_PRED, pxn, pxn, pf, FN) || is_ins(NV_NOT_PRED, pint, pint, FN, FN)) return 1;
    if (is_ins(NV_AND_PRED, pxn, pxn, pint, FN)) return 1;
    if (is_ins(NV_SELP_F32, res, FF(FM_QNAN), res, pxn)) return 1;
    if (FM2(NV_SETP_EQ_F32, RF_P, &p1, y, FF(0)) || FM2(NV_SETP_EQ_F32, RF_P, &p2, x, FF(FM_ONE)))
        return 1;
    if (FM2(NV_SETP_EQ_F32, RF_P, &p3, a, FF(FM_ONE)) || FM2(NV_SETP_EQ_F32, RF_P, &p4, ya, FF(FM_INF)))
        return 1;
    if (is_ins(NV_AND_PRED, p3, p3, p4, FN) || is_ins(NV_OR_PRED, p1, p1, p2, FN)) return 1;
    if (is_ins(NV_OR_PRED, p1, p1, p3, FN)) return 1;
    return is_ins(NV_SELP_F32, dst, FF(FM_ONE), res, p1);
}

/* x^y = 2^(y * log2 |x|), the product carried in two parts so a large
 * result keeps its low bits, then the sign and the special cases. */
int fm_pow(nv_opnd_t dst, nv_opnd_t x, nv_opnd_t y)
{
    nv_opnd_t a, lh, ll, th, n, tl, res;

    if (FM1(NV_ABS_F32, RF_F, &a, x) || fm_lg2d(a, &lh, &ll)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &th, y, lh) || FM1(NV_NEG_F32, RF_F, &n, th)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &tl, y, lh, n) || is_ins(NV_FMA_RN_F32, tl, y, ll, tl)) return 1;
    if (rg_new(RF_F, &res) || fm_exp2(res, th, tl)) return 1;
    return fm_pfix(dst, res, x, a, y);
}
