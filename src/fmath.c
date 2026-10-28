/* fmath.c -- fm_*: exp, tanh, sin, cos and pow to the device library's accuracy.
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
#define FM_TWO_N   0xC0000000u      /* -2.0 */
#define FM_HALF_N  0xBF000000u      /* -0.5 */
#define FM_INF     0x7F800000u      /* +inf */
#define FM_QNAN    0x7FFFFFFFu      /* a quiet NaN */
#define FM_P252    0x437C0000u      /* 252.0: 2^252 overflows f32 whatever ex2 gives */
#define FM_N252    0xC37C0000u      /* -252.0: 2^-252 underflows to 0 */
#define FM_2P23    0x4B000000u      /* 2^23, lifts a subnormal into the normal range */
#define FM_N23     0xC1B80000u      /* -23.0, the exponent that lift adds */
#define FM_2P24    0x4B800000u      /* 2^24: every f32 at or above it is an even integer */
#define FM_MINN    0x00800000u      /* 2^-126, the smallest normal */
#define FM_L2E_HI  0x3FB8AA3Bu      /* log2(e) */
#define FM_L2E_LO  0x32A57060u
#define FM_K_HI    0x4038AA3Bu      /* 2/ln(2), the atanh series' factor for log2 */
#define FM_K_LO    0x33257060u
#define FM_2DPI    0x3F22F983u      /* 2/pi */
#define FM_P1_N    0xBFC90FDAu      /* -(pi/2) as three parts: pi/2 rounded down to f32, */
#define FM_P2_N    0xB3A22169u      /* the rest rounded, */
#define FM_P3_N    0x26F72CEDu      /* and what that leaves, rounded */
#define FM_CW_MAX  0x47CE4780u      /* 105615.0: past it the three parts lose bits */
#define FM_PH_HI   0x1FC90FDBu      /* pi/2 * 2^-64, the weight of the 64-bit fraction */
#define FM_PH_LO   0x933BBD2Eu
#define FM_LN2_HI  0x3F317218u      /* ln(2) */
#define FM_LN2_LO  0xB102E308u
#define FM_TH_SW   0x3F0CCCCDu      /* 0.55: tanh switches from the polynomial to exp */

/* tanh(x) = x + x^3 * Q(x^2) on [0, 0.55], max relative error 3.5e-9. */
static const uint32_t FM_TQ[5] = { 0xBEAAAAA8u, 0x3E08874Eu, 0xBD5CB14Au, 0x3CAD8248u, 0xBBD2688Au };
/* sin(r) = r + r^3 * S(r^2) on [0, pi/4], 8.3e-9. */
static const uint32_t FM_SS[3] = { 0xBE2AAAA6u, 0x3C0884A7u, 0xB94CEE71u };
/* cos(r) = 1 - r^2/2 + r^4 * C(r^2) on [0, pi/4], 1.2e-10. */
static const uint32_t FM_SC[3] = { 0x3D2AAAA5u, 0xBAB6061Au, 0x37CCF5CEu };
/* atanh(s) / s = 1 + s^2 * (1/3 + s^2/5 + ...): the series, which for
 * |s| <= 3 - 2*sqrt(2) leaves out less than 2^-34 of the total. */
static const uint32_t FM_AT[5] = { 0x3EAAAAABu, 0x3E4CCCCDu, 0x3E124925u, 0x3DE38E39u, 0x3DBA2E8Cu };
/* 2/pi as six 32-bit words, most significant first: bits 1..192 of the fraction. */
static const uint32_t FM_2PI[6] = {
    0xA2F9836Eu, 0x4E441529u, 0xFC2757D1u, 0xF534DDC0u, 0xDB629599u, 0x3C439041u
};

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

/* A 32-bit immediate given as its bit pattern (the words of 2/pi). */
static nv_opnd_t fm_u(uint32_t bits)
{
    nv_opnd_t o = is_oimm(0);

    o.v = bits;
    return o;
}

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

/* e^x = 2^(x * log2 e), the product kept as th + tl by an fma. */
int fm_exp(nv_opnd_t dst, nv_opnd_t x)
{
    nv_opnd_t th, n, tl;

    if (FM2(NV_MUL_RN_F32, RF_F, &th, x, FF(FM_L2E_HI))) return 1;
    if (FM1(NV_NEG_F32, RF_F, &n, th)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &tl, x, FF(FM_L2E_HI), n)) return 1;
    if (is_ins(NV_FMA_RN_F32, tl, x, FF(FM_L2E_LO), tl)) return 1;
    return fm_exp2(dst, th, tl);
}

/* tanh: below 0.55 the odd polynomial; from there 1 - 2 / (e^(2|x|) + 1)
 * with x's sign, which is 1 once e^(2|x|) passes 2^25 or overflows. NaN
 * fails the switch test and goes through the polynomial, staying NaN. */
int fm_tanh(nv_opnd_t dst, nv_opnd_t x)
{
    nv_opnd_t a, t, e, d, r, b, z, q, c, s, p;

    if (FM1(NV_ABS_F32, RF_F, &a, x) || FM2(NV_ADD_RN_F32, RF_F, &t, a, a)) return 1;
    if (rg_new(RF_F, &e) || fm_exp(e, t)) return 1;
    if (FM2(NV_ADD_RN_F32, RF_F, &d, e, FF(FM_ONE)) || FM1(NV_RCP_RN_F32, RF_F, &r, d)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &b, r, FF(FM_TWO_N), FF(FM_ONE))) return 1;
    if (is_ins(NV_COPYSIGN_F32, b, x, b, FN)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &z, x, x) || fm_horn(&q, z, FM_TQ, 5u)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &c, z, x) || FM3(NV_FMA_RN_F32, RF_F, &s, c, q, x)) return 1;
    if (FM2(NV_SETP_GE_F32, RF_P, &p, a, FF(FM_TH_SW))) return 1;
    return is_ins(NV_SELP_F32, dst, b, s, p);
}

/* ---- Sine and cosine ---- */

/* The 224-bit product of the 32-bit significand m and the 192 bits of 2/pi,
 * as words w[0] (lowest) .. w[6], by one carry chain of mad.wide.u32. */
static int fm_phw(nv_opnd_t m, nv_opnd_t *w)
{
    nv_opnd_t t, c = is_onone();
    uint32_t i;

    for (i = 0; i < 6u; i++) {
        if (i == 0) {
            if (FM2(NV_MUL_WIDE_U32, RF_RD, &t, m, fm_u(FM_2PI[5]))) return 1;
        } else if (FM3(NV_MAD_WIDE_U32, RF_RD, &t, m, fm_u(FM_2PI[5u - i]), c)) {
            return 1;
        }
        if (FM1(NV_CVT_U32_U64, RF_R, &w[i], t) || FM2(NV_SHR_U64, RF_RD, &c, t, FI(32))) return 1;
    }
    return FM1(NV_CVT_U32_U64, RF_R, &w[6], c);
}

/* *o = w[k + j] for the word index j in 1..4 that p[0..2] (j == 2, 3, 4) say. */
static int fm_phs(const nv_opnd_t *w, const nv_opnd_t *p, uint32_t k, nv_opnd_t *o)
{
    if (FM3(NV_SELP_S32, RF_R, o, w[k + 2u], w[k + 1u], p[0])) return 1;
    if (is_ins(NV_SELP_S32, *o, w[k + 3u], *o, p[1])) return 1;
    return is_ins(NV_SELP_S32, *o, w[k + 4u], *o, p[2]);
}

/* From the 64-bit window hi:lo of x * 2/pi (2 integer bits, 62 fraction
 * bits): the quadrant into q and the remainder times pi/2 into r. A
 * fraction of one half or more rounds the quadrant up and is read as the
 * negative fraction it then is, which the s64 view of the shifted window
 * gives directly. Both flip with the sign of x (ix is its bit pattern). */
static int fm_phf(nv_opnd_t hi, nv_opnd_t lo, nv_opnd_t ix, nv_opnd_t r, nv_opnd_t q)
{
    nv_opnd_t fh, fl, qq, rb, a, b, rf, rr, pn, nr, nq;

    if (FM3(NV_SHF_L_CLAMP_B32, RF_R, &fh, lo, hi, FI(2)) || FM2(NV_SHL_B32, RF_R, &fl, lo, FI(2)))
        return 1;
    if (FM2(NV_SHR_U32, RF_R, &qq, hi, FI(30)) || FM2(NV_SHR_U32, RF_R, &rb, fh, FI(31))) return 1;
    if (is_ins(NV_ADD_S32, qq, qq, rb, FN)) return 1;
    if (FM1(NV_CVT_U64_U32, RF_RD, &a, fh) || is_ins(NV_SHL_B64, a, a, FI(32), FN)) return 1;
    if (FM1(NV_CVT_U64_U32, RF_RD, &b, fl) || is_ins(NV_OR_B64, a, a, b, FN)) return 1;
    if (FM1(NV_CVT_RN_F32_S64, RF_F, &rf, a)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &rr, rf, FF(FM_PH_HI))) return 1;
    if (is_ins(NV_FMA_RN_F32, rr, rf, FF(FM_PH_LO), rr)) return 1;
    if (FM2(NV_SETP_LT_S32, RF_P, &pn, ix, FI(0))) return 1;
    if (FM1(NV_NEG_F32, RF_F, &nr, rr) || is_ins(NV_SELP_F32, r, nr, rr, pn)) return 1;
    if (FM1(NV_NEG_S32, RF_R, &nq, qq)) return 1;
    return is_ins(NV_SELP_S32, q, nq, qq, pn);
}

/* Payne-Hanek reduction for a finite |x| >= 105615, so its biased exponent
 * e is 143..254. With m the significand as a 32-bit integer (top bit set),
 * x * 2/pi = P * 2^(e - 350) for P = m * (2/pi * 2^192), so the two integer
 * bits and 62 fraction bits that matter are bits b .. b + 63 of P, with
 * b = 288 - e in 34..145: words b/32 .. b/32 + 2, shifted by b % 32. The
 * bits of 2/pi past 2^-192 move the result by under a quarter of its last
 * bit. Writes r and q in place of the fast path's. */
static int fm_ph(nv_opnd_t x, nv_opnd_t r, nv_opnd_t q)
{
    nv_opnd_t ix, e, m, w[7], b, wi, s, p[3], lo, mid, top, hi, ol;
    uint32_t i;

    if (FM1(NV_MOV_B32, RF_R, &ix, x) || FM2(NV_SHR_U32, RF_R, &e, ix, FI(23))) return 1;
    if (is_ins(NV_AND_B32, e, e, FI(255), FN)) return 1;
    if (FM2(NV_SHL_B32, RF_R, &m, ix, FI(8)) || is_ins(NV_OR_B32, m, m, FI(INT32_MIN), FN)) return 1;
    if (fm_phw(m, w)) return 1;
    if (FM2(NV_SUB_S32, RF_R, &b, FI(288), e)) return 1;
    if (FM2(NV_SHR_U32, RF_R, &wi, b, FI(5)) || FM2(NV_AND_B32, RF_R, &s, b, FI(31))) return 1;
    for (i = 0; i < 3u; i++)
        if (FM2(NV_SETP_EQ_S32, RF_P, &p[i], wi, FI((int32_t)i + 2))) return 1;
    if (fm_phs(w, p, 0, &lo) || fm_phs(w, p, 1, &mid) || fm_phs(w, p, 2, &top)) return 1;
    if (FM3(NV_SHF_R_CLAMP_B32, RF_R, &hi, mid, top, s)) return 1;
    if (FM3(NV_SHF_R_CLAMP_B32, RF_R, &ol, lo, mid, s)) return 1;
    return fm_phf(hi, ol, ix, r, q);
}

/* x = q * pi/2 + r with |r| <= pi/4 (a little over, at worst), r in *r and
 * the quadrant in *q. Below 105615, Cody-Waite in three fma steps: the first
 * is exact, because x and q * P1 are both multiples of 2^-23 and their
 * difference is under 1. Above it, and finite, Payne-Hanek, branched around
 * since it is long and rare. inf and NaN stay on the fast path and give NaN. */
static int fm_red(nv_opnd_t x, nv_opnd_t *r, nv_opnd_t *q)
{
    nv_opnd_t t, k, a, p, pf;
    uint32_t l = 0;

    if (FM2(NV_MUL_RN_F32, RF_F, &t, x, FF(FM_2DPI))) return 1;
    if (FM1(NV_CVT_RNI_F32_F32, RF_F, &k, t)) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, r, k, FF(FM_P1_N), x)) return 1;
    if (is_ins(NV_FMA_RN_F32, *r, k, FF(FM_P2_N), *r)) return 1;
    if (is_ins(NV_FMA_RN_F32, *r, k, FF(FM_P3_N), *r)) return 1;
    if (FM1(NV_CVT_RZI_S32_F32, RF_R, q, k)) return 1;
    if (FM1(NV_ABS_F32, RF_F, &a, x) || FM2(NV_SETP_GE_F32, RF_P, &p, a, FF(FM_CW_MAX))) return 1;
    if (FM2(NV_SETP_LT_F32, RF_P, &pf, a, FF(FM_INF)) || is_ins(NV_AND_PRED, p, p, pf, FN)) return 1;
    if (is_lbl(&l)) return 1;
    if (is_ins(NV_BRA, FN, is_olbl(l), FN, FN) || is_guard(p, 1)) return 1;
    if (fm_ph(x, *r, *q)) return 1;
    return is_ins(NV_LABEL, is_olbl(l), FN, FN, FN);
}

/* sin and cos of the reduced r, then the quadrant picks one and its sign:
 * sin x = sin r, cos r, -sin r, -cos r for q mod 4 = 0..3, and cos x is
 * sin x one quadrant on. */
int fm_sin(nv_opnd_t dst, nv_opnd_t x, int cosine)
{
    nv_opnd_t r, q, z, sp, c, sn, cp, cs, o, po, v, h, ph, nv;

    if (fm_red(x, &r, &q)) return 1;
    if (cosine && is_ins(NV_ADD_S32, q, q, FI(1), FN)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &z, r, r)) return 1;
    if (fm_horn(&sp, z, FM_SS, 3u)) return 1;
    if (FM2(NV_MUL_RN_F32, RF_F, &c, z, r) || FM3(NV_FMA_RN_F32, RF_F, &sn, c, sp, r)) return 1;
    if (fm_horn(&cp, z, FM_SC, 3u)) return 1;
    if (is_ins(NV_FMA_RN_F32, cp, z, cp, FF(FM_HALF_N))) return 1;
    if (FM3(NV_FMA_RN_F32, RF_F, &cs, z, cp, FF(FM_ONE))) return 1;
    if (FM2(NV_AND_B32, RF_R, &o, q, FI(1)) || FM2(NV_SETP_NE_S32, RF_P, &po, o, FI(0))) return 1;
    if (FM3(NV_SELP_F32, RF_F, &v, cs, sn, po)) return 1;
    if (FM2(NV_AND_B32, RF_R, &h, q, FI(2)) || FM2(NV_SETP_NE_S32, RF_P, &ph, h, FI(0))) return 1;
    if (FM1(NV_NEG_F32, RF_F, &nv, v)) return 1;
    return is_ins(NV_SELP_F32, dst, nv, v, ph);
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

/* log2 x, or ln x when `natural` is set, from the two-part log2 |x|: one
 * rounding at the end, and a negative x (not -0) is NaN. For ln the small
 * terms are dropped when lh is infinite (x = 0 or inf), where they are NaN. */
int fm_log(nv_opnd_t dst, nv_opnd_t x, int natural)
{
    nv_opnd_t a, lh, ll, r, c, ah, pf, pn;

    if (FM1(NV_ABS_F32, RF_F, &a, x) || fm_lg2d(a, &lh, &ll)) return 1;
    if (natural) {
        if (FM2(NV_MUL_RN_F32, RF_F, &c, ll, FF(FM_LN2_HI))) return 1;
        if (is_ins(NV_FMA_RN_F32, c, lh, FF(FM_LN2_LO), c)) return 1;
        if (FM1(NV_ABS_F32, RF_F, &ah, lh) || FM2(NV_SETP_LT_F32, RF_P, &pf, ah, FF(FM_INF))) return 1;
        if (is_ins(NV_SELP_F32, c, c, FF(0), pf)) return 1;
        if (FM3(NV_FMA_RN_F32, RF_F, &r, lh, FF(FM_LN2_HI), c)) return 1;
    } else if (FM2(NV_ADD_RN_F32, RF_F, &r, lh, ll)) {
        return 1;
    }
    if (FM2(NV_SETP_LT_F32, RF_P, &pn, x, FF(0))) return 1;
    return is_ins(NV_SELP_F32, dst, FF(FM_QNAN), r, pn);
}
