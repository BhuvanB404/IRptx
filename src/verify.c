/* verify.c -- vf_*: ASCII, labels, register bounds, barrier uniformity, size caps.
 *
 * Runs on the instruction records after isel and before anything is printed
 * (ASCII, and so P12: a dash is a non-ASCII byte), P5 (register bounds), P6
 * (labels) and the barrier rule P8 / ruling M7, plus the size caps. P1-P3 and
 * P11 are properties of the text the emitter writes, not of the records.
 *
 * Codes: E061 non-ASCII, E060 label trouble, E079 for a register out of
 * range or a barrier problem, E083 for too many records (output size, E084,
 * is the emitter's). Each failure prints one line naming the record index
 * and what is wrong.
 *
 * Divergent region (vf_bar). Walking the record list, a branch is
 * "divergent" when it carries a guard predicate, goes forward (its label is
 * defined later in the list) and is neither
 *   - the whole-kernel guard exit: its target is the exit label, the label
 *     that sits right before the final `ret` (every thread of a block takes
 *     it together under H4, and is_tile only emits it when no barrier can
 *     follow for part of a block); nor
 *   - a loop exit: the record just before the target label (comments and
 *     nops skipped) is an unguarded branch back to a label defined before
 *     this branch. That is the shape `L_top: ... @p bra L_done; ...;
 *     bra L_top; L_done:`. Loop bounds are launch constants, so the exit is
 *     taken by every thread of the block on the same trip.
 * The records after a divergent branch and before its target label are its
 * region. A bar.sync, a shfl, or any guarded barrier inside a region is
 * refused. A Cond lowers to `@!p bra L_else`, so its then and else bodies
 * are regions (the else body is reached through the unguarded `bra L_end`
 * that ends the then body, which lies inside the first region, so it opens
 * the second). Backward branches never open one.
 *
 * bar.sync operands (M7): src[0] is the OP_IMM id, 0..15; the optional
 * src[1] is the OP_IMM thread count, a multiple of 32 in 32..1024. No
 * count means the whole block. bar.warp.sync is not checked here: its mask
 * comes from activemask and it is legal under divergence. */

#include <stdio.h>
#include <string.h>
#include "irptx.h"
#include "isel.h"
#include "kir.h"

#define VF_BARMAX 1024u             /* most threads a block can hold */
#define VF_BARID  15u               /* highest barrier id */
#define VF_MSG    160u

static uint32_t VF_POS[IR_MAX_REG + 1u];    /* label number to record index + 1; 0 = undefined */
static int32_t  VF_COV[IR_MAX_INS + 2u];    /* difference array of divergent regions */
static char     VF_WHAT[VF_MSG];

/* ---- Helpers ---- */

/* Print a failure whose text is `fmt` with up to two numbers. */
static int vf_fail(int code, const char *fmt, uint32_t a, uint32_t b)
{
    int n = snprintf(VF_WHAT, sizeof VF_WHAT, fmt, (unsigned)a, (unsigned)b);

    if (n < 0 || (size_t)n >= sizeof VF_WHAT) return er_fail(code, "a failed module check");
    return er_fail(code, VF_WHAT);
}

/* 0 when every byte of `s` is printable ASCII (0x20-0x7E). */
static int vf_prt(const char *s)
{
    IX_GUARD(g, IR_MAX_OUT);

    while (*s && g--) {
        unsigned char c = (unsigned char)*s++;
        if (c < 0x20u || c > 0x7Eu) return 1;
    }
    return 0;
}

/* Fill VF_POS from the records. Returns 0, or 1 with *at the index of the
 * first label that is out of range or defined twice. Silent. */
static int vf_fill(const nv_ins_t *r, uint32_t n, uint32_t *at)
{
    uint32_t i;

    memset(VF_POS, 0, sizeof VF_POS);
    for (i = 0; i < n; i++) {
        uint32_t v;
        if (r[i].op != NV_LABEL) continue;
        v = r[i].dst.v;
        if (r[i].dst.kind != OP_LBL || v == 0 || IX_CHK(v, IR_MAX_REG + 1u) || VF_POS[v]) {
            *at = i;
            return 1;
        }
        VF_POS[v] = i + 1u;
    }
    return 0;
}

/* The exit label number: the label right before the final ret (comments and
 * nops skipped), or 0 when the list does not end that way. */
static uint32_t vf_exit(const nv_ins_t *r, uint32_t n)
{
    uint32_t k = n;
    int seen_ret = 0;
    IX_GUARD(g, n + 1u);

    while (k > 0 && g--) {
        const nv_ins_t *q = &r[--k];
        if (q->op == NV_COMMENT || q->op == NV_NOP) continue;
        if (!seen_ret) {
            if (q->op != NV_RET) return 0;
            seen_ret = 1;
            continue;
        }
        return (q->op == NV_LABEL && q->dst.kind == OP_LBL) ? q->dst.v : 0;
    }
    return 0;
}

/* ---- Checks ---- */

/* P4, P12: every string the emitter prints is printable ASCII. */
static int vf_ascii(void)
{
    const kir_t *k = kr_get();
    uint32_t i, n;
    const nv_ins_t *r = is_recs(&n);

    if (vf_prt(k->name)) return er_fail(E061, "a kernel name with a non-ASCII byte");
    for (i = 0; i < k->narg && i < IR_MAX_ARG; i++)
        if (vf_prt(kr_str(k->arg[i].name)))
            return vf_fail(E061, "a parameter name (argument %u) with a non-ASCII byte", i, 0);
    for (i = 0; i < n; i++)
        if (r[i].cmt && vf_prt(kr_str(r[i].cmt)))
            return vf_fail(E061, "a comment on record %u with a non-ASCII byte", i, 0);
    return 0;
}

/* P6: each label defined once, each branch target defined. */
static int vf_lbl(void)
{
    uint32_t n, i, at = 0;
    const nv_ins_t *r = is_recs(&n);

    if (vf_fill(r, n, &at))
        return vf_fail(E060, "label defined twice or out of range at record %u", at, 0);
    for (i = 0; i < n; i++) {
        uint32_t t = r[i].src[0].v;
        if (r[i].op != NV_BRA) continue;
        if (r[i].src[0].kind != OP_LBL || t == 0 || IX_CHK(t, IR_MAX_REG + 1u) || !VF_POS[t])
            return vf_fail(E060, "branch at record %u to label L%u that is not defined", i, t);
    }
    return 0;
}

/* P5: every register operand is numbered 1..rg_count of its file. */
static int vf_regs(void)
{
    uint32_t n, i, j;
    const nv_ins_t *r = is_recs(&n);

    for (i = 0; i < n; i++) {
        const nv_opnd_t *o[6];
        o[0] = &r[i].dst; o[1] = &r[i].prd;
        for (j = 0; j < 4; j++) o[2 + j] = &r[i].src[j];
        for (j = 0; j < 6; j++) {
            if (o[j]->kind > OP_LBL)
                return vf_fail(E079, "record %u has an operand of unknown kind", i, 0);
            if (o[j]->kind != OP_REG) continue;
            if (o[j]->rf >= RF_N || o[j]->v < 1u || o[j]->v > rg_count((rf_t)o[j]->rf))
                return vf_fail(E079, "record %u uses register %u outside its declared range",
                               i, o[j]->v);
        }
    }
    return 0;
}

/* M7: bar.sync id 0..15, count a multiple of 32 up to 1024. */
static int vf_barops(const nv_ins_t *q, uint32_t i)
{
    if (q->src[0].kind != OP_IMM || q->src[0].v > VF_BARID)
        return vf_fail(E079, "bar.sync at record %u without a barrier id in 0..15", i, 0);
    if (q->nsrc < 2) return 0;
    if (q->src[1].kind != OP_IMM || q->src[1].v == 0 || q->src[1].v > VF_BARMAX ||
        q->src[1].v % 32u != 0)
        return vf_fail(E079, "bar.sync at record %u with a count that is not 32..1024 in steps of 32",
                       i, 0);
    return 0;
}

/* 1 when the guarded forward branch at `i` to record index `j` is a loop
 * exit: the record before label `j` is an unguarded branch to a label
 * defined before `i`. */
static int vf_lexit(const nv_ins_t *r, uint32_t i, uint32_t j)
{
    uint32_t k = j;
    IX_GUARD(g, j + 1u);

    while (k > 0 && g--) {
        const nv_ins_t *q = &r[--k];
        if (q->op == NV_COMMENT || q->op == NV_NOP) continue;
        if (q->op != NV_BRA || q->prd.kind != OP_NONE || q->src[0].kind != OP_LBL) return 0;
        if (IX_CHK(q->src[0].v, IR_MAX_REG + 1u) || !VF_POS[q->src[0].v]) return 0;
        return VF_POS[q->src[0].v] - 1u < i;
    }
    return 0;
}

/* Mark the records strictly between each divergent forward branch and its
 * target in VF_COV (a difference array, summed by the caller). An unguarded
 * forward branch met inside a region is the `bra L_end` that closes a then
 * body, so the else body between it and L_end is a region too. */
static void vf_regions(const nv_ins_t *r, uint32_t n)
{
    uint32_t xl = vf_exit(r, n), i;
    int32_t depth = 0;

    memset(VF_COV, 0, ((size_t)n + 2u) * sizeof VF_COV[0]);
    for (i = 0; i < n; i++) {
        uint32_t t = r[i].src[0].v, j;
        int grd = r[i].prd.kind == OP_REG;
        depth += VF_COV[i];
        if (r[i].op != NV_BRA || (!grd && depth <= 0)) continue;
        if (r[i].src[0].kind != OP_LBL || IX_CHK(t, IR_MAX_REG + 1u) || !VF_POS[t]) continue;
        if (xl && t == xl) continue;
        j = VF_POS[t] - 1u;
        if (j <= i || (grd && vf_lexit(r, i, j))) continue;
        VF_COV[i + 1u]++;
        VF_COV[j]--;
    }
}

/* P8, M7: no bar.sync or shfl in a divergent region or under its own guard,
 * and bar.sync operands in range. */
static int vf_bar(void)
{
    uint32_t n, i, at = 0;
    int32_t depth = 0;
    const nv_ins_t *r = is_recs(&n);

    (void)vf_fill(r, n, &at);           /* vf_lbl has already refused a bad table */
    vf_regions(r, n);
    for (i = 0; i < n; i++) {
        int bar = r[i].op == NV_BAR_SYNC;
        depth += VF_COV[i];
        if (!bar && r[i].op != NV_SHFL_BFLY) continue;
        if (depth > 0 || r[i].prd.kind != OP_NONE)
            return vf_fail(E079, bar ? "bar.sync at record %u is under a divergent branch or guard"
                                     : "shfl at record %u is under a divergent branch or guard",
                           i, 0);
        if (bar && vf_barops(&r[i], i)) return 1;
    }
    return 0;
}

/* Record count only. The printed size is the emitter's to enforce: its
 * buffer sets an overflow flag and em_run refuses with E084, so no estimate
 * here can refuse a module the emitter would have printed. */
static int vf_size(void)
{
    uint32_t n;

    (void)is_recs(&n);
    if (n > IR_MAX_INS) return er_fail(E083, "more than IR_MAX_INS instruction records");
    return 0;
}

/* ---- Entry ---- */

int vf_run(void)
{
    if (vf_ascii()) return 1;
    if (vf_lbl()) return 1;
    if (vf_regs()) return 1;
    if (vf_bar()) return 1;
    if (vf_size()) return 1;
    return 0;
}
