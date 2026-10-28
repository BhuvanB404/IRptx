/* t_is.c -- selection fixtures (is01..): small kernels through the real
 * reader and is_run, then the record list is read back with is_recs. Load
 * lowering first (is01..is25), then the reductions: accumulators and strided
 * loops (is26..), branches and shared memory (is29..), the warp shuffle
 * (is32..) and the shared tree (is34..), then Let bindings and the
 * multi-state combines (is36..). T064 continues the numbering with the
 * nested loop fixtures. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tharns.h"
#include "irptx.h"
#include "isel.h"
#include "kir.h"

static char ib[4096];               /* generated KIR text */
static char ie[512];                /* captured stderr */

/* A kernel with the argument lines `args` and a Tile of one axis a0 of
 * extent `ext`, launched as one block of `blk`. The warp and tree fixtures
 * need whole warps in the block, which a fixed 8 can never offer. */
static int is_gox(const char *args, const char *body, int ext, int blk)
{
    int rc;

    snprintf(ib, sizeof ib, "kir 1\nkernel k\nlaunch 1 1 1 %d 1 1 0\n%sbody\n"
        "[(Tile axes=[(Axis name=\"a0\" extent=(Dim expr=(Literal value=%d dtype=\"int\") hint=none) window=none)] "
        "body=[%s] block_threads=%d aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n",
        blk, args, ext, body, blk);
    if (kr_load(ib, strlen(ib)) != 0) return -1;
    th_cap_on();
    rc = is_run(kr_get());
    th_cap_off(ie, sizeof ie);
    return rc;
}

/* The eight-thread kernel the load and store fixtures use. */
static int is_go(const char *args, const char *body)
{
    return is_gox(args, body, 8, 8);
}

/* The n-th (from 0) ld.global.f32 record, or NULL. */
static const nv_ins_t *is_nld(uint32_t n)
{
    uint32_t cnt, i;
    const nv_ins_t *r = is_recs(&cnt);

    for (i = 0; i < cnt; i++)
        if (r[i].op == NV_LD_GLOBAL_F32 && n-- == 0) return &r[i];
    return NULL;
}

/* How many records have opcode `op`. */
static uint32_t is_cnt(nv_op_t op)
{
    uint32_t cnt, i, n = 0;
    const nv_ins_t *r = is_recs(&cnt);

    for (i = 0; i < cnt; i++) n += r[i].op == op;
    return n;
}

#define XIN "arg 0 x0 in f32 1 8\narg 1 y out f32 1 8\n"
#define LD(nm, idx, dt) "(Load names=[\"" nm "\"] input=\"x0\" index=[" idx "] dtype=" dt ")"
#define A0 "(Var name=\"a0\")"
#define LIT(n) "(Literal value=" #n " dtype=\"int\")"

static void is01(void)
{
    const nv_ins_t *a, *b;

    CHEQ(is_go(XIN, LD("p", A0, "dt:f32") LD("q", A0, "dt:f32")), 0);
    a = is_nld(0);
    b = is_nld(1);
    CHECK(a && b);
    CHECK(a->dst.kind == OP_REG && a->dst.rf == RF_F);
    CHECK(b->dst.kind == OP_REG && b->dst.rf == RF_F && b->dst.v != a->dst.v);
    CHECK(a->src[0].kind == OP_REG && a->src[0].rf == RF_RD);
    CHEQ(a->src[1].kind, OP_NONE);
    CHEQ(is_cnt(NV_MUL_WIDE_S32), 2u);
    CHEQ(is_cnt(NV_ADD_S64), 2u);
    PASS();
}
TH_REG("is", 1, "a Load with a thread-varying index is mul.wide, add.s64, ld.global.f32 into a fresh %f", is01)

static void is02(void)
{
    const nv_ins_t *a;

    CHEQ(is_go(XIN, LD("p", LIT(3), "dt:f32")), 0);
    a = is_nld(0);
    CHECK(a);
    CHECK(a->src[0].kind == OP_REG && a->src[0].rf == RF_RD);
    CHECK(a->src[1].kind == OP_IMM && a->src[1].v == 12u);
    CHEQ(is_cnt(NV_MUL_WIDE_S32), 0u);
    CHEQ(is_cnt(NV_ADD_S64), 0u);
    PASS();
}
TH_REG("is", 2, "a constant index folds into [%rd+imm] and emits no address arithmetic", is02)

static void is03(void)
{
    const nv_ins_t *a;

    CHEQ(is_go("arg 0 c const f32 0\narg 1 y out f32 1 8\n",
        "(Load names=[\"c0\"] input=\"c\" index=[] dtype=dt:f32)"), 0);
    a = is_nld(0);
    CHECK(a);
    CHEQ(a->src[1].kind, OP_NONE);
    PASS();
}
TH_REG("is", 3, "a rank-0 constant buffer loads at offset 0 with an empty index", is03)

static void is04(void)
{
    const nv_ins_t *a;

    CHEQ(is_go("arg 0 c const f32 1 1\narg 1 y out f32 1 8\n",
        "(Load names=[\"c0\"] input=\"c\" index=[" LIT(0) "] dtype=dt:f32)"), 0);
    a = is_nld(0);
    CHECK(a);
    CHEQ(a->src[1].kind, OP_NONE);
    PASS();
}
TH_REG("is", 4, "a 1-element constant buffer indexed [0] loads like a global input", is04)

static void is05(void)
{
    CHEQ(is_go(XIN, "(Load names=[\"p\"] input=\"nope\" index=[" LIT(0) "] dtype=dt:f32)"), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 5, "a Load of a name that is not a parameter is refused E022 with no load emitted", is05)

static void is06(void)
{
    CHEQ(is_go(XIN, LD("p", A0, "none")), 0);
    CHECK(is_nld(0) != NULL);
    PASS();
}
TH_REG("is", 6, "a Load with dtype none is the inherited f32", is06)

static void is07(void)
{
    CHEQ(is_go(XIN, LD("p", A0, "dt:i32")), 1);
    CHECK(strncmp(ie, "error[E024]", 11) == 0);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 7, "a Load with a non-f32 dtype is refused E024", is07)

static void is08(void)
{
    CHEQ(is_go(XIN, "(Load names=[\"p\" \"q\"] input=\"x0\" index=[" A0 "] dtype=dt:f32)"), 0);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 2u);     /* index a0 is not provably aligned: two scalars */
    CHEQ(is_cnt(NV_LD_GLOBAL_V2_F32), 0u);
    PASS();
}
TH_REG("is", 8, "a Load of two names at an unprovable index is two scalar loads", is08)

static void is09(void)
{
    CHEQ(is_go(XIN, "(Load names=[\"p\"] input=\"y\" index=[" A0 "] dtype=dt:f32)"), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    PASS();
}
TH_REG("is", 9, "a Load from an output buffer is refused E022", is09)

/* ---- Assign (is10..) ---- */

#define LDS LD("p", A0, "dt:f32") LD("q", A0, "dt:f32")
#define ASG(nm, op, args, dt) "(Assign name=\"" nm "\" op=op:" op " args=[" args "] dtype=" dt ")"

/* The n-th (from 0) record with opcode `op`, or NULL. */
static const nv_ins_t *is_nth(nv_op_t op, uint32_t n)
{
    uint32_t cnt, i;
    const nv_ins_t *r = is_recs(&cnt);

    for (i = 0; i < cnt; i++)
        if (r[i].op == op && n-- == 0) return &r[i];
    return NULL;
}

static void is10(void)
{
    const nv_ins_t *m;

    CHEQ(is_go(XIN, LDS ASG("v", "multiply", "\"p\" \"q\"", "dt:f32")), 0);
    m = is_nth(NV_MUL_RN_F32, 0);
    CHECK(m);
    CHECK(m->dst.kind == OP_REG && m->dst.rf == RF_F);
    CHECK(m->src[0].v == is_nld(0)->dst.v && m->src[1].v == is_nld(1)->dst.v);
    CHECK(m->dst.v != m->src[0].v && m->dst.v != m->src[1].v);
    PASS();
}
TH_REG("is", 10, "a binary Assign is one .rn record over the two arg registers into a fresh %f", is10)

static void is11(void)
{
    const nv_ins_t *m;

    CHEQ(is_go(XIN, LDS ASG("v", "relu", "\"p\"", "dt:f32")), 0);
    m = is_nth(NV_MAX_F32, 0);
    CHECK(m);
    CHECK(m->src[0].v == is_nld(0)->dst.v);
    PASS();
}
TH_REG("is", 11, "a unary Assign reads one arg register", is11)

static void is12(void)
{
    /* gelu needs erf, so it is deliberately not in the table (ops.h). */
    CHEQ(is_go(XIN, LDS ASG("v", "gelu", "\"p\"", "dt:f32")), 1);
    CHECK(strncmp(ie, "error[E021]", 11) == 0);
    CHECK(strstr(ie, "gelu") != NULL);
    PASS();
}
TH_REG("is", 12, "an op outside the table is refused E021 naming the op", is12)

static void is13(void)
{
    CHEQ(is_go(XIN, LDS ASG("v", "multiply", "\"p\"", "dt:f32")), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_go(XIN, LDS ASG("v", "relu", "\"p\" \"q\"", "dt:f32")), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    PASS();
}
TH_REG("is", 13, "an Assign whose arg count is not the op's arity is refused E020", is13)

static void is14(void)
{
    CHEQ(is_go(XIN, LDS ASG("v", "add", "\"p\" \"zz\"", "dt:f32")), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    CHECK(strstr(ie, "zz") != NULL);
    PASS();
}
TH_REG("is", 14, "an Assign of an unbound arg is refused E022 naming it", is14)

static void is15(void)
{
    CHEQ(is_go(XIN, LDS ASG("v", "add", "\"p\" \"q\"", "none")), 0);
    CHECK(is_nth(NV_ADD_RN_F32, 0) != NULL);
    CHEQ(is_go(XIN, LDS ASG("v", "add", "\"p\" \"q\"", "dt:i32")), 1);
    CHECK(strncmp(ie, "error[E024]", 11) == 0);
    PASS();
}
TH_REG("is", 15, "an Assign with dtype none is the inherited f32; another dtype is E024", is15)

static void is16(void)
{
    const nv_ins_t *a, *b;

    CHEQ(is_go(XIN, LDS ASG("v", "add", "\"p\" \"q\"", "dt:f32")
        ASG("w", "multiply", "\"v\" \"p\"", "dt:f32")), 0);
    a = is_nth(NV_ADD_RN_F32, 0);
    b = is_nth(NV_MUL_RN_F32, 0);
    CHECK(a && b && b->src[0].v == a->dst.v);
    PASS();
}
TH_REG("is", 16, "an Assign result is bound so a later Assign reads it", is16)

/* ---- Write (T030) ---- */

#define WR(out, idx, vals, extra) "(Write output=\"" out "\" index=[" idx "] values=[" vals "] value_dtype=dt:f32 " extra ")"
#define WOK "atomic=false swizzle=\"NONE\""

static const nv_ins_t *is_nst(uint32_t n)
{
    uint32_t cnt, i;
    const nv_ins_t *r = is_recs(&cnt);

    for (i = 0; i < cnt; i++)
        if (r[i].op == NV_ST_GLOBAL_F32 && n-- == 0) return &r[i];
    return NULL;
}

static void is17(void)
{
    const nv_ins_t *s, *l;

    CHEQ(is_go(XIN, LD("p", A0, "dt:f32") WR("y", A0, "\"p\"", WOK)), 0);
    s = is_nst(0);
    l = is_nld(0);
    CHECK(s && l);
    CHECK(s->dst.kind == OP_NONE);
    CHECK(s->src[0].kind == OP_REG && s->src[0].rf == RF_RD);
    CHEQ(s->src[1].kind, OP_NONE);
    CHECK(s->src[2].kind == OP_REG && s->src[2].rf == RF_F && s->src[2].v == l->dst.v);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 1u);
    PASS();
}
TH_REG("is", 17, "a Write is one st.global.f32 [%rd], %f with the value register", is17)

static void is18(void)
{
    const nv_ins_t *s;

    CHEQ(is_go(XIN, LDS WR("y", LIT(3), "\"p\" \"q\"", WOK)), 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 2u);
    s = is_nst(0);
    CHECK(s && s->src[1].kind == OP_IMM && s->src[1].v == 12);
    s = is_nst(1);
    CHECK(s && s->src[1].kind == OP_IMM && s->src[1].v == 16);
    PASS();
}
TH_REG("is", 18, "a constant-index Write folds into [base+imm], value i at +4*i", is18)

static void is19(void)
{
    CHEQ(is_go(XIN, LDS WR("nope", A0, "\"p\"", WOK)), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
    CHEQ(is_go(XIN, LDS WR("x0", A0, "\"p\"", WOK)), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 19, "a Write to an unknown or non-output buffer is refused E022, nothing stored", is19)

static void is20(void)
{
    CHEQ(is_go(XIN, LDS WR("y", A0, "\"p\" \"zz\"", WOK)), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    CHECK(strstr(ie, "zz") != NULL);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 20, "a Write of an unbound value is refused E022 with no partial store", is20)

static void is21(void)
{
    CHEQ(is_go(XIN, LDS WR("y", A0, "\"p\"", "atomic=true swizzle=\"NONE\"")), 0);
    CHECK(is_cnt(NV_RED_GLOBAL_ADD_F32) == 1u);
    CHEQ(is_go(XIN, LDS WR("y", A0, "\"p\"", "atomic=false swizzle=\"B128\"")), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    PASS();
}
TH_REG("is", 21, "an atomic or swizzled Write is refused E020", is21)

/* ---- Vector accesses (T032; is08 now covers the scalar fallback) ---- */

#define XIN16 "arg 0 x0 in f32 1 64\narg 1 y out f32 1 64\n"
#define A2 "(BinaryExpr op=\"*\" left=" A0 " right=" LIT(2) ")"
#define A4 "(BinaryExpr op=\"*\" left=" A0 " right=" LIT(4) ")"
#define A2P1 "(BinaryExpr op=\"+\" left=" A2 " right=" LIT(1) ")"
#define LD2(i) "(Load names=[\"p\" \"q\"] input=\"x0\" index=[" i "] dtype=dt:f32)"
#define LD4(i) "(Load names=[\"p\" \"q\" \"r\" \"s\"] input=\"x0\" index=[" i "] dtype=dt:f32)"

static void is22(void)
{
    const nv_ins_t *l;

    CHEQ(is_go(XIN16, LD2(A2) WR("y", A0, "\"p\"", WOK)), 0);
    l = is_nth(NV_LD_GLOBAL_V2_F32, 0);
    CHECK(l && l->dst.kind == OP_REG && l->dst.rf == RF_F);
    CHEQ(is_cnt(NV_LD_GLOBAL_V2_F32), 1u);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 0u);
    CHEQ(is_cnt(NV_LD_GLOBAL_V4_F32), 0u);
    CHECK(l->src[0].kind == OP_REG && l->src[0].rf == RF_RD);
    CHEQ(is_go(XIN16, LD4(A4) WR("y", A0, "\"s\"", WOK)), 0);
    l = is_nth(NV_LD_GLOBAL_V4_F32, 0);
    CHECK(l);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 22, "a Load of 2 or 4 names at an aligned index is one ld.global.v2/v4.f32", is22)

static void is23(void)
{
    const nv_ins_t *s0, *s1;

    CHEQ(is_go(XIN16, LD2(A2P1) WR("y", A0, "\"p\"", WOK)), 0);        /* a0 * 2 + 1: misaligned */
    CHEQ(is_cnt(NV_LD_GLOBAL_V2_F32), 0u);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 2u);
    s0 = is_nld(0);
    s1 = is_nld(1);
    CHECK(s0 && s1 && s1->dst.v == s0->dst.v + 1u);
    CHECK(s0->src[1].kind == OP_NONE && s1->src[1].kind == OP_IMM && s1->src[1].v == 4);
    CHEQ(is_go(XIN16, LD2(A0) WR("y", A0, "\"p\"", WOK)), 0);          /* index a0: misaligned */
    CHEQ(is_cnt(NV_LD_GLOBAL_V2_F32), 0u);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 2u);
    CHEQ(is_go(XIN16, "(Load names=[\"p\" \"q\" \"r\"] input=\"x0\" index=[" A0 "] dtype=dt:f32)"), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);                         /* 3 names: refused */
    PASS();
}
TH_REG("is", 23, "a misaligned or non-affine Load of 2 names is two scalar loads; 3 names is E020", is23)

static void is24(void)
{
    const nv_ins_t *s;

    /* Loaded run is already consecutive: no mov, one st.v4. */
    CHEQ(is_go(XIN16, LD4(A4) WR("y", A4, "\"p\" \"q\" \"r\" \"s\"", WOK)), 0);
    s = is_nth(NV_ST_GLOBAL_V4_F32, 0);
    CHECK(s && s->dst.kind == OP_NONE && s->src[2].kind == OP_REG && s->src[2].rf == RF_F);
    CHECK(s->src[2].v == is_nth(NV_LD_GLOBAL_V4_F32, 0)->dst.v);
    CHEQ(is_cnt(NV_MOV_F32), 0u);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
    /* Values out of order: copied into a fresh run first. */
    CHEQ(is_go(XIN16, LD2(A2) WR("y", A2, "\"q\" \"p\"", WOK)), 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_V2_F32), 1u);
    CHEQ(is_cnt(NV_MOV_F32), 2u);
    s = is_nth(NV_ST_GLOBAL_V2_F32, 0);
    CHECK(s && s->src[2].v == is_nth(NV_MOV_F32, 0)->dst.v);
    CHECK(is_nth(NV_MOV_F32, 1)->dst.v == s->src[2].v + 1u);
    PASS();
}
TH_REG("is", 24, "a Write of 4 or 2 aligned values is one st.global.v4/v2.f32 over a consecutive run", is24)

static void is25(void)
{
    CHEQ(is_go(XIN16, LD2(A2) WR("y", A2P1, "\"p\" \"q\"", WOK)), 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_V2_F32), 0u);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 2u);
    CHEQ(is_cnt(NV_MOV_F32), 0u);
    CHEQ(is_go(XIN16, LDS WR("y", A0, "\"p\" \"q\" \"p\"", WOK)), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 25, "a misaligned Write of 2 values is two scalar stores; 3 values is E020", is25)

/* ---- Reductions: accumulators, loops, branches, shared, warp, tree ---- */

#define XINR "arg 0 x0 in f32 1 128\narg 1 y out f32 1 128\n"
#define A1V "(Var name=\"a1\")"
#define AXR(nm, n) "(Axis name=\"" nm "\" extent=(Dim expr=" LIT(n) " hint=none) window=none)"
#define SL(ax, st, sp, body) "(StridedLoop axis=" ax " start=" st " step=" sp \
    " body=[" body "] unroll=false end=none seed=true)"
#define ACCX(nm, v, op, axes) "(Accum name=\"" nm "\" value=\"" v "\" op=op:" op \
    " dtype=dt:f32 axes=[" axes "] base=none)"
#define ACC(nm, v, op) ACCX(nm, v, op, "\"a1\"")
#define CND(c, body, els) "(Cond cond=" c " body=[" body "] else_body=[" els "])"
#define EQA0 "(BinaryExpr op=\"==\" left=" A0 " right=" LIT(0) ")"
#define LANE0 "(BinaryExpr op=\"==\" left=(Var name=\"lane\") right=" LIT(0) ")"
#define SM(nm, n) "(Smem name=\"" nm "\" extents=[" LIT(n) "] dtype=\"float\" align=0)"
#define SYN(i) "(Sync barrier_id=" #i " count=none warp=false)"
#define WS(n, cm) "(WarpShuffle state=[\"acc\"] state_b=[\"acc__o\"] combine_states=" cm \
    " length=" #n " dtype=dt:f32)"
#define TRE(l, inner) "(TreeHalve bufs=[\"sm\"] state=[\"acc\"] state_b=[\"acc__o\"] " \
    "combine_states=[(Assign name=\"acc\" op=op:add args=[\"acc\" \"acc__o\"] dtype=none)] " \
    "length=" #l " tid_var=\"warp\" dtype=dt:f32 barrier_id=0 barrier_count=none inner=" inner ")"

/* The r01 shape: a partial per warp, into shared, a barrier, a tree, a write. */
#define TREEK(l, inner) LD("p", A0, "dt:f32") ACCX("acc", "p", "add", "\"a1\"") \
    SM("sm", 4) CND(LANE0, WR("sm", "(Var name=\"warp\")", "\"acc\"", WOK), "") \
    SYN(0) TRE(l, inner) CND(EQA0, WR("y", A0, "\"acc\"", WOK), "")

/* Record index of the n-th (from 0) record with opcode `op`, or -1. */
static long is_pos(nv_op_t op, uint32_t n)
{
    uint32_t cnt, i;
    const nv_ins_t *r = is_recs(&cnt);

    for (i = 0; i < cnt; i++)
        if (r[i].op == op && n-- == 0) return (long)i;
    return -1;
}

static void is26(void)
{
    CHEQ(is_gox(XIN, SL(AXR("a1", 64), "(Var name=\"a0\")", LIT(8),
        LD("p", A1V, "dt:f32") ACC("acc", "p", "add"))
        WR("y", A0, "\"acc\"", WOK), 8, 8), 0);
    CHEQ(is_cnt(NV_LABEL), 2u);
    CHEQ(is_cnt(NV_BRA), 2u);
    CHEQ(is_cnt(NV_SETP_GE_S32), 1u);
    CHEQ(is_cnt(NV_MOV_S32), 1u);          /* i = start */
    CHEQ(is_cnt(NV_ADD_S32), 1u);          /* i += step, at the back edge only */
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 1u);
    CHEQ(is_cnt(NV_ADD_RN_F32), 1u);
    CHECK(is_pos(NV_MOV_F32, 0) >= 0 && is_pos(NV_LABEL, 0) >= 0);
    CHECK(is_pos(NV_MOV_F32, 0) < is_pos(NV_LABEL, 0));    /* seeded before the loop */
    PASS();
}
TH_REG("is", 26, "a StridedLoop emits the guard, its body and one back edge, seeded first", is26)

static void is27(void)
{
    const nv_ins_t *m;

    CHEQ(is_gox(XIN, SL(AXR("a1", 64), "(Var name=\"a0\")", LIT(8),
        LD("p", A1V, "dt:f32") ACC("acc", "p", "multiply"))
        WR("y", A0, "\"acc\"", WOK), 8, 8), 0);
    m = is_nth(NV_MOV_F32, 0);
    CHECK(m && m->src[0].kind == OP_FIMM && m->src[0].v == 0x3F800000u);
    PASS();
}
TH_REG("is", 27, "an Accum of multiply seeds 1.0f, not 0.0", is27)

static void is28(void)
{
    CHEQ(is_gox(XIN, SL(AXR("a1", 64), "(Var name=\"a0\")", LIT(8),
        LD("p", A1V, "dt:f32") ACC("acc", "p", "subtract"))
        WR("y", A0, "\"acc\"", WOK), 8, 8), 1);
    CHECK(strncmp(ie, "error[E021]", 11) == 0);
    CHEQ(is_cnt(NV_SUB_RN_F32), 0u);
    PASS();
}
TH_REG("is", 28, "an Accum whose op has no identity and no base is refused E021", is28)

static void is29(void)
{
    const nv_ins_t *b0, *b1;

    CHEQ(is_gox(XIN, LD("p", A0, "dt:f32") CND(EQA0, WR("y", A0, "\"p\"", WOK), ""),
        8, 8), 0);
    CHEQ(is_cnt(NV_LABEL), 2u);
    CHEQ(is_cnt(NV_BRA), 2u);
    b0 = is_nth(NV_BRA, 0);
    CHECK(b0 && b0->prd.kind == OP_REG && b0->prd.rf == RF_P && b0->pneg == 1u);
    b1 = is_nth(NV_BRA, 1);
    CHECK(b1 && b1->prd.kind == OP_NONE);
    CHECK(is_pos(NV_ST_GLOBAL_F32, 0) > is_pos(NV_BRA, 0));
    CHECK(is_pos(NV_ST_GLOBAL_F32, 0) < is_pos(NV_BRA, 1));
    PASS();
}
TH_REG("is", 29, "a Cond branches over its body with @!%p and rejoins at the end label", is29)

static void is30(void)
{
    CHEQ(is_gox(XIN, LD("p", A0, "dt:f32")
        CND(EQA0, WR("sm", A0, "\"p\"", WOK), "") SM("sm", 8)
        WR("y", A0, "\"p\"", WOK), 8, 8), 0);
    CHEQ(is_cnt(NV_ST_SHARED_F32), 1u);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 1u);
    CHECK(sm_get()->narr == 1u);
    CHECK(sm_get()->arr[0].elems == 8u);
    CHECK(sm_get()->arr[0].rank == 1u);
    CHECK(sm_get()->arr[0].dim[0] == 8);
    PASS();
}
TH_REG("is", 30, "a Write into a shared array is st.shared.f32 even before its Smem statement", is30)

static void is31(void)
{
    const nv_ins_t *s;

    CHEQ(is_gox(XIN, LD("p", A0, "dt:f32") SYN(3) WR("y", A0, "\"p\"", WOK), 8, 8), 0);
    s = is_nth(NV_BAR_SYNC, 0);
    CHECK(s && s->src[0].kind == OP_IMM && s->src[0].v == 3u);
    PASS();
}
TH_REG("is", 31, "a Sync is bar.sync with the IR's barrier id", is31)

static void is32(void)
{
    const nv_ins_t *h;
    uint32_t i;

    CHEQ(is_gox(XINR, LD("p", A0, "dt:f32") ACCX("acc", "p", "add", "\"a1\"")
        WS(32, "[]") WR("y", A0, "\"acc\"", WOK), 32, 32), 0);
    CHEQ(is_cnt(NV_SHFL_BFLY), 5u);
    for (i = 0; i < 5u; i++) {
        h = is_nth(NV_SHFL_BFLY, i);
        CHECK(h && h->src[1].kind == OP_IMM && h->src[1].v == (16u >> i));
        CHECK(h && h->src[2].v == 31u && h->src[3].kind == OP_REG);
    }
    CHEQ(is_cnt(NV_ADD_RN_F32), 6u);       /* the Accum, then one combine per step */
    PASS();
}
TH_REG("is", 32, "a WarpShuffle of 32 halves the offset 16..1, shuffling then combining", is32)

static void is33(void)
{
    CHEQ(is_gox(XINR, LD("p", A0, "dt:f32") ACCX("acc", "p", "add", "\"a1\"")
        WS(3, "[]") WR("y", A0, "\"acc\"", WOK), 32, 32), 1);
    CHECK(strncmp(ie, "error[E034]", 11) == 0);
    CHEQ(is_gox(XIN, LD("p", A0, "dt:f32") ACCX("acc", "p", "add", "\"a1\"")
        WS(32, "[]") WR("y", A0, "\"acc\"", WOK), 8, 8), 1);
    CHECK(strncmp(ie, "error[E034]", 11) == 0);
    CHEQ(is_cnt(NV_SHFL_BFLY), 0u);
    PASS();
}
TH_REG("is", 33, "a shuffle of length 3, or in a block that is not whole warps, is E034", is33)

static void is34(void)
{
    CHEQ(is_gox(XINR, TREEK(4, "none"), 128, 128), 0);
    CHEQ(is_cnt(NV_BAR_SYNC), 2u);         /* the Sync, and one in the tree body */
    CHEQ(is_cnt(NV_LD_SHARED_F32), 2u);    /* the fold, and the broadcast of buf[0] */
    CHEQ(is_cnt(NV_ST_SHARED_F32), 2u);    /* the Cond's store, and the fold's */
    CHEQ(is_cnt(NV_SHR_S32), 2u);          /* warp = tid >> 5, and the halving */
    PASS();
}
TH_REG("is", 34, "a TreeHalve of 4 emits a barrier per step and broadcasts buf[0] into the state", is34)

static void is35(void)
{
    CHEQ(is_gox(XINR, TREEK(4, LIT(0)), 128, 128), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_gox(XINR, TREEK(3, "none"), 128, 128), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    /* A refusal leaves the records it had already made (the Sync above the
     * tree); what it must not leave is a poisoned state, so the good tree has
     * to compile again from scratch. */
    CHEQ(is_gox(XINR, TREEK(4, "none"), 128, 128), 0);
    CHEQ(is_cnt(NV_BAR_SYNC), 2u);
    PASS();
}
TH_REG("is", 35, "a tree with an inner state or a length of 3 is refused E020", is35)

/* ---- Let bindings and multi-state combines (T057, T060) ---- */

#define LET(nm, val, dt) "(Let name=\"" nm "\" value=" val " dtype=" dt ")"
#define LETF "(Literal value=0x1.0000000000000p+0f dtype=\"float\")"
#define TWOACC ACCX("acc0", "p", "add", "\"a1\"") ACCX("acc1", "p", "add", "\"a1\"")
#define CM2 "combine_states=[(Assign name=\"acc0\" op=op:add args=[\"acc0\" \"acc0__o\"] dtype=none) " \
    "(Assign name=\"acc1\" op=op:add args=[\"acc1\" \"acc1__o\"] dtype=none)]"
#define WSC2 "(WarpShuffle state=[\"acc0\" \"acc1\"] state_b=[\"acc0__o\" \"acc1__o\"] " \
    "combine_states=[(Assign name=\"acc0\" op=op:add args=[\"acc0\" \"acc0__o\"] dtype=none)] " \
    "length=32 dtype=dt:f32)"
#define TRE2 "(TreeHalve bufs=[\"sm0\" \"sm1\"] state=[\"acc0\" \"acc1\"] " \
    "state_b=[\"acc0__o\" \"acc1__o\"] " CM2 \
    " length=4 tid_var=\"warp\" dtype=dt:f32 barrier_id=0 barrier_count=none inner=none)"

/* The float half of the contract: 0f hex into a fresh %f, read back by a
 * later Assign. `one` is 1.0 so the mul's second source is that register. */
static void is36(void)
{
    const nv_ins_t *m;

    CHEQ(is_gox(XIN, LET("one", LETF, "dt:f32") LD("p", A0, "dt:f32")
        "(Assign name=\"q\" op=op:multiply args=[\"p\" \"one\"] dtype=dt:f32)"
        WR("y", A0, "\"q\"", WOK), 8, 8), 0);
    CHEQ(is_cnt(NV_MOV_F32), 1u);
    m = is_nth(NV_MOV_F32, 0);
    CHECK(m && m->dst.kind == OP_REG && m->dst.rf == RF_F);
    CHECK(m && m->src[0].kind == OP_FIMM && m->src[0].v == 0x3F800000u);
    CHEQ(is_cnt(NV_MUL_RN_F32), 1u);
    PASS();
}
TH_REG("is", 36, "a float Let is mov.f32 %f, 0f... and its name reads as that register", is36)

/* The integer half, and the one value the contract refuses. */
static void is37(void)
{
    CHEQ(is_gox(XIN, LET("k", "(Literal value=3 dtype=\"int\")", "dt:int")
        LD("p", "(Var name=\"k\")", "dt:f32") WR("y", A0, "\"p\"", WOK), 8, 8), 0);
    CHEQ(is_gox(XIN, LET("q", "(FlatIndex)", "dt:f32") WR("y", A0, "\"q\"", WOK), 8, 8), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    /* A refusal must not poison the state the next kernel starts from. */
    CHEQ(is_gox(XIN, LD("p", A0, "dt:f32") WR("y", A0, "\"p\"", WOK), 8, 8), 0);
    PASS();
}
TH_REG("is", 37, "an int Let folds and binds; a FlatIndex value is refused E020", is37)

/* Two states: both are shuffled, and the combine list is one list per step,
 * not one per state -- the difference is 7 adds against 12. */
static void is38(void)
{
    CHEQ(is_gox(XINR, LD("p", A0, "dt:f32") TWOACC WSC2
        WR("y", A0, "\"acc0\"", WOK), 32, 32), 0);
    CHEQ(is_cnt(NV_SHFL_BFLY), 10u);       /* 2 states x 5 steps */
    CHEQ(is_cnt(NV_MOV_F32), 2u);          /* one identity seed per accumulator */
    CHEQ(is_cnt(NV_ADD_RN_F32), 7u);       /* the 2 Accums, then the list once a step */
    PASS();
}
TH_REG("is", 38, "a two-state WarpShuffle shuffles both and lowers one combine list per step", is38)

/* Two buffers folded side by side: every state loads and stores its own cell,
 * and each buf[j][0] is broadcast back at the end. */
static void is39(void)
{
    CHEQ(is_gox(XINR, LD("p", A0, "dt:f32") TWOACC SM("sm0", 4) SM("sm1", 4)
        CND(LANE0, WR("sm0", "(Var name=\"warp\")", "\"acc0\"", WOK)
                   WR("sm1", "(Var name=\"warp\")", "\"acc1\"", WOK), "")
        SYN(0) TRE2 WR("y", A0, "\"acc0\"", WOK), 128, 128), 0);
    CHEQ(is_cnt(NV_LD_SHARED_F32), 4u);    /* the fold, and the broadcast of buf[j][0] */
    CHEQ(is_cnt(NV_ST_SHARED_F32), 4u);    /* the Cond's two stores, and the fold's */
    CHEQ(is_cnt(NV_BAR_SYNC), 2u);         /* the Sync, and one in the loop body */
    CHEQ(is_cnt(NV_ADD_RN_F32), 4u);       /* the 2 Accums, then the 2 combines */
    PASS();
}
TH_REG("is", 39, "a two-state TreeHalve folds both buffers and broadcasts buf[j][0]", is39)

/* ---- Barriers in loops ---- */

/* Run the verifier over what is_run left; its E-code, or 0. */
static int is_vf(void)
{
    int rc;

    th_cap_on();
    rc = vf_run();
    th_cap_off(ie, sizeof ie);
    return rc ? (ie[7] - '0') * 100 + (ie[8] - '0') * 10 + (ie[9] - '0') : 0;
}

#define SYNLOOP(st) SL(AXR("a1", 500), st, LIT(128), LD("p", A0, "dt:f32") SYN(0))

/* A loop from the thread's own index: the trip count differs, so the Sync
 * in its body would hang the block. A loop from a literal, or from a block
 * axis, runs the same trips everywhere and may hold one. */
static void is40(void)
{
    CHEQ(is_gox(XINR, SYNLOOP(A0), 128, 128), 0);
    CHEQ(is_vf(), 62);
    CHEQ(is_gox(XINR, SYNLOOP(LIT(0)), 128, 128), 0);
    CHEQ(is_vf(), 0);
    CHEQ(is_gox(XINR, "(Loop axis=" AXR("j", 4) " body=[" SYN(0) "] unroll=false seed=true)",
        128, 128), 0);
    CHEQ(is_vf(), 0);
    PASS();
}
TH_REG("is", 40, "a Sync in a StridedLoop of thread-varying start is E062; literal starts and Loops pass", is40)

/* Uniformity follows names: a block-level axis and a Let of one are
 * uniform, the thread's own axis is not. Two axes over a block of 32: a0
 * (stride 32) is the block's, a1 (stride 1) the thread's. */
static void is41(void)
{
    static char k2[2048];
    const char *hd = "kir 1\nkernel k\nlaunch 4 1 1 32 1 1 0\n" XINR "body\n"
        "[(Tile axes=[" "(Axis name=\"a0\" extent=(Dim expr=" LIT(4) " hint=none) window=none) "
        "(Axis name=\"a1\" extent=(Dim expr=" LIT(32) " hint=none) window=none)] body=[%s] "
        "block_threads=32 aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n";
    const char *ok = LET("s", A0, "dt:int") SL(AXR("i", 64), "(Var name=\"s\")", LIT(8), SYN(0));
    const char *bad = SL(AXR("i", 64), A1V, LIT(8), SYN(0));

    snprintf(k2, sizeof k2, hd, ok);
    CHEQ(kr_load(k2, strlen(k2)), 0);
    CHEQ(is_run(kr_get()), 0);
    CHEQ(is_vf(), 0);
    snprintf(k2, sizeof k2, hd, bad);
    CHEQ(kr_load(k2, strlen(k2)), 0);
    CHEQ(is_run(kr_get()), 0);
    CHEQ(is_vf(), 62);
    PASS();
}
TH_REG("is", 41, "a loop from a block-level axis (or a Let of one) may hold a Sync; from the thread axis it is E062", is41)


/* ---- StridedLoop steps and bounds ---- */

#define SLS(sp) SL(AXR("a1", 64), A0, sp, LD("p", A1V, "dt:f32"))
#define SLE(n, en) "(StridedLoop axis=" AXR("a1", n) " start=" A0 " step=" LIT(8) \
    " body=[" LD("p", A1V, "dt:f32") "] unroll=false end=" en " seed=true)"

/* The exit test is `i >= bound`, so only a positive constant step ends. */
static void is42(void)
{
    CHEQ(is_gox(XINR, SLS(LIT(-8)), 8, 8), 1);
    CHECK(strncmp(ie, "error[E026]", 11) == 0);
    CHEQ(is_gox(XINR, SLS(LIT(0)), 8, 8), 1);
    CHECK(strncmp(ie, "error[E026]", 11) == 0);
    CHEQ(is_gox(XINR, SLS(A0), 8, 8), 1);
    CHECK(strncmp(ie, "error[E026]", 11) == 0);
    CHEQ(is_cnt(NV_LABEL), 0u);            /* refused before any loop record */
    CHEQ(is_gox(XINR, SLS("(BinaryExpr op=\"*\" left=" LIT(2) " right=" LIT(4) ")"), 8, 8), 0);
    CHECK(is_nth(NV_ADD_S32, 0) && is_nth(NV_ADD_S32, 0)->src[1].kind == OP_IMM);
    CHECK(is_nth(NV_ADD_S32, 0)->src[1].v == 8u);
    PASS();
}
TH_REG("is", 42, "a StridedLoop step that is negative, zero or not a constant is E026; 2*4 folds to 8", is42)

/* The bound is min(end, extent), and the last step must stay in s32. */
static void is43(void)
{
    const nv_ins_t *s;

    CHEQ(is_gox(XINR, SLE(64, LIT(100)), 8, 8), 0);       /* past the extent: clamped */
    s = is_nth(NV_SETP_GE_S32, 0);
    CHECK(s && s->src[1].kind == OP_IMM && s->src[1].v == 64u);
    CHEQ(is_gox(XINR, SLE(64, LIT(40)), 8, 8), 0);
    s = is_nth(NV_SETP_GE_S32, 0);
    CHECK(s && s->src[1].kind == OP_IMM && s->src[1].v == 40u);
    CHEQ(is_gox(XINR, SLE(64, A0), 8, 8), 0);              /* a register end: min.s32 */
    CHEQ(is_cnt(NV_MIN_S32), 1u);
    CHEQ(is_gox(XINR, SL(AXR("a1", 2147483647), A0, LIT(8), ""), 8, 8), 1);
    CHECK(strncmp(ie, "error[E059]", 11) == 0);
    PASS();
}
TH_REG("is", 43, "a StridedLoop bound is min(end, extent), and a step past s32 at the end is E059", is43)

/* ---- combine_states ---- */

#define CMA "(Assign name=\"acc\" op=op:add args=[\"acc\" \"acc__o\"] dtype=none)"
#define WSK(cm) LD("p", A0, "dt:f32") ACCX("acc", "p", "add", "\"a1\"") WS(32, cm) \
    WR("y", A0, "\"acc\"", WOK)

/* A combine is a pure merge program of Assigns. A loop or a Cond inside one
 * used to be lowered through the walk's single request slot, out of order
 * (or E000); now it is E027 before any combine record. */
static void is44(void)
{
    CHEQ(is_gox(XINR, WSK("[" SL(AXR("j", 1), LIT(0), LIT(1), CMA) "]"), 32, 32), 1);
    CHECK(strncmp(ie, "error[E027]", 11) == 0);
    CHEQ(is_gox(XINR, WSK("[" CND(EQA0, CMA, "") "]"), 32, 32), 1);
    CHECK(strncmp(ie, "error[E027]", 11) == 0);
    CHEQ(is_cnt(NV_ADD_RN_F32), 1u);       /* the Accum only */
    CHEQ(is_gox(XINR, WSK("[" CMA "]"), 32, 32), 0);
    CHEQ(is_cnt(NV_ADD_RN_F32), 6u);       /* the Accum, then one Assign per step */
    PASS();
}
TH_REG("is", 44, "a combine_states holding a loop or a Cond is E027; plain Assigns lower once per step", is44)

/* ---- Overflow in the compiler's own arithmetic ---- */

/* Three -2^31 constants times strides of 2^31 - 1 overflow an int64 sum;
 * the sum is checked term by term, so this is E059, not undefined
 * behaviour that happens to land on a refusal. */
static void is45(void)
{
    CHEQ(is_gox("arg 0 x0 in f32 4 1 1 1 2147483647\narg 1 y out f32 1 8\n",
        "(Load names=[\"v\"] input=\"x0\" index=[" LIT(-2147483648) " " LIT(-2147483648) " "
        LIT(-2147483648) " " A0 "] dtype=dt:f32)", 8, 8), 1);
    CHECK(strncmp(ie, "error[E059]", 11) == 0);
    CHEQ(is_gox("arg 0 x0 in f32 2 4 8\narg 1 y out f32 1 8\n",
        "(Load names=[\"v\"] input=\"x0\" index=[" LIT(3) " " A0 "] dtype=dt:f32)", 8, 8), 0);
    CHECK(is_nld(0) && is_nld(0)->src[1].kind == OP_NONE);
    PASS();
}
TH_REG("is", 45, "a constant index whose int64 sum would overflow is E059; an ordinary one folds", is45)

/* Four extents of 65536 multiply to 2^64, which wrapped the element count
 * to 0 and passed; the size is now checked at every multiply. */
static void is46(void)
{
    CHEQ(is_gox(XIN, "(Smem name=\"big\" extents=[" LIT(65536) " " LIT(65536) " " LIT(65536) " "
        LIT(65536) "] dtype=\"float\" align=0)", 8, 8), 1);
    CHECK(strncmp(ie, "error[E059]", 11) == 0);
    CHEQ(is_gox(XIN, "(Smem name=\"ok\" extents=[" LIT(4) " " LIT(8) "] dtype=\"float\" align=0)",
        8, 8), 0);
    CHECK(sm_get()->narr == 1u && sm_get()->arr[0].elems == 32u);
    PASS();
}
TH_REG("is", 46, "a shared array whose extents multiply past 64 bits is E059, not a wrapped 0", is46)
