/* t_is.c -- selection fixtures (is01..): small kernels through the real
 * reader and is_run, then the record list is read back with is_recs. Load
 * lowering first (is01..is09); T064 continues the numbering with the nested
 * loop fixtures. */

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
 * extent 8 whose body is `body`; the launch is one block of 8. */
static int is_go(const char *args, const char *body)
{
    int rc;

    snprintf(ib, sizeof ib, "kir 1\nkernel k\nlaunch 1 1 1 8 1 1 0\n%sbody\n"
        "[(Tile axes=[(Axis name=\"a0\" extent=(Dim expr=(Literal value=8 dtype=\"int\") hint=none) window=none)] "
        "body=[%s] block_threads=none aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n",
        args, body);
    if (kr_load(ib, strlen(ib)) != 0) return -1;
    th_cap_on();
    rc = is_run(kr_get());
    th_cap_off(ie, sizeof ie);
    return rc;
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
    CHEQ(is_go(XIN, LDS ASG("v", "exp", "\"p\"", "dt:f32")), 1);
    CHECK(strncmp(ie, "error[E021]", 11) == 0);
    CHECK(strstr(ie, "exp") != NULL);
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
    CHEQ(is_go(XIN, LDS WR("y", A0, "\"p\"", "atomic=true swizzle=\"NONE\"")), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_go(XIN, LDS WR("y", A0, "\"p\"", "atomic=false swizzle=\"B128\"")), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_cnt(NV_ST_GLOBAL_F32), 0u);
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
