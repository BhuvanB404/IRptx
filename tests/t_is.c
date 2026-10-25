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
    CHEQ(is_go(XIN, "(Load names=[\"p\" \"q\"] input=\"x0\" index=[" A0 "] dtype=dt:f32)"), 1);
    CHECK(strncmp(ie, "error[E020]", 11) == 0);
    CHEQ(is_cnt(NV_LD_GLOBAL_F32), 0u);
    PASS();
}
TH_REG("is", 8, "a Load of two names is refused E020 until the vector loads land", is08)

static void is09(void)
{
    CHEQ(is_go(XIN, "(Load names=[\"p\"] input=\"y\" index=[" A0 "] dtype=dt:f32)"), 1);
    CHECK(strncmp(ie, "error[E022]", 11) == 0);
    PASS();
}
TH_REG("is", 9, "a Load from an output buffer is refused E022", is09)
