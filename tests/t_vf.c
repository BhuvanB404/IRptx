/* t_vf.c -- the verifier (vf01..): every check fires, with its exact E-code
 * and one error line, on a hand-built bad record list, and stays silent on a
 * good one. Lists are built with the isel record helpers on a kernel loaded
 * through the real reader, so the records follow the selector's conventions
 * (NV_LABEL has the OP_LBL in dst; NV_BRA has it in src[0], guard in prd). */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tharns.h"
#include "irptx.h"
#include "isel.h"
#include "kir.h"

/* The real empty Tile: one rank-1 axis of 200, two f32 arguments. */
#define VF_KIR \
    "kir 1\nkernel k_empty\nlaunch 2 1 1 128 1 1 0\narg 0 x in f32 1 200\narg 1 y out f32 1 200\nbody\n" \
    "[(Tile axes=[(Axis name=\"a0\" extent=(Dim expr=(Literal value=200 dtype=\"int\") hint=none) window=none)] " \
    "body=[] block_threads=none aux_threads=0 raster_axes=none raster_group=none raster_orient=\"m\")]\nend\n"

static char eb[512];                /* captured stderr */

/* Load the empty kernel and clear the record list; 0 on success. */
static int vf_fresh(void)
{
    if (kr_load(VF_KIR, strlen(VF_KIR)) != 0) return 1;
    is_reset();
    return 0;
}

/* E-code of the single error line a failing call printed (rc 1), 0 for rc 0
 * with a silent stderr, -1 for anything else. */
static int vf_code(int rc, size_t len)
{
    int i, nl = 0;

    if (rc == 0) return len == 0 ? 0 : -1;
    for (i = 0; eb[i]; i++) nl += eb[i] == '\n';
    if (rc != 1 || nl != 1 || strncmp(eb, "error[E", 7) != 0 || eb[10] != ']') return -1;
    return (eb[7] - '0') * 100 + (eb[8] - '0') * 10 + (eb[9] - '0');
}

/* Run the verifier, capturing stderr, and return its vf_code. Every test goes
 * through vf_run and asserts the code of the stage its list is built to hit
 * (the earlier stages stay quiet on it): the check functions are static in
 * verify.c. */
static int vf_chk(void)
{
    int rc;
    size_t len;

    th_cap_on();
    rc = vf_run();
    len = th_cap_off(eb, sizeof eb);
    return vf_code(rc, len);
}

/* ---- Record helpers ---- */

static nv_opnd_t vf_r(void) { nv_opnd_t o = is_onone(); rg_new(RF_R, &o); return o; }
static nv_opnd_t vf_p(void) { nv_opnd_t o = is_onone(); rg_new(RF_P, &o); return o; }
static void vf_mov(nv_opnd_t d) { is_ins(NV_MOV_U32, d, is_oimm(1), is_onone(), is_onone()); }
static void vf_lab(uint32_t n) { is_ins(NV_LABEL, is_olbl(n), is_onone(), is_onone(), is_onone()); }
static void vf_bra(uint32_t n) { is_ins(NV_BRA, is_onone(), is_olbl(n), is_onone(), is_onone()); }
static void vf_gbra(uint32_t n, nv_opnd_t p) { vf_bra(n); is_guard(p, 0); }
static void vf_ret(void) { is_ins(NV_RET, is_onone(), is_onone(), is_onone(), is_onone()); }
static void vf_bar(int id, int cnt, int two)
{
    is_ins(NV_BAR_SYNC, is_onone(), is_oimm(id), two ? is_oimm(cnt) : is_onone(), is_onone());
}

/* Intern `s` at the end of the string pool (the reader's pool is read only
 * through kr_get, the test appends to it) and return its offset. */
static uint32_t vf_pool(const char *s)
{
    kir_t *k = (kir_t *)kr_get();
    uint32_t off = k->nstr;
    size_t n = strlen(s) + 1u;

    memcpy(k->str + off, s, n);
    k->nstr += (uint32_t)n;
    return off;
}

/* ---- E061: ASCII ---- */

static void vf01(void)
{
    CHEQ(vf_fresh(), 0);
    vf_mov(vf_r());
    is_cmt(vf_pool("plain ascii: x[i] + 1"));
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 1, "ASCII: a plain comment passes", vf01)

static void vf02(void)
{
    CHEQ(vf_fresh(), 0);
    vf_mov(vf_r());
    is_cmt(vf_pool("bad \xe2\x80\x94 dash"));
    CHEQ(vf_chk(), 61);
    PASS();
}
TH_REG("vf", 2, "ASCII: an em dash in a comment is E061", vf02)

static void vf03(void)
{
    CHEQ(vf_fresh(), 0);
    vf_mov(vf_r());
    is_cmt(vf_pool("tab\there"));
    CHEQ(vf_chk(), 61);
    PASS();
}
TH_REG("vf", 3, "ASCII: a tab in a comment is E061", vf03)

static void vf04(void)
{
    CHEQ(vf_fresh(), 0);
    strcpy(((kir_t *)kr_get())->name, "k\xc3\xa9");
    CHEQ(vf_chk(), 61);
    PASS();
}
TH_REG("vf", 4, "ASCII: a non-ASCII kernel name is E061", vf04)

/* ---- E060: labels ---- */

static void vf05(void)
{
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    is_lbl(&l1); is_lbl(&l2);
    vf_lab(l1); vf_bra(l1); vf_lab(l2);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 5, "labels: defined once and branched to passes", vf05)

static void vf06(void)
{
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    is_lbl(&l1);
    vf_lab(l1); vf_lab(l1);
    CHEQ(vf_chk(), 60);
    PASS();
}
TH_REG("vf", 6, "labels: a label defined twice is E060", vf06)

static void vf07(void)
{
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    is_lbl(&l1); is_lbl(&l2);
    vf_lab(l1); vf_bra(l2);
    CHEQ(vf_chk(), 60);
    PASS();
}
TH_REG("vf", 7, "labels: a branch to an undefined label is E060", vf07)

static void vf08(void)
{
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    is_lbl(&l1);
    is_ins(NV_BRA, is_onone(), is_oimm(3), is_onone(), is_onone());
    CHEQ(vf_chk(), 60);
    PASS();
}
TH_REG("vf", 8, "labels: a branch target that is not a label is E060", vf08)

/* ---- E079: registers ---- */

static void vf09(void)
{
    nv_opnd_t a, p;

    CHEQ(vf_fresh(), 0);
    a = vf_r(); p = vf_p();
    vf_mov(a);
    is_ins(NV_SETP_EQ_S32, p, a, is_oimm(0), is_onone());
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 9, "registers: operands inside 1..count pass", vf09)

static void vf10(void)
{
    nv_opnd_t a;

    CHEQ(vf_fresh(), 0);
    a = vf_r();
    vf_mov(a);
    is_ins(NV_MOV_U32, is_oreg(RF_R, a.v + 1u), is_oimm(1), is_onone(), is_onone());
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 10, "registers: one above the count is E079", vf10)

static void vf11(void)
{
    CHEQ(vf_fresh(), 0);
    (void)vf_r();
    is_ins(NV_MOV_U32, is_oreg(RF_R, 0), is_oimm(1), is_onone(), is_onone());
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 11, "registers: index zero is E079", vf11)

static void vf12(void)
{
    nv_opnd_t a;

    CHEQ(vf_fresh(), 0);
    a = vf_r();
    is_ins(NV_MOV_U32, a, is_oreg(RF_F, 1), is_onone(), is_onone());
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 12, "registers: a register of a file with none handed out is E079", vf12)

static void vf13(void)
{
    nv_opnd_t a, p;

    CHEQ(vf_fresh(), 0);
    a = vf_r(); p = vf_p();
    vf_mov(a);
    is_guard(is_oreg(RF_P, p.v + 1u), 0);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 13, "registers: a guard predicate above the count is E079", vf13)

/* ---- E079: barriers ---- */

static void vf14(void)
{
    nv_opnd_t p;
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1);
    vf_gbra(l1, p); vf_bar(0, 64, 1); vf_lab(l1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 14, "barriers: bar.sync inside a Cond region is E079", vf14)

static void vf15(void)
{
    nv_opnd_t p;
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1);
    vf_gbra(l1, p);
    is_ins(NV_SHFL_BFLY, is_oreg(RF_F, 1), is_oreg(RF_F, 1), is_oimm(1), is_onone());
    vf_lab(l1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 15, "barriers: a shuffle inside a Cond region is E079", vf15)

static void vf16(void)
{
    nv_opnd_t p;
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2);
    vf_gbra(l1, p); vf_mov(vf_r()); vf_gbra(l2, p); vf_bar(0, 64, 1); vf_lab(l2); vf_lab(l1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 16, "barriers: bar.sync inside nested Cond regions is E079", vf16)

static void vf17(void)
{
    nv_opnd_t p;

    CHEQ(vf_fresh(), 0);
    p = vf_p();
    vf_bar(0, 64, 1); is_guard(p, 0);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 17, "barriers: a bar.sync under its own guard is E079", vf17)

static void vf18(void)
{
    CHEQ(vf_fresh(), 0);
    vf_bar(16, 64, 1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 18, "barriers: barrier id 16 is E079", vf18)

static void vf19(void)
{
    CHEQ(vf_fresh(), 0);
    vf_bar(0, 48, 1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 19, "barriers: a count of 48 is E079", vf19)

static void vf20(void)
{
    CHEQ(vf_fresh(), 0);
    vf_bar(0, 0, 1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 20, "barriers: a count of 0 is E079", vf20)

static void vf21(void)
{
    CHEQ(vf_fresh(), 0);
    vf_bar(0, 2048, 1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 21, "barriers: a count of 2048 is E079", vf21)

static void vf22(void)
{
    CHEQ(vf_fresh(), 0);
    vf_bar(15, 1024, 1); vf_bar(1, 32, 1); vf_bar(0, 0, 0);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 22, "barriers: ids 0 and 15, counts 32 and 1024, no count all pass", vf22)

static void vf23(void)
{
    nv_opnd_t p;
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1);
    vf_gbra(l1, p); vf_mov(vf_r()); vf_lab(l1); vf_bar(0, 64, 1);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 23, "barriers: a bar.sync after the Cond closes passes", vf23)

static void vf24(void)
{
    nv_opnd_t p;
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1);
    vf_gbra(l1, p); vf_bar(0, 64, 1); vf_lab(l1); vf_ret();
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 24, "barriers: the final guard exit does not open a region", vf24)

static void vf25(void)
{
    nv_opnd_t p;
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2);
    vf_lab(l1); vf_gbra(l2, p); vf_bar(0, 64, 1); vf_bra(l1); vf_lab(l2);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 25, "barriers: a loop exit after an unguarded back-branch is exempt", vf25)

static void vf26(void)
{
    nv_opnd_t p;
    uint32_t l1, l2, l3;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2); is_lbl(&l3);
    vf_gbra(l3, p); vf_lab(l1); vf_gbra(l2, p); vf_bar(0, 64, 1); vf_bra(l1); vf_lab(l2); vf_lab(l3);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 26, "barriers: a looped bar.sync inside a Cond is E079", vf26)

static void vf27(void)
{
    nv_opnd_t p;
    uint32_t l1, l2, l3;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2); is_lbl(&l3);
    vf_gbra(l3, p); vf_lab(l1); vf_mov(vf_r()); vf_bra(l1); vf_lab(l2); vf_lab(l3); vf_bar(0, 64, 1);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 27, "barriers: a bar.sync after a Cond that ends in a loop passes", vf27)

static void vf28(void)
{
    nv_opnd_t p;
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2);
    vf_gbra(l1, p); vf_mov(vf_r()); vf_bra(l2); vf_lab(l1); vf_bar(0, 64, 1); vf_lab(l2);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 28, "barriers: bar.sync in the else body is E079", vf28)

static void vf29(void)
{
    nv_opnd_t p;
    uint32_t l1, l2;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1); is_lbl(&l2);
    vf_gbra(l1, p); vf_mov(vf_r()); vf_bra(l2); vf_lab(l1); vf_mov(vf_r()); vf_lab(l2); vf_bar(0, 64, 1);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 29, "barriers: a bar.sync after an if/else passes", vf29)

/* ---- E083: size ---- */

static void vf30(void)
{
    CHEQ(vf_fresh(), 0);
    vf_mov(vf_r());
    is_cmt(vf_pool("short"));
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 30, "size: a short list passes", vf30)

/* is_ins refuses past IR_MAX_INS (E083 from the selector), so the list is
 * filled to the cap and the check for one record more is reached through the
 * selector's own refusal; the verifier then still accepts the full list. */
static void vf31(void)
{
    uint32_t i, n;
    nv_opnd_t a;

    CHEQ(vf_fresh(), 0);
    a = vf_r();
    th_cap_on();
    for (i = 0; i <= IR_MAX_INS; i++) vf_mov(a);
    (void)th_cap_off(eb, sizeof eb);
    (void)is_recs(&n);
    CHECK(n <= IR_MAX_INS);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 31, "size: the selector stops at IR_MAX_INS and the full list verifies", vf31)

/* ---- vf_run end to end ---- */

static void vf32(void)
{
    nv_opnd_t p;
    uint32_t l1;

    CHEQ(vf_fresh(), 0);
    p = vf_p(); is_lbl(&l1);
    vf_gbra(l1, p); vf_bar(0, 64, 1); vf_lab(l1);
    CHEQ(vf_chk(), 79);
    PASS();
}
TH_REG("vf", 32, "run: a list with a bad barrier refuses", vf32)

static void vf33(void)
{
    CHEQ(vf_fresh(), 0);
    vf_mov(vf_r());
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 33, "run: a good list passes", vf33)

static void vf34(void)
{
    CHEQ(vf_fresh(), 0);
    CHEQ(is_run(kr_get()), 0);
    CHEQ(vf_chk(), 0);
    PASS();
}
TH_REG("vf", 34, "run: the real empty-Tile record list passes", vf34)
