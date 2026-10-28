/* emit.c -- em_*: records to PTX text; capped buffer with overflow flag.
 *
 * The module is built in one static buffer of IR_MAX_OUT bytes. em_apnd is the
 * only writer: an append that would not fit sets the overflow flag and adds
 * nothing (never a cut-off line), and em_run turns the flag into E084 so no
 * partial module ever leaves. This emitter does not write to stdout; main.c
 * does the single fwrite.
 *
 * Operand conventions the selector must follow (the mnemonic table below
 * says which shape each opcode prints with):
 *   ld/st/red  address = src[0] register plus optional constant byte offset
 *              in src[1] (OP_NONE = none); ld: dst is the value; st/red:
 *              the value is src[2].
 *   v2/v4      the value operand names the FIRST of a run of consecutive
 *              registers of its file, printed as {%f5, %f6, %f7, %f8}.
 *   cp.async   shared address src[0] (+ src[1]), global address src[2]
 *              (+ src[3]); copy size 4 bytes (.ca) or 16 bytes (.cg).
 *   OP_SYM     [param_i] in an ld.param record, otherwise smem_<i>, or
 *              smem_pool for a slot of IR_MAX_SMEM.
 *   NV_LABEL   OP_LBL in dst, printed `L_<n>:`. NV_BRA: OP_LBL in src[0].
 * Labels are always spelled L_<n>. */

#include <stdio.h>
#include <string.h>
#include "irptx.h"
#include "isel.h"
#include "kir.h"

/* ---- Buffer ---- */

static struct {
    char     buf[IR_MAX_OUT];
    uint32_t len;
    uint8_t  ovf;                   /* 1: an append did not fit */
    uint8_t  bad;                   /* 1: a record we cannot print (our bug) */
} EM;

/* Append `s` whole, or set the overflow flag and append nothing. */
static void em_apnd(const char *s)
{
    size_t n = strlen(s);

    if (EM.ovf) return;
    if (n > (size_t)IR_MAX_OUT - EM.len) {
        EM.ovf = 1;
        return;
    }
    memcpy(EM.buf + EM.len, s, n);
    EM.len += (uint32_t)n;
}

/* Append a number with a printf format taking one int64/uint32 value. */
static void em_num(const char *fmt, long long v)
{
    char t[40];
    int n = snprintf(t, sizeof t, fmt, v);

    if (n < 0 || (size_t)n >= sizeof t) { EM.bad = 1; return; }
    em_apnd(t);
}

/* An f32 immediate from its bit pattern: 0fXXXXXXXX. */
static void em_hex(uint32_t bits)
{
    char t[16];
    int n = snprintf(t, sizeof t, "0f%08X", (unsigned)bits);

    if (n < 0 || (size_t)n >= sizeof t) { EM.bad = 1; return; }
    em_apnd(t);
}

/* ---- Mnemonic table ---- */

typedef enum {
    SH_GEN = 0,                     /* mnemonic, then dst and sources, then tail */
    SH_NOP, SH_LABEL, SH_BRA, SH_PARAM,
    SH_LD, SH_LDV, SH_ST, SH_STV, SH_CPA
} em_sh_t;

typedef struct {
    const char *mn;
    uint8_t     sh;                 /* em_sh_t */
    uint8_t     w;                  /* vector width for SH_LDV / SH_STV */
    const char *tail;               /* fixed last operand or size, or NULL */
} em_ent_t;

#define G(m)      { m, SH_GEN, 0, NULL }
#define GT(m, t)  { m, SH_GEN, 0, t }

/* Indexed by nv_op_t. A missing entry leaves mn NULL and refuses at print. */
static const em_ent_t EM_TAB[] = {
    [NV_NOP] = { NULL, SH_NOP, 0, NULL },
    [NV_LABEL] = { NULL, SH_LABEL, 0, NULL },
    [NV_RET] = G("ret"),
    [NV_BRA] = { "bra", SH_BRA, 0, NULL },
    [NV_LD_PARAM_U64] = { "ld.param.u64", SH_PARAM, 0, NULL },
    [NV_CVTA_TO_GLOBAL] = G("cvta.to.global.u64"),
    [NV_MOV_U32] = G("mov.u32"), [NV_MOV_B32] = G("mov.b32"),
    [NV_MOV_F32] = G("mov.f32"), [NV_MOV_S32] = G("mov.s32"),
    [NV_CVT_RN_F32_S32] = G("cvt.rn.f32.s32"),
    [NV_CVT_RNI_F32_F32] = G("cvt.rni.f32.f32"), [NV_CVT_RZI_F32_F32] = G("cvt.rzi.f32.f32"),
    [NV_CVT_RZI_S32_F32] = G("cvt.rzi.s32.f32"), [NV_CVT_U32_U64] = G("cvt.u32.u64"),
    [NV_CVT_U64_U32] = G("cvt.u64.u32"), [NV_CVT_RN_F32_S64] = G("cvt.rn.f32.s64"),
    [NV_LD_GLOBAL_F32] = { "ld.global.f32", SH_LD, 0, NULL },
    [NV_LD_GLOBAL_V2_F32] = { "ld.global.v2.f32", SH_LDV, 2, NULL },
    [NV_LD_GLOBAL_V4_F32] = { "ld.global.v4.f32", SH_LDV, 4, NULL },
    [NV_ST_GLOBAL_F32] = { "st.global.f32", SH_ST, 0, NULL },
    [NV_ST_GLOBAL_V2_F32] = { "st.global.v2.f32", SH_STV, 2, NULL },
    [NV_ST_GLOBAL_V4_F32] = { "st.global.v4.f32", SH_STV, 4, NULL },
    [NV_LD_SHARED_F32] = { "ld.shared.f32", SH_LD, 0, NULL },
    [NV_LD_SHARED_V4_F32] = { "ld.shared.v4.f32", SH_LDV, 4, NULL },
    [NV_ST_SHARED_F32] = { "st.shared.f32", SH_ST, 0, NULL },
    [NV_ST_SHARED_V4_F32] = { "st.shared.v4.f32", SH_STV, 4, NULL },
    [NV_RED_GLOBAL_ADD_F32] = { "red.global.add.f32", SH_ST, 0, NULL },
    [NV_CP_ASYNC_CA] = { "cp.async.ca.shared.global", SH_CPA, 0, "4" },
    [NV_CP_ASYNC_CA8] = { "cp.async.ca.shared.global", SH_CPA, 0, "8" },
    [NV_CP_ASYNC_CG] = { "cp.async.cg.shared.global", SH_CPA, 0, "16" },
    [NV_CP_ASYNC_COMMIT] = G("cp.async.commit_group"),
    [NV_CP_ASYNC_WAIT] = G("cp.async.wait_group"),
    [NV_ADD_S32] = G("add.s32"), [NV_SUB_S32] = G("sub.s32"),
    [NV_MUL_LO_S32] = G("mul.lo.s32"), [NV_MAD_LO_S32] = G("mad.lo.s32"),
    [NV_MUL_WIDE_S32] = G("mul.wide.s32"), [NV_ADD_S64] = G("add.s64"),
    [NV_MIN_S32] = G("min.s32"),
    [NV_DIV_S32] = G("div.s32"), [NV_REM_S32] = G("rem.s32"),
    [NV_AND_B32] = G("and.b32"), [NV_OR_B32] = G("or.b32"), [NV_XOR_B32] = G("xor.b32"),
    [NV_SHL_B32] = G("shl.b32"), [NV_SHR_U32] = G("shr.u32"), [NV_SHR_S32] = G("shr.s32"),
    [NV_SELP_S32] = G("selp.s32"), [NV_SELP_F32] = G("selp.f32"), [NV_NEG_S32] = G("neg.s32"),
    [NV_MUL_WIDE_U32] = G("mul.wide.u32"), [NV_MAD_WIDE_U32] = G("mad.wide.u32"),
    [NV_SHR_U64] = G("shr.u64"), [NV_SHL_B64] = G("shl.b64"), [NV_OR_B64] = G("or.b64"),
    [NV_SHF_L_CLAMP_B32] = G("shf.l.clamp.b32"), [NV_SHF_R_CLAMP_B32] = G("shf.r.clamp.b32"),
    [NV_SETP_EQ_S32] = G("setp.eq.s32"), [NV_SETP_NE_S32] = G("setp.ne.s32"),
    [NV_SETP_LT_S32] = G("setp.lt.s32"), [NV_SETP_LE_S32] = G("setp.le.s32"),
    [NV_SETP_GT_S32] = G("setp.gt.s32"), [NV_SETP_GE_S32] = G("setp.ge.s32"),
    [NV_SETP_LT_F32] = G("setp.lt.f32"), [NV_SETP_GT_F32] = G("setp.gt.f32"),
    [NV_SETP_GE_F32] = G("setp.ge.f32"), [NV_SETP_EQ_F32] = G("setp.eq.f32"),
    [NV_AND_PRED] = G("and.pred"), [NV_OR_PRED] = G("or.pred"), [NV_NOT_PRED] = G("not.pred"),
    [NV_ADD_RN_F32] = G("add.rn.f32"), [NV_SUB_RN_F32] = G("sub.rn.f32"),
    [NV_MUL_RN_F32] = G("mul.rn.f32"), [NV_DIV_RN_F32] = G("div.rn.f32"),
    [NV_FMA_RN_F32] = G("fma.rn.f32"), [NV_SQRT_RN_F32] = G("sqrt.rn.f32"),
    [NV_RCP_RN_F32] = G("rcp.rn.f32"),
    [NV_MAX_F32] = G("max.f32"), [NV_MIN_F32] = G("min.f32"),
    [NV_ABS_F32] = G("abs.f32"), [NV_NEG_F32] = G("neg.f32"),
    [NV_COPYSIGN_F32] = G("copysign.f32"),
    [NV_EX2_APPROX_F32] = G("ex2.approx.f32"), [NV_LG2_APPROX_F32] = G("lg2.approx.f32"),
    [NV_RSQRT_APPROX_F32] = G("rsqrt.approx.f32"),
    [NV_MOV_TID_X] = GT("mov.u32", "%tid.x"), [NV_MOV_CTAID_X] = GT("mov.u32", "%ctaid.x"),
    [NV_ACTIVEMASK] = G("activemask.b32"), [NV_SHFL_BFLY] = G("shfl.sync.bfly.b32"),
    [NV_BAR_SYNC] = G("bar.sync"), [NV_BAR_WARP_SYNC] = G("bar.warp.sync")
};

/* The table must have exactly one slot per opcode: a negative array size
 * stops the build when nv_op_t grows (or shrinks) without this table. */
typedef char em_tab_len[(sizeof EM_TAB / sizeof EM_TAB[0] == (size_t)NV_NCOUNT) ? 1 : -1];

/* ---- Operands ---- */

static const char *const EM_RF[RF_N] = { "%r", "%rd", "%f", "%p" };

static void em_reg(uint8_t rf, uint32_t n)
{
    if (rf >= RF_N) { EM.bad = 1; return; }
    em_apnd(EM_RF[rf]);
    em_num("%lld", (long long)n);
}

/* One operand in generic position. */
static void em_opd(const nv_opnd_t *o)
{
    switch (o->kind) {
    case OP_REG: em_reg(o->rf, o->v); break;
    case OP_IMM: em_num("%lld", (long long)(int32_t)o->v); break;
    case OP_FIMM: em_hex(o->v); break;
    case OP_LBL: em_num("L_%lld", (long long)o->v); break;
    case OP_SYM:
        if (o->v >= IR_MAX_SMEM) em_apnd("smem_pool");
        else em_num("smem_%lld", (long long)o->v);
        break;
    default: EM.bad = 1; break;
    }
}

/* A run of `w` consecutive registers starting at `o`: {%f5, %f6}. */
static void em_run_regs(const nv_opnd_t *o, unsigned w)
{
    unsigned i;

    if (o->kind != OP_REG || o->rf >= RF_N || o->v == 0 || o->v + w - 1u > rg_count((rf_t)o->rf)) {
        EM.bad = 1;
        return;
    }
    em_apnd("{");
    for (i = 0; i < w; i++) {
        if (i) em_apnd(", ");
        em_reg(o->rf, o->v + i);
    }
    em_apnd("}");
}

/* [base] or [base+off]; `off` may be OP_NONE or negative. */
static void em_addr(const nv_opnd_t *base, const nv_opnd_t *off)
{
    long long v;

    if (base->kind != OP_REG) { EM.bad = 1; return; }
    em_apnd("[");
    em_reg(base->rf, base->v);
    if (off->kind == OP_IMM) {
        v = (long long)(int32_t)off->v;
        em_num("+%lld", v);             /* ptxas takes [%r+-4], not [%r-4] */
    } else if (off->kind != OP_NONE) {
        EM.bad = 1;
    }
    em_apnd("]");
}

/* ---- Records ---- */

/* The text after the mnemonic for each shape. */
static void em_body(const nv_ins_t *r, const em_ent_t *e)
{
    unsigned i;

    switch (e->sh) {
    case SH_BRA: em_opd(&r->src[0]); return;
    case SH_PARAM:
        em_opd(&r->dst);
        em_num(", [param_%lld]", (long long)r->src[0].v);
        return;
    case SH_LD:
        em_opd(&r->dst); em_apnd(", ");
        em_addr(&r->src[0], &r->src[1]);
        return;
    case SH_LDV:
        em_run_regs(&r->dst, e->w); em_apnd(", ");
        em_addr(&r->src[0], &r->src[1]);
        return;
    case SH_ST:
        em_addr(&r->src[0], &r->src[1]); em_apnd(", ");
        em_opd(&r->src[2]);
        return;
    case SH_STV:
        em_addr(&r->src[0], &r->src[1]); em_apnd(", ");
        em_run_regs(&r->src[2], e->w);
        return;
    case SH_CPA:
        em_addr(&r->src[0], &r->src[1]); em_apnd(", ");
        em_addr(&r->src[2], &r->src[3]); em_apnd(", ");
        em_apnd(e->tail);
        return;
    default: break;
    }
    if (r->dst.kind != OP_NONE) em_opd(&r->dst);
    for (i = 0; i < 4 && r->src[i].kind != OP_NONE; i++) {
        if (r->dst.kind != OP_NONE || i) em_apnd(", ");
        em_opd(&r->src[i]);
    }
    if (e->tail) {
        if (r->dst.kind != OP_NONE || r->src[0].kind != OP_NONE) em_apnd(", ");
        em_apnd(e->tail);
    }
}

/* Print one record as lines of text. */
static void em_ins(const nv_ins_t *r)
{
    const em_ent_t *e;

    if (r->op >= NV_NCOUNT) { EM.bad = 1; return; }
    e = &EM_TAB[r->op];
    if (e->sh == SH_NOP) return;
    if (e->sh == SH_LABEL) {
        em_opd(&r->dst);
        em_apnd(":\n");
        return;
    }
    if (!e->mn) { EM.bad = 1; return; }
    em_apnd("    ");
    if (r->prd.kind == OP_REG) {
        em_apnd(r->pneg ? "@!" : "@");
        em_opd(&r->prd);
        em_apnd(" ");
    } else if (r->prd.kind != OP_NONE) {
        EM.bad = 1;
    }
    em_apnd(e->mn);
    if (e->sh != SH_GEN || r->dst.kind != OP_NONE || r->src[0].kind != OP_NONE || e->tail)
        em_apnd(" ");
    em_body(r, e);
    em_apnd(";");
    if (r->cmt) { em_apnd("  // "); em_apnd(kr_str(r->cmt)); }
    em_apnd("\n");
}

/* ---- Module ---- */

/* Version stamp (P11) and the three directives (P1). */
static void em_head(const kir_t *k)
{
    em_apnd("//\n// irptx " IRPTX_VER " ");
    em_apnd(k->name);
    em_apnd("\n//\n.version 8.0\n.target sm_89\n.address_size 64\n\n");
}

/* `.reg .f32 %f<N>;` for file `rf`, nothing when it holds no registers. */
static void em_regs(rf_t rf, const char *decl)
{
    uint32_t n = rg_count(rf);

    if (n == 0) return;
    em_apnd("    .reg ");
    em_apnd(decl);
    em_num("<%lld>;\n", (long long)n + 1);
}

/* The entry: parameters with their IR names as comments, register and
 * shared declarations. The dynamic pool is declared at module scope, before
 * the entry, because ptxas only takes .extern .shared there. */
static void em_ent(const kir_t *k)
{
    const is_smem_t *sm = sm_get();
    uint32_t i;

    if (sm->dyn) em_apnd(".extern .shared .align 16 .b8 smem_pool[];\n\n");
    em_apnd(".visible .entry ");
    em_apnd(k->name);
    em_apnd("(\n");
    for (i = 0; i < k->narg && i < IR_MAX_ARG; i++) {
        em_num("    .param .u64 param_%lld", (long long)i);
        em_apnd(i + 1u < k->narg ? "," : " ");
        em_apnd("        // ");
        em_apnd(kr_str(k->arg[i].name));
        em_apnd("\n");
    }
    em_apnd(")\n");
    if (k->block[0] > 0) {
        em_num(".maxntid %lld", (long long)k->block[0]);
        em_num(", %lld", (long long)(k->block[1] > 0 ? k->block[1] : 1));
        em_num(", %lld\n", (long long)(k->block[2] > 0 ? k->block[2] : 1));
    }
    em_apnd("{\n");
    em_regs(RF_P, ".pred %p");
    em_regs(RF_R, ".b32  %r");
    em_regs(RF_RD, ".b64  %rd");
    em_regs(RF_F, ".f32  %f");
    if (!sm->dyn) {
        for (i = 0; i < sm->narr && i < IR_MAX_SMEM; i++) {
            em_num("    .shared .align %lld", (long long)sm->arr[i].align);
            em_num(" .f32 smem_%lld", (long long)i);
            em_num("[%lld];\n", (long long)sm->arr[i].elems);
        }
    }
    em_apnd("\n");
}

/* ---- Entry ---- */

int em_run(const char **out, size_t *len)
{
    const kir_t *k = kr_get();
    const nv_ins_t *r;
    uint32_t n, i;

    if (out) *out = "";
    if (len) *len = 0;
    memset(&EM, 0, sizeof EM);
    r = is_recs(&n);
    em_head(k);
    em_ent(k);
    for (i = 0; i < n; i++) em_ins(&r[i]);
    em_apnd("}\n");
    if (EM.bad) return er_fail(E000, "records emission can print");
    if (EM.ovf) return er_fail(E084, "a module within IR_MAX_OUT characters");
    if (out) *out = EM.buf;
    if (len) *len = EM.len;
    return 0;
}
