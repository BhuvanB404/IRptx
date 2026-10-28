/* isel.c -- one lowering per Kernel IR statement.
 *
 * labels, the record array, the scoped name bindings, home registers and the
 * accumulator table. Sections below hold the state helpers; the walk that
 * lowers statements lands in T019 and goes above Entry. */

#include <string.h>
#include "irptx.h"
#include "isel.h"
#include "kir.h"
#include "kexpr.h"
#include "ops.h"

/* ---- State ---- */

/* A name's single home register (ruling H2). Kept apart from the scoped
 * bindings: leaving a loop drops its binding but not its register. */
typedef struct {
    uint32_t  name;
    nv_opnd_t reg;
} is_home_t;

static struct {
    nv_ins_t  ins[IR_MAX_INS];
    uint32_t  nins;
    uint32_t  nreg[RF_N];           /* last number handed out per file; 0 = none */
    uint32_t  nlbl;
    is_bind_t bind[IR_MAX_BIND];
    uint32_t  nbind;
    uint32_t  mark[IR_MAX_DEP];     /* nbind at each open scope's start */
    uint32_t  dep;
    is_home_t home[IR_MAX_BIND];
    uint32_t  nhome;
    is_acc_t  acc[IR_MAX_ACC];
    uint32_t  nacc;
    const kir_arg_t *arg;           /* the kernel's parameters, set by mm_params */
    uint32_t  narg;
    nv_opnd_t base[IR_MAX_ARG];     /* %rd holding each global base */
    is_smem_t sm;                   /* shared layout, filled by sm_add */
    uint32_t  xlbl;                 /* exit label of the tile guard; 0 = none */
    uint32_t  ncell;                /* N, the product of the tile's extents */
    nv_opnd_t tid;                  /* %tid.x, the base of every thread index */
    uint32_t  block;                /* block size, for the warp-width checks */
} S;

void is_reset(void)
{
    memset(&S, 0, sizeof S);
}

/* Names are string pool offsets and the reader does not intern, so equal
 * offsets or equal text both mean the same name. */
static int is_same(uint32_t a, uint32_t b)
{
    return a == b || strcmp(kr_str(a), kr_str(b)) == 0;
}

/* ---- Registers ---- */

int rg_new(rf_t rf, nv_opnd_t *out)
{
    uint32_t n;

    if (IX_CHK(rf, RF_N)) return er_fail(E000, "a register file that does not exist");
    n = IX_PNEW(S.nreg[rf], IR_MAX_REG);
    if (n == 0) return er_fail(E081, "more registers in one file than the pool holds");
    *out = is_oreg(rf, n);
    return 0;
}

int rg_home(uint32_t name, rf_t rf, nv_opnd_t *out)
{
    uint32_t i, h;

    for (i = 0; i < S.nhome; i++) {
        if (!is_same(S.home[i].name, name)) continue;
        if (S.home[i].reg.rf != (uint8_t)rf)
            return er_fail(E029, "a name used both as an integer loop variable and as an f32 value");
        *out = S.home[i].reg;
        return 0;
    }
    h = IX_PNEW(S.nhome, IR_MAX_BIND);
    if (h == 0) return er_fail(E081, "more home registers than the table holds");
    if (rg_new(rf, &S.home[h - 1u].reg)) {
        S.nhome--;                  /* give the slot back: no half-made home */
        return 1;
    }
    S.home[h - 1u].name = name;
    *out = S.home[h - 1u].reg;
    return 0;
}

uint32_t rg_count(rf_t rf)
{
    return IX_CHK(rf, RF_N) ? 0u : S.nreg[rf];
}

/* ---- Scopes ---- */

int is_push(void)
{
    if (S.dep >= IR_MAX_DEP) return er_fail(E082, "scopes nested deeper than the limit");
    S.mark[S.dep++] = S.nbind;
    return 0;
}

int is_pop(void)
{
    if (S.dep == 0) return er_fail(E000, "a scope closed that was never opened");
    S.nbind = S.mark[--S.dep];
    return 0;
}

/* Bind, with the uniformity flag a loop's trip-count proof reads back. */
static int is_bind(uint32_t name, nv_opnd_t opd, uint8_t uni)
{
    uint32_t b = IX_PNEW(S.nbind, IR_MAX_BIND);

    if (b == 0) return er_fail(E081, "more live name bindings than the table holds");
    S.bind[b - 1u].name = name;
    S.bind[b - 1u].opd = opd;
    S.bind[b - 1u].uni = uni;
    return 0;
}

int is_bset(uint32_t name, nv_opnd_t opd) { return is_bind(name, opd, 0); }
int is_bsetu(uint32_t name, nv_opnd_t opd) { return is_bind(name, opd, 1); }

int is_funi(uint32_t name)
{
    uint32_t i = S.nbind;

    while (i > 0) {
        i--;
        if (is_same(S.bind[i].name, name)) return S.bind[i].uni;
    }
    return 0;
}

/* Newest-first lookup that reports instead of refusing: 1 and *out on a hit,
 * 0 when the name is not bound. A handler that can take two paths (a buffer
 * that may be a parameter or a shared array) asks here first, because is_find
 * prints on a miss and the other path has not been tried yet. */
static int is_try(uint32_t name, nv_opnd_t *out)
{
    uint32_t i = S.nbind;

    while (i > 0) {
        i--;
        if (is_same(S.bind[i].name, name)) {
            *out = S.bind[i].opd;
            return 1;
        }
    }
    return 0;
}

int is_find(uint32_t name, nv_opnd_t *out)
{
    char msg[IR_MAX_NAME + 64];

    if (is_try(name, out)) return 0;
    strcpy(msg, "name '");
    strncat(msg, kr_str(name), IR_MAX_NAME);
    strcat(msg, "' that is bound in no enclosing scope");
    return er_fail(E022, msg);
}

/* ---- Accumulators ---- */

is_acc_t *is_afind(uint32_t name)
{
    uint32_t i;

    for (i = 0; i < S.nacc; i++)
        if (is_same(S.acc[i].name, name)) return &S.acc[i];
    return NULL;
}

int is_aadd(uint32_t name, nv_opnd_t reg, uint8_t op, uint32_t ident,
            uint8_t has_init, is_acc_t **out)
{
    uint32_t a;
    is_acc_t *p;

    if (is_afind(name)) return er_fail(E028, "an accumulator or Init name declared twice");
    a = IX_PNEW(S.nacc, IR_MAX_ACC);
    if (a == 0) return er_fail(E081, "more accumulators than the table holds");
    p = &S.acc[a - 1u];
    p->name = name;
    p->reg = reg;
    p->op = op;
    p->ident = ident;
    p->has_init = has_init;
    p->seeded = 0;
    *out = p;
    return 0;
}

/* ---- Operands ---- */

static nv_opnd_t is_opnd(op_kind_t kind, rf_t rf, uint32_t v)
{
    nv_opnd_t o;

    o.kind = (uint8_t)kind;
    o.rf = (uint8_t)rf;
    o.v = v;
    return o;
}

nv_opnd_t is_onone(void) { return is_opnd(OP_NONE, RF_R, 0); }
nv_opnd_t is_oreg(rf_t rf, uint32_t n) { return is_opnd(OP_REG, rf, n); }
nv_opnd_t is_oimm(int32_t v) { return is_opnd(OP_IMM, RF_R, (uint32_t)v); }
nv_opnd_t is_ofim(uint32_t bits) { return is_opnd(OP_FIMM, RF_R, bits); }
nv_opnd_t is_olbl(uint32_t n) { return is_opnd(OP_LBL, RF_R, n); }
nv_opnd_t is_osym(uint32_t id) { return is_opnd(OP_SYM, RF_R, id); }

/* ---- Records ---- */

nv_ins_t *is_new(nv_op_t op)
{
    uint32_t n = IX_PNEW(S.nins, IR_MAX_INS);
    nv_ins_t *r;

    if (n == 0) {
        er_fail(E083, "more instructions than the record array holds");
        return NULL;
    }
    r = &S.ins[n - 1u];
    memset(r, 0, sizeof *r);
    r->op = (uint16_t)op;
    return r;
}

int is_ins(nv_op_t op, nv_opnd_t d, nv_opnd_t a, nv_opnd_t b, nv_opnd_t c)
{
    nv_ins_t *r = is_new(op);

    if (!r) return 1;
    r->dst = d;
    r->src[0] = a;
    r->src[1] = b;
    r->src[2] = c;
    r->nsrc = (c.kind != OP_NONE) ? 3u : (b.kind != OP_NONE) ? 2u : (a.kind != OP_NONE) ? 1u : 0u;
    return 0;
}

int is_guard(nv_opnd_t prd, int neg)
{
    if (S.nins == 0) return er_fail(E000, "a guard with no instruction to guard");
    S.ins[S.nins - 1u].prd = prd;
    S.ins[S.nins - 1u].pneg = neg ? 1u : 0u;
    return 0;
}

/* Mark the last record, a loop's exit branch, as taken on the same trip by
 * every thread of the block, so verify may allow a barrier in its body. */
static void is_muni(void)
{
    if (S.nins > 0) S.ins[S.nins - 1u].uni = 1;
}

int is_cmt(uint32_t off)
{
    if (S.nins == 0) return er_fail(E000, "a comment with no instruction to attach to");
    S.ins[S.nins - 1u].cmt = off;
    return 0;
}

int is_lbl(uint32_t *n)
{
    uint32_t l = IX_PNEW(S.nlbl, IR_MAX_REG);

    if (l == 0) return er_fail(E081, "more labels than the pool holds");
    *n = l;
    return 0;
}

const nv_ins_t *is_recs(uint32_t *n)
{
    *n = S.nins;
    return S.ins;
}

uint32_t is_nlbl(void)
{
    return S.nlbl;
}

/* ---- Memory ---- */

#define MM_ELEM 4                   /* bytes in an f32 element */
#define MM_STATIC_MAX 49152u        /* static shared limit; past it one pool */
#define SM_MAX_BYTES  (1u << 30)    /* layout cap, far past any device: keeps sums in 32 bits */

/* Is `o` an s32 value: a %r register or an immediate? */
static int mm_ints(nv_opnd_t o)
{
    return o.kind == OP_IMM || (o.kind == OP_REG && o.rf == RF_R);
}

int mm_params(const kir_t *k)
{
    uint32_t i;
    nv_opnd_t raw, base;

    if (!k) return er_fail(E000, "a parameter prologue with no kernel");
    if (IX_CHK(k->narg, IR_MAX_ARG + 1u)) return er_fail(E081, "more parameters than the table holds");
    S.arg = k->arg;
    S.narg = k->narg;
    for (i = 0; i < k->narg; i++) {
        if (k->arg[i].dt != KD_F32) return er_fail(E024, "a buffer whose dtype is not f32");
        if (rg_new(RF_RD, &raw) || rg_home(k->arg[i].name, RF_RD, &base)) return 1;
        if (is_ins(NV_LD_PARAM_U64, raw, is_osym(i), is_onone(), is_onone())) return 1;
        if (is_cmt(k->arg[i].name)) return 1;
        if (is_ins(NV_CVTA_TO_GLOBAL, base, raw, is_onone(), is_onone())) return 1;
        S.base[i] = base;
    }
    return 0;
}

int mm_arg(uint32_t name, uint32_t *idx)
{
    uint32_t i;
    char msg[IR_MAX_NAME + 64];

    for (i = 0; i < S.narg; i++) {
        if (!is_same(S.arg[i].name, name)) continue;
        *idx = i;
        return 0;
    }
    strcpy(msg, "buffer '");
    strncat(msg, kr_str(name), IR_MAX_NAME);
    strcat(msg, "' that is not a kernel parameter");
    return er_fail(E022, msg);
}

/* Row-major strides of `dim` into st[]: st[k] = product of dim[k+1..]. Each
 * must stay within s32 or the index math would not be s32 (R6). */
static int mm_strides(const int32_t *dim, uint32_t rank, int64_t *st)
{
    int64_t s = 1;
    uint32_t k = rank;

    while (k > 0) {
        k--;
        if (dim[k] < 1) return er_fail(E059, "an extent below 1");
        st[k] = s;
        s *= dim[k];
        if (s > INT32_MAX) return er_fail(E059, "a buffer with more than 2^31 elements");
    }
    return 0;
}

/* Add `r * stride` (a non-constant term) into the running sum *acc. */
static int mm_term(nv_opnd_t r, int64_t stride, nv_opnd_t *acc)
{
    nv_opnd_t t;

    if (stride == 1 && acc->kind == OP_NONE) { *acc = r; return 0; }
    if (rg_new(RF_R, &t)) return 1;
    if (stride == 1) {
        if (is_ins(NV_ADD_S32, t, *acc, r, is_onone())) return 1;
    } else if (acc->kind == OP_NONE) {
        if (is_ins(NV_MUL_LO_S32, t, r, is_oimm((int32_t)stride), is_onone())) return 1;
    } else {
        if (is_ins(NV_MAD_LO_S32, t, r, is_oimm((int32_t)stride), *acc)) return 1;
    }
    *acc = t;
    return 0;
}

/* *cst += v * stride. v is s32 and the stride at most 2^31 - 1, so the
 * product fits in 63 bits; the sum is checked before it is made, since a
 * rank-8 index of extreme constants would otherwise wrap an int64. */
static int mm_cadd(int64_t *cst, int32_t v, int64_t stride)
{
    int64_t p = (int64_t)v * stride;

    if (p > 0 ? *cst > INT64_MAX - p : *cst < INT64_MIN - p)
        return er_fail(E059, "a constant index outside s32");
    *cst += p;
    return 0;
}

/* One index item: fold it into the constant, or lower it into the sum. */
static int mm_item(const kir_val_t *it, int64_t stride, int64_t *cst, nv_opnd_t *acc)
{
    int32_t v = 0;
    int isc = 0;
    nv_opnd_t r;

    if (kx_fold(it, &v, &isc)) return 1;
    if (isc) return mm_cadd(cst, v, stride);
    if (kx_eval(it, &r)) return 1;
    if (!mm_ints(r)) return er_fail(E059, "an index that is not an integer");
    if (r.kind == OP_IMM) return mm_cadd(cst, (int32_t)r.v, stride);
    return mm_term(r, stride, acc);
}

int mm_flat(const int32_t *dim, uint32_t rank, const kir_val_t *idx, nv_opnd_t *out)
{
    int64_t st[IR_MAX_DIM], cst = 0;
    nv_opnd_t acc = is_onone(), t;
    uint32_t k;

    if (!idx || idx->tag != KV_LIST) return er_fail(E059, "an index that is not a list");
    if (rank > IR_MAX_DIM) return er_fail(E081, "a buffer of rank above the limit");
    if (idx->n != rank)
        return er_fail(E059, "an index list whose length is not the buffer rank");
    if (mm_strides(dim, rank, st)) return 1;
    for (k = 0; k < rank; k++)
        if (mm_item(kr_kid(idx, k), st[k], &cst, &acc)) return 1;
    if (cst > INT32_MAX || cst < INT32_MIN) return er_fail(E059, "a constant index outside s32");
    if (acc.kind == OP_NONE) { *out = is_oimm((int32_t)cst); return 0; }
    if (cst != 0) {
        if (rg_new(RF_R, &t)) return 1;
        if (is_ins(NV_ADD_S32, t, acc, is_oimm((int32_t)cst), is_onone())) return 1;
        acc = t;
    }
    *out = acc;
    return 0;
}

int mm_gflat(uint32_t arg, const kir_val_t *idx, nv_opnd_t *out)
{
    if (IX_CHK(arg, S.narg)) return er_fail(E000, "a parameter index past the table");
    return mm_flat(S.arg[arg].dim, S.arg[arg].rank, idx, out);
}

/* 1 when a width-`w` (2 or 4) vector access at the flat index of `idx` is
 * provably aligned (ruling M6): every per-dimension index is affine and its
 * coefficients and constant, times that dimension's stride, are all divisible
 * by `w`. Each term then is, so the sum is, for any value of the variables.
 * The base is 16-byte aligned (cudaMalloc, and shared arrays are placed at
 * max(align, 16)). Silent: 0 means "use scalars", never an error. Call it
 * after mm_flat accepted the same arguments. */
int mm_vec(const int32_t *dim, uint32_t rank, const kir_val_t *idx, uint32_t w)
{
    int64_t st[IR_MAX_DIM];
    kx_aff_t a;
    uint32_t k, j;

    if (w != 2u && w != 4u) return 0;
    if (!idx || idx->tag != KV_LIST || idx->n != rank || rank > IR_MAX_DIM) return 0;
    if (mm_strides(dim, rank, st)) return 0;
    for (k = 0; k < rank; k++) {
        if (!kx_aff(kr_kid(idx, k), &a)) return 0;
        if (a.cst * st[k] % (int64_t)w != 0) return 0;
        for (j = 0; j < a.nvar; j++)
            if (a.coef[j] * st[k] % (int64_t)w != 0) return 0;
    }
    return 1;
}

/* Byte offset of element `flat` as an s32 immediate, or E059. */
static int mm_bytes(int32_t flat, int32_t *out)
{
    int64_t b = (int64_t)flat * MM_ELEM;

    if (b > INT32_MAX || b < INT32_MIN) return er_fail(E059, "a byte offset outside s32");
    *out = (int32_t)b;
    return 0;
}

int mm_gaddr(uint32_t arg, nv_opnd_t flat, nv_opnd_t *areg, int32_t *boff)
{
    nv_opnd_t off, sum;

    if (IX_CHK(arg, S.narg)) return er_fail(E000, "a parameter index past the table");
    if (!mm_ints(flat)) return er_fail(E000, "a flat index that is not an s32");
    if (flat.kind == OP_IMM) {
        *areg = S.base[arg];
        return mm_bytes((int32_t)flat.v, boff);
    }
    if (rg_new(RF_RD, &off) || rg_new(RF_RD, &sum)) return 1;
    if (is_ins(NV_MUL_WIDE_S32, off, flat, is_oimm(MM_ELEM), is_onone())) return 1;
    if (is_ins(NV_ADD_S64, sum, S.base[arg], off, is_onone())) return 1;
    *areg = sum;
    *boff = 0;
    return 0;
}

int sm_add(uint32_t name, const int32_t *dim, uint32_t rank, uint32_t align, uint32_t *id)
{
    is_smem_t *m = &S.sm;
    uint32_t al = (align < 16u) ? 16u : align;
    uint64_t off, end, elems = 1;
    is_sarr_t *a;
    uint32_t i;

    if (m->narr >= IR_MAX_SMEM) return er_fail(E081, "more shared arrays than the table holds");
    if (!dim || rank == 0 || rank > IR_MAX_DIM) return er_fail(E059, "a shared array with no extents");
    /* checked at every multiply: four extents of 65536 make 2^64, which
     * wraps a uint64 to 0 and would pass any check made after the loop */
    for (i = 0; i < rank; i++) {
        if (dim[i] < 1) return er_fail(E059, "a shared array with an extent below 1");
        elems *= (uint64_t)dim[i];
        if (elems > SM_MAX_BYTES / MM_ELEM)
            return er_fail(E059, "a shared array of zero, huge or oddly aligned size");
    }
    off = ((uint64_t)m->total + al - 1u) / al * al;
    end = off + elems * MM_ELEM;
    if (al > 4096u || (align & (align - 1u)) != 0 || end > SM_MAX_BYTES)
        return er_fail(E059, "a shared array of zero, huge or oddly aligned size");
    a = &m->arr[m->narr];
    a->name = name;
    a->elems = (uint32_t)elems;
    a->align = al;
    a->off = (uint32_t)off;
    a->rank = (uint8_t)rank;
    for (i = 0; i < rank; i++) a->dim[i] = dim[i];
    m->total = (uint32_t)end;
    m->dyn = (m->total > MM_STATIC_MAX) ? 1u : 0u;
    *id = m->narr++;
    return 0;
}

/* Silent sm_find: 1 and *id when a shared array called `name` exists. A Write
 * or a Load may target either a parameter or a shared array, and only one of
 * the two lookups is allowed to print. */
static int is_shv(uint32_t name, uint32_t *id)
{
    uint32_t i;

    for (i = 0; i < S.sm.narr; i++) {
        if (!is_same(S.sm.arr[i].name, name)) continue;
        *id = i;
        return 1;
    }
    return 0;
}

/* One Smem statement: add its array to the layout. A Write can store into a
 * shared array several statements before the Smem that declares it (r01 does
 * exactly that), so this runs in the pre-pass, not in the walk. */
static int is_sdecl(const kir_node_t *nd, uint32_t *id)
{
    const kir_val_t *nm = kr_fld(nd, "name"), *ex = kr_fld(nd, "extents");
    const kir_val_t *dt = kr_fld(nd, "dtype"), *al = kr_fld(nd, "align");
    int32_t dim[IR_MAX_DIM], v = 0;
    uint32_t i, align = 0;
    int isc = 0;

    if (!nm || !ex || !dt) return er_fail(E015, "a Smem missing name, extents or dtype");
    if (nm->tag != KV_STR || ex->tag != KV_LIST)
        return er_fail(E015, "a Smem whose name or extents have the wrong shape");
    if (dt->tag != KV_STR || (strcmp(kr_str(dt->ref), "float") != 0 &&
                              strcmp(kr_str(dt->ref), "f32") != 0))
        return er_fail(E024, "a Smem whose dtype is not f32");
    if (al && al->tag == KV_INT) {
        if (al->i < 0 || al->i > INT32_MAX) return er_fail(E059, "a Smem alignment below 0");
        align = (uint32_t)al->i;
    }
    if (ex->n == 0 || ex->n > IR_MAX_DIM)
        return er_fail(E059, "a Smem with no extents or more than IR_MAX_DIM of them");
    for (i = 0; i < ex->n; i++) {
        if (kx_fold(kr_kid(ex, i), &v, &isc)) return 1;
        if (!isc || v < 1) return er_fail(E059, "a Smem extent that is not a constant above 0");
        dim[i] = v;
    }
    return sm_add(nm->ref, dim, ex->n, align, id);
}



int sm_find(uint32_t name, uint32_t *id)
{
    uint32_t i;
    char msg[IR_MAX_NAME + 64];

    for (i = 0; i < S.sm.narr; i++) {
        if (!is_same(S.sm.arr[i].name, name)) continue;
        *id = i;
        return 0;
    }
    strcpy(msg, "shared array '");
    strncat(msg, kr_str(name), IR_MAX_NAME);
    strcat(msg, "' that was never declared");
    return er_fail(E022, msg);
}

const is_smem_t *sm_get(void)
{
    return &S.sm;
}

int sm_addr(uint32_t id, nv_opnd_t flat, nv_opnd_t *areg, int32_t *boff)
{
    const is_sarr_t *a;
    nv_opnd_t base, sum;
    int64_t pool, b = 0;
    int32_t b32 = 0;

    if (IX_CHK(id, S.sm.narr)) return er_fail(E000, "a shared array id past the table");
    if (!mm_ints(flat)) return er_fail(E000, "a flat index that is not an s32");
    a = &S.sm.arr[id];
    pool = S.sm.dyn ? (int64_t)a->off : 0;
    if (flat.kind == OP_IMM) {
        if (mm_bytes((int32_t)flat.v, &b32)) return 1;
        b = b32;
    }
    if (b + pool > INT32_MAX || b + pool < INT32_MIN)
        return er_fail(E059, "a shared byte offset outside s32");
    if (rg_new(RF_R, &base)) return 1;
    if (is_ins(NV_MOV_U32, base, is_osym(S.sm.dyn ? IR_MAX_SMEM : id), is_onone(), is_onone()))
        return 1;
    *areg = base;
    *boff = (int32_t)(b + pool);
    if (flat.kind == OP_IMM) return 0;
    if (rg_new(RF_R, &sum)) return 1;
    if (is_ins(NV_MAD_LO_S32, sum, flat, is_oimm(MM_ELEM), base)) return 1;
    *areg = sum;
    return 0;
}

/* ---- Thread Model ---- */

#define TM_WARP 32                  /* lanes per warp; lane = tid & 31, warp = tid >> 5 */

/* What a scan of the tile body found. */
typedef struct {
    uint8_t bar;                    /* a Sync, TreeHalve, WarpShuffle or CpAsyncWait */
    uint8_t smem;                   /* an Smem declaration */
} is_tflg_t;

/* One cursor of the explicit scan stack: a node (walking its fields) or a
 * list (walking its items). */
typedef struct {
    const kir_node_t *nd;
    const kir_val_t  *lst;
    uint32_t          i;
} is_cur_t;

/* Note what node `nd` is, for the guard policy. */
static void is_kind(const kir_node_t *nd, is_tflg_t *f)
{
    switch ((kr_kind_t)nd->kind) {
    case KR_SYNC: case KR_TREEHALVE: case KR_WARPSHUFFLE: case KR_CPASYNCWAIT:
        f->bar = 1;
        break;
    case KR_SMEM:
        f->smem = 1;
        break;
    default:
        break;
    }
}

/* Push the node or list `v` points at on the scan stack; any other value is
 * a leaf. E082 when the nesting passes IR_MAX_DEP (the reader limits it too,
 * so this is a second fence, not the first). */
static int is_spush(is_cur_t *st, uint32_t *sp, const kir_val_t *v, is_tflg_t *f)
{
    is_cur_t c;

    c.nd = NULL;
    c.lst = NULL;
    c.i = 0;
    if (v->tag == KV_NODE) {
        c.nd = kr_nod(v);
        if (!c.nd) return er_fail(E011, "a node reference past the node table");
        is_kind(c.nd, f);
    } else if (v->tag == KV_LIST) {
        c.lst = v;
    } else {
        return 0;
    }
    if (*sp >= IR_MAX_DEP) return er_fail(E082, "nesting deeper than the limit in a tile body");
    st[(*sp)++] = c;
    return 0;
}

/* Scan the subtree under `root` (a value) for barriers and Smem. No
 * recursion: a stack of cursors bounded by IR_MAX_DEP, and a step budget of
 * twice every field and item (one step per value, one per pop), so a
 * malformed pool cannot spin it. */
static int is_scan(const kir_t *k, const kir_val_t *root, is_tflg_t *f)
{
    is_cur_t st[IR_MAX_DEP];
    uint32_t sp = 0;
    IX_GUARD(g, 2u * ((uint64_t)k->nfld + k->nitem) + 2u);

    f->bar = 0;
    f->smem = 0;
    if (is_spush(st, &sp, root, f)) return 1;
    while (sp > 0 && g--) {
        is_cur_t *c = &st[sp - 1u];
        const kir_val_t *v;

        if (c->nd) {
            if (c->i >= c->nd->nfld) { sp--; continue; }
            if (IX_CHK(c->nd->fld0 + c->i, k->nfld + 1u))   /* fields are 1-based */
                return er_fail(E011, "a field index past the field table");
            v = &k->fld[c->nd->fld0 + c->i++].val;
        } else {
            if (c->i >= c->lst->n) { sp--; continue; }
            v = kr_kid(c->lst, c->i++);
            if (!v) return er_fail(E011, "a list item past the item table");
        }
        if (is_spush(st, &sp, v, f)) return 1;
    }
    if (sp > 0) return er_fail(E000, "a tile scan that ran out of steps");
    return 0;
}

/* Declare every Smem in the tile body, in document order, before the walk:
 * sm_addr reads static-or-dynamic off the finished layout, so every array has
 * to be placed before the first access to any of them. A value is pushed
 * exactly once, so checking the node a push just landed on declares each
 * array exactly once. */
static int is_spre(const kir_t *k, const kir_val_t *root)
{
    is_cur_t st[IR_MAX_DEP];
    is_tflg_t f;
    uint32_t sp = 0, id;
    IX_GUARD(g, 2u * ((uint64_t)k->nfld + k->nitem) + 2u);

    if (is_spush(st, &sp, root, &f)) return 1;
    while (sp > 0 && g--) {
        is_cur_t *c = &st[sp - 1u];
        const kir_val_t *v;

        if (c->nd) {
            if (c->i >= c->nd->nfld) { sp--; continue; }
            if (IX_CHK(c->nd->fld0 + c->i, k->nfld + 1u))   /* fields are 1-based */
                return er_fail(E011, "a field index past the field table");
            v = &k->fld[c->nd->fld0 + c->i++].val;
        } else {
            if (c->i >= c->lst->n) { sp--; continue; }
            v = kr_kid(c->lst, c->i++);
            if (!v) return er_fail(E011, "a list item past the item table");
        }
        if (is_spush(st, &sp, v, &f)) return 1;
        if (v->tag == KV_NODE && st[sp - 1u].nd && st[sp - 1u].nd->kind == KR_SMEM &&
            is_sdecl(st[sp - 1u].nd, &id)) return 1;
    }
    if (sp > 0) return er_fail(E000, "a shared pre-pass that ran out of steps");
    return 0;
}

/* Offset in the string pool of the text `s`, 0 when the kernel never spells
 * it (then no expression can name it either). Strings sit end to end. */
static uint32_t is_pool(const kir_t *k, const char *s)
{
    uint32_t i = 1u;
    IX_GUARD(g, k->nstr);

    while (i < k->nstr && g--) {
        if (strcmp(k->str + i, s) == 0) return i;
        i += (uint32_t)strlen(k->str + i) + 1u;
    }
    return 0;
}

static int is_pow2(int32_t v) { return v > 0 && (v & (v - 1)) == 0; }

static int32_t is_log2(int32_t v)
{
    int32_t n = 0;

    while (v > 1) { v >>= 1; n++; }
    return n;
}

/* Tile extents: fold each axis' Dim to a constant >= 1. Fills ext[] and nm[]
 * (name offsets) and returns the axis count in *na, N in *n. */
static int is_axes(const kir_node_t *tile, int32_t *ext,
                   uint32_t *nm, uint32_t *na, int64_t *n)
{
    const kir_val_t *ax = kr_fld(tile, "axes");
    uint32_t i;

    *n = 1;
    if (!ax || ax->tag != KV_LIST) return er_fail(E015, "a Tile with no axes list");
    if (ax->n > IR_MAX_DIM) return er_fail(E081, "a Tile with more axes than the limit");
    for (i = 0; i < ax->n; i++) {
        const kir_node_t *a = kr_nod(kr_kid(ax, i));
        const kir_val_t *name = a ? kr_fld(a, "name") : NULL;
        const kir_val_t *dim = a ? kr_fld(a, "extent") : NULL;
        int32_t e = 0;
        int isc = 0;

        if (!name || !dim || name->tag != KV_STR) return er_fail(E015, "an Axis with no name or extent");
        if (kx_fold(dim, &e, &isc)) return 1;
        if (!isc) return er_fail(E059, "a Tile axis whose extent is not a constant");
        if (e < 1) return er_fail(E059, "a Tile axis whose extent is below 1");
        *n *= e;
        if (*n > INT32_MAX) return er_fail(E059, "a Tile with more than 2^31 cells");
        ext[i] = e;
        nm[i] = name->ref;
    }
    *na = ax->n;
    return 0;
}

/* gid = ctaid.x * block + tid.x. The launch block size is known, so it is an
 * immediate and %ntid.x is never read; a one-block grid is just tid. */
static int is_gid(const kir_t *k, nv_opnd_t *tid, nv_opnd_t *gid)
{
    nv_opnd_t cta;

    if (rg_new(RF_R, tid)) return 1;
    if (is_ins(NV_MOV_TID_X, *tid, is_onone(), is_onone(), is_onone())) return 1;
    S.tid = *tid;                   /* a tree with no `tid` name still needs it */
    if (k->grid[0] == 1) { *gid = *tid; return 0; }
    if (rg_new(RF_R, &cta) || rg_new(RF_R, gid)) return 1;
    if (is_ins(NV_MOV_CTAID_X, cta, is_onone(), is_onone(), is_onone())) return 1;
    return is_ins(NV_MAD_LO_S32, *gid, cta, is_oimm(k->block[0]), *tid);
}

/* Launch shape and guard policy (audit H4). Surplus threads (gid >= N) exit
 * early. With a barrier in the body they must make up whole blocks, so a
 * block exits all together before any barrier; the test is the same
 * `gid >= N` either way because N is then a multiple of the block size. */
static int is_pol(const kir_t *k, int64_t n, const is_tflg_t *f, int *guard)
{
    int64_t blk = k->block[0], tot = (int64_t)k->grid[0] * blk;

    if (k->grid[1] != 1 || k->grid[2] != 1 || k->block[1] != 1 || k->block[2] != 1)
        return er_fail(E025, "a launch with y or z beyond 1");
    if (k->grid[0] < 1 || blk < 1) return er_fail(E059, "a launch with an empty grid or block");
    if (tot > INT32_MAX) return er_fail(E059, "a launch with more threads than s32 holds");
    if (n > tot) return er_fail(E059, "a Tile with more cells than the launch has threads");
    *guard = n < tot;
    if (*guard && f->bar && n % blk != 0)
        return er_fail(E059, "surplus threads that are not whole blocks while the body has a barrier");
    return 0;
}

/* With Smem the block-level axes must stay constant inside a block, which
 * holds when some suffix of the extents multiplies to the block size (R4). */
static int is_r4(const int32_t *ext, uint32_t na, int32_t block)
{
    int64_t p = 1;
    uint32_t i = na;

    while (i > 0) {
        i--;
        p *= ext[i];
        if (p == block) return 0;
    }
    return er_fail(E059, "shared memory in a Tile whose block does not match a suffix of its axes");
}

/* Axis `i` of the row-major decode: (gid / stride) % ext, the first axis
 * without the remainder. Powers of two become shr and and; gid is never
 * negative so the signed forms agree with the unsigned ones. */
static int is_dec(nv_opnd_t gid, int32_t ext, int64_t stride, int first, nv_opnd_t *out)
{
    nv_opnd_t t;

    *out = gid;
    if (ext == 1) { *out = is_oimm(0); return 0; }
    if (stride > 1) {
        if (rg_new(RF_R, &t)) return 1;
        if (is_pow2((int32_t)stride)) {
            if (is_ins(NV_SHR_S32, t, *out, is_oimm(is_log2((int32_t)stride)), is_onone())) return 1;
        } else if (is_ins(NV_DIV_S32, t, *out, is_oimm((int32_t)stride), is_onone())) {
            return 1;
        }
        *out = t;
    }
    if (first) return 0;
    if (rg_new(RF_R, &t)) return 1;
    if (is_pow2(ext)) {
        if (is_ins(NV_AND_B32, t, *out, is_oimm(ext - 1), is_onone())) return 1;
    } else if (is_ins(NV_REM_S32, t, *out, is_oimm(ext), is_onone())) {
        return 1;
    }
    *out = t;
    return 0;
}

/* lane = tid & 31 and warp = tid >> 5, bound only when the kernel spells the
 * name (a name absent from the string pool cannot be referenced). */
static int is_lw(const kir_t *k, nv_opnd_t tid)
{
    uint32_t nl = is_pool(k, "lane"), nw = is_pool(k, "warp");
    nv_opnd_t r;

    if (nl) {
        if (rg_new(RF_R, &r)) return 1;
        if (is_ins(NV_AND_B32, r, tid, is_oimm(TM_WARP - 1), is_onone()) || is_bset(nl, r)) return 1;
    }
    if (nw) {
        if (rg_new(RF_R, &r)) return 1;
        if (is_ins(NV_SHR_S32, r, tid, is_oimm(is_log2(TM_WARP)), is_onone()) || is_bset(nw, r)) return 1;
    }
    return 0;
}

/* setp.ge gid, N; @p bra exit. The label is placed by is_tend. */
static int is_exit(nv_opnd_t gid, int64_t n)
{
    nv_opnd_t p;

    if (is_lbl(&S.xlbl) || rg_new(RF_P, &p)) return 1;
    if (is_ins(NV_SETP_GE_S32, p, gid, is_oimm((int32_t)n), is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(S.xlbl), is_onone(), is_onone())) return 1;
    if (S.block && n % S.block == 0) is_muni();     /* whole blocks leave together */
    return is_guard(p, 0);
}

int is_tile(const kir_t *k, const kir_node_t *tile, const kir_val_t **body)
{
    int32_t ext[IR_MAX_DIM];
    uint32_t nm[IR_MAX_DIM], na = 0, i;
    int64_t n = 1, stride;
    const kir_val_t *aux;
    nv_opnd_t tid, gid, ax;
    is_tflg_t fl;
    int guard = 0;

    if (!k || !tile || !body) return er_fail(E000, "a thread prologue with no kernel or tile");
    *body = kr_fld(tile, "body");
    if (!*body || (*body)->tag != KV_LIST) return er_fail(E015, "a Tile with no body list");
    aux = kr_fld(tile, "aux_threads");
    if (aux && aux->tag == KV_INT && aux->i != 0) return er_fail(E059, "a Tile with aux_threads");
    if (is_axes(tile, ext, nm, &na, &n) || is_scan(k, *body, &fl)) return 1;
    if (is_pol(k, n, &fl, &guard)) return 1;
    if (fl.smem && is_r4(ext, na, k->block[0])) return 1;
    S.ncell = (uint32_t)n;
    S.xlbl = 0;
    S.block = (uint32_t)k->block[0];
    if (is_gid(k, &tid, &gid)) return 1;
    if (guard && is_exit(gid, n)) return 1;
    for (i = 0, stride = n; i < na; i++) {
        stride /= ext[i];
        if (is_dec(gid, ext[i], stride, i == 0, &ax)) return 1;
        /* gid / stride is one value across a block when the stride is a
         * multiple of the block size (blocks start at multiples of it) */
        if (ext[i] == 1 || stride % k->block[0] == 0) {
            if (is_bsetu(nm[i], ax)) return 1;
        } else if (is_bset(nm[i], ax)) {
            return 1;
        }
    }
    return is_lw(k, tid);
}

uint32_t is_ncell(void)
{
    return S.ncell;
}

int is_tend(void)
{
    if (S.xlbl && is_ins(NV_LABEL, is_olbl(S.xlbl), is_onone(), is_onone(), is_onone())) return 1;
    return is_ins(NV_RET, is_onone(), is_onone(), is_onone(), is_onone());
}


/* ---- Walk ---- */

/* The walk's one request slot: a handler or a close asks for a child list
 * with is_wvisit and the walk pushes it once the caller has returned. */
struct is_wk_s {
    is_frm_t  req;
    int       has;
    uint32_t *sp;                   /* the walk's frame count, for the depth check */
    uint8_t   shut;                 /* 1 while a close runs: its scope is still open */
};

int is_wvisit(is_wk_t *w, const kir_val_t *lst, is_cls_t cls, int scope,
              const kir_node_t *nd, uint32_t a, uint32_t b)
{
    if (!w || !lst || !nd) return er_fail(E000, "a child list requested with no walk, list or node");
    if (lst->tag != KV_LIST) return er_fail(E015, "a statement body that is not a list");
    if (w->has) return er_fail(E000, "two child lists requested by one statement");
    if (*w->sp >= IR_MAX_DEP) return er_fail(E082, "statement lists nested deeper than the limit");
    if (scope) {
        /* the finishing list's scope goes first, so a chained list (the else
         * of a Cond) gets a fresh scope of its own */
        if (w->shut && is_pop()) return 1;
        w->shut = 0;
        if (is_push()) return 1;
    }
    w->req.lst = lst;
    w->req.next = 0;
    w->req.cls = (uint8_t)cls;
    w->req.scope = scope ? 1u : 0u;
    w->req.nd = nd;
    w->req.a = a;
    w->req.b = b;
    w->has = 1;
    return 0;
}

/* Push frame `f`. Its scope (if any) was opened by is_wvisit already. */
static int is_wpush(is_frm_t *st, uint32_t *sp, const is_frm_t *f)
{
    if (*sp >= IR_MAX_DEP) return er_fail(E082, "statement lists nested deeper than the limit");
    st[(*sp)++] = *f;
    return 0;
}

/* The statement kind named in a refusal. Kinds with no lowering yet (the
 * stubs below) and kinds that never will have one share this message. */
static int is_nyet(const kir_node_t *nd)
{
    char msg[IR_MAX_NAME + 64];

    strcpy(msg, "a ");
    strncat(msg, kr_str(nd->kname), IR_MAX_NAME);
    strcat(msg, " statement");
    return er_fail(E020, msg);
}

/* ---- Statement handlers ----
 * One per kind, in the order is_stmt dispatches them. Each keeps the signature
 * and the walk contract in isel.h: do the work on entry, then is_wvisit any
 * child list. A kind with no lowering yet refuses E020 by its IR name. */

/* Load(names, input, index, dtype): `ld.global.f32` from a kernel parameter
 * (a global input or a constant buffer, both just params) into fresh %f
 * registers bound to the names; name i is element index+i (the KIR's
 * consecutive layout). One name is one scalar load; 2 or 4 names are one
 * `.v2`/`.v4` load into consecutive %f registers when mm_vec proves the
 * access aligned (M6), else that many scalar loads off one address. Any other
 * count is E020. A constant index folds into the address: [%rd+imm], no
 * instruction. Everything is checked before the first record, so a refusal
 * leaves no half-made load. An input that names a shared array loads with
 * ld.shared.f32 instead, flattened against the array's own extents; shared
 * has no .v2 form, so a pair falls back to two scalars. Not lowered here:
 * dtype other than f32 (E024; `none` is the inherited f32, ruling C2) and an
 * input that is neither a parameter nor a shared array (E022). */
#define IS_MAX_VEC 4u

static int is_load(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nms = kr_fld(nd, "names"), *in = kr_fld(nd, "input");
    const kir_val_t *idx = kr_fld(nd, "index"), *dt = kr_fld(nd, "dtype");
    const kir_val_t *nm;
    const int32_t *dim;
    uint32_t arg = 0, sid = 0, rank, i, n;
    int32_t boff;
    nv_opnd_t flat, areg, dst[IS_MAX_VEC];
    nv_op_t lop;
    int sh;

    (void)w;
    if (!nms || !in || !idx || !dt) return er_fail(E015, "a Load missing names, input, index or dtype");
    if (nms->tag != KV_LIST || nms->n == 0) return er_fail(E015, "a Load whose names are not a non-empty list");
    if (in->tag != KV_STR) return er_fail(E015, "a Load whose input is not a name");
    n = nms->n;
    for (i = 0; i < n && i < IS_MAX_VEC; i++) {
        nm = kr_kid(nms, i);
        if (!nm || nm->tag != KV_STR) return er_fail(E015, "a Load name that is not a string");
    }
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "a Load whose dtype is not f32");
    if (n != 1u && n != 2u && n != 4u) return er_fail(E020, "a Load of a number of values other than 1, 2 or 4");
    sh = is_shv(in->ref, &sid);
    if (sh) {
        dim = S.sm.arr[sid].dim;
        rank = S.sm.arr[sid].rank;
    } else {
        if (mm_arg(in->ref, &arg)) return 1;
        if (S.arg[arg].kind == KA_OUT) return er_fail(E022, "a Load from an output buffer");
        dim = S.arg[arg].dim;
        rank = S.arg[arg].rank;
    }
    if (mm_flat(dim, rank, idx, &flat)) return 1;
    if (sh) {
        if (sm_addr(sid, flat, &areg, &boff)) return 1;
    } else if (mm_gaddr(arg, flat, &areg, &boff)) return 1;
    if (boff > INT32_MAX - 4 * (int32_t)IS_MAX_VEC) return er_fail(E059, "a Load offset out of range");
    for (i = 0; i < n; i++)
        if (rg_new(RF_F, &dst[i])) return 1;     /* consecutive: nothing else allocates here */
    if (n > 1u && mm_vec(dim, rank, idx, n) && (!sh || n == 4u)) {
        lop = sh ? NV_LD_SHARED_V4_F32 : (n == 2u ? NV_LD_GLOBAL_V2_F32 : NV_LD_GLOBAL_V4_F32);
        if (is_ins(lop, dst[0], areg, boff ? is_oimm(boff) : is_onone(), is_onone())) return 1;
    } else {
        lop = sh ? NV_LD_SHARED_F32 : NV_LD_GLOBAL_F32;
        for (i = 0; i < n; i++) {
            int32_t o = boff + 4 * (int32_t)i;
            if (is_ins(lop, dst[i], areg, o ? is_oimm(o) : is_onone(), is_onone())) return 1;
        }
    }
    for (i = 0; i < n; i++) {
        nm = kr_kid(nms, i);
        if (is_cmt(nm->ref) || is_bset(nm->ref, dst[i])) return 1;
    }
    return 0;
}

/* ---- Arithmetic ---- */

/* name = op(args): the op comes from the table by its Emmy name, so an op
 * outside it (exp, sqrt, reciprocal until T034) is E021 naming it. Arity must
 * match the arg count (E020); args are names looked up in the bindings
 * (E022); dtype is f32 or none (the inherited f32, ruling C2; E024
 * otherwise). A target that is an accumulator's name writes straight into its
 * home register (H2); anything else gets a fresh %f. */
static int is_asgn(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nm = kr_fld(nd, "name"), *op = kr_fld(nd, "op");
    const kir_val_t *as = kr_fld(nd, "args"), *dt = kr_fld(nd, "dtype");
    const kir_val_t *a0, *a1;
    opn_t o;
    nv_opnd_t x, y, dst;
    is_acc_t *acc;
    char msg[IR_MAX_NAME + 64];

    (void)w;
    if (!nm || !op || !as || !dt) return er_fail(E015, "an Assign missing name, op, args or dtype");
    if (nm->tag != KV_STR || op->tag != KV_OP || as->tag != KV_LIST)
        return er_fail(E015, "an Assign whose name, op or args have the wrong shape");
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "an Assign whose dtype is not f32");
    o = op_find(kr_str(op->ref));
    if (o == OPN_NONE) {
        strcpy(msg, "the op '");
        strncat(msg, kr_str(op->ref), IR_MAX_NAME);
        strcat(msg, "' is not in the op table");
        return er_fail(E021, msg);
    }
    if (as->n != op_arity(o)) return er_fail(E020, "an Assign whose arg count is not the op's arity");
    a0 = kr_kid(as, 0);
    a1 = as->n > 1 ? kr_kid(as, 1) : a0;
    if (!a0 || !a1 || a0->tag != KV_STR || a1->tag != KV_STR)
        return er_fail(E020, "an Assign arg that is not a name");
    if (is_find(a0->ref, &x) || is_find(a1->ref, &y)) return 1;
    acc = is_afind(nm->ref);
    if (acc) dst = acc->reg;
    else if (rg_new(RF_F, &dst)) return 1;
    if (op_emit(o, dst, x, y)) return 1;
    return is_bset(nm->ref, dst);
}

/* Let(name, value, dtype): bind a name to a value (Arithmetic). A value that
 * lowers to an f32 immediate becomes `mov.f32 %f, 0fXXXXXXXX` into a fresh %f
 * so the name reads as a register everywhere afterwards; an integer
 * expression lowers to an s32 register or immediate and is bound as it
 * stands. A FlatIndex value is not in the open set and kx_eval refuses it by
 * kind name (E020). dtype carries no rule of its own: the value's own dtype
 * already chose the file, and refusing `int` here would break the integer
 * Lets the contract asks for. */
static int is_let(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nm = kr_fld(nd, "name"), *vl = kr_fld(nd, "value");
    nv_opnd_t v, dst;

    (void)w;
    if (!nm || !vl) return er_fail(E015, "a Let missing name or value");
    if (nm->tag != KV_STR) return er_fail(E015, "a Let whose name is not a name");
    if (kx_eval(vl, &v)) return 1;
    if (v.kind == OP_FIMM) {
        if (rg_new(RF_F, &dst)) return 1;
        if (is_ins(NV_MOV_F32, dst, v, is_onone(), is_onone())) return 1;
    } else {
        dst = v;
    }
    return kx_uni(vl) ? is_bsetu(nm->ref, dst) : is_bset(nm->ref, dst);
}

/* Write(output, index, values, value_dtype, atomic, swizzle): value i goes
 * to element index+i (the KIR's consecutive layout). One value is one
 * `st.global.f32`; 2 or 4 values are one `st.global.v2/v4.f32` when mm_vec
 * proves the access aligned (M6), else that many scalar stores. The vector
 * form needs its values in consecutive %f registers, so unless they already
 * are, each is copied (`mov.f32`) into a fresh run first. The address is
 * computed once and the per-value offset folds into the immediate. An output
 * that names a shared array stores with st.shared.f32, flattened against the
 * array's own extents; an output that is neither an `out` parameter nor a
 * shared array is E022. Every check and every value name lookup comes before
 * the first record. Refused: atomic (E020; T071 lowers red.global.add), a
 * swizzle other than NONE (E020), a value count other than 1, 2 or 4 (E020,
 * the simulator's rule), a value_dtype other than f32 or none (E024). */
/* The stores of one Write: n values from [areg+boff]. `vec` asks for the
 * single vector record, which needs its values in consecutive %f registers,
 * so a run that is not already consecutive is copied into a fresh one first.
 * Shared memory has a v4 form but no v2 (embarrassing, but true), which the
 * caller has already accounted for in `vec`. */
static int is_sto(nv_opnd_t *val, uint32_t n, nv_opnd_t areg, int32_t boff, int vec, int sh)
{
    nv_opnd_t run[IS_MAX_VEC];
    nv_op_t op;
    uint32_t i;
    int seq = 1;

    if (!vec) {
        op = sh ? NV_ST_SHARED_F32 : NV_ST_GLOBAL_F32;
        for (i = 0; i < n; i++) {
            int32_t o = boff + 4 * (int32_t)i;
            if (is_ins(op, is_onone(), areg, o ? is_oimm(o) : is_onone(), val[i])) return 1;
        }
        return 0;
    }
    for (i = 0; i < n; i++)
        seq = seq && val[i].kind == OP_REG && val[i].rf == RF_F && val[i].v == val[0].v + i;
    if (seq) {
        run[0] = val[0];
    } else {
        for (i = 0; i < n; i++) {
            if (rg_new(RF_F, &run[i])) return 1;
            if (is_ins(NV_MOV_F32, run[i], val[i], is_onone(), is_onone())) return 1;
        }
    }
    if (sh)
        return is_ins(NV_ST_SHARED_V4_F32, is_onone(), areg,
                      boff ? is_oimm(boff) : is_onone(), run[0]);
    return is_ins(n == 2u ? NV_ST_GLOBAL_V2_F32 : NV_ST_GLOBAL_V4_F32, is_onone(), areg,
                  boff ? is_oimm(boff) : is_onone(), run[0]);
}

static int is_wrt(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *out = kr_fld(nd, "output"), *idx = kr_fld(nd, "index");
    const kir_val_t *vs = kr_fld(nd, "values"), *dt = kr_fld(nd, "value_dtype");
    const kir_val_t *at = kr_fld(nd, "atomic"), *sw = kr_fld(nd, "swizzle");
    nv_opnd_t val[IS_MAX_VEC], flat, areg;
    const int32_t *dim;
    uint32_t arg = 0, sid = 0, rank, i, n;
    int32_t boff;
    int vec, sh;

    (void)w;
    if (!out || !idx || !vs || !dt) return er_fail(E015, "a Write missing output, index, values or value_dtype");
    if (out->tag != KV_STR) return er_fail(E015, "a Write whose output is not a name");
    if (vs->tag != KV_LIST || vs->n == 0) return er_fail(E015, "a Write whose values are not a non-empty list");
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "a Write whose value_dtype is not f32");
    if (sw && sw->tag == KV_STR && strcmp(kr_str(sw->ref), "NONE") != 0)
        return er_fail(E020, "a Write with a swizzle");
    n = vs->n;
    if (n != 1u && n != 2u && n != 4u) return er_fail(E020, "a Write of a number of values other than 1, 2 or 4");
    sh = is_shv(out->ref, &sid);
    if (sh) {
        dim = S.sm.arr[sid].dim;
        rank = S.sm.arr[sid].rank;
    } else {
        if (mm_arg(out->ref, &arg)) return 1;
        if (S.arg[arg].kind != KA_OUT) return er_fail(E022, "a Write to a buffer that is not an output");
        dim = S.arg[arg].dim;
        rank = S.arg[arg].rank;
    }
    for (i = 0; i < n; i++) {
        const kir_val_t *v = kr_kid(vs, i);
        if (!v || v->tag != KV_STR) return er_fail(E015, "a Write value that is not a name");
        if (is_find(v->ref, &val[i])) return 1;
    }
    if (mm_flat(dim, rank, idx, &flat)) return 1;
    if (sh) {
        if (sm_addr(sid, flat, &areg, &boff)) return 1;
    } else if (mm_gaddr(arg, flat, &areg, &boff)) return 1;
    if (boff > INT32_MAX - 4 * (int32_t)IS_MAX_VEC) return er_fail(E059, "a Write offset out of range");
    if (at && at->tag == KV_BOOL && at->i) {
        if (sh) return er_fail(E020, "an atomic Write to shared memory");
        for (i = 0; i < n; i++) {
            int32_t o = boff + 4 * (int32_t)i;
            if (is_ins(NV_RED_GLOBAL_ADD_F32, is_onone(), areg, o ? is_oimm(o) : is_onone(), val[i])) return 1;
        }
        return 0;
    }
    vec = n > 1u && mm_vec(dim, rank, idx, n) && (!sh || n == 4u);
    return is_sto(val, n, areg, boff, vec, sh);
}
/* ---- Accumulators (T037) ---- */

/* The accumulator an Accum statement names, registered on first sight, plus
 * the value its first update folds in: the op's identity, or the IR's `base`
 * (a live name) when it gives one. Emits nothing, so a caller can still
 * refuse after this and leave no half-made seed behind. */
static int is_areg(const kir_node_t *nd, is_acc_t **out)
{
    const kir_val_t *nm = kr_fld(nd, "name"), *op = kr_fld(nd, "op");
    const kir_val_t *dt = kr_fld(nd, "dtype");
    is_acc_t *a;
    nv_opnd_t reg;
    opn_t o;
    uint32_t bits = 0;

    if (!nm || !op || !dt) return er_fail(E015, "an Accum missing name, op or dtype");
    if (nm->tag != KV_STR || op->tag != KV_OP)
        return er_fail(E015, "an Accum whose name or op has the wrong shape");
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "an Accum whose dtype is not f32");
    o = op_find(kr_str(op->ref));
    if (o == OPN_NONE) return er_fail(E021, "an Accum whose op is not in the op table");
    if (op_arity(o) != 2u) return er_fail(E020, "an Accum whose op is not binary");
    if (!op_ident(o, &bits)) return er_fail(E021, "an Accum whose op has no identity");
    a = is_afind(nm->ref);
    if (!a) {
        if (rg_home(nm->ref, RF_F, &reg)) return 1;
        if (is_aadd(nm->ref, reg, (uint8_t)o, bits, 0, &a)) return 1;
    } else if (a->op == OPN_NONE) {
        a->op = (uint8_t)o;
        a->ident = bits;
    }
    *out = a;
    return 0;
}

/* is_areg plus the update's left operand (R8.3): the bound `base` name when
 * the IR gives one, else the accumulator's own home register. `base` is a
 * name the body computes for this trip (f02 folds acc1 by exp(old_max-new_max)
 * into it), so it is looked up here, at the Accum statement, and never at
 * seed time. */
static int is_afrom(const kir_node_t *nd, is_acc_t **out, nv_opnd_t *src)
{
    const kir_val_t *bs = kr_fld(nd, "base");

    if (is_areg(nd, out)) return 1;
    if (!bs || bs->tag == KV_NONE) {
        *src = (*out)->reg;
    } else if (bs->tag == KV_STR) {
        if (is_find(bs->ref, src)) return 1;
    } else {
        return er_fail(E020, "an Accum whose base is not a name");
    }
    return 0;
}

/* Emit the op's identity into the accumulator's one home register, once, and
 * bind the name where the accumulator is visible. H1: the seed goes down
 * before the loop body runs, not on first sight of the Accum inside it, and
 * it is always the identity (R8.2) whatever `base` says. */
static int is_aseed(is_acc_t *a)
{
    if (a->seeded) return 0;
    if (is_ins(NV_MOV_F32, a->reg, is_ofim(a->ident), is_onone(), is_onone())) return 1;
    a->seeded = 1;
    return is_bset(a->name, a->reg);
}

/* Accum(name, value, op, dtype, axes, base): acc = op(base or acc, value) in
 * the accumulator's home register (H2, R8.3). A loop that owns it has already
 * seeded it (is_sseed); one that does not gets its identity here, on first
 * sight. */
static int is_acc(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *vl = kr_fld(nd, "value");
    is_acc_t *a;
    nv_opnd_t src, v;

    (void)w;
    if (!vl || vl->tag != KV_STR) return er_fail(E015, "an Accum whose value is not a name");
    if (is_afrom(nd, &a, &src)) return 1;
    if (is_find(vl->ref, &v)) return 1;
    if (is_aseed(a)) return 1;
    return op_emit((opn_t)a->op, a->reg, src, v);
}

/* ---- Control Flow (T038, T039) ---- */

/* 1 when `nd`'s axes list names the axis `ax`, or when it has no axes at all
 * (then the innermost enclosing loop owns it). */
static int is_onax(const kir_node_t *nd, uint32_t ax)
{
    const kir_val_t *as = kr_fld(nd, "axes"), *v;
    uint32_t i;

    if (!as || as->tag == KV_NONE) return 1;
    if (as->tag != KV_LIST) return 0;
    for (i = 0; i < as->n; i++) {
        v = kr_kid(as, i);
        if (v && v->tag == KV_STR && is_same(v->ref, ax)) return 1;
    }
    return 0;
}

/* Seed every accumulator of the loop's immediate body that reduces this axis,
 * before the loop starts. An accumulator inside a loop must not restart each
 * trip, and a combine that reads the state before its Accum statement (f02
 * does) has to find a real value there (research R8, ruling H1). */
static int is_sseed(const kir_node_t *sl, uint32_t ax)
{
    const kir_val_t *bd = kr_fld(sl, "body"), *sd = kr_fld(sl, "seed");
    const kir_node_t *nd;
    is_acc_t *a;
    uint32_t i;

    if (sd && sd->tag == KV_BOOL && !sd->i) return 0;
    if (!bd || bd->tag != KV_LIST) return er_fail(E015, "a loop with no body list");
    for (i = 0; i < bd->n; i++) {
        nd = kr_nod(kr_kid(bd, i));
        if (!nd || nd->kind != KR_ACCUM || !is_onax(nd, ax)) continue;
        if (is_areg(nd, &a)) return 1;
        if (a->has_init) continue;
        if (is_aseed(a)) return 1;
    }
    return 0;
}

/* The axis of a loop statement: its name offset in *nm, its constant extent
 * in *ext. An extent that is not a constant is not something the launch
 * geometry can be checked against, so it refuses (FR-013). */
static int is_lax(const kir_node_t *nd, uint32_t *nm, int32_t *ext)
{
    const kir_val_t *ax = kr_fld(nd, "axis"), *an, *ex;
    const kir_node_t *a = kr_nod(ax);
    int isc = 0;

    an = a ? kr_fld(a, "name") : NULL;
    ex = a ? kr_fld(a, "extent") : NULL;
    if (!an || an->tag != KV_STR || !ex) return er_fail(E015, "a loop axis with no name or extent");
    if (kx_fold(ex, ext, &isc)) return 1;
    if (!isc || *ext < 1) return er_fail(E059, "a loop whose extent is not a constant above 0");
    *nm = an->ref;
    return 0;
}

/* StridedLoop(axis, start, step, body, unroll, end, seed): one thread strides
 * an axis, shape per contracts/lowering.md:
 *   i = start; L_top: setp.ge i, bound; @%p bra L_done; body; i += step;
 *   bra L_top; L_done:
 * `unroll` is a scheduling hint: the loop is correct either way and the
 * generated kernel's speed is only the last tiebreak. */

/* The step: a constant above 0, or E026. The `i >= bound` exit only stops a
 * loop that counts up, and a step held in a register could be zero or
 * negative and spin forever. The bound is at most the extent (is_sbnd),
 * so the last `i += step` stays in s32 when extent - 1 + step does. */
static int is_sstep(const kir_val_t *sp, int32_t ext, int32_t *stp)
{
    int isc = 0;

    if (kx_fold(sp, stp, &isc)) return 1;
    if (!isc || *stp < 1) return er_fail(E026, "a StridedLoop whose step is not a constant above 0");
    if ((int64_t)ext - 1 + *stp > INT32_MAX)
        return er_fail(E059, "a StridedLoop whose last step would pass s32");
    return 0;
}

/* The bound: the extent, or `end` when the IR gives one. Emmy promises end
 * never exceeds the extent; min(end, extent) keeps that true here rather
 * than trusting it, since is_sstep's overflow proof rests on it. */
static int is_sbnd(const kir_val_t *en, int32_t ext, nv_opnd_t *bnd)
{
    nv_opnd_t e;

    *bnd = is_oimm(ext);
    if (!en || en->tag == KV_NONE) return 0;
    if (kx_eval(en, &e)) return 1;
    if (!mm_ints(e)) return er_fail(E059, "a StridedLoop whose end is not an integer");
    if (e.kind == OP_IMM) {
        if ((int32_t)e.v < ext) *bnd = e;
        return 0;
    }
    if (rg_new(RF_R, bnd)) return 1;
    return is_ins(NV_MIN_S32, *bnd, e, is_oimm(ext), is_onone());
}

static int is_sloop(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *st = kr_fld(nd, "start"), *sp = kr_fld(nd, "step");
    const kir_val_t *bd = kr_fld(nd, "body"), *en = kr_fld(nd, "end");
    nv_opnd_t home, s, bnd, pv;
    uint32_t ax = 0, ltop = 0, lend = 0;
    int32_t ext = 0, stp = 0;
    int uni;

    if (!st || !sp || !bd) return er_fail(E015, "a StridedLoop missing start, step or body");
    if (bd->tag != KV_LIST) return er_fail(E015, "a StridedLoop whose body is not a list");
    if (is_lax(nd, &ax, &ext) || is_sstep(sp, ext, &stp)) return 1;
    if (kx_eval(st, &s)) return 1;
    if (!mm_ints(s)) return er_fail(E059, "a StridedLoop whose start is not an integer");
    if (is_sbnd(en, ext, &bnd)) return 1;
    /* the same trip count in every thread of the block only when start and
     * bound are (the step is a constant); a barrier in the body needs that */
    uni = kx_uni(st) && (!en || en->tag == KV_NONE || kx_uni(en));
    if (is_sseed(nd, ax)) return 1;
    if (is_lbl(&ltop) || is_lbl(&lend)) return 1;
    if (is_wvisit(w, bd, IC_LOOP, 1, nd, ltop, lend)) return 1;
    if (rg_home(ax, RF_R, &home) || (uni ? is_bsetu(ax, home) : is_bset(ax, home))) return 1;
    if (is_ins(NV_MOV_S32, home, s, is_onone(), is_onone())) return 1;
    if (is_ins(NV_LABEL, is_olbl(ltop), is_onone(), is_onone(), is_onone())) return 1;
    if (rg_new(RF_P, &pv)) return 1;
    if (is_ins(NV_SETP_GE_S32, pv, home, bnd, is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(lend), is_onone(), is_onone())) return 1;
    if (uni) is_muni();
    return is_guard(pv, 0);
}

/* The back edge of a StridedLoop: step the loop variable in its home
 * register, then close the loop. The variable's binding is still open here.
 * is_sloop has already proved the step a constant above 0. */
static int is_cloop(const is_frm_t *f)
{
    const kir_val_t *ax = kr_fld(f->nd, "axis");
    const kir_node_t *a = kr_nod(ax);
    const kir_val_t *an = a ? kr_fld(a, "name") : NULL;
    nv_opnd_t home;

    if (!an || an->tag != KV_STR) return er_fail(E015, "a loop axis with no name");
    if (rg_home(an->ref, RF_R, &home)) return 1;
    if (f->nd->kind == KR_LOOP) {
        if (is_ins(NV_ADD_S32, home, home, is_oimm(1), is_onone())) return 1;
    } else {
        const kir_val_t *sp = kr_fld(f->nd, "step");
        int32_t stp = 0;
        int isc = 0;
        if (kx_fold(sp, &stp, &isc)) return 1;
        if (!isc) return er_fail(E000, "a StridedLoop step that stopped being a constant");
        if (is_ins(NV_ADD_S32, home, home, is_oimm(stp), is_onone())) return 1;
    }
    if (is_ins(NV_BRA, is_onone(), is_olbl(f->a), is_onone(), is_onone())) return 1;
    return is_ins(NV_LABEL, is_olbl(f->b), is_onone(), is_onone(), is_onone());
}

/* Cond(cond, body, else_body): @!%p bra L_else; then; bra L_end; L_else:;
 * else; L_end:. Each body gets its own scope. A barrier inside one would
 * strand the threads that took the other way, and verify refuses that shape
 * (P8, ruling M7), so this handler does not test for it twice. */
static int is_cond(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *cd = kr_fld(nd, "cond"), *bd = kr_fld(nd, "body");
    nv_opnd_t p;
    uint32_t lelse = 0, lend = 0;

    if (!cd || !bd) return er_fail(E015, "a Cond missing cond or body");
    if (bd->tag != KV_LIST) return er_fail(E015, "a Cond whose body is not a list");
    if (kx_eval(cd, &p)) return 1;
    if (p.kind != OP_REG || p.rf != RF_P)
        return er_fail(E059, "a Cond whose condition is not a comparison");
    if (is_lbl(&lelse) || is_lbl(&lend)) return 1;
    if (is_wvisit(w, bd, IC_CTHEN, 1, nd, lelse, lend)) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(lelse), is_onone(), is_onone())) return 1;
    return is_guard(p, 1);
}

static int is_cend(const is_frm_t *f);

/* End of a Cond's then list: skip the else, open it, or close straight away
 * when the IR has no else list for it to open. */
static int is_cthen(is_wk_t *w, const is_frm_t *f)
{
    const kir_val_t *eb = kr_fld(f->nd, "else_body");

    if (is_ins(NV_BRA, is_onone(), is_olbl(f->b), is_onone(), is_onone())) return 1;
    if (is_ins(NV_LABEL, is_olbl(f->a), is_onone(), is_onone(), is_onone())) return 1;
    if (!eb || eb->tag == KV_NONE) return is_cend(f);
    if (eb->tag != KV_LIST) return er_fail(E015, "a Cond whose else_body is not a list");
    if (eb->n == 0) return is_cend(f);          /* no else: no frame to push */
    return is_wvisit(w, eb, IC_CEND, 1, f->nd, 0, f->b);
}

/* End of a Cond's else list: the end label. */
static int is_cend(const is_frm_t *f)
{
    return is_ins(NV_LABEL, is_olbl(f->b), is_onone(), is_onone(), is_onone());
}

/* ---- Shared memory statements (T040) ---- */

/* Smem(name, extents, dtype, align): the pre-pass already placed it, so the
 * walk only confirms the layout knows the name. Declaring it here would be
 * too late anyway: whether the kernel has static arrays or one pooled block
 * depends on every array in it. */
static int is_smem(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nm = kr_fld(nd, "name");
    uint32_t id;

    (void)w;
    if (!nm || nm->tag != KV_STR) return er_fail(E015, "a Smem whose name is not a string");
    if (!is_shv(nm->ref, &id)) return er_fail(E000, "a Smem the pre-pass did not declare");
    return 0;
}

/* ---- Barriers (T041) ---- */

/* Sync(barrier_id, count, warp): bar.sync id[, count]; a warp sync takes this
 * warp's activemask, because the IR does not narrow the set further and
 * verify only excuses bar.warp.sync when its mask comes from there (R11). */
static int is_sync(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *bi = kr_fld(nd, "barrier_id"), *ct = kr_fld(nd, "count");
    const kir_val_t *wp = kr_fld(nd, "warp");
    nv_opnd_t id, cnt = is_onone(), mask;

    (void)w;
    if (!bi || bi->tag != KV_INT) return er_fail(E015, "a Sync with no barrier id");
    if (bi->i < 0 || bi->i > 15) return er_fail(E059, "a barrier id outside 0..15");
    if (ct && ct->tag != KV_NONE) {
        if (ct->tag != KV_INT || ct->i < 32 || ct->i > 1024 || ct->i % 32 != 0)
            return er_fail(E059, "a barrier count that is not 32..1024 in steps of 32");
        cnt = is_oimm((int32_t)ct->i);
    }
    id = is_oimm((int32_t)bi->i);
    if (!wp || wp->tag != KV_BOOL || !wp->i)
        return is_ins(NV_BAR_SYNC, is_onone(), id, cnt, is_onone());
    if (rg_new(RF_R, &mask)) return 1;
    if (is_ins(NV_ACTIVEMASK, mask, is_onone(), is_onone(), is_onone())) return 1;
    return is_ins(NV_BAR_WARP_SYNC, is_onone(), mask, is_onone(), is_onone());
}

/* ---- Warp Ops and Tree (T042, T043) ---- */

/* The three name lists of a warp or tree statement, checked once. `cm` may be
 * empty: then the combine is the accumulator's own op folded with state_b. */
typedef struct {
    const kir_val_t *st;            /* state names */
    const kir_val_t *sb;            /* state_b names, one per state */
    const kir_val_t *cm;            /* combine_states, may be empty */
    uint32_t         n;             /* pairs */
} is_ws_t;

static int is_wsget(const kir_node_t *nd, is_ws_t *x)
{
    const kir_val_t *st = kr_fld(nd, "state"), *sb = kr_fld(nd, "state_b");
    const kir_val_t *cm = kr_fld(nd, "combine_states");
    uint32_t i;

    if (!st || !sb || st->tag != KV_LIST || sb->tag != KV_LIST || st->n == 0)
        return er_fail(E015, "a warp or tree statement with no state list");
    if (st->n != sb->n) return er_fail(E020, "a state list and a state_b list of different lengths");
    if (cm && cm->tag != KV_LIST && cm->tag != KV_NONE)
        return er_fail(E015, "combine_states that is not a list");
    for (i = 0; i < st->n; i++) {
        const kir_val_t *a = kr_kid(st, i), *b = kr_kid(sb, i);
        if (!a || !b || a->tag != KV_STR || b->tag != KV_STR)
            return er_fail(E015, "a state name that is not a string");
    }
    x->st = st;
    x->sb = sb;
    x->cm = cm;
    x->n = st->n;
    return 0;
}

/* One shfl.sync.bfly.b32: four sources, so the record is filled by hand. */
static int is_shfl1(nv_opnd_t d, nv_opnd_t a, uint32_t half, nv_opnd_t mask)
{
    nv_ins_t *r = is_new(NV_SHFL_BFLY);

    if (!r) return 1;
    r->dst = d;
    r->src[0] = a;
    r->src[1] = is_oimm((int32_t)half);
    r->src[2] = is_oimm(TM_WARP - 1);
    r->src[3] = mask;
    r->nsrc = 4u;
    return 0;
}

/* The combine of one shuffle or tree step: the IR's own combine statements
 * when it has them, lowered once per step because one list already names
 * every state, and in a fresh scope so the names it binds die with the step;
 * else each state's accumulator op folded with that state's state_b.
 * Emmy types combine_states as Assigns only (a pure merge program), and
 * that is all this takes: straight-line code lowered here, in place, with
 * no call back into the statement walk, so nothing recurses and nothing
 * can ask the walk for a child list in the middle of a step. Any other
 * kind is E027. */
static int is_comb(is_wk_t *w, const is_ws_t *x)
{
    const kir_node_t *nd;
    const kir_val_t *sn, *bn;
    nv_opnd_t v;
    is_acc_t *a;
    uint32_t i, j;
    int rc = 0;

    if (x->cm && x->cm->n) {
        for (i = 0; i < x->cm->n; i++) {
            nd = kr_nod(kr_kid(x->cm, i));
            if (!nd) return er_fail(E015, "a combine_states item that is not a statement");
            if (nd->kind != KR_ASSIGN)
                return er_fail(E027, "a combine_states statement that is not an Assign");
        }
        if (is_push()) return 1;
        for (i = 0; i < x->cm->n && !rc; i++)
            rc = is_asgn(w, kr_nod(kr_kid(x->cm, i)));
        if (is_pop()) return 1;
        return rc;
    }
    for (j = 0; j < x->n; j++) {
        sn = kr_kid(x->st, j);
        bn = kr_kid(x->sb, j);
        a = is_afind(sn->ref);
        if (!a) return er_fail(E021, "a combine with no statements and no accumulator");
        if (is_find(bn->ref, &v)) return 1;
        if (op_emit((opn_t)a->op, a->reg, a->reg, v)) return 1;
    }
    return 0;
}

/* One butterfly step at `half`: shuffle every state into its state_b, then
 * combine. shfl moves bits, so a value crosses the warp as a %r and comes
 * back as a %f. */
static int is_shstep(is_wk_t *w, const is_ws_t *x, uint32_t half, nv_opnd_t mask)
{
    nv_opnd_t s, sb, tr, sr;
    uint32_t j;

    for (j = 0; j < x->n; j++) {
        if (is_find(kr_kid(x->st, j)->ref, &s)) return 1;
        if (rg_new(RF_R, &tr) || rg_new(RF_R, &sr)) return 1;
        if (is_ins(NV_MOV_B32, tr, s, is_onone(), is_onone())) return 1;
        if (is_shfl1(sr, tr, half, mask)) return 1;
        if (rg_home(kr_kid(x->sb, j)->ref, RF_F, &sb)) return 1;
        if (is_ins(NV_MOV_B32, sb, sr, is_onone(), is_onone())) return 1;
        if (is_bset(kr_kid(x->sb, j)->ref, sb)) return 1;
    }
    return is_comb(w, x);
}

/* WarpShuffle(state, state_b, combine_states, length, dtype): halve the
 * length from length/2 down to 1, shuffling and combining at each step. Five
 * steps cover a warp, and the guard makes that a hard bound rather than a
 * promise.
 *
 * The member mask is 0xffffffff, not activemask. Emmy's reference takes
 * activemask because it may launch a partial warp; here every lane of every
 * warp is provably present and converged at the shuffle: the block is whole
 * warps (checked below), surplus threads leave in whole blocks whenever the
 * body has a shuffle (is_pol), and verify refuses a shuffle under any
 * divergent branch, guard, or loop of thread-varying trip count. */
static int is_shfl(is_wk_t *w, const kir_node_t *nd)
{
    is_ws_t x;
    nv_opnd_t mask;
    int32_t len = 0;
    int isc = 0;
    uint32_t half;
    IX_GUARD(g, 6);

    if (is_wsget(nd, &x)) return 1;
    if (kx_fold(kr_fld(nd, "length"), &len, &isc)) return 1;
    if (!isc || len < 1 || len > TM_WARP || !is_pow2(len))
        return er_fail(E034, "a warp shuffle whose length is not a power of two up to 32");
    if (S.block == 0 || S.block % TM_WARP != 0)
        return er_fail(E034, "a warp shuffle in a block that is not a multiple of 32");
    mask = is_oimm(-1);
    for (half = (uint32_t)len / 2u; half >= 1u && g--; half /= 2u)
        if (is_shstep(w, &x, half, mask)) return 1;
    return 0;
}

/* The thread index a tree folds with: the name the IR gives, or %tid.x when
 * it names `tid` but the body never bound it. */
static int is_tid(const kir_val_t *tv, nv_opnd_t *out)
{
    if (!tv || tv->tag != KV_STR) return er_fail(E015, "a tree with no tid variable");
    if (is_try(tv->ref, out)) return 0;
    if (strcmp(kr_str(tv->ref), "tid") == 0 && S.tid.kind == OP_REG) {
        *out = S.tid;
        return 0;
    }
    return er_fail(E022, "a tree tid variable that is not bound");
}

/* One tree step: each thread under the half point loads the upper half's
 * cell out of every state's own buffer, combines, and writes its own cell
 * back, so the next step halves all of them at once (R10). `tid + rlen` is
 * computed once and reused as each buffer's upper index. */
static int is_tstep(is_wk_t *w, const is_ws_t *x, const uint32_t *ids,
                    nv_opnd_t tid, nv_opnd_t rlen)
{
    const kir_val_t *sn, *bn;
    nv_opnd_t flat, areg, sb, s;
    int32_t boff;
    uint32_t j;

    if (rg_new(RF_R, &flat)) return 1;
    if (is_ins(NV_ADD_S32, flat, tid, rlen, is_onone())) return 1;
    for (j = 0; j < x->n; j++) {
        bn = kr_kid(x->sb, j);
        if (sm_addr(ids[j], flat, &areg, &boff)) return 1;
        if (rg_home(bn->ref, RF_F, &sb)) return 1;
        if (is_ins(NV_LD_SHARED_F32, sb, areg, boff ? is_oimm(boff) : is_onone(), is_onone())) return 1;
        if (is_bset(bn->ref, sb)) return 1;
    }
    if (is_comb(w, x)) return 1;
    for (j = 0; j < x->n; j++) {
        sn = kr_kid(x->st, j);
        if (is_find(sn->ref, &s)) return 1;
        if (sm_addr(ids[j], tid, &areg, &boff)) return 1;
        if (is_ins(NV_ST_SHARED_F32, is_onone(), areg, boff ? is_oimm(boff) : is_onone(), s)) return 1;
    }
    return 0;
}

/* TreeHalve(bufs, state, state_b, combine_states, length, tid_var, dtype,
 * barrier_id, barrier_count, inner): fold the partials of bufs[0] by halving,
 * with one barrier after each step and outside its predicate, so no thread
 * waits for one that branched away (R10). When the loop ends buf[0] holds the
 * total and every state is loaded from it, since the IR may write from any
 * thread afterwards. */
static int is_tree(is_wk_t *w, const kir_node_t *nd)
{
    is_ws_t x;
    const kir_val_t *bf = kr_fld(nd, "bufs"), *tv = kr_fld(nd, "tid_var");
    const kir_val_t *in = kr_fld(nd, "inner"), *bc = kr_fld(nd, "barrier_count");
    const kir_val_t *bi = kr_fld(nd, "barrier_id"), *bn;
    nv_opnd_t tid, rlen, pv, pq, areg, st;
    uint32_t ids[IR_MAX_SMEM];
    int32_t len = 0, boff = 0, bid = 0;
    uint32_t ltop = 0, lend = 0, lskip = 0, j;
    int isc = 0;

    if (is_wsget(nd, &x)) return 1;
    if (x.n > IR_MAX_SMEM)
        return er_fail(E020, "a tree over more states than the layout has arrays");
    if (in && in->tag != KV_NONE) return er_fail(E020, "a tree with an inner state");
    if (bc && bc->tag != KV_NONE) return er_fail(E020, "a tree with a barrier count");
    if (!bf || bf->tag != KV_LIST || bf->n != x.n)
        return er_fail(E020, "a tree whose buffers and states do not pair up");
    if (kx_fold(kr_fld(nd, "length"), &len, &isc)) return 1;
    if (!isc || len < 1 || !is_pow2(len))
        return er_fail(E020, "a tree whose length is not a power of two");
    if (bi && bi->tag == KV_INT) bid = (int32_t)bi->i;
    if (bid < 0 || bid > 15) return er_fail(E059, "a tree barrier id outside 0..15");
    for (j = 0; j < x.n; j++) {
        bn = kr_kid(bf, j);
        if (!bn || bn->tag != KV_STR) return er_fail(E015, "a tree buffer that is not a name");
        if (sm_find(bn->ref, &ids[j])) return 1;
        if (S.sm.arr[ids[j]].elems < (uint32_t)len)
            return er_fail(E059, "a tree longer than the shared array it folds");
    }
    if (is_tid(tv, &tid)) return 1;
    if (is_lbl(&ltop) || is_lbl(&lend) || is_lbl(&lskip)) return 1;
    if (rg_new(RF_R, &rlen) || rg_new(RF_P, &pv) || rg_new(RF_P, &pq)) return 1;
    if (is_ins(NV_MOV_S32, rlen, is_oimm(len / 2), is_onone(), is_onone())) return 1;
    if (is_ins(NV_LABEL, is_olbl(ltop), is_onone(), is_onone(), is_onone())) return 1;
    if (is_ins(NV_SETP_LT_S32, pv, rlen, is_oimm(1), is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(lend), is_onone(), is_onone())) return 1;
    is_muni();                          /* len/2 halving to 0: the same trip count everywhere */
    if (is_guard(pv, 0)) return 1;
    if (is_ins(NV_SETP_LT_S32, pq, tid, rlen, is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(lskip), is_onone(), is_onone())) return 1;
    if (is_guard(pq, 1)) return 1;
    if (is_tstep(w, &x, ids, tid, rlen)) return 1;
    if (is_ins(NV_LABEL, is_olbl(lskip), is_onone(), is_onone(), is_onone())) return 1;
    if (is_ins(NV_BAR_SYNC, is_onone(), is_oimm(bid), is_onone(), is_onone())) return 1;
    if (is_ins(NV_SHR_S32, rlen, rlen, is_oimm(1), is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(ltop), is_onone(), is_onone())) return 1;
    if (is_ins(NV_LABEL, is_olbl(lend), is_onone(), is_onone(), is_onone())) return 1;
    for (j = 0; j < x.n; j++) {          /* buf[j][0] holds the total: broadcast it */
        if (is_find(kr_kid(x.st, j)->ref, &st)) return 1;
        if (sm_addr(ids[j], is_oimm(0), &areg, &boff)) return 1;
        if (is_ins(NV_LD_SHARED_F32, st, areg, boff ? is_oimm(boff) : is_onone(), is_onone())) return 1;
    }
    return 0;
}

/* ---- Statement kinds not lowered yet ---- */
static int is_loop(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *bd = kr_fld(nd, "body");
    nv_opnd_t home, pv;
    uint32_t ax = 0, ltop = 0, lend = 0;
    int32_t ext = 0;

    if (!bd || bd->tag != KV_LIST) return er_fail(E015, "a Loop with no body list");
    if (is_lax(nd, &ax, &ext)) return 1;
    if (is_sseed(nd, ax)) return 1;
    if (is_lbl(&ltop) || is_lbl(&lend)) return 1;
    if (is_wvisit(w, bd, IC_LOOP, 1, nd, ltop, lend)) return 1;
    if (rg_home(ax, RF_R, &home) || is_bsetu(ax, home)) return 1;     /* 0 .. ext: uniform */
    if (is_ins(NV_MOV_S32, home, is_oimm(0), is_onone(), is_onone())) return 1;
    if (is_ins(NV_LABEL, is_olbl(ltop), is_onone(), is_onone(), is_onone())) return 1;
    if (rg_new(RF_P, &pv)) return 1;
    if (is_ins(NV_SETP_GE_S32, pv, home, is_oimm(ext), is_onone())) return 1;
    if (is_ins(NV_BRA, is_onone(), is_olbl(lend), is_onone(), is_onone())) return 1;
    is_muni();
    return is_guard(pv, 0);
}
static int is_init(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nm = kr_fld(nd, "name");
    const kir_val_t *iv = kr_fld(nd, "identity");
    const kir_val_t *dt = kr_fld(nd, "dtype");
    is_acc_t *a;
    nv_opnd_t reg;
    uint32_t bits = 0;
    float f;

    (void)w;
    if (!nm || !iv || !dt) return er_fail(E015, "an Init missing name, identity or dtype");
    if (nm->tag != KV_STR) return er_fail(E015, "an Init whose name is not a string");
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "an Init whose dtype is not f32");
    if (iv->tag == KV_FLT) {
        bits = iv->fbits;
    } else if (iv->tag == KV_INT) {
        f = (float)iv->i;
        memcpy(&bits, &f, sizeof bits);
    } else {
        return er_fail(E015, "an Init whose identity is not a number");
    }
    if (is_afind(nm->ref)) return er_fail(E028, "an Init whose name is already an accumulator");
    if (rg_home(nm->ref, RF_F, &reg)) return 1;
    if (is_aadd(nm->ref, reg, OPN_NONE, bits, 1, &a)) return 1;
    if (is_ins(NV_MOV_F32, a->reg, is_ofim(bits), is_onone(), is_onone())) return 1;
    a->seeded = 1;
    return is_bset(nm->ref, a->reg);
}
static int is_cpadd(nv_opnd_t a, nv_opnd_t b, nv_opnd_t *out)
{
    if (a.kind == OP_IMM && b.kind == OP_IMM) {
        *out = is_oimm((int32_t)(a.v + b.v));
        return 0;
    }
    if (rg_new(RF_R, out)) return 1;
    if (a.kind == OP_IMM)
        return is_ins(NV_ADD_S32, *out, b, is_oimm((int32_t)a.v), is_onone());
    if (b.kind == OP_IMM)
        return is_ins(NV_ADD_S32, *out, a, is_oimm((int32_t)b.v), is_onone());
    return is_ins(NV_ADD_S32, *out, a, b, is_onone());
}

static int is_cpa(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *sm = kr_fld(nd, "smem");
    const kir_val_t *si = kr_fld(nd, "smem_index");
    const kir_val_t *src = kr_fld(nd, "src");
    const kir_val_t *ri = kr_fld(nd, "src_index");
    const kir_val_t *nb = kr_fld(nd, "nbytes");
    const kir_val_t *sw = kr_fld(nd, "swizzle");
    const kir_val_t *li = kr_fld(nd, "lane_index");
    uint32_t sid = 0, arg = 0;
    int32_t nbytes = 0, boff = 0, src_boff = 0;
    const int32_t *dim, *src_dim;
    uint32_t rank, src_rank;
    nv_opnd_t flat_sm, flat_ln, flat_comb, flat_src;
    nv_opnd_t areg, src_areg;
    nv_op_t op;
    nv_ins_t *r;

    (void)w;
    if (!sm || sm->tag != KV_STR) return er_fail(E015, "a CpAsyncCopy with no smem name");
    if (!si || si->tag != KV_LIST) return er_fail(E015, "a CpAsyncCopy with no smem_index");
    if (!src || src->tag != KV_STR) return er_fail(E015, "a CpAsyncCopy with no src name");
    if (!ri || ri->tag != KV_LIST) return er_fail(E015, "a CpAsyncCopy with no src_index");
    if (!nb || nb->tag != KV_INT) return er_fail(E015, "a CpAsyncCopy with no nbytes");
    nbytes = (int32_t)nb->i;
    if (nbytes != 4 && nbytes != 8 && nbytes != 16)
        return er_fail(E020, "a CpAsyncCopy whose nbytes is not 4, 8 or 16");
    if (sw && sw->tag == KV_STR && strcmp(kr_str(sw->ref), "NONE") != 0)
        return er_fail(E020, "a CpAsyncCopy with a swizzle");
    if (!is_shv(sm->ref, &sid)) return er_fail(E022, "a CpAsyncCopy whose smem is not declared");
    dim = S.sm.arr[sid].dim;
    rank = S.sm.arr[sid].rank;
    if (mm_flat(dim, rank, si, &flat_sm)) return 1;
    flat_comb = flat_sm;
    if (li && li->tag == KV_LIST) {
        if (mm_flat(dim, rank, li, &flat_ln)) return 1;
        if (is_cpadd(flat_comb, flat_ln, &flat_comb)) return 1;
    }
    if (sm_addr(sid, flat_comb, &areg, &boff)) return 1;
    if (mm_arg(src->ref, &arg)) return 1;
    src_dim = S.arg[arg].dim;
    src_rank = S.arg[arg].rank;
    if (mm_flat(src_dim, src_rank, ri, &flat_src)) return 1;
    if (mm_gaddr(arg, flat_src, &src_areg, &src_boff)) return 1;
    if (nbytes == 16) op = NV_CP_ASYNC_CG;
    else if (nbytes == 8) op = NV_CP_ASYNC_CA8;
    else op = NV_CP_ASYNC_CA;
    r = is_new(op);
    if (!r) return 1;
    r->dst = is_onone();
    r->src[0] = areg;
    r->src[1] = (boff ? is_oimm(boff) : is_onone());
    r->src[2] = src_areg;
    r->src[3] = (src_boff ? is_oimm(src_boff) : is_onone());
    r->nsrc = 4u;
    return 0;
}
static int is_cpc(is_wk_t *w, const kir_node_t *nd)
{
    (void)w; (void)nd;
    return is_ins(NV_CP_ASYNC_COMMIT, is_onone(), is_onone(), is_onone(), is_onone());
}
static int is_cpw(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *g = kr_fld(nd, "group");
    int32_t grp = 0;
    (void)w;
    if (g && g->tag == KV_INT) grp = (int32_t)g->i;
    return is_ins(NV_CP_ASYNC_WAIT, is_onone(), is_oimm(grp), is_onone(), is_onone());
}

/* One statement: dispatch on its kind. Anything not listed (KR_OTHER, the
 * tensor-core and TMA kinds, expressions, data nodes) is E020 by name. */
static int is_stmt(is_wk_t *w, const kir_node_t *nd)
{
    switch ((kr_kind_t)nd->kind) {
    case KR_LOAD:           return is_load(w, nd);
    case KR_ASSIGN:         return is_asgn(w, nd);
    case KR_WRITE:          return is_wrt(w, nd);
    case KR_ACCUM:          return is_acc(w, nd);
    case KR_COND:           return is_cond(w, nd);
    case KR_STRIDEDLOOP:    return is_sloop(w, nd);
    case KR_SMEM:           return is_smem(w, nd);
    case KR_SYNC:           return is_sync(w, nd);
    case KR_WARPSHUFFLE:    return is_shfl(w, nd);
    case KR_TREEHALVE:      return is_tree(w, nd);
    case KR_LET:            return is_let(w, nd);
    case KR_LOOP:           return is_loop(w, nd);
    case KR_INIT:           return is_init(w, nd);
    case KR_CPASYNCCOPY:    return is_cpa(w, nd);
    case KR_CPASYNCCOMMIT:  return is_cpc(w, nd);
    case KR_CPASYNCWAIT:    return is_cpw(w, nd);
    case KR_TILE:
        return er_fail(E021, "a Tile nested inside a body");
    default:
        return is_nyet(nd);
    }
}

/* The pending close of a finished list: a loop's back edge, or the branch
 * and label that split a Cond's then list from its else list. */
static int is_close(is_wk_t *w, const is_frm_t *f)
{
    switch ((is_cls_t)f->cls) {
    case IC_NONE:
        return 0;
    case IC_LOOP:
        return is_cloop(f);
    case IC_CTHEN:
        return is_cthen(w, f);
    case IC_CEND:
        return is_cend(f);
    default:
        return er_fail(E000, "a pending close of an unknown kind");
    }
}

/* Finish the top frame: run its close with its scope still open, pop the
 * scope, then push whatever list the close asked for. */
static int is_wend(is_wk_t *w, is_frm_t *st, uint32_t *sp)
{
    is_frm_t f = st[--*sp];

    w->has = 0;
    w->shut = f.scope;
    if (is_close(w, &f)) return 1;
    if (w->shut && is_pop()) return 1;
    w->shut = 0;
    return w->has ? is_wpush(st, sp, &w->req) : 0;
}

/* Lower the statements of `body`, a list. Every pass of the loop takes an
 * item or finishes a frame; a frame is pushed per child list, so the budget
 * is every item plus two per node (generous: the reader builds trees). */
static int is_walk(const kir_t *k, const kir_val_t *body)
{
    is_frm_t st[IR_MAX_DEP], f0;
    is_wk_t w;
    uint32_t sp = 0;
    IX_GUARD(g, (uint64_t)k->nitem + 2u * (uint64_t)k->nnode + 8u);

    memset(&f0, 0, sizeof f0);
    f0.lst = body;
    w.has = 0;
    w.shut = 0;
    w.sp = &sp;
    if (is_wpush(st, &sp, &f0)) return 1;
    while (sp > 0 && g--) {
        is_frm_t *f = &st[sp - 1u];
        const kir_val_t *it;
        const kir_node_t *nd;

        if (f->next >= f->lst->n) {
            if (is_wend(&w, st, &sp)) return 1;
            continue;
        }
        it = kr_kid(f->lst, f->next++);
        nd = it ? kr_nod(it) : NULL;
        if (!nd) return er_fail(E015, "a statement list item that is not a node");
        w.has = 0;
        w.shut = 0;
        if (is_stmt(&w, nd)) return 1;
        if (w.has && is_wpush(st, &sp, &w.req)) return 1;
    }
    if (sp > 0) return er_fail(E000, "a statement walk that ran out of steps");
    return 0;
}

/* ---- Entry ---- */

int is_run(const kir_t *k)
{
    const kir_node_t *tile;
    const kir_val_t *body = NULL;

    is_reset();
    if (!k) return er_fail(E000, "a lowering run with no kernel");
    /* nodes are 1-based: 1 .. nnode are real, 0 is the sentinel */
    if (k->root == 0 || IX_CHK(k->root, k->nnode + 1u) || k->node[k->root].kind != KR_TILE)
        return er_fail(E021, "a body that is not exactly one Tile");
    tile = &k->node[k->root];
    if (is_push() || mm_params(k) || is_tile(k, tile, &body)) return 1;
    if (is_spre(k, body)) return 1;
    if (is_walk(k, body)) return 1;
    return is_pop() || is_tend();
}
