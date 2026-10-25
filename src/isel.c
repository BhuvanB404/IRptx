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
            return er_fail(E000, "a name given homes in two register files");
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

int is_bset(uint32_t name, nv_opnd_t opd)
{
    uint32_t b = IX_PNEW(S.nbind, IR_MAX_BIND);

    if (b == 0) return er_fail(E081, "more live name bindings than the table holds");
    S.bind[b - 1u].name = name;
    S.bind[b - 1u].opd = opd;
    return 0;
}

int is_find(uint32_t name, nv_opnd_t *out)
{
    uint32_t i = S.nbind;
    char msg[IR_MAX_NAME + 64];

    while (i > 0) {
        i--;
        if (is_same(S.bind[i].name, name)) {
            *out = S.bind[i].opd;
            return 0;
        }
    }
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

    if (is_afind(name)) return er_fail(E000, "an accumulator registered twice");
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

/* One index item: fold it into the constant, or lower it into the sum. */
static int mm_item(const kir_val_t *it, int64_t stride, int64_t *cst, nv_opnd_t *acc)
{
    int32_t v = 0;
    int isc = 0;
    nv_opnd_t r;

    if (kx_fold(it, &v, &isc)) return 1;
    if (isc) { *cst += (int64_t)v * stride; return 0; }
    if (kx_eval(it, &r)) return 1;
    if (!mm_ints(r)) return er_fail(E059, "an index that is not an integer");
    if (r.kind == OP_IMM) { *cst += (int64_t)(int32_t)r.v * stride; return 0; }
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

int sm_add(uint32_t name, uint32_t elems, uint32_t align, uint32_t *id)
{
    is_smem_t *m = &S.sm;
    uint32_t al = (align < 16u) ? 16u : align;
    uint64_t off, end;
    is_sarr_t *a;

    if (m->narr >= IR_MAX_SMEM) return er_fail(E081, "more shared arrays than the table holds");
    off = ((uint64_t)m->total + al - 1u) / al * al;
    end = off + (uint64_t)elems * MM_ELEM;
    if (elems == 0 || al > 4096u || (align & (align - 1u)) != 0 || end > (1u << 30))
        return er_fail(E059, "a shared array of zero, huge or oddly aligned size");
    a = &m->arr[m->narr];
    a->name = name;
    a->elems = elems;
    a->align = al;
    a->off = (uint32_t)off;
    m->total = (uint32_t)end;
    m->dyn = (m->total > MM_STATIC_MAX) ? 1u : 0u;
    *id = m->narr++;
    return 0;
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
            if (IX_CHK(c->nd->fld0 + c->i, k->nfld))
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
    if (is_gid(k, &tid, &gid)) return 1;
    if (guard && is_exit(gid, n)) return 1;
    for (i = 0, stride = n; i < na; i++) {
        stride /= ext[i];
        if (is_dec(gid, ext[i], stride, i == 0, &ax) || is_bset(nm[i], ax)) return 1;
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

/* ---- Statement handlers: written by later tasks ----
 * Each stub refuses with E020 naming its kind. A task replaces its stub with
 * the real lowering and keeps the signature (see "The walk" in isel.h). */

/* Load(names, input, index, dtype): one scalar `ld.global.f32` from a kernel
 * parameter (a global input or a constant buffer, both just params) into a
 * fresh %f register bound to the name. A constant index folds into the
 * address: [%rd+imm], no instruction. Everything is checked before the first
 * record, so a refusal leaves no half-made load. Not lowered here: dtype
 * other than f32 (E024; `none` is the inherited f32, ruling C2), a load of
 * 2 or 4 names (T032 makes those .v2/.v4; guessing consecutive-element
 * scalar loads now would bake in a layout nobody has checked: E020), and an
 * input that is not a parameter (E022; T040 adds Smem). */
static int is_load(is_wk_t *w, const kir_node_t *nd)
{
    const kir_val_t *nms = kr_fld(nd, "names"), *in = kr_fld(nd, "input");
    const kir_val_t *idx = kr_fld(nd, "index"), *dt = kr_fld(nd, "dtype");
    const kir_val_t *nm;
    uint32_t arg;
    int32_t boff;
    nv_opnd_t flat, areg, dst;

    (void)w;
    if (!nms || !in || !idx || !dt) return er_fail(E015, "a Load missing names, input, index or dtype");
    if (nms->tag != KV_LIST || nms->n == 0) return er_fail(E015, "a Load whose names are not a non-empty list");
    if (in->tag != KV_STR) return er_fail(E015, "a Load whose input is not a name");
    nm = kr_kid(nms, 0);
    if (!nm || nm->tag != KV_STR) return er_fail(E015, "a Load name that is not a string");
    if (dt->tag != KV_NONE && !(dt->tag == KV_DT && strcmp(kr_str(dt->ref), "f32") == 0))
        return er_fail(E024, "a Load whose dtype is not f32");
    if (nms->n != 1) return er_fail(E020, "a Load of several values");
    if (mm_arg(in->ref, &arg)) return 1;
    if (S.arg[arg].kind == KA_OUT) return er_fail(E022, "a Load from an output buffer");
    if (mm_gflat(arg, idx, &flat) || mm_gaddr(arg, flat, &areg, &boff)) return 1;
    if (rg_new(RF_F, &dst)) return 1;
    if (is_ins(NV_LD_GLOBAL_F32, dst, areg, boff ? is_oimm(boff) : is_onone(), is_onone())) return 1;
    if (is_cmt(nm->ref)) return 1;
    return is_bset(nm->ref, dst);
}
static int is_asgn(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_wrt(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }
static int is_acc(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }
static int is_cond(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_sloop(is_wk_t *w, const kir_node_t *nd) { (void)w; return is_nyet(nd); }
static int is_smem(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_sync(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_shfl(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_tree(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_let(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }
static int is_loop(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_init(is_wk_t *w, const kir_node_t *nd)  { (void)w; return is_nyet(nd); }
static int is_cpa(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }
static int is_cpc(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }
static int is_cpw(is_wk_t *w, const kir_node_t *nd)   { (void)w; return is_nyet(nd); }

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

/* The pending close of a finished list. IC_NONE has nothing to do; the
 * others belong to the control-flow tasks and nothing registers them yet, so
 * reaching one is our bug. */
static int is_close(is_wk_t *w, const is_frm_t *f)
{
    (void)w;
    switch ((is_cls_t)f->cls) {
    case IC_NONE:
        return 0;
    case IC_LOOP: case IC_CTHEN: case IC_CEND:
        return er_fail(E000, "a pending close with no code behind it");
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
    if (k->root == 0 || IX_CHK(k->root, k->nnode) || k->node[k->root].kind != KR_TILE)
        return er_fail(E021, "a body that is not exactly one Tile");
    tile = &k->node[k->root];
    if (is_push() || mm_params(k) || is_tile(k, tile, &body)) return 1;
    if (is_walk(k, body)) return 1;
    return is_pop() || is_tend();
}
