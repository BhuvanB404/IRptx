/* isel.h -- selection state and record helpers shared by isel.c, kexpr.c and
 * (later) verify.c and emit.c.
 *
 * isel.c holds the state in one static struct. Everything outside reaches it
 * through these calls. Anything that can fail has already printed one
 * error line when it returns 1. Names bound here are string pool offsets from
 * the KIR reader; the reader does not intern, so two offsets can spell the
 * same name and lookups compare the text. */

#ifndef ISEL_H
#define ISEL_H

#include "irptx.h"

/* ---- Reset ---- */

/* Forget everything: records, registers, labels, bindings, homes,
 * accumulators. is_run calls it first, so a process (or a test) can run the
 * selector again from a clean slate. */
void is_reset(void);

/* ---- Registers ---- */

/* Allocate the next register of file `rf`. Numbers start at 1 (index 0 is
 * the pool sentinel, so %r0 is never used and a declaration is <count+1>).
 * Returns 0 and sets *out, or E081 when IR_MAX_REG is reached (the counter
 * is left alone). E000 for a bad `rf`. */
int rg_new(rf_t rf, nv_opnd_t *out);

/* The single home register of `name` in file `rf`. The first call
 * allocates, later calls return the same register, whatever scope they are
 * in. A name asked for in two different files is E029: the IR used one name
 * for an integer loop variable and an f32 value. E081 when the home table
 * is full. Does not bind the name: use is_bset. */
int rg_home(uint32_t name, rf_t rf, nv_opnd_t *out);

/* How many registers file `rf` has handed out (so %r<count+1> declares them). */
uint32_t rg_count(rf_t rf);

/* ---- Scopes and bindings ---- */

/* Open a scope. E082 when IR_MAX_DEP scopes are already open. */
int is_push(void);

/* Close the innermost scope, dropping its bindings. E000 with none open. */
int is_pop(void);

/* Bind `name` to `opd` in the innermost scope (it may shadow). E081 when the
 * binding table is full. */
int is_bset(uint32_t name, nv_opnd_t opd);

/* is_bset for a value known to be the same in every thread of a block. */
int is_bsetu(uint32_t name, nv_opnd_t opd);

/* Newest-first lookup. Returns 0 and sets *out, or E022 naming the name. */
int is_find(uint32_t name, nv_opnd_t *out);

/* 1 when the newest binding of `name` was made by is_bsetu, else 0 (also for
 * a name that is not bound). Silent. */
int is_funi(uint32_t name);

/* ---- Accumulators ---- */

/* The accumulator called `name`, or NULL. The pointer is valid until the next
 * is_aadd or is_reset. */
is_acc_t *is_afind(uint32_t name);

/* Register a new accumulator (seeded = 0). E081 when the table is full, E028
 * for a name already present (two Inits, or an Init of an Accum's name). */
int is_aadd(uint32_t name, nv_opnd_t reg, uint8_t op, uint32_t ident,
            uint8_t has_init, is_acc_t **out);

/* ---- Operands ---- */

nv_opnd_t is_onone(void);
nv_opnd_t is_oreg(rf_t rf, uint32_t n);
nv_opnd_t is_oimm(int32_t v);
nv_opnd_t is_ofim(uint32_t bits);       /* f32 bit pattern */
nv_opnd_t is_olbl(uint32_t n);
nv_opnd_t is_osym(uint32_t id);         /* shared array or parameter index */

/* ---- Memory ---- */

/* Parameter prologue: for each kernel parameter i,
 * in order, `ld.param.u64 %rdA, [param_i]` (OP_SYM i; the IR name rides in
 * the record's comment) then `cvta.to.global.u64 %rdB, %rdA`. %rdB is the
 * parameter's home register, kept for every later access. E024 for a
 * parameter that is not f32. Call once, before any access. */
int mm_params(const kir_t *k);

/* Index of the parameter called `name` (compared by text), or E022. */
int mm_arg(uint32_t name, uint32_t *idx);

/* Row-major flat element index of the index list `idx` (a KV_LIST) against
 * extents dim[0..rank-1]: sum(idx[k] * prod(dim[k+1..])). The list length
 * must equal `rank`, else E059; rank 0 with an empty list is the constant 0
 * (a scalar). A constant index (kx_fold) is folded to an OP_IMM and emits
 * nothing. Otherwise *out is an s32 register and the multiplies are
 * `mad.lo.s32`. Works for Smem extents too (pass the array's extents). */
int mm_flat(const int32_t *dim, uint32_t rank, const kir_val_t *idx, nv_opnd_t *out);

/* mm_flat against the dims of parameter `arg`. */
int mm_gflat(uint32_t arg, const kir_val_t *idx, nv_opnd_t *out);

/* 1 when a width-w (2 or 4) vector access at the flat index of `idx` is
 * provably aligned (all per-dimension affine coefficients and
 * constants, times the strides, divisible by w; base 16-aligned), else 0
 * (not affine, misaligned, or w not 2 or 4). Silent. Call after mm_flat. */
int mm_vec(const int32_t *dim, uint32_t rank, const kir_val_t *idx, uint32_t w);

/* Address of f32 element `flat` (OP_IMM or an s32 register) of parameter
 * `arg`, as register *areg (RF_RD) plus constant byte offset *boff, ready
 * for `[%rd+boff]`. A constant flat index emits nothing: *areg is the
 * parameter's base and *boff is flat*4. Otherwise `mul.wide.s32 off, flat, 4`
 * and `add.s64 addr, base, off`, with *boff 0. E059 if the byte offset
 * leaves s32. */
int mm_gaddr(uint32_t arg, nv_opnd_t flat, nv_opnd_t *areg, int32_t *boff);

/* Shared layout, built in document order by the pre-pass (is_spre). Adds an
 * array of the extents dim[0..rank-1] f32 with the IR's `align` (0 allowed),
 * keeping those extents so an index list can be flattened against them. Every
 * array is declared and placed at max(align, 16) and its offset is the running
 * total rounded up to that, so every shared offset is 16-aligned and a v2/v4
 * access needs only the element-index divisibility check. *id is its slot.
 * E059 for an empty or absurd extent, a non-power-of-two align or a size past
 * 1 GiB, E081 when IR_MAX_SMEM arrays exist. The pool is dynamic
 * (is_smem_t.dyn) once the total passes 48 KiB. */
int sm_add(uint32_t name, const int32_t *dim, uint32_t rank, uint32_t align, uint32_t *id);

/* Slot of the shared array called `name`, or E022. */
int sm_find(uint32_t name, uint32_t *id);

/* The layout so far, for the emitter to declare. */
const is_smem_t *sm_get(void);

/* Address of f32 element `flat` of shared array `id`, as s32 register *areg
 * plus constant byte offset *boff for `[%r+boff]`. Emits `mov.u32 base, sym`
 * where OP_SYM is the array slot in static mode, or IR_MAX_SMEM (the pool)
 * in dynamic mode, where *boff also carries the array's pool offset. A
 * register flat index adds `mad.lo.s32 a, flat, 4, base`. The layout must be
 * complete before the first call (the mode depends on the total). */
int sm_addr(uint32_t id, nv_opnd_t flat, nv_opnd_t *areg, int32_t *boff);

/* ---- Thread model ---- */

/* The thread prologue of one Tile. Emits
 *   mov.u32 tid, %tid.x ; [mov.u32 cta, %ctaid.x ; mad.lo.s32 gid, cta, block, tid]
 *   [setp.ge.s32 p, gid, N ; @p bra L_exit]
 * then decodes gid into the axes row-major (last fastest; shr/and for powers
 * of two, div/rem otherwise; an extent of 1 binds the immediate 0) and binds
 * each axis name in the innermost scope (the caller opens one first). `lane`
 * (tid & 31) and `warp` (tid >> 5) are bound when the kernel spells those
 * names. Raster hints and Axis.window are ignored (audit). *body is the
 * Tile's body list for the walk to visit.
 *
 * Guard: N = product of the extents. N > grid*block is E059; y or z beyond 1
 * in the launch is E025; aux_threads != 0, a non-constant extent, or Smem in
 * a body whose block size matches no suffix product of the extents is E059.
 * N < grid*block exits surplus threads early; when the body has a barrier
 * (Sync, TreeHalve, WarpShuffle, CpAsyncWait) N must be a multiple of the
 * block size, so whole blocks exit together before any barrier, else E059.
 * Record conventions: NV_LABEL has the OP_LBL in dst; NV_BRA has it in
 * src[0] and the guard predicate in prd. */
int is_tile(const kir_t *k, const kir_node_t *tile, const kir_val_t **body);

/* N of the last is_tile. */
uint32_t is_ncell(void);

/* The epilogue: the exit label (only if is_tile branched to it) and `ret`. */
int is_tend(void);

/* ---- The walk ---- */

/* The statement walk. One explicit stack of frames,
 * no recursion, at most IR_MAX_DEP deep (E082). A frame is a statement list
 * being visited: the list, the next item, and a "pending close" saying what
 * to do when the list is finished. is_run drives it; handlers only ask.
 *
 * Contract for a statement handler `int is_xxx(is_wk_t *w, const kir_node_t *nd)`
 * (one per statement kind, in isel.c):
 *   (a) it does its work on entry: emits records, binds names, pre-scans the
 *       immediate body to seed its accumulators, allocates labels;
 *   (b) to have a child list visited it calls is_wvisit once and returns 0.
 *       The walk visits that list after the handler returns, so the child
 *       statements come after every record the handler emitted. With
 *       scope = 1 is_wvisit opens the child scope at once (E082 if the
 *       nesting is full, nothing opened then): call is_wvisit FIRST, then
 *       bind loop variables, so they live in the child scope and vanish
 *       when it closes;
 *   (c) the close action `cls` is run by the walk when that list is done,
 *       with the frame (list, node, a, b) handed back. It runs with the
 *       list's scope still open, so a loop's back edge can still see its
 *       variable. If `scope` was set, the walk then pops the scope (unless a
 *       chained is_wvisit already did). A close
 *       may itself call is_wvisit (Cond: the then list closes with IC_CTHEN,
 *       which emits `bra end; else:` and visits else_body with IC_CEND);
 *       is_wvisit pops the finishing list's scope before opening the new one.
 * A handler that does not call is_wvisit is a leaf statement. Return 1 only
 * after printing one error (as everywhere). Handlers must not call is_push
 * or is_pop for a child list: pass scope = 1 instead. The Tile's body
 * frame has no scope of its own (is_run opened the Tile's). Nesting is allowed up to IR_MAX_DEP
 * frames, counting the Tile body.
 *
 * Pending-close kinds. IC_NONE does nothing. IC_LOOP steps a StridedLoop's
 * variable and branches back; IC_CTHEN closes a Cond's then list and visits
 * its else list; IC_CEND closes that else list. `a` and `b` carry two label
 * numbers, `nd` the statement node. */
typedef enum {
    IC_NONE = 0,
    IC_LOOP,                        /* back edge, step, branch, done label */
    IC_CTHEN,                       /* Cond: end of the then list */
    IC_CEND                         /* Cond: end of the else list, end label */
} is_cls_t;

typedef struct {
    const kir_val_t  *lst;          /* the statement list being visited */
    uint32_t          next;         /* next item to lower */
    uint8_t           cls;          /* is_cls_t, run when the list is done */
    uint8_t           scope;        /* 1: is_wvisit opened a scope; the walk pops it after cls */
    const kir_node_t *nd;           /* the statement that asked */
    uint32_t          a, b;         /* two words for the close (labels) */
} is_frm_t;

typedef struct is_wk_s is_wk_t;

/* Ask the walk to visit the statement list `lst` next (after the current
 * handler or close returns). E015 for a non-list, E000 for a second request
 * from one handler. The depth check (E082) happens when the walk pushes it. */
int is_wvisit(is_wk_t *w, const kir_val_t *lst, is_cls_t cls, int scope,
              const kir_node_t *nd, uint32_t a, uint32_t b);

/* ---- Records ---- */

/* Append a zeroed record with opcode `op`. NULL after printing E083 when
 * IR_MAX_INS is reached. The pointer is valid until the next append. */
nv_ins_t *is_new(nv_op_t op);

/* Append op with destination `d` and up to three sources. nsrc counts up to
 * the last source that is not OP_NONE. 0 on success, 1 with E083 when full. */
int is_ins(nv_op_t op, nv_opnd_t d, nv_opnd_t a, nv_opnd_t b, nv_opnd_t c);

/* Guard the most recent record with @p / @!p, and attach an IR comment
 * (string pool offset). E000 when there is no record yet. */
int is_guard(nv_opnd_t prd, int neg);
int is_cmt(uint32_t off);

/* A fresh label number (1-based). E081 past IR_MAX_REG. */
int is_lbl(uint32_t *n);

/* Read side for verify and emit. */
const nv_ins_t *is_recs(uint32_t *n);
uint32_t is_nlbl(void);

#endif /* ISEL_H */
