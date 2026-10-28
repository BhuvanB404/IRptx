/* irptx.h -- version, limits (IR_MAX_*), IX_GUARD / IX_CHK / IX_PNEW and shared types.
 *
 * Every .c file includes this and nothing else of ours that it can avoid.
 * Everything here is plain data: no function bodies, no globals, no malloc.
 * Each table in the compiler is a fixed array sized from the limits below,
 * so "how much memory does this use" is answered by reading this file.
 * Built with -std=c99 -Wall -Wextra -pedantic -Werror, so it has to be clean. */

#ifndef IRPTX_H
#define IRPTX_H

#include <stdint.h>
#include <stddef.h>

/* The one place the version lives. Printed by --version and stamped into the
 * first comment line of every PTX file (contracts/ptx-module.md, P11). */
#define IRPTX_VER "0.1.0"

/* ---- Limits ---- */

/* Hitting any of these refuses with an E08x code. Nothing is truncated and no
 * counter is wound back (data-model.md, "Limits"). */
#define IR_MAX_KIR   (1u << 20)     /* KIR text bytes; largest open one is 23 KB */
#define IR_MAX_NODE  65536u         /* KIR nodes */
#define IR_MAX_FLD   262144u        /* node fields */
#define IR_MAX_ITEM  262144u        /* list elements */
#define IR_MAX_STR   (1u << 20)     /* string pool bytes */
#define IR_MAX_DEP   64u            /* walk and expression stack depth; open set nests 6 */
#define IR_MAX_ARG   32u            /* kernel parameters */
#define IR_MAX_DIM   8u             /* buffer rank */
#define IR_MAX_INS   131072u        /* instruction records */
#define IR_MAX_REG   65535u         /* registers per file */
#define IR_MAX_BIND  16384u         /* live name bindings across all scopes */
#define IR_MAX_ACC   1024u          /* accumulators */
#define IR_MAX_SMEM  16u            /* shared arrays */
#define IR_MAX_OUT   2000000u       /* output characters, the runner's PTX limit */

#define IR_MAX_NAME  63u            /* longest KIR string (contracts/kir-text.md) */
#define IR_MAX_AFF   4u             /* variables in an affine index (kx_aff_t) */

/* ---- Guards ---- */

/* Bounded loop. Declares a down-counter; write `while (cond && g--)` so a
 * loop that never converges stops instead of spinning. Our own macro, three
 * lines, same idea as everyone's. */
#define IX_GUARD(g, n) uint32_t g = (uint32_t)(n)

/* Untrusted index check. Evaluates to 1 when `i` is out of bounds for a table
 * holding `n` items, so it reads as "if that's rubbish, refuse". The cast
 * makes a negative signed index huge rather than clever. Both arguments are
 * evaluated once. */
#define IX_CHK(i, n) ((uint64_t)(i) >= (uint64_t)(n))

/* Pool allocation. Index 0 is the sentinel, so a result of 0 means the pool is
 * full. The counter starts at 0, the first allocation returns 1, and a full
 * pool leaves the counter alone (no wound-back counters, no wraparound).
 * Evaluates `c` twice: pass a plain lvalue. */
#define IX_PNEW(c, max) (((c) < (uint32_t)(max)) ? ++(c) : 0u)

/* ---- Enums ---- */

/* Kernel IR node kinds, one per Emmy class that can appear in a KIR dump,
 * named after the class. Statements first (the ones isel lowers, then the
 * ones it refuses with E020), then expressions, then the plain data nodes.
 * The reader maps a name to one of these with its table in kir.c; anything
 * it does not know is KR_OTHER with kname kept, so isel can refuse by the
 * real name. Source: emmy.compiler.ir.{kernel,stmt,expr,axis}. */
typedef enum {
    KR_OTHER = 0,                   /* unknown name: refuse with E020 */
    /* lowered statements */
    KR_TILE, KR_LOAD, KR_ASSIGN, KR_WRITE, KR_ACCUM, KR_COND,
    KR_STRIDEDLOOP, KR_SMEM, KR_SYNC, KR_WARPSHUFFLE, KR_TREEHALVE,
    KR_LET, KR_LOOP, KR_INIT,
    KR_CPASYNCCOPY, KR_CPASYNCCOMMIT, KR_CPASYNCWAIT,
    /* statements we recognise by name and refuse (E020) */
    KR_PRE, KR_CARRY, KR_SELECT, KR_ZEROPROLOGUE,
    KR_TMADESCRIPTOR, KR_TMALOAD,
    KR_MBARRIERINIT, KR_MBARRIERARRIVEEXPECTTX, KR_MBARRIERARRIVE,
    KR_MBARRIERWAIT, KR_SETMAXNREG,
    KR_FRAGMENTAPPLY, KR_FRAGMENTROWREDUCE, KR_REGFRAGMENT,
    KR_BLOCKSCALELOAD, KR_LDMATRIXLOAD, KR_MMASYNCPTX,
    KR_FRAGMENTPROMOTE, KR_FRAGMENTREPACK,
    KR_WGMMADESCRIPTOR, KR_WGMMAMMA, KR_WGMMAFENCE, KR_WGMMACOMMIT,
    KR_WGMMAWAIT, KR_REGSTORE,
    /* integer and predicate expressions */
    KR_VAR, KR_LITERAL, KR_BINARYEXPR, KR_TERNARYEXPR, KR_BUILTIN,
    KR_FUNCCALLEXPR, KR_CASTEXPR, KR_FLATINDEX,
    /* plain data nodes */
    KR_AXIS, KR_DIM, KR_WINDOW, KR_FRAGLAYOUT, KR_KERNELOP,
    KR_NKIND                        /* count, keep last */
} kr_kind_t;

/* Value tags of a KIR field (contracts/kir-text.md, "S-expression values"). */
typedef enum {
    KV_NONE = 0,                    /* `none` */
    KV_BOOL, KV_INT, KV_FLT, KV_STR, KV_LIST, KV_NODE,
    KV_OP,                          /* op:<name> */
    KV_DT                           /* dt:<name> */
} kv_tag_t;

/* Kernel parameter kinds, from the `arg` lines. */
typedef enum { KA_IN = 0, KA_CONST, KA_OUT } ka_kind_t;

/* Register files. Four, as in the PTX module header: %r s32, %rd u64,
 * %f f32, %p pred. Every Accum, Init and loop variable has one home register
 * in one of these and rebinds are `mov` into it (ruling H2). */
typedef enum { RF_R = 0, RF_RD, RF_F, RF_P, RF_N } rf_t;

/* Operand kinds. */
typedef enum {
    OP_NONE = 0,
    OP_REG,                         /* register: rf + number */
    OP_IMM,                         /* s32 immediate */
    OP_FIMM,                        /* f32 immediate, as its 32-bit pattern */
    OP_SYM,                         /* shared array or parameter id */
    OP_LBL                          /* label number */
} op_kind_t;

/* PTX instruction opcodes. They index the mnemonic table in emit.c, and the
 * order here must match it; an unknown value is our bug (E000). Typed
 * variants are separate opcodes so emit never has to guess a suffix. This
 * is the set the audited PTX uses (audit/ptx-tests) plus the ops table. */
typedef enum {
    NV_NOP = 0,
    NV_LABEL, NV_RET, NV_BRA, NV_COMMENT,
    /* moves, params, conversions */
    NV_LD_PARAM_U64, NV_CVTA_TO_GLOBAL, NV_MOV_U32, NV_MOV_B32, NV_MOV_F32,
    NV_MOV_S32, NV_MOV_U64, NV_CVT_RN_F32_S32, NV_CVT_S32_U32,
    NV_CVT_RNI_F32_F32, NV_CVT_RZI_F32_F32, NV_CVT_RZI_S32_F32,
    NV_CVT_U32_U64, NV_CVT_U64_U32, NV_CVT_RN_F32_S64,
    /* memory */
    NV_LD_GLOBAL_F32, NV_LD_GLOBAL_V2_F32, NV_LD_GLOBAL_V4_F32,
    NV_ST_GLOBAL_F32, NV_ST_GLOBAL_V2_F32, NV_ST_GLOBAL_V4_F32,
    NV_LD_SHARED_F32, NV_LD_SHARED_V4_F32,
    NV_ST_SHARED_F32, NV_ST_SHARED_V4_F32,
    NV_RED_GLOBAL_ADD_F32,
    NV_CP_ASYNC_CA, NV_CP_ASYNC_CA8, NV_CP_ASYNC_CG, NV_CP_ASYNC_COMMIT, NV_CP_ASYNC_WAIT,
    /* integer arithmetic */
    NV_ADD_S32, NV_SUB_S32, NV_MUL_LO_S32, NV_MAD_LO_S32, NV_MUL_WIDE_S32,
    NV_ADD_S64, NV_MIN_S32, NV_DIV_S32, NV_REM_S32, NV_DIV_U32, NV_REM_U32,
    NV_AND_B32, NV_OR_B32, NV_XOR_B32, NV_SHL_B32, NV_SHR_U32, NV_SHR_S32,
    NV_SELP_S32, NV_SELP_F32, NV_NEG_S32,
    NV_MUL_WIDE_U32, NV_MAD_WIDE_U32, NV_SHR_U64, NV_SHL_B64, NV_OR_B64,
    NV_SHF_L_CLAMP_B32, NV_SHF_R_CLAMP_B32,
    /* predicates */
    NV_SETP_EQ_S32, NV_SETP_NE_S32, NV_SETP_LT_S32, NV_SETP_LE_S32,
    NV_SETP_GT_S32, NV_SETP_GE_S32, NV_SETP_GE_U32, NV_SETP_EQ_U32,
    NV_SETP_LT_F32, NV_SETP_GT_F32, NV_SETP_GE_F32, NV_SETP_EQ_F32,
    NV_AND_PRED, NV_OR_PRED, NV_NOT_PRED,
    /* float arithmetic */
    NV_ADD_RN_F32, NV_SUB_RN_F32, NV_MUL_RN_F32, NV_DIV_RN_F32,
    NV_FMA_RN_F32, NV_SQRT_RN_F32, NV_RCP_RN_F32,
    NV_MAX_F32, NV_MIN_F32, NV_ABS_F32, NV_NEG_F32, NV_COPYSIGN_F32,
    NV_EX2_APPROX_F32, NV_LG2_APPROX_F32, NV_RSQRT_APPROX_F32,
    /* thread, warp and barrier */
    NV_MOV_TID_X, NV_MOV_CTAID_X, NV_MOV_NTID_X, NV_MOV_LANEID,
    NV_ACTIVEMASK, NV_SHFL_BFLY, NV_BAR_SYNC, NV_BAR_WARP_SYNC,
    NV_NCOUNT                       /* count, keep last */
} nv_op_t;

/* ---- Shared structs: KIR side ---- */

/* One value. `ref` and `n` mean different things per tag (data-model.md). */
typedef struct {
    uint8_t  tag;                   /* kv_tag_t */
    int64_t  i;                     /* KV_INT, KV_BOOL */
    uint32_t fbits;                 /* KV_FLT: the f32 bit pattern; no float survives the reader */
    uint32_t ref;                   /* STR/OP/DT: string offset; NODE: node index; LIST: first item */
    uint32_t n;                     /* KV_LIST: item count */
} kir_val_t;

typedef struct {
    uint32_t  name;                 /* string pool offset */
    kir_val_t val;
} kir_fld_t;

typedef struct {
    uint16_t kind;                  /* kr_kind_t */
    uint32_t kname;                 /* string pool offset of the kind name */
    uint32_t fld0, nfld;            /* slice of the field pool */
} kir_node_t;

/* One kernel parameter. The parameter index is the position of its `arg`
 * line. Rank 0 is legal: a scalar output is one element, empty index is
 * offset 0 (ruling C2). */
typedef struct {
    uint32_t name;                  /* IR buffer name; never a PTX identifier */
    uint8_t  kind;                  /* ka_kind_t */
    uint8_t  dt;                    /* only f32 is lowered (E024) */
    uint8_t  rank;                  /* 0 .. IR_MAX_DIM */
    int32_t  dim[IR_MAX_DIM];       /* each at least 1 */
} kir_arg_t;

/* The whole kernel. Written by kr_load, read-only after. Big, so it lives in a
 * static, never on the stack. */
typedef struct {
    char       name[IR_MAX_NAME + 1];
    int32_t    grid[3], block[3];
    int32_t    smem;                /* dynamic shared bytes from `launch` */
    kir_arg_t  arg[IR_MAX_ARG];
    uint32_t   narg;
    uint32_t   zero[IR_MAX_ARG];    /* zero_outputs, string offsets; informational */
    uint32_t   nzero;
    uint32_t   root;                /* node index of the single Tile */
    kir_node_t node[IR_MAX_NODE];
    uint32_t   nnode;
    kir_fld_t  fld[IR_MAX_FLD];
    uint32_t   nfld;
    kir_val_t  item[IR_MAX_ITEM];
    uint32_t   nitem;
    char       str[IR_MAX_STR];
    uint32_t   nstr;
} kir_t;

/* ---- Shared structs: selection side ---- */

/* One operand. `v` is a register number, an immediate, f32 bits, a symbol id
 * or a label number depending on `kind`. */
typedef struct {
    uint8_t  kind;                  /* op_kind_t */
    uint8_t  rf;                    /* rf_t, OP_REG only */
    uint32_t v;
} nv_opnd_t;

/* One PTX instruction, label or directive before printing. */
typedef struct {
    uint16_t  op;                   /* nv_op_t */
    nv_opnd_t dst;
    nv_opnd_t src[4];
    uint8_t   nsrc;
    nv_opnd_t prd;                  /* optional guard predicate, @%p or @!%p */
    uint8_t   pneg;
    uint32_t  cmt;                  /* optional `// IR:` comment, string offset, 0 = none */
    uint8_t   uni;                  /* NV_BRA: a loop exit whose trip count isel proved
                                     * the same for every thread of a block */
} nv_ins_t;

/* One live name binding: an IR name (string offset) to an operand. */
typedef struct {
    uint32_t  name;
    nv_opnd_t opd;
    uint8_t   uni;                  /* 1: the value is the same in every thread of a block */
} is_bind_t;

/* One accumulator, bound once to a home register for its whole life. */
typedef struct {
    uint32_t  name;                 /* IR accumulator name */
    nv_opnd_t reg;
    uint8_t   op;                   /* reducing op, from the ops.c enum */
    uint32_t  ident;                /* f32 bits of the identity (max is -1e30, never inf) */
    uint8_t   has_init;             /* declared by Init: seeded once, never re-seeded */
    uint8_t   seeded;               /* 1 once an identity mov has been emitted for it (H1) */
} is_acc_t;

/* One shared array. Offsets are rounded up to the alignment. */
typedef struct {
    uint32_t name;
    uint32_t elems;                 /* product of the extents */
    uint32_t align;                 /* max(IR align, 16); declared at this align */
    uint32_t off;                   /* byte offset in the pool */
    uint8_t  rank;                  /* 1 .. IR_MAX_DIM */
    int32_t  dim[IR_MAX_DIM];       /* extents as the IR wrote them, for flattening */
} is_sarr_t;

/* The shared-memory layout, built by a pre-pass over the Tile body. */
typedef struct {
    is_sarr_t arr[IR_MAX_SMEM];
    uint32_t  narr;
    uint32_t  total;                /* bytes including padding */
    uint8_t   dyn;                  /* 1 when total > 49152: one extern pool */
} is_smem_t;

/* An affine index: sum(coef[i] * var[i]) + cst, at most IR_MAX_AFF variables.
 * Built only from +, * by literals, Var and Literal. Used to prove vector
 * alignment (M6) and to fold constants. Not affine means no vector access. */
typedef struct {
    uint32_t var[IR_MAX_AFF];       /* string offsets */
    int64_t  coef[IR_MAX_AFF];
    uint32_t nvar;
    int64_t  cst;
} kx_aff_t;

/* ---- E-codes ---- */

/* One macro per code; the meanings live in the table in err.c. Callers write
 * er_fail(E011, "..."). Ranges are fixed in data-model.md. */
/* internal */
#define E000 0
/* front end (printed by compile.py) */
#define E001 1
#define E002 2
#define E003 3
#define E004 4
#define E005 5
#define E009 9
/* KIR reader */
#define E010 10
#define E011 11
#define E012 12
#define E013 13
#define E014 14
#define E015 15
#define E019 19
/* isel */
#define E020 20
#define E021 21
#define E022 22
#define E024 24
#define E025 25
#define E026 26
#define E027 27
#define E028 28
#define E029 29
#define E034 34
#define E059 59
/* verify */
#define E060 60
#define E061 61
#define E062 62
#define E079 79
/* limits */
#define E080 80
#define E081 81
#define E082 82
#define E083 83
#define E084 84
#define E099 99

/* ---- Errors (err.c) ---- */

/* Print `error[E0NN]: irptx: cannot lower <what>` to stderr and return 1.
 * `what` may be NULL and is bounded; an unlisted code prints as E000. */
int er_fail(int code, const char *what);

/* One-line meaning of an E-code from the table, NULL if unlisted. */
const char *er_text(int code);

/* ---- Stage entry points ---- */

/* The pipeline main.c runs after kr_load: isel, verify, emission. Each
 * returns 0 on success or prints one error line and returns 1. */

/* isel.c: lower the loaded kernel to instruction records (is_*). */
int is_run(const kir_t *k);

/* verify.c: check the records against the module contract (vf_*). */
int vf_run(void);

/* emit.c: print the records into the IR_MAX_OUT buffer. On success `*out`
 * points at the module text and `*len` is its length; on failure (E084 when
 * the buffer would overflow) nothing has been written anywhere. */
int em_run(const char **out, size_t *len);

#endif /* IRPTX_H */
