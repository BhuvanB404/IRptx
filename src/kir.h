/* kir.h -- the KIR reader interface: node pools and kr_* declarations.
 *
 * One kernel is read into a single static kir_t inside kir.c. After a
 * successful kr_load it is read-only: isel and friends reach it through the
 * accessors below and never hold a pointer across a second kr_load. */

#ifndef KIR_H
#define KIR_H

#include <stddef.h>
#include "irptx.h"

/* ---- Argument dtypes ---- */

/* kir_arg_t.dt. Only f32 is lowered; every other dtype name the front end
 * writes becomes KD_OTHER and isel refuses it with E024. The reader does not
 * police dtypes because that is a lowering decision, not a syntax one. */
enum { KD_NONE = 0, KD_F32, KD_OTHER };

/* ---- Loading ---- */

/* Read `len` bytes of KIR text (header lines, then one S-expression, then
 * `end`) into the static pools. Returns 0 on success. On any failure it has
 * already printed one `error[E0NN]` line to stderr (E010-E015, E080-E082),
 * writes nothing to stdout and returns 1. `txt` need not be NUL terminated. */
int kr_load(const char *txt, size_t len);

/* The loaded kernel, read-only. Valid after a successful kr_load. */
const kir_t *kr_get(void);

/* ---- Lookup ---- */

/* The field called `name` of node `nd`, or NULL when absent. Callers check
 * and refuse with E015 when a required field is missing. */
const kir_val_t *kr_fld(const kir_node_t *nd, const char *name);

/* The n-th item (from 0) of list value `lst`, or NULL when `lst` is not a
 * list or n is past its end. */
const kir_val_t *kr_kid(const kir_val_t *lst, uint32_t n);

/* The node a KV_NODE value points at, or NULL for any other value. */
const kir_node_t *kr_nod(const kir_val_t *v);

/* The string at pool offset `off`; "" when the offset is out of range. */
const char *kr_str(uint32_t off);

#endif /* KIR_H */
