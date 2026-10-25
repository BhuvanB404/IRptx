/* tharns.h -- the C test harness: TH_REG, assertions, stderr capture.
 *
 * pointer table, because tests/rules.sh bans function pointers. TH_REG does
 * not store the test anywhere. It defines a small global wrapper, th_run_<fn>,
 * that asks the runner whether to run this test (th_pick), calls it by name
 * and reports (th_done). The Makefile greps the TH_REG lines of tests/t_*.c
 * into build/tcases.h, which declares every wrapper and calls them in file
 * order (tests/tmain.c includes it), so a new test needs no other edit.
 *
 * Rules the generator enforces: one TH_REG per line, the function is named
 * <fam><num padded to two digits> (kx01), the family matches the file name
 * (t_kx.c), a family and number appear once. */

#ifndef THARNS_H
#define THARNS_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* What the command line asked for. */
typedef struct {
    int         list;               /* --list: print names, run nothing */
    const char *fam;                /* --fam <fam>: only that family, or NULL */
} th_sel_t;

/* ---- State (tests/tmain.c) ---- */

extern int  th_pass, th_fail;       /* totals for the summary */
extern int  th_ok;                  /* PASS() reached in the running test */
extern char th_msg[160];            /* first failed check of the running test */

/* True when the test should run now. Sets up the running test's state. */
int  th_pick(const th_sel_t *s, const char *fam, int num, const char *desc, const char *fn);
/* Print the result line of the test that just ran and count it. */
void th_done(void);

/* ---- Registration ---- */

/* Put after the test, on one line:
 *   TH_REG("kx", 1, "folds a literal times a literal", kx01) */
#define TH_REG(fam, num, desc, fn) \
    void th_run_##fn(const th_sel_t *s); \
    void th_run_##fn(const th_sel_t *s) \
    { \
        if (th_pick(s, fam, num, desc, #fn)) { fn(); th_done(); } \
    }

/* ---- Assertions ---- */

/* Stops the test at the first failure; the runner prints where. A test that
 * returns without reaching PASS() fails too, so a stray `return` cannot pass. */
#define CHECK(x) do { if (!(x)) { \
    snprintf(th_msg, sizeof th_msg, "%.30s:%d: %.100s", __FILE__, __LINE__, #x); \
    return; } } while (0)
#define CHEQ(a, b)  CHECK((a) == (b))
#define CHSTR(a, b) CHECK(strcmp((a), (b)) == 0)
#define PASS()      do { th_ok = 1; } while (0)

/* ---- Capturing stderr ---- */

/* Everything the code under test writes to stderr between th_cap_on and
 * th_cap_off lands in `buf` (NUL terminated, cut to `sz`) instead of the
 * terminal, so refusals leave the output clean and their text can be checked.
 * Returns the byte count. Not nestable. */
void   th_cap_on(void);
size_t th_cap_off(char *buf, size_t sz);

#endif /* THARNS_H */
