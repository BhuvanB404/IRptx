/* t_er.c -- er_fail: the one printed line, byte for byte. */

#include <string.h>
#include "tharns.h"
#include "irptx.h"

static char eb[1024];

/* er_fail(code, what), its stderr text in eb, its return value. */
static int er_run(int code, const char *what)
{
    int rc;

    th_cap_on();
    rc = er_fail(code, what);
    th_cap_off(eb, sizeof eb);
    return rc;
}

static void er01(void)
{
    CHEQ(er_run(E011, "an index"), 1);
    CHSTR(eb, "error[E011]: irptx: cannot lower an index\n");
    PASS();
}
TH_REG("er", 1, "prints one error line and returns 1", er01)

static void er02(void)
{
    CHEQ(er_run(E000, NULL), 1);
    CHSTR(eb, "error[E000]: irptx: cannot lower something\n");
    PASS();
}
TH_REG("er", 2, "a NULL what prints 'something'", er02)

static void er03(void)
{
    CHEQ(er_run(777, "x"), 1);
    CHSTR(eb, "error[E000]: irptx: cannot lower x (unlisted error code 777)\n");
    PASS();
}
TH_REG("er", 3, "an unlisted code prints as E000 and names itself", er03)

static void er04(void)
{
    er_run(E013, "a\xc3\xa9\tz");
    CHSTR(eb, "error[E013]: irptx: cannot lower a???z\n");
    PASS();
}
TH_REG("er", 4, "bytes outside printable ASCII become '?'", er04)

static void er05(void)
{
    char what[400];
    size_t n;

    memset(what, 'w', sizeof what - 1u);
    what[sizeof what - 1u] = '\0';
    er_run(E081, what);
    n = strlen(eb);
    CHEQ(n, strlen("error[E081]: irptx: cannot lower ") + 300u + 1u);
    CHEQ(eb[n - 1u], '\n');
    CHEQ(eb[n - 2u], 'w');
    PASS();
}
TH_REG("er", 5, "a long what is cut at 300 characters", er05)

static void er06(void)
{
    CHECK(er_text(E082) != NULL);
    CHECK(er_text(E005) != NULL);
    CHECK(er_text(777) == NULL);
    PASS();
}
TH_REG("er", 6, "er_text finds listed codes and only those", er06)
