/* t_kr.c -- the KIR reader: good input read back field by field, then input
 * it must refuse, each with its exact E-code and exactly one error line. */

#include <stdint.h>
#include <string.h>
#include "tharns.h"
#include "irptx.h"
#include "kir.h"

/* Header every body test shares: one rank-1 input of 4 elements. */
#define HDR  "kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0\narg 0 x in f32 1 4\nbody\n"
#define HDR0 "kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0\n"

static char kb[(2u << 20)];         /* generated inputs, over IR_MAX_KIR on purpose */
static char eb[512];                /* captured stderr of the last load */

/* Load `t`; 0 on success with a silent stderr, else the E-code of the single
 * error line it printed, or -1 when the return value, the line count or the
 * shape of the line is wrong. */
static int kr_try(const char *t, size_t n)
{
    int rc, code = 0, i, nl = 0;
    size_t len;

    th_cap_on();
    rc = kr_load(t, n);
    len = th_cap_off(eb, sizeof eb);
    if (rc == 0) return len == 0 ? 0 : -1;
    for (i = 0; eb[i]; i++) nl += eb[i] == '\n';
    if (rc != 1 || nl != 1 || strncmp(eb, "error[E", 7) != 0 || eb[10] != ']') return -1;
    code = (eb[7] - '0') * 100 + (eb[8] - '0') * 10 + (eb[9] - '0');
    return code;
}

static int kr_trys(const char *t) { return kr_try(t, strlen(t)); }

/* HDR + body + "\nend\n" into kb; returns the length. */
static size_t kr_mk(const char *hdr, const char *body)
{
    size_t h = strlen(hdr), b = strlen(body);

    memcpy(kb, hdr, h);
    memcpy(kb + h, body, b);
    memcpy(kb + h + b, "\nend\n", 5);
    return h + b + 5u;
}

static int kr_body(const char *body) { return kr_try(kb, kr_mk(HDR, body)); }

/* ---- Good input ---- */

static void kr01(void)
{
    const kir_t *K;

    CHEQ(kr_trys(HDR "(Tile axes=[] body=[])\nend\n"), 0);
    K = kr_get();
    CHSTR(K->name, "k");
    CHEQ(K->root, 1u);
    CHEQ(K->node[K->root].kind, KR_TILE);
    CHSTR(kr_str(K->node[K->root].kname), "Tile");
    CHEQ(K->node[K->root].nfld, 2u);
    PASS();
}
TH_REG("kr", 1, "reads the smallest kernel", kr01)

static void kr02(void)
{
    const kir_t *K;

    CHEQ(kr_trys("kir 1\nkernel my_k2\nlaunch 4 3 2 128 1 1 2048\n"
                 "arg 0 a in f32 2 8 16\narg 1 b const f32 0\narg 2 o out i32 3 1 2 3\n"
                 "zero o\nzero b\nbody\n(Tile)\nend\n"), 0);
    K = kr_get();
    CHSTR(K->name, "my_k2");
    CHEQ(K->grid[0], 4);
    CHEQ(K->grid[1], 3);
    CHEQ(K->grid[2], 2);
    CHEQ(K->block[0], 128);
    CHEQ(K->block[2], 1);
    CHEQ(K->smem, 2048);
    CHEQ(K->narg, 3u);
    CHSTR(kr_str(K->arg[0].name), "a");
    CHEQ(K->arg[0].kind, KA_IN);
    CHEQ(K->arg[0].dt, KD_F32);
    CHEQ(K->arg[0].rank, 2);
    CHEQ(K->arg[0].dim[1], 16);
    CHEQ(K->arg[1].kind, KA_CONST);
    CHEQ(K->arg[1].rank, 0);
    CHEQ(K->arg[2].kind, KA_OUT);
    CHEQ(K->arg[2].dt, KD_OTHER);
    CHEQ(K->arg[2].dim[2], 3);
    CHEQ(K->nzero, 2u);
    CHSTR(kr_str(K->zero[0]), "o");
    CHSTR(kr_str(K->zero[1]), "b");
    PASS();
}
TH_REG("kr", 2, "reads launch, args and zero lines", kr02)

static void kr03(void)
{
    const kir_node_t *n, *d;
    const kir_val_t *v;

    CHEQ(kr_trys(HDR "[(Weird a=0x1.8p+3f b=-9223372036854775808 c=9223372036854775807 "
                 "d=-0x1.0p+0f e=[[] [1 2] (Dim expr=(Literal value=64 dtype=\"int\") hint=none)] "
                 "f=op:add g=dt:f32 h=true i=none j=false s=\"a b\")]\nend\n"), 0);
    n = &kr_get()->node[kr_get()->root];
    CHEQ(n->kind, KR_OTHER);
    CHSTR(kr_str(n->kname), "Weird");
    CHEQ(n->nfld, 11u);
    CHEQ(kr_fld(n, "a")->tag, KV_FLT);
    CHEQ(kr_fld(n, "a")->fbits, 0x41400000u);
    CHEQ(kr_fld(n, "d")->fbits, 0xbf800000u);
    CHEQ(kr_fld(n, "b")->i, INT64_MIN);
    CHEQ(kr_fld(n, "c")->i, INT64_MAX);
    CHEQ(kr_fld(n, "f")->tag, KV_OP);
    CHSTR(kr_str(kr_fld(n, "f")->ref), "add");
    CHEQ(kr_fld(n, "g")->tag, KV_DT);
    CHSTR(kr_str(kr_fld(n, "g")->ref), "f32");
    CHEQ(kr_fld(n, "h")->tag, KV_BOOL);
    CHEQ(kr_fld(n, "h")->i, 1);
    CHEQ(kr_fld(n, "j")->i, 0);
    CHEQ(kr_fld(n, "i")->tag, KV_NONE);
    CHSTR(kr_str(kr_fld(n, "s")->ref), "a b");
    v = kr_fld(n, "e");
    CHEQ(v->n, 3u);
    CHEQ(kr_kid(v, 0)->n, 0u);
    CHEQ(kr_kid(v, 1)->n, 2u);
    CHEQ(kr_kid(kr_kid(v, 1), 1)->i, 2);
    d = kr_nod(kr_kid(v, 2));
    CHECK(d != NULL);
    CHEQ(d->kind, KR_DIM);
    CHEQ(kr_nod(kr_fld(d, "expr"))->kind, KR_LITERAL);
    CHEQ(kr_fld(d, "hint")->tag, KV_NONE);
    PASS();
}
TH_REG("kr", 3, "reads every value kind with exact numbers", kr03)

static void kr04(void)
{
    const kir_node_t *n;
    const kir_val_t *v;

    CHEQ(kr_trys(HDR "(Tile axes=[1 2] x=5)\nend\n"), 0);
    n = &kr_get()->node[kr_get()->root];
    v = kr_fld(n, "axes");
    CHEQ(v->n, 2u);
    CHECK(kr_kid(v, 2) == NULL);
    CHECK(kr_kid(kr_fld(n, "x"), 0) == NULL);
    CHECK(kr_kid(NULL, 0) == NULL);
    CHECK(kr_fld(n, "zz") == NULL);
    CHECK(kr_fld(NULL, "x") == NULL);
    CHECK(kr_fld(n, NULL) == NULL);
    CHECK(kr_nod(kr_fld(n, "x")) == NULL);
    CHECK(kr_nod(NULL) == NULL);
    CHSTR(kr_str(0xfffffff0u), "");
    PASS();
}
TH_REG("kr", 4, "lookups on absent things answer NULL", kr04)

static void kr05(void)
{
    CHEQ(kr_trys(HDR "(Tile a=1 b=2 c=3)\nend\n"), 0);
    CHEQ(kr_trys(HDR0 "body\n[]\nend\n"), 0);
    CHEQ(kr_get()->nnode, 0u);
    CHEQ(kr_get()->nfld, 0u);
    CHEQ(kr_get()->narg, 0u);
    CHEQ(kr_get()->root, 0u);
    PASS();
}
TH_REG("kr", 5, "a second load starts from nothing", kr05)

static void kr06(void)
{
    CHEQ(kr_trys("kir 1\r\nkernel k\r\nlaunch 1 1 1 1 1 1 0\r\nbody\r\n(Tile)\r\nend\r\n"), 0);
    CHEQ(kr_get()->node[kr_get()->root].kind, KR_TILE);
    PASS();
}
TH_REG("kr", 6, "CRLF line ends are fine", kr06)

static void kr07(void)
{
    CHEQ(kr_trys(HDR "[(Tile)]\nend\n"), 0);
    CHEQ(kr_get()->node[kr_get()->root].kind, KR_TILE);
    CHEQ(kr_trys(HDR "[(Tile) (Tile)]\nend\n"), 0);
    CHEQ(kr_get()->root, 0u);
    CHEQ(kr_trys(HDR "[]\nend\n"), 0);
    CHEQ(kr_get()->root, 0u);
    PASS();
}
TH_REG("kr", 7, "root is the lone Tile, else left at 0 for isel", kr07)

static size_t kr_nest(size_t depth)
{
    size_t h = strlen(HDR);

    memcpy(kb, HDR, h);
    memset(kb + h, '[', depth);
    memset(kb + h + depth, ']', depth);
    memcpy(kb + h + 2u * depth, "\nend\n", 5);
    return h + 2u * depth + 5u;
}

static void kr08(void)
{
    CHEQ(kr_try(kb, kr_nest(IR_MAX_DEP)), 0);
    PASS();
}
TH_REG("kr", 8, "nesting of 64 is accepted", kr08)

static void kr09(void)
{
    CHEQ(kr_try(kb, kr_nest(IR_MAX_DEP + 1u)), 82);
    PASS();
}
TH_REG("kr", 9, "nesting of 65 refuses E082", kr09)

/* ---- Truncated and malformed text ---- */

static void kr10(void)
{
    static const char t[] = HDR "(Tile axes=[] body=[(Var name=\"a\")] ";

    CHEQ(kr_try(t, sizeof t - 1u), 10);
    PASS();
}
TH_REG("kr", 10, "text cut inside the body refuses E010", kr10)

static void kr11(void)
{
    CHEQ(kr_trys(HDR "(Tile)\n"), 10);
    CHEQ(kr_trys(HDR "(Tile)\nen"), 10);
    PASS();
}
TH_REG("kr", 11, "a missing or cut 'end' refuses E010", kr11)

static void kr12(void)
{
    CHEQ(kr_trys(""), 10);
    CHEQ(kr_try(NULL, 0), 10);
    CHEQ(kr_trys("kir 1\n"), 10);
    CHEQ(kr_trys("kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0\n"), 10);
    CHEQ(kr_trys("kir 1\nkernel k\nlaunch 1 1 1 1 1 1"), 10);
    PASS();
}
TH_REG("kr", 12, "empty and header-only input refuse E010", kr12)

static void kr13(void)
{
    CHEQ(kr_trys("kir 2\nkernel k\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys("kernel k\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys("kir\nkernel k\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys("kir 1 x\nkernel k\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 10);
    PASS();
}
TH_REG("kr", 13, "a wrong or missing 'kir 1' line refuses E010", kr13)

static void kr14(void)
{
    CHEQ(kr_trys(HDR0 "frob 1\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys(HDR0 "launch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys(HDR0 "arg 1 x in f32 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys(HDR0 "arg 0 x sideways f32 0\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys(HDR0 "body extra\n[]\nend\n"), 10);
    PASS();
}
TH_REG("kr", 14, "unknown, repeated or out-of-order header lines refuse E010", kr14)

static void kr15(void)
{
    CHEQ(kr_body("[(Var name)]"), 10);
    CHEQ(kr_body("[(Var =5)]"), 10);
    CHEQ(kr_body("[(Var name=)]"), 10);
    CHEQ(kr_body("[(Var 5)]"), 10);
    CHEQ(kr_body("[(Var name=\"a\" name2)]"), 10);
    PASS();
}
TH_REG("kr", 15, "a malformed field token refuses E010", kr15)

static void kr16(void)
{
    CHEQ(kr_body("[(Var name=\"a\"]]"), 10);
    CHEQ(kr_body("[(Var name=\"a\")"), 10);
    CHEQ(kr_body("[(Var name=\"a\"))]"), 10);
    CHEQ(kr_body("("), 10);
    CHEQ(kr_body("(1)"), 10);
    PASS();
}
TH_REG("kr", 16, "mismatched or unclosed brackets refuse E010", kr16)

static void kr17(void)
{
    CHEQ(kr_body("[maybe]"), 10);
    CHEQ(kr_body("[(Var name=maybe)]"), 10);
    CHEQ(kr_trys(HDR "[(Var name=\"unterminated"), 10);
    CHEQ(kr_body("[(Var name=\xc3\xa9)]"), 10);
    PASS();
}
TH_REG("kr", 17, "unknown bare words, a cut string, high bytes refuse E010", kr17)

static void kr18(void)
{
    CHEQ(kr_body("[] []"), 10);
    CHEQ(kr_try(kb, kr_mk(HDR, "[]") ) , 0);
    CHEQ(kr_trys(HDR "[]\nend\njunk\n"), 10);
    CHEQ(kr_trys(HDR "[]\nend junk\n"), 10);
    PASS();
}
TH_REG("kr", 18, "a second expression or text after 'end' refuses E010", kr18)

/* ---- Strings ---- */

static void kr19(void)
{
    char s[100];
    size_t i;

    memcpy(s, "[(Var name=\"", 12);
    for (i = 0; i < 63; i++) s[12 + i] = 'a';
    memcpy(s + 75, "\")]", 4);
    CHEQ(kr_body(s), 0);
    CHEQ(strlen(kr_str(kr_fld(&kr_get()->node[kr_get()->root], "name")->ref)), 63u);
    PASS();
}
TH_REG("kr", 19, "a 63 character string is accepted", kr19)

static void kr20(void)
{
    char s[100];

    memcpy(s, "[(Var name=\"", 12);
    memset(s + 12, 'a', 64);
    memcpy(s + 76, "\")]", 4);
    CHEQ(kr_body(s), 13);
    PASS();
}
TH_REG("kr", 20, "a 64 character string refuses E013", kr20)

static void kr21(void)
{
    CHEQ(kr_body("[(Var name=\"a\"b\")]"), 13);
    CHEQ(kr_body("[(Var name=\"a\\\\b\")]"), 13);
    CHEQ(kr_body("[(Var name=\"a\tb\")]"), 13);
    CHEQ(kr_body("[(Var name=\"a\xc3\xa9\")]"), 13);
    PASS();
}
TH_REG("kr", 21, "quote, backslash, tab or high byte in a string refuse E013", kr21)

/* ---- Numbers ---- */

static void kr22(void)
{
    CHEQ(kr_body("[(L v=0x1.8p+3)]"), 14);
    CHEQ(kr_body("[(L v=0x1.zzp+3f)]"), 14);
    CHEQ(kr_body("[(L v=0x1.8f)]"), 14);
    CHEQ(kr_body("[(L v=0xf)]"), 14);
    CHEQ(kr_body("[(L v=0x)]"), 14);
    PASS();
}
TH_REG("kr", 22, "bad hex floats refuse E014", kr22)

static void kr23(void)
{
    CHEQ(kr_body("[(L v=0x1.0p+200f)]"), 14);
    CHEQ(kr_body("[(L v=-0x1.0p+200f)]"), 14);
    CHEQ(kr_body("[(L v=0x1.fffffep+127f)]"), 0);
    PASS();
}
TH_REG("kr", 23, "a hex float past f32 refuses E014, FLT_MAX fits", kr23)

static void kr24(void)
{
    CHEQ(kr_body("[(L v=99999999999999999999)]"), 14);
    CHEQ(kr_body("[(L v=9223372036854775808)]"), 14);
    CHEQ(kr_body("[(L v=-9223372036854775809)]"), 14);
    CHEQ(kr_body("[(L v=9223372036854775807)]"), 0);
    CHEQ(kr_body("[(L v=-9223372036854775808)]"), 0);
    PASS();
}
TH_REG("kr", 24, "an integer outside int64 refuses E014", kr24)

static void kr25(void)
{
    CHEQ(kr_body("[(L v=-)]"), 14);
    CHEQ(kr_body("[(L v=1x)]"), 14);
    CHEQ(kr_body("[(L v=1.5)]"), 14);
    CHEQ(kr_body("[(L v=--1)]"), 14);
    PASS();
}
TH_REG("kr", 25, "a lone minus or a non-digit in an integer refuses E014", kr25)

/* ---- Header content ---- */

static void kr26(void)
{
    CHEQ(kr_trys("kir 1\nkernel 9k\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 12);
    CHEQ(kr_trys(HDR0 "arg 0 x-y in f32 0\nbody\n[]\nend\n"), 12);
    CHEQ(kr_trys(HDR0 "zero a.b\nbody\n[]\nend\n"), 12);
    PASS();
}
TH_REG("kr", 26, "a bad kernel, arg or zero name refuses E012", kr26)

static void kr27(void)
{
    static const char tl[] = "\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n";
    char s[200];

    memcpy(s, "kir 1\nkernel ", 13);
    memset(s + 13, 'n', 64);
    memcpy(s + 77, tl, sizeof tl - 1u);
    CHEQ(kr_try(s, 77u + sizeof tl - 1u), 12);
    memcpy(s + 76, tl, sizeof tl - 1u);
    CHEQ(kr_try(s, 76u + sizeof tl - 1u), 0);
    PASS();
}
TH_REG("kr", 27, "a 63 character kernel name fits, 64 refuses E012", kr27)

static void kr28(void)
{
    CHEQ(kr_trys("kir 1\nkernel k\nbody\n[]\nend\n"), 15);
    CHEQ(kr_trys("kir 1\nlaunch 1 1 1 1 1 1 0\nbody\n[]\nend\n"), 15);
    PASS();
}
TH_REG("kr", 28, "body before kernel or launch refuses E015", kr28)

static void kr29(void)
{
    CHEQ(kr_trys(HDR0 "arg 0 x in f32 0\nbody\n[]\nend\n"), 0);
    CHEQ(kr_trys("kir 1\nkernel k\nlaunch 0 1 1 1 1 1 0\nbody\n[]\nend\n"), 14);
    CHEQ(kr_trys("kir 1\nkernel k\nlaunch 1 1 1 1 1 1 -1\nbody\n[]\nend\n"), 14);
    CHEQ(kr_trys("kir 1\nkernel k\nlaunch 1 1 1 1 1 1 0 9\nbody\n[]\nend\n"), 10);
    CHEQ(kr_trys(HDR0 "arg 0 x in f32 1 0\nbody\n[]\nend\n"), 14);
    PASS();
}
TH_REG("kr", 29, "launch and dim numbers are range checked", kr29)

/* ---- Limits ---- */

static size_t kr_args(unsigned n, unsigned rank)
{
    size_t p = strlen(HDR0);
    unsigned i, d;

    memcpy(kb, HDR0, p);
    for (i = 0; i < n; i++) {
        p += (size_t)snprintf(kb + p, 64, "arg %u x%u in f32 %u", i, i, rank);
        for (d = 0; d < rank; d++) p += (size_t)snprintf(kb + p, 8, " 1");
        kb[p++] = '\n';
    }
    memcpy(kb + p, "body\n[]\nend\n", 12);
    return p + 12u;
}

static void kr30(void)
{
    CHEQ(kr_try(kb, kr_args(32, 0)), 0);
    CHEQ(kr_get()->narg, 32u);
    CHEQ(kr_try(kb, kr_args(33, 0)), 81);
    PASS();
}
TH_REG("kr", 30, "32 args fit, the 33rd refuses E081", kr30)

static void kr31(void)
{
    CHEQ(kr_try(kb, kr_args(1, 8)), 0);
    CHEQ(kr_get()->arg[0].rank, 8);
    CHEQ(kr_try(kb, kr_args(1, 9)), 81);
    PASS();
}
TH_REG("kr", 31, "rank 8 fits, rank 9 refuses E081", kr31)

static void kr32(void)
{
    size_t h = strlen(HDR);

    memset(kb, ' ', sizeof kb);
    memcpy(kb, "kir 1\n", 6);
    CHEQ(kr_try(kb, IR_MAX_KIR + 1u), 80);
    CHEQ(kr_try(kb, IR_MAX_KIR), 10);       /* in bounds: refused for what it lacks */
    memcpy(kb, HDR "[]", h + 2u);
    memcpy(kb + h + 2u, "\nend\n", 5);
    CHEQ(kr_try(kb, h + 7u), 0);
    PASS();
}
TH_REG("kr", 32, "input over 1 MiB refuses E080, exactly 1 MiB is read", kr32)

static size_t kr_rep(const char *unit, size_t times)
{
    size_t h = strlen(HDR), u = strlen(unit), i, p = h;

    memcpy(kb, HDR, h);
    kb[p++] = '[';
    for (i = 0; i < times; i++) { memcpy(kb + p, unit, u); p += u; }
    kb[p++] = ']';
    memcpy(kb + p, "\nend\n", 5);
    return p + 5u;
}

static void kr33(void)
{
    CHEQ(kr_try(kb, kr_rep("(Var)", IR_MAX_NODE - 1u)), 0);
    CHEQ(kr_get()->nnode, IR_MAX_NODE - 1u);
    CHEQ(kr_try(kb, kr_rep("(Var)", IR_MAX_NODE)), 81);
    PASS();
}
TH_REG("kr", 33, "65535 nodes fit, one more refuses E081", kr33)

static void kr34(void)
{
    CHEQ(kr_try(kb, kr_rep("1 ", IR_MAX_ITEM - 1u)), 0);
    CHEQ(kr_get()->nitem, IR_MAX_ITEM - 1u);
    CHEQ(kr_try(kb, kr_rep("1 ", IR_MAX_ITEM)), 81);
    PASS();
}
TH_REG("kr", 34, "list items: the pool boundary refuses E081", kr34)

static void kr35(void)
{
    size_t p = strlen(HDR);
    unsigned i;

    memcpy(kb, HDR, p);
    p += (size_t)snprintf(kb + p, 16, "[(Tile n=0");
    for (i = 1; i < 6000u; i++) p += (size_t)snprintf(kb + p, 32, " f%u=%u", i, i);
    p += (size_t)snprintf(kb + p, 16, ")]\nend\n");
    CHEQ(kr_try(kb, p), 0);
    CHEQ(kr_get()->nfld, 6000u);
    CHEQ(kr_fld(&kr_get()->node[kr_get()->root], "f5999")->i, 5999);
    PASS();
}
TH_REG("kr", 35, "a node with thousands of fields reads back", kr35)

static void kr36(void)
{
    size_t p = strlen(HDR);
    unsigned i;
    const kir_val_t *v;

    memcpy(kb, HDR, p);
    memcpy(kb + p, "[(Tile s=[", 10);
    p += 10u;
    for (i = 0; i < 40000u; i++) p += (size_t)snprintf(kb + p, 40, "\"s%u_padpadpadpad\" ", i);
    memcpy(kb + p, "])]\nend\n", 8);
    CHEQ(kr_try(kb, p + 8u), 0);
    v = kr_fld(&kr_get()->node[kr_get()->root], "s");
    CHEQ(v->n, 40000u);
    CHSTR(kr_str(kr_kid(v, 39999)->ref), "s39999_padpadpadpad");
    PASS();
}
TH_REG("kr", 36, "forty thousand distinct strings read back", kr36)

static void kr37(void)
{
    CHEQ(kr_try(kb, kr_mk(HDR, "[(Tile a=[1 2 3] b=(Dim x=1) c=\"s\")]")), 0);
    CHEQ(kr_get()->nnode, 2u);
    CHEQ(kr_get()->nfld, 4u);
    CHEQ(kr_get()->nitem, 4u);        /* 3 in a, 1 in the root list */
    PASS();
}
TH_REG("kr", 37, "node, field and item counts match the text", kr37)

/* The range is checked on the double, before it is narrowed: narrowing an
 * out-of-range double to float is undefined, so 2^128 and a value a
 * hair past FLT_MAX must refuse without ever being converted. */
static void kr38(void)
{
    CHEQ(kr_body("[(L v=0x1p128f)]"), 14);
    CHEQ(kr_body("[(L v=0x1.fffffe0000001p+127f)]"), 14);
    CHEQ(kr_body("[(L v=-0x1.fffffep+127f)]"), 0);
    CHEQ(kr_body("[(L v=0x1p-149f)]"), 0);
    PASS();
}
TH_REG("kr", 38, "a hex float is range-checked as a double before it is narrowed", kr38)
