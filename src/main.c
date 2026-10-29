/* main.c -- reads KIR, runs the stages, the single fwrite of the module.
 *
 * irptx [file]   KIR from the file, or from stdin when no argument
 * irptx --version
 * Exit 0 with the module on stdout, or 1 with one error line on stderr and
 * nothing on stdout. */

#include <stdio.h>
#include <string.h>
#include "irptx.h"
#include "kir.h"

/* ---- Input ---- */

#define RD_CHUNK 65536u

/* +1 so an input of exactly IR_MAX_KIR bytes is told apart from a longer one */
static char KIR_BUF[IR_MAX_KIR + 1u];

/* Read all of `fp` into KIR_BUF. Returns 0 and sets *len, or prints E080
 * (over IR_MAX_KIR) or E001 (read error) and returns 1. The loop is bounded
 * by the buffer size: each pass either reads a byte or stops. */
static int rd_all(FILE *fp, size_t *len)
{
    size_t n = 0;
    IX_GUARD(g, IR_MAX_KIR + 2u);

    while (g--) {
        size_t want = (size_t)IR_MAX_KIR + 1u - n;
        size_t got;

        if (want > RD_CHUNK) want = RD_CHUNK;
        if (want == 0) return er_fail(E080, "input larger than the KIR limit");
        got = fread(KIR_BUF + n, 1, want, fp);
        n += got;
        if (n > IR_MAX_KIR) return er_fail(E080, "input larger than the KIR limit");
        if (got < want) {
            if (ferror(fp)) return er_fail(E001, "KIR input that could be read");
            *len = n;
            return 0;
        }
    }
    return er_fail(E080, "input larger than the KIR limit");
}

/* ---- Main ---- */

int main(int argc, char **argv)
{
    FILE *fp = stdin;
    size_t len = 0, outlen = 0;
    const char *out = NULL;
    int rc;

    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        if (fputs("irptx " IRPTX_VER "\n", stdout) == EOF || fflush(stdout) != 0) return 1;
        return 0;
    }
    if (argc > 2) return er_fail(E001, "more than one input file argument");
    if (argc == 2) {
        fp = fopen(argv[1], "rb");
        if (!fp) return er_fail(E001, "an input file that could be opened");
    }
    rc = rd_all(fp, &len);
    if (fp != stdin) fclose(fp);
    if (rc) return 1;

    if (kr_load(KIR_BUF, len)) return 1;
    if (is_run(kr_get())) return 1;
    if (vf_run()) return 1;
    if (em_run(&out, &outlen)) return 1;

    if (fwrite(out, 1, outlen, stdout) != outlen || fflush(stdout) != 0)
        return er_fail(E000, "a module that could be written to stdout");
    return 0;
}
