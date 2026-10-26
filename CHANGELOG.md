irptx Changelog
===============

## Unreleased

### Docs

- the spec, plan, research, data model, contracts and task list for a compiler
  from Emmy Kernel IR to PTX, written with GitHub Spec Kit (Bhuvan B, 2026-10-22)

- an independent audit of that spec: a numpy simulation of its lowering rules
  against all 15 open problems, about 45 ptxas test kernels, a rules
  traceability table and a re-plan (Bhuvan B, 2026-10-24)

  short prefixed names, why-comments with `# ---- Section ----` headers, ASCII
  only with no em dashes, E-coded refusals, plain-English lower-case commit
  subjects, `0.x` versions per category milestone, and this changelog
  (Bhuvan B, 2026-10-25)

  Python front end that only loads the IR and writes it as KIR text. The
  core ships as a static x86-64 Linux binary, `bin/irptx`, with a one-time
  build from source as the fallback (Bhuvan B, 2026-10-25)

### Front end

- `fe/kirdump.py` loads one kernel IR with Emmy and writes it as KIR text, a generic
  field-by-field dump with no per-kind code; `fe_meta` refuses with E002 when
  `meta.json` is missing (Bhuvan B, 2026-10-25)

- `compile.py` writes the KIR text, runs the core on it with a 20 s limit and relays its stdout,
  stderr and exit code unchanged. It finds the core as `bin/irptx`, else a one-time `cc` build
  into `.irptx-cache/` (keyed on the sources, renamed into place atomically), else E004.
  `--kir` prints the KIR text and `--version` prints the version (Bhuvan B, 2026-10-25)

### Core

- `src/main.c` reads KIR from stdin or a file argument (E080 past 1 MiB), prints
  `irptx 0.1.0` for `--version`, and runs read, select, verify, emit with one `fwrite`
  only when all succeed. Selection, verify and emission are placeholders for now, so
  every kernel refuses with E020 and stdout stays empty (Bhuvan B, 2026-10-26)

- `src/ops.c`, `src/ops.h`: the op table for the pointwise basics (`add subtract multiply
  divide true_divide maximum minimum relu copy negative abs`) with `op_find`, arity,
  reducing identities (-1e30 for maximum, never infinity) and `op_emit` in `.rn` forms.
  `sub` is left out: Emmy cannot construct it (Bhuvan B, 2026-10-26)

- `is_load` lowers a scalar Load to `ld.global.f32` from a global input or a constant buffer
  (a constant index folds into `[%rd+imm]`, rank-0 and 1-element buffers load at offset 0) and
  binds the name to a fresh `%f`. Several names (T032), a non-f32 dtype and non-parameter inputs
  refuse. The four pointwise problems now stop at their first Assign (Bhuvan B, 2026-10-26)

- `is_asgn` lowers an Assign through `op_find` and `op_emit` into a fresh `%f` bound to the
  name (an accumulator's name writes into its home register). An op outside the table is E021
  naming it, an arg count off the arity E020, an unbound arg E022, a non-f32 dtype E024; dtype
  none is the inherited f32. The four pointwise problems now stop at their first Write
  (Bhuvan B, 2026-10-26)

### Build

- `Makefile`: `make` builds `build/irptx` with `-std=c99 -Wall -Wextra -pedantic
  -Werror -O2` (CC overridable), `make static` links `bin/irptx` statically,
  `make test` runs the C unit tests, the smoke test and the Python unittests
  (skipping the first two until they exist), `make rules` runs the rules check
  and `make clean` removes build outputs but never the committed `bin/irptx`
  (Bhuvan B, 2026-10-26)

### Tools

- `tools/sweep.py` sends every problem through the hackathon runner with
  `submit.py --compiler compile.py`, records pass, wrong, ptxas, compile_error
  or run_error per problem in `results/sweep.json`, and `--compare` exits 1 on
  any pass-to-fail regression. Today every open problem is compile_error, as
  the compiler is still a stub (Bhuvan B, 2026-10-25)
- `is_wrt` lowers Write to one `st.global.f32` per value (value i at index+i, offset folded into the immediate); the output must be an `out` parameter (E022), atomic and swizzle are E020. p00 to p03 now compile (Bhuvan B, 2026-10-26)
