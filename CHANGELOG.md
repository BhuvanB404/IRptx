irptx Changelog
===============

## Unreleased

`f02_row_softmax` PASSes, so category 4 is complete and the sweep stands at
eleven of fifteen: every pointwise kernel, every reduction and every fused one.
What was missing was two rules the spec had already written down.

### Lowering

- seeding follows R8.2 exactly: an owning loop writes the op's identity and
  nothing else. Resolving `base` at loop entry was wrong twice over -- f02's
  `base` (`in0__acc1_sa`) is a name the body computes inside the loop, so the
  lookup raised E022 before the loop could start, and the identity is what
  Emmy seeds anyway. `base` keeps its R8.3 job as the update's left operand,
  so `acc1 = add(acc1 * exp(old_max - new_max), exp(x - new_max))` rescales
  instead of simply adding, which is what made online softmax come out right
  (Bhuvan B, 2026-10-26)

- `is_let` lowers a Let: a float literal to `mov.f32 %f, 0fXXXXXXXX` in a
  fresh `%f` that later statements read as a register, and an integer
  expression to the s32 value it folds to. A `FlatIndex` value is not in the
  open set and refuses E020 by kind name (Bhuvan B, 2026-10-26)

- `is_tree` folds one shared array per state side by side: each step loads
  `buf[j][tid+rlen]` for every state, combines once, and stores `buf[j][tid]`
  back, then broadcasts `buf[j][0]` into state j. Buffers and states must pair
  up (E020) and may not outnumber the layout's arrays. A `combine_states`
  list is lowered once per step instead of once per state -- it already names
  every state, so the old loop doubled it -- and it runs in a fresh scope so
  its intermediate names die with the step (R9). `f02_row_softmax` PASS on
  the runner (Bhuvan B, 2026-10-26)

### Tests

- `is36` to `is39`: the float Let's `0f` hex and the register it binds, the
  int Let and the refused FlatIndex, a two-state shuffle's ten butterflies
  against its seven adds (one combine list per step, not two), and a
  two-state tree's four shared loads, four stores and two broadcasts
  (Bhuvan B, 2026-10-26)

## irptx 0.1.0

Ten of the fifteen open problems now pass on the runner: every pointwise kernel,
every reduction, and two of the three fused ones. The compiler is a C99 core in
it back out as text, so the whole lowering is in `src/` and a reviewer can read
it without reading any Python. That reaches the hackathon's MVP bar (categories 1
and 2) and a little past it. Everything below this line is what it took.

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
- `mm_vec` proves a 2 or 4 wide access aligned (ruling M6: every per-dimension affine coefficient and constant, times its stride, divisible by the width); `is_load` and `is_wrt` use `.v2`/`.v4` then and N scalars otherwise, and refuse counts other than 1, 2 or 4 (E020). Vector stores copy values into a consecutive `%f` run unless they already are one. p00 to p03 now store with `st.global.v2/v4.f32`; ptxas accepts them for sm_89 and sm_120 (Bhuvan B, 2026-10-26)

- `is_acc`, `is_sloop` and `is_cond` lower category 2's control flow. An Accum is registered
  on first sight with its op's identity and updated in one home register; a StridedLoop seeds
  the accumulators that reduce its axis before the top label, so they do not restart each trip,
  and steps at the back edge; a Cond branches with `@!%p` and gives each body its own scope.
  An op with no identity and no base refuses E021, a zero step E059 (Bhuvan B, 2026-10-26)

- shared memory is placed by a pre-pass over the Tile body in document order, because r01
  stores into `acc0_smem` four statements before the `Smem` that declares it. `is_load` and
  `is_wrt` route an array name to `ld/st.shared.f32` flattened against the array's own extents,
  and `is_sync` emits `bar.sync id[, count]`, or a warp sync on this warp's activemask. The four
  reductions now reach their WarpShuffle (Bhuvan B, 2026-10-26)

- `is_shfl` folds across a warp with `shfl.sync.bfly.b32` from length/2 down to 1, carrying the
  value through `%r` and back, and runs the IR's combine statements (or the accumulator's own op
  when the list is empty). `is_tree` folds the per-warp partials in shared memory with one
  barrier per step and outside the predicate, then broadcasts `buf[0]` into every state, so
  whichever thread the IR writes from sees the total. r01 to r04 PASS on the runner, max abs err
  1.526e-05, ptxas clean for sm_89 and sm_120: categories 1 and 2 are complete, the MVP
  (Bhuvan B, 2026-10-26)

### Ops

- the op table grows from 11 names to 25. `sqrt`, `rsqrt`, `reciprocal`, `tanh`, `sin`,
  `cos` and `log2` are one record each; `exp`, `exp_fast`, `log`, `sigmoid`, `silu`,
  `softplus` and `pow` are a short sequence written out from ex2 and lg2
  (`exp(x) = ex2(x*log2e)`, `ln(x) = lg2(x)*ln2`, `sigmoid(x) = 1/(1+exp(-x))`), with `dst`
  written last so an accumulator's home register can be both argument and destination.
  `where`, `fma`, `gelu`, `gelu_tanh`, `erf` and `sign` stay out: the first two need a
  three-argument op shape and the rest a float predicate or an opcode the record set does
  not have (Bhuvan B, 2026-10-26)

- `f01_rmsnorm` and `f03_l2norm` PASS with the new ops (max abs err 1.907e-06 and 1.490e-08),
  so category 4 is 2 of 3 and the sweep is 10 of 15. `f02_row_softmax` needs the two coupled
  softmax states (T060) (Bhuvan B, 2026-10-26)

### Tests

- `is26` to `is35` cover the reduction shapes: the loop guard and back edge, the identity an
  Accum seeds, the refused op with no identity and the refused zero-free step, the Cond branch
  pair, a Write into a shared array that precedes its Smem statement, the barrier id, the
  shuffle ladder 16..1 with its two refusals, and the tree's barrier, broadcast and refusals.
  `op08` and `op09` pin the two composite shapes, exp's `mul` by log2 e then `ex2.approx` and
  sigmoid's four records into `dst` (Bhuvan B, 2026-10-26)
