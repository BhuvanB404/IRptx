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

### Tools

- `tools/sweep.py` sends every problem through the hackathon runner with
  `submit.py --compiler compile.py`, records pass, wrong, ptxas, compile_error
  or run_error per problem in `results/sweep.json`, and `--compare` exits 1 on
  any pass-to-fail regression. Today every open problem is compile_error, as
  the compiler is still a stub (Bhuvan B, 2026-10-25)
