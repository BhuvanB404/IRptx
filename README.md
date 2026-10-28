irptx
=====

Emmy's Kernel IR in, PTX out. A compiler written for the
[Emmy Hackathon 2026](https://github.com/cloudrift-ai/emmy-hackathon-2026).

Emmy compiles PyTorch to CUDA C++ and builds it with `nvcc`. The step after that,
`ptxas`, turns PTX into the same GPU binary about 19x faster, and Emmy's autotuner
compiles thousands of variants. So this project skips the C++ in the middle and
writes the PTX itself.

Status: **`irptx 0.1.0`**. All 15 open problems pass on the current sweep across pointwise, reduction, matmul, fused, and attention categories.

Using it
--------

```bash
pip install emmy-ml==0.3.11 numpy requests
python compile.py path/to/kernel.ir.json > out.ptx
```

That invocation is the whole contract, and it is the one the hidden set is run
with. Two more flags are for reading rather than for grading:

```bash
python compile.py --kir path/to/kernel.ir.json    # the text the C core reads
python compile.py --version                       # irptx 0.1.0
```

Python 3.12 is the only requirement. `bin/irptx` is committed, statically linked
for x86-64 Linux, so no C compiler is needed to run the compiler. If that binary
cannot run on the machine it finds itself on, `compile.py` builds `src/*.c` once
with the system `cc` into `.irptx-cache/` and refuses with E004 if that fails
too.

Results
-------

From `results/sweep.json`, every problem through the hackathon's own
`submit.py --compiler compile.py` on the hosted RTX 5090 runner. `median_us` is
the kernel time, so it says as much about the launch as about the code.

| problem | category | result | median_us |
|---|---|---|---|
| a01_attention | 5 | PASS | 127.46 |
| a02_attention_128 | 5 | PASS | 49.73 |
| f01_rmsnorm | 4 | PASS | 17.34 |
| f02_row_softmax | 4 | PASS | 17.34 |
| f03_l2norm | 4 | PASS | 16.51 |
| m01_matmul_512 | 3 | PASS | 116.86 |
| m02_matmul_rect | 3 | PASS | 16.38 |
| p00_scalar_add | 1 | PASS | 24.90 |
| p01_vector_add | 1 | PASS | 24.03 |
| p02_scale_shift | 1 | PASS | 17.02 |
| p03_relu_mul | 1 | PASS | 17.06 |
| r01_array_sum | 2 | PASS | 23.97 |
| r02_big_sum | 2 | PASS | 41.28 |
| r03_mean | 2 | PASS | 32.13 |
| r04_row_sum | 2 | PASS | 24.48 |

Every module also assembles under `ptxas` for `sm_89` (what the problems are
scheduled for) and `sm_120` (the runner's card). A refusal is a deliberate
answer: the compiler prints one `error[E0NN]` line and nothing on stdout, because
a module that assembles and computes the wrong answer is worse than no module.

How it is built
---------------

```
kernel.ir.json  ->  compile.py + fe/kirdump.py  ->  KIR text  ->  bin/irptx  ->  PTX
                    (Emmy loads, we serialize)      (ASCII)      (C99 core)
```

Two parts, with one rule between them: **the front end decides nothing.**

The hackathon allows Emmy to *deserialize* the IR but not to lower it. Its own
renderer plus `nvcc -ptx` would produce working PTX for any IR, which is exactly
what is not allowed. So the Python side loads one kernel with
`Graph.from_dict` and writes every field out as KIR text, a generic walk of the
dataclasses in declaration order. It does not fold, reorder, drop or rename
anything. Every lowering decision happens in C, in `src/`, where a reviewer can
read it: `kir.c` parses, `isel.c` selects, `verify.c` checks the machine form
before any text exists, `emit.c` prints.

That split is also why the core is C and committed as a binary. It is small
enough to have no dependencies at all (no allocation, no recursion, every loop
bounded, every table sized in `src/irptx.h`), and it is the part a judge is
invited to review.

House style
-----------

The C core is written in a defensive C99 style: short prefixed names, comments that say why, ASCII only, and an E-code for every
refusal. `make rules` holds the code to them.

```bash
make          # build/irptx, for development
make static   # bin/irptx, statically linked; this is the committed binary
make test     # trunner, the smoke test over every open IR, the Python tests
make rules    # the house rules, as a ratchet
```
