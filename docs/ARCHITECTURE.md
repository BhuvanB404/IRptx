# irptx architecture

are written (task T008). T049 fills in the remaining sections.

## Pipeline (to be filled by T049)

kir (load and normalise the Kernel IR) -> isel (choose PTX forms, assign
registers) -> verify (check the machine form before text exists) -> emit
(print PTX text).

## Module map (to be filled by T049)

## Data model (to be filled by T049)

## Error codes (to be filled by T049)

## PTX idioms

Facts here come from the hackathon example
`examples/k_sum_1_reduce.ptx` (it passes on the runner) and from the
audit in `specs/001-kernel-ir-ptx-compiler/audit/audit_ptx.md`. Snippets
marked "checked" were assembled with `ptxas -arch=sm_89` and
`-arch=sm_120` (both exit 0); see the end of this section.

### Module header

```
.version 8.0
.target sm_89
.address_size 64
```

`.version 8.0` is the PTX ISA version; `.target sm_89` is the virtual
architecture; `.address_size 64` makes pointers 64 bit. irptx always
writes these three lines first. The runner is sm_120 but PTX 8.0 does
not know sm_120 (ptxas rejects that pair), and sm_89 PTX assembles for
both (audit row 1). Never raise the target.

### Entry and register declarations

```
.visible .entry k_sum_1_reduce(
        .param .u64 param_x,
        .param .u64 param_sum_1
)
{
        .reg .pred      %p<4>;
        .reg .f32       %f<16>;
        .reg .b32       %r<16>;
        .reg .b64       %rd<8>;
```

`.visible .entry` makes a kernel the host can launch; its name and
parameter order follow the contract. `.reg .T %name<N>` declares N
registers `%name0` to `%name{N-1}` of type T. irptx uses four banks:
`%p` predicates, `%f` 32-bit floats, `%r` 32-bit integers and raw
bits, `%rd` 64-bit addresses. `N` is the high-water count the isel
pass allocated, plus one, because numbering starts at 0 (the example
uses `%r1` upward and still declares `<16>`). Every accumulator, Init
and loop variable has one home register that later statements rebind
with `mov` or an in-place op (audit H2), so declared counts stay small.

### Param loads and cvta

```
        ld.param.u64    %rd1, [param_x];
        cvta.to.global.u64      %rd3, %rd1;
```

`ld.param.u64 d, [sym]` copies the kernel argument `sym` (a device
pointer) into register `d`. `cvta.to.global.u64 d, a` converts a generic
address `a` into a global-state-space address `d`, which is what
`ld.global` and `st.global` expect. irptx emits one `ld.param.u64` plus
one `cvta` per buffer parameter, in the prologue, in parameter order, so
every later access uses the `%rd` holding the global base. Scalar sizes
are baked into the code as immediates (the example compares against
`1024`), so only buffers become parameters unless the contract adds a
scalar.

### Thread, block and global index

```
        mov.u32         %r1, %ctaid.x;
        mov.u32         %r2, %ntid.x;
        mov.u32         %r3, %tid.x;
        mad.lo.s32      %r4, %r1, %r2, %r3;
```

`%tid.x` is the thread index in its block, `%ntid.x` the block size,
`%ctaid.x` the block index in the grid. The special registers must be
copied with `mov.u32` before use. `mad.lo.s32 d, a, b, c` computes
`a*b + c` keeping the low 32 bits, so `%r4` is the global index
`gid = ctaid*ntid + tid`. For a launch whose grid is 1 (as in r01) the
example only reads `%tid.x`; irptx skips the unused reads. Grid and
block y and z are expected to be 1 (audit H4). Checked.

### Addressing a float element

```
        mul.wide.s32    %rd5, %r4, 4;
        add.s64         %rd6, %rd3, %rd5;
        ld.global.f32   %f2, [%rd6];
        st.global.f32   [%rd6], %f2;
```

`mul.wide.s32 d64, a32, imm` multiplies and widens to 64 bits, so the
byte offset is `index * sizeof(float)` without overflow. Then add it to
the `cvta`'d base and load or store. Vector forms (`ld.global.v4.f32`)
are used only when every index coefficient and the constant are divisible
by the width (rule M6).

### Guarded early exit

```
        setp.ge.s32     %p1, %r4, 1000;
        @%p1 bra        DONE;
        ...
DONE:
        ret;
```

`setp.ge.s32 p, a, b` sets predicate `p` when `a >= b`; `@%p1 bra L`
branches only for threads where `p` is true; `ret` ends the thread.
irptx uses this when the grid has more threads than elements. Policy
(audit H4): if the body has no barrier, exit early when `gid >= N`. If
the body has a barrier, a thread that exits early would leave the others
waiting at `bar.sync`, so irptx requires the surplus threads to form
whole blocks, exits per block, and otherwise refuses with an E-code. Checked.

### Warp butterfly with shfl.sync.bfly.b32

```
        activemask.b32  %r5;
        shfl.sync.bfly.b32      %f3, %f1, 16, 0x1f, %r5;
        add.rn.f32      %f1, %f1, %f3;
        shfl.sync.bfly.b32      %f3, %f1, 8, 0x1f, %r5;
        add.rn.f32      %f1, %f1, %f3;
```

Operand order (audit row 7): `shfl.sync.bfly.b32 d, a, b, c, membermask`.
`d` receives the value of `a` from the lane whose id is
`laneid XOR b` (here `b` is the offset 16, 8, 4, 2, 1). `c = 0x1f` packs
clamp 31 and segment mask 0, i.e. the whole 32-lane warp. `membermask`
names the lanes taking part. `activemask.b32 d` returns the mask of
lanes currently executing, so partial warps (block 8, say) do not hang
or read dead lanes (audit H3); irptx does not assert `block % 32 == 0`
and does not hard-code `0xffffffff`. The `.b32` instruction accepts a
`.f32` register directly (audit row 7), so no `mov.b32` round trip is
needed; the hackathon example uses the round trip
(`mov.b32 %r5, %f1` then `mov.b32 %f3, %r6`), which is equally valid.
A full reduction repeats the pair for offsets 16, 8, 4, 2, 1 (fewer for
shorter segments). Never place a shuffle under a divergent condition
(rule M7). Checked.

### Shared-memory tree with bar.sync

```
.shared .align 4 .b8 tree_smem[512];
        ...
        mov.u32         %r8, tree_smem;
        shl.b32         %r9, %r3, 2;
        add.s32         %r10, %r8, %r9;
        st.shared.f32   [%r10], %f1;
        bar.sync        0;
        setp.ge.u32     %p2, %r3, 64;
        @%p2 bra        SKIP;
        ld.shared.f32   %f4, [%r10];
        ld.shared.f32   %f5, [%r10+256];
        add.rn.f32      %f4, %f4, %f5;
        st.shared.f32   [%r10], %f4;
SKIP:
        bar.sync        0;
```

A module-scope `.shared .align A .b8 name[BYTES]` declares a static
array; `mov.u32 r, name` yields its 32-bit shared address, which is then
offset by `tid*4` (`shl.b32` by 2) and used with `[reg]` or `[reg+imm]`.
`bar.sync 0` is a block-wide barrier on barrier id 0: every thread of
the block must reach it, and it orders the shared stores before the
loads that follow. One tree step: threads below the half-width load
their slot and the slot half-width away, add, and store back; the
branch skips the others, and both paths meet again at the second
`bar.sync`. irptx repeats this halving (128, 64, ..., 1 floats) and reads
the answer at slot 0. The barrier count must be a multiple of 32 and the
id at most 15 (rule M7). Static shared memory is capped at 48 KiB
(49152 bytes, inclusive; audit row 21): above that irptx switches to one
`.extern .shared .align 16 .b8 smem_pool[];` and adds constant offsets
(audit row 22). Declare arrays `.align max(IR align, 4)` (ruling C2).
Checked.

The hackathon example folds four warp partials in thread 0 instead of
looping the tree; that is the same idiom with the loop unrolled.

### .rn arithmetic

```
        mul.rn.f32      %f2, %f1, 0f3DCCCCCD;
        add.rn.f32      %f1, %f2, %f1;
```

`.rn` is round-to-nearest-even. On `add`, `sub` and `mul` it is optional
syntax, and without it ptxas may contract `x*c + d` into one FMA, which
rounds once and changes the low bit (audit row 18; SASS showed `FFMA`
for plain and `FMUL`+`FADD` for `.rn` on both arches). irptx therefore
writes `.rn` on every float `add`, `sub` and `mul`, so results match the
unfused reference bit for bit. For `div`, `sqrt`, `rcp` and `fma` the
modifier is mandatory syntax (plain forms are syntax errors; audit rows
12 and 20), and `.rn` there gives the IEEE-correct result. Float
literals are written as `0fXXXXXXXX` (IEEE single bits;
`0f3DCCCCCD` is 0.1f). Min and max use `min.NaN.f32` and `max.NaN.f32`
(audit). The identity for max is `-1e30`, never infinity. Checked.

### How the scratch file was checked

The snippets marked "checked" are lines of one kernel, `k_idioms`
(params x, y, n; early exit, `.rn` ops, `activemask` plus two shuffles,
a shared-memory tree step, two barriers). It was assembled with:

```
/opt/cuda/bin/ptxas -arch=sm_89  idioms.ptx -o /dev/null   # exit 0
/opt/cuda/bin/ptxas -arch=sm_120 idioms.ptx -o /dev/null   # exit 0
```

and the hackathon example also assembles on both targets.


driver file `nv_be.c` registers a table of function pointers (`isel`,
`emit`, option parsing) and runs them in order. The `isel` step
(`isel.c`, `nv_compile`) walks the compiler's SSA IR and produces a
PTX-shaped machine module in memory: virtual registers, instructions as
data, no text yet. It leaves register allocation to ptxas. A separate
check pass (`verify.c`, `nv_vchk`) runs over that machine module and
rejects bad register classes or malformed calls before any output exists;
a failure there aborts the compile. Only then does `emit.c` print the
machine module as PTX text (header, per-function declarations, each
instruction, globals). The split gives three properties irptx wants:
selection logic that never formats strings, a verifier that sees the exact
form being printed, and an emitter that is a dumb printer. irptx mirrors
this as kir -> isel -> verify -> emit, with the Kernel IR loader (kir) in
copied.
