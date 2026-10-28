"""Held-out kernels for tests/generalize.py: shapes that are NOT in the open set.

Same format as the hackathon's tools/expressions_open.py (a list named OPEN,
which is what tools/generate.py reads). Every shape here is deliberately off
the open set: odd and prime sizes, rectangular and non-square matmuls, other
reduction lengths, softmax / norm widths that are not 256, 512 or 1024,
attention with unequal head and sequence sizes, and chained pointwise ops.

Keep it that way: never add an open-set shape here, and never tune the
compiler against one entry of this list. It checks generality, it is not a
second open set.
"""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _f64(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _attn(q, kt, v, scale):
    return [(_softmax((_f64(q) @ _f64(kt)) / scale) @ _f64(v)).astype(np.float32)]


def _mm(a, b):
    return [(_f64(a) @ _f64(b)).astype(np.float32)]


def _sum(x, **kw):
    return [_f64(x).sum(**kw).astype(np.float32)]


def _rms(x, w):
    r = np.sqrt((_f64(x) ** 2).mean(-1, keepdims=True) + 1e-5)
    return [(x / r * w).astype(np.float32)]


def _ln(x):
    d = _f64(x) - _f64(x).mean(-1, keepdims=True)
    return [(d / np.sqrt((d * d).mean(-1, keepdims=True) + 1e-5)).astype(np.float32)]


def _l2(x):
    return [(x / np.sqrt((_f64(x) ** 2).sum(-1, keepdims=True))).astype(np.float32)]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-_f64(x)))


PW = dict(rtol=1e-6, atol=1e-6)
TR = dict(rtol=1e-5, atol=1e-6)
RD = dict(rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS)
MM = dict(rtol=1e-3, atol=1e-3)
FU = dict(rtol=1e-3, atol=1e-4)
SM = dict(rtol=1e-3, atol=1e-5)

OPEN = [
    # ---- pointwise: odd / prime lengths, 2-D, chains --------------------
    dict(id="g_add_1000", code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="g_mul_4099", code="torch.randn(4099) * torch.randn(4099)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="g_axpb_3000", code="torch.randn(3000) * 3.0 - 0.5",
         ref=lambda x: [x * np.float32(3.0) - np.float32(0.5)], **PW),
    dict(id="g_relu_mul_777", code="x = torch.randn(777); y = torch.randn(777); torch.relu(x) * y",
         ref=lambda x, y: [np.maximum(x, 0) * y], rtol=0, atol=0),
    dict(id="g_chain_2560",
         code="x = torch.randn(2560); y = torch.randn(2560); z = torch.randn(2560); "
              "torch.relu(x * y + z) * 0.5",
         ref=lambda x, y, z: [(np.maximum(_f64(x) * y + z, 0) * 0.5).astype(np.float32)], **PW),
    dict(id="g_sig_tanh_1536",
         code="x = torch.randn(1536); y = torch.randn(1536); torch.sigmoid(x) * torch.tanh(y)",
         ref=lambda x, y: [(_sig(x) * np.tanh(_f64(y))).astype(np.float32)], **TR),
    dict(id="g_add2d_96x80", code="torch.randn(96, 80) + torch.randn(96, 80)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),

    # ---- reductions: other lengths, other row shapes --------------------
    dict(id="g_sum_1000", code="torch.randn(1000).sum()", ref=lambda x: _sum(x), **RD),
    dict(id="g_sum_100003", code="torch.randn(100003).sum()", ref=lambda x: _sum(x),
         rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="g_mean_12345", code="torch.randn(12345).mean()",
         ref=lambda x: [_f64(x).mean().astype(np.float32)], **RD),
    dict(id="g_rowsum_48x300", code="torch.randn(48, 300).sum(-1)",
         ref=lambda x: _sum(x, axis=-1), **RD),
    dict(id="g_rowsum_33x777", code="torch.randn(33, 777).sum(-1)",
         ref=lambda x: _sum(x, axis=-1), **RD),

    # ---- matmul: rectangular, odd, non-multiple-of-tile ------------------
    dict(id="g_mm_96x80x48", code="torch.randn(96, 80) @ torch.randn(80, 48)", ref=_mm, **MM),
    dict(id="g_mm_200x128x72", code="torch.randn(200, 128) @ torch.randn(128, 72)", ref=_mm, **MM),
    dict(id="g_mm_64x192x320", code="torch.randn(64, 192) @ torch.randn(192, 320)", ref=_mm, **MM),
    dict(id="g_mm_33x65x17", code="torch.randn(33, 65) @ torch.randn(65, 17)", ref=_mm, **MM),
    dict(id="g_mm_256", code="torch.randn(256, 256) @ torch.randn(256, 256)", ref=_mm, **MM),

    # ---- fused: norms and softmax at other widths -----------------------
    dict(id="g_rms_48x768",
         code="x = torch.randn(48, 768); w = torch.randn(768); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=_rms, **FU),
    dict(id="g_softmax_32x100", code="torch.softmax(torch.randn(32, 100), dim=-1)",
         ref=lambda x: [_softmax(_f64(x)).astype(np.float32)], **SM),
    dict(id="g_softmax_17x1000", code="torch.softmax(torch.randn(17, 1000), dim=-1)",
         ref=lambda x: [_softmax(_f64(x)).astype(np.float32)], **SM),
    dict(id="g_l2_40x384",
         code="x = torch.randn(40, 384); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=_l2, **FU),
    dict(id="g_layernorm_32x512",
         # Inline on purpose: a name bound to an expression (d = x - m) traces as a
         # fresh graph input, not as the expression.
         code="x = torch.randn(32, 512); (x - x.mean(-1, keepdim=True)) / torch.sqrt("
              "((x - x.mean(-1, keepdim=True)) * (x - x.mean(-1, keepdim=True)))"
              ".mean(-1, keepdim=True) + 1e-5)",
         ref=_ln, **FU),

    # ---- attention: unequal sequence and head sizes ---------------------
    dict(id="g_attn_32x16x96",
         code="q = torch.randn(32, 16); kt = torch.randn(16, 96); v = torch.randn(96, 16); "
              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 4.0), **FU),
    dict(id="g_attn_96x64",
         code="q = torch.randn(96, 64); kt = torch.randn(64, 96); v = torch.randn(96, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 8.0), **FU),
    dict(id="g_attn_48x32x80",
         code="q = torch.randn(48, 32); kt = torch.randn(32, 80); v = torch.randn(80, 24); "
              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 5.656854), **FU),
]
