# kirdump.py -- fe_*: Graph.from_dict, buffers, meta.json, generic KIR writer (no decisions).
#
# The front end loads one kernel IR through Emmy's own classes and writes it as KIR text.
# It never folds, reorders, drops or
# renames anything: the body is a generic dataclass walk, field by field, in declaration order.
#
# Refusals: every function raises FeError(code, what). compile.py catches it and calls
# fe_report(err), which prints `error[E0NN]: irptx: <what>` and lets the caller exit 1. An
# exception (not print-and-exit) keeps these functions callable from tests.
import contextlib
import dataclasses
import json
import sys
from pathlib import Path


class FeError(Exception):
    """A front-end refusal: `code` is the E-code string, `what` the message tail."""

    def __init__(self, code, what):
        super().__init__(what)
        self.code = code
        self.what = what


def fe_report(err):
    """Print a FeError in the house format on stderr."""
    print(f"error[{err.code}]: irptx: {err.what}", file=sys.stderr)


# ---- Loading ----
def fe_load(path):
    """Return (graph, kernel_node) for the single KernelOp in the IR file at `path`."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise FeError("E001", f"cannot read IR file {path}: {exc}") from exc
    # Emmy may chat on stdout at import time; stdout belongs to the PTX, so divert it.
    with contextlib.redirect_stdout(sys.stderr):
        from emmy.compiler.graph import Graph
        from emmy.compiler.ir.kernel.ir import KernelOp
    try:
        graph = Graph.from_dict(data)
    except Exception as exc:  # Emmy raises KeyError/ValueError/...; any of them is a load failure
        raise FeError("E009", f"Emmy could not load {path}: {type(exc).__name__}: {exc}") from exc
    kernels = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if len(kernels) != 1:
        raise FeError("E009", f"expected exactly one KernelOp in {path}, found {len(kernels)}")
    return graph, kernels[0]


def fe_meta(path, entry):
    """Return the meta.json kernel dict whose `entry` matches, from the file beside `path`."""
    meta_path = Path(path).resolve().parent / "meta.json"
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        raise FeError("E002", f"cannot read {meta_path}: {exc}") from exc
    for kern in meta.get("kernels", []):
        if kern.get("entry") == entry:
            return kern
    raise FeError("E002", f"{meta_path} has no kernel entry named {entry}")


def fe_bufs(graph, kernel_node, meta_kern):
    """Return [(name, kind, dtype, shape)] in meta `arg_order`, typed from the graph outputs."""
    from emmy.compiler.ir.base import ConstantOp
    tensors = {t.name: (nid, t) for nid, node in graph.nodes.items() for t in node.outputs}
    bufs = []
    for name in meta_kern["arg_order"]:
        if name not in tensors:
            raise FeError("E002", f"arg_order names {name}, which the graph does not define")
        nid, tens = tensors[name]
        if nid == kernel_node.id:
            kind = "out"
        elif isinstance(graph.nodes[nid].op, ConstantOp):
            kind = "const"
        else:
            kind = "in"
        if not all(isinstance(d, int) or d.is_static for d in tens.shape):
            raise FeError("E009", f"buffer {name} has a symbolic shape {list(tens.shape)}")
        shape = [d if isinstance(d, int) else d.as_static() for d in tens.shape]
        bufs.append((name, kind, str(tens.dtype), shape))
    return bufs


# ---- Writing ----
def fe_dump(name, meta_kern, bufs, body):
    """Return the KIR text for one kernel. `body` is KernelOp.body."""
    lines = ["kir 1", f"kernel {name}"]
    lines.append("launch " + " ".join(str(v) for v in
                 [*meta_kern["grid"], *meta_kern["block"], meta_kern["smem_bytes"]]))
    for i, (bname, kind, dtype, shape) in enumerate(bufs):
        lines.append(f"arg {i} {bname} {kind} {dtype} {len(shape)} " + " ".join(map(str, shape)))
    lines += [f"zero {z}" for z in meta_kern.get("zero_outputs", [])]
    lines += ["body", fe_sexp(body), "end", ""]
    text = "\n".join(line.rstrip() for line in lines)
    bad = next((c for c in text if not (c == "\n" or " " <= c <= "~")), None)
    if bad is not None:
        raise FeError("E003", f"KIR dump has a non-ASCII or control character {bad!r}")
    return text


def fe_leaf(val):
    """Text of one non-container value; the type test order matters (bool is an int)."""
    from emmy.compiler.dtype import DataType
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    if val is None:
        return "none"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return val.hex() + "f"
    if isinstance(val, str):
        if '"' in val or "\\" in val:
            raise FeError("E003", f"string {val!r} holds a quote or backslash")
        return f'"{val}"'
    if isinstance(val, ElementwiseImpl):
        return "op:" + val.name
    if isinstance(val, DataType):
        return "dt:" + val.name
    raise FeError("E009", f"cannot dump a value of type {type(val).__name__}")


def fe_sexp(root):
    """Serialize `root` with an explicit stack: Python recursion depth is not ours to bound."""
    from emmy.compiler.dtype import DataType
    out = []
    stack = [root]
    while stack:
        item = stack.pop()
        if isinstance(item, _Text):
            out.append(item.s)
        elif isinstance(item, (tuple, list)):
            stack.append(_Text("]"))
            stack.extend(reversed([x for v in item for x in (v, _Text(" "))][:-1]))
            out.append("[")
        elif dataclasses.is_dataclass(item) and not isinstance(item, (type, DataType)):
            stack.append(_Text(")"))
            for fld in reversed(dataclasses.fields(item)):
                stack.append(getattr(item, fld.name))
                stack.append(_Text(f" {fld.name}="))
            out.append("(" + type(item).__name__)
        else:
            out.append(fe_leaf(item))
    return "".join(out)


class _Text:
    """Literal output queued on the walk stack, distinct from any IR value (even a str)."""

    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s
