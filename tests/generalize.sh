#!/usr/bin/env bash
# generalize.sh -- held-out kernels: generate, compile, ptxas, and run on the GPU if present.
# Thin wrapper over tests/generalize.py; every argument is passed through
# (--regen, --no-gpu, --strict, --time N, --json F, --baseline F, --only a,b).
# Knobs: PY (default .venv/bin/python), HK (hackathon checkout), PTXAS, NVCC.
set -u
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$SELF")
PY=${PY:-$ROOT/.venv/bin/python}
export HK=${HK:-$ROOT/../emmy-hackathon-2026}
if [ ! -f "$HK/tools/generate.py" ]; then
    echo "generalize: skipped, no hackathon checkout at $HK"
    exit 0
fi
exec "$PY" "$SELF/generalize.py" "$@"
