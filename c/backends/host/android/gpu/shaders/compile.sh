#!/usr/bin/env bash
# Compile GLSL compute shaders to SPIR-V and embed as shaders_spv.h.
# Prefers glslangValidator; falls back to emit_spv.py (no extra tools).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if command -v glslangValidator >/dev/null 2>&1; then
  for f in linear rmsnorm silu vec_add vec_mul softmax rope pack merge cache_store attn gather linear_f16 gather_f16 linear_gemm argmax linear_q8 silu_mul; do
    glslangValidator -V --target-env vulkan1.0 -o "${f}.spv" "${f}.comp"
  done
  python3 - "$DIR" <<'PY'
import os, struct, sys
here = sys.argv[1]
names = ["linear","rmsnorm","silu","vec_add","vec_mul","softmax","rope","pack","merge","cache_store","attn","gather","linear_f16","gather_f16","linear_gemm","argmax","linear_q8","silu_mul"]
blobs = {}
for n in names:
    blobs[n] = open(os.path.join(here, n + ".spv"), "rb").read()
sys.path.insert(0, here)
from emit_spv import embed_header
embed_header(blobs, os.path.join(here, "shaders_spv.h"))
print("wrote shaders_spv.h from glslang")
PY
else
  python3 "$DIR/emit_spv.py"
fi
