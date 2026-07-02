#!/usr/bin/env bash
# Regenerates the SPIR-V modules from the GLSL sources with debug info for Nsight Graphics,
# then validates them. Requires the Vulkan SDK (glslangValidator, spirv-opt, spirv-val on PATH).
#
# Two steps:
#   1. glslang -gVS  => NonSemantic.Shader.DebugInfo.100 + embedded GLSL source.
#   2. spirv-opt --merge-return --inline-entry-points-exhaustive --eliminate-dead-functions
#      => fully inlines `main` while emitting DebugInlinedAt records. Nsight builds its nested
#      call tree from DebugInlinedAt; without this step (glslang alone emits none) the driver
#      inlines at execution time and Nsight shows only a FLAT list of call sites.
#
# Usage:  ./build.sh   (from this directory)
set -euo pipefail
cd "$(dirname "$0")"
for s in restir_initial restir_temporal restir_spatial_and_shade; do
    echo "Compiling $s.comp -> $s.spv"
    glslangValidator --target-env vulkan1.2 -S comp -gVS "$s.comp" -o "$s.spv"
    spirv-opt --target-env=vulkan1.2 --merge-return --inline-entry-points-exhaustive \
        --eliminate-dead-functions "$s.spv" -o "$s.spv"
    spirv-val --target-env vulkan1.2 "$s.spv"
done
echo "All shaders compiled, inlined (with debug info) and validated."
