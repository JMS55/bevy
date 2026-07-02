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
# Usage:  ./build.ps1        (from this directory)
$ErrorActionPreference = "Stop"
$shaders = @("restir_initial", "restir_temporal", "restir_spatial_and_shade")
foreach ($s in $shaders) {
    Write-Host "Compiling $s.comp -> $s.spv"
    glslangValidator --target-env vulkan1.2 -S comp -gVS "$s.comp" -o "$s.spv"
    spirv-opt --target-env=vulkan1.2 --merge-return --inline-entry-points-exhaustive `
        --eliminate-dead-functions "$s.spv" -o "$s.spv"
    spirv-val --target-env vulkan1.2 "$s.spv"
}
Write-Host "All shaders compiled, inlined (with debug info) and validated."
