# GLSL port of `restir.wgsl` for Nsight Graphics shader debugging

Hand-written, close line-by-line GLSL translation of `restir.wgsl` and the functions it
transitively reaches. Compiled to SPIR-V with **function/source-level debug info** and
loaded through **wgpu SPIR-V passthrough**, so the ReSTIR passes can be single-stepped at
source level in [Nsight Graphics](https://docs.nvidia.com/nsight-graphics/UserGuide/configure-application.html#configuring-your-application-shaders).

WGSL → naga → SPIR-V yields little/no source-correlated debug info; glslang's `-gVS` emits
`NonSemantic.Shader.DebugInfo.100` plus the embedded GLSL source, which Nsight consumes.

## Files
- `restir_common.glsl` — shared translation unit (bindings, structs, all helpers). `#include`d.
- `restir_initial.comp` / `restir_temporal.comp` / `restir_spatial_and_shade.comp` — the three
  compute entry points (`main`). `restir_initial.comp` defines `DLSS_RR_GUIDE_BUFFERS` so it
  matches bevy_solari's `initial_with_psr_pipeline` (set=2 guide buffers + PSR code paths).
- `restir_*.spv` — prebuilt SPIR-V, checked in and embedded by `realtime/mod.rs`.

## Regenerate the SPIR-V
Requires the Vulkan SDK (`glslangValidator`, `spirv-val`):
```
./build.ps1     # Windows
./build.sh      # bash
```

## How it's loaded
`realtime/mod.rs` embeds the `.spv` via `embedded_asset!`; `realtime/node.rs` points the
`initial_with_psr`, `temporal`, and `spatial_and_shade` pipelines at them with entry point
`main`. The shader loader maps `.spv` → `Source::SpirV`, and
`RenderDevice::create_shader_module` routes SPIR-V to `create_shader_module_passthrough`
when the device has `PASSTHROUGH_SHADERS`.

**Build & run with:**
```
cargo run --example solari --features "spirv_shader_passthrough shader_format_spirv dlss"
```
(`dlss` is required because `initial` is the DLSS/PSR variant; run with DLSS Ray
Reconstruction enabled so `initial_with_psr_pipeline` is the one actually dispatched.)

## Keeping in sync
This is a manual port — if `restir.wgsl` or any function it reaches changes, update the
GLSL here and rerun the build script.
