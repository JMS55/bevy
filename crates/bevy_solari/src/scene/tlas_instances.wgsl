// Packs TLAS instance descriptors on the GPU, so that moving an instance costs nothing beyond the
// transform write that already happens.
//
// The layout is `VkAccelerationStructureInstanceKHR`, which is byte-identical to DXR's
// `D3D12_RAYTRACING_INSTANCE_DESC`. Those are the only backends Solari builds a TLAS on; see
// `scene::tlas_build`.
//
// `transforms` holds each instance as three `vec4<f32>` rows of a row-major 3x4 matrix, which is
// exactly what both APIs want.
//
// Acceleration structure references are 64 bit, but nothing here does arithmetic on one — they are
// only copied — so they travel as a `vec2<u32>` of little-endian halves rather than a `u64`. That
// keeps `SHADER_INT64` out of `SolariPlugins::required_wgpu_features`, where it would be a hard
// requirement bought for no benefit.

@group(0) @binding(0) var<storage, read> transforms: array<array<vec4<f32>, 3>>;

// Per-slot acceleration structure reference, as the two halves of a `u64`. Zero means the slot is
// dead: either it was never handed out, or its instance isn't currently drawable. Vulkan and DXR
// both define a zero reference as an inactive instance that the build discards, so a hole costs
// nothing and needs no stand-in structure to point at.
@group(0) @binding(1) var<storage, read> blas_refs: array<vec2<u32>>;

// `VkAccelerationStructureInstanceKHR`, 64 bytes.
struct TlasInstance {
    transform: array<vec4<f32>, 3>,
    // Instance slot in the low 24 bits, visibility mask in the high 8. Slots past 2^24 would alias;
    // `RaytracingSceneBindings::advance_tlas` asserts they can't happen.
    custom_data_and_mask: u32,
    // Shader binding table offset in the low 24 bits, instance flags in the high 8. Solari uses
    // neither.
    sbt_offset_and_flags: u32,
    blas_ref: vec2<u32>,
}

@group(0) @binding(2) var<storage, read_write> instances: array<TlasInstance>;

@compute @workgroup_size(64)
fn pack_tlas_instances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot = global_id.x;
    if slot >= arrayLength(&instances)
        || slot >= arrayLength(&blas_refs)
        || slot >= arrayLength(&transforms) {
        return;
    }

    let blas_ref = blas_refs[slot];
    // A dead slot is masked off as well as left pointing at nothing, so it stays unhittable under
    // an API that honours only one of the two.
    let mask = select(0u, 0xFFu, any(blas_ref != vec2(0u)));

    instances[slot] = TlasInstance(
        transforms[slot],
        (slot & 0xFFFFFFu) | (mask << 24u),
        0u,
        blas_ref,
    );
}
