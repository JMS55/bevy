// Packs TLAS instance descriptors on the GPU, so that moving an instance costs nothing beyond the
// transform write that already happens.
//
// `transforms` holds each instance as three `vec4<f32>` rows of a row-major 3x4 matrix, which is
// exactly what Vulkan and DXR want and a transpose away from what Metal wants.
//
// See `scene::tlas_build::InstanceLayout` for the two descriptor layouts and which backend uses
// which.

@group(0) @binding(0) var<storage, read> transforms: array<array<vec4<f32>, 3>>;

// Per-slot acceleration structure reference. Zero means the slot is dead: either it was never
// handed out, or its instance isn't currently drawable.
@group(0) @binding(1) var<storage, read> blas_refs: array<u64>;

// What a dead slot points at.
//
// Vulkan and DXR both define a zero acceleration structure reference as an inactive instance that
// the build discards, so it's a constant there and nothing has to be passed in. Metal references
// its structures by `MTLResourceID`, an opaque handle rather than an address, and documents
// nothing about a zero one — so it gets the address of a degenerate dummy instead, which only the
// CPU knows.
#ifdef TLAS_INSTANCE_LAYOUT_METAL

struct PackConfig {
    dead_blas_ref: u64,
}
var<immediate> config: PackConfig;

fn dead_blas_ref() -> u64 {
    return config.dead_blas_ref;
}

#else

fn dead_blas_ref() -> u64 {
    return 0lu;
}

#endif

#ifdef TLAS_INSTANCE_LAYOUT_METAL

// `MTLIndirectAccelerationStructureInstanceDescriptor`, 72 bytes.
struct TlasInstance {
    // `MTLPackedFloat4x3`: four packed float3 columns, so 12 tightly packed floats.
    transform: array<f32, 12>,
    options: u32,
    mask: u32,
    intersection_function_table_offset: u32,
    user_id: u32,
    blas_ref: u64,
}

fn make_instance(rows: array<vec4<f32>, 3>, slot: u32, mask: u32, blas_ref: u64) -> TlasInstance {
    // Column j is (row0[j], row1[j], row2[j]).
    let r = rows;
    return TlasInstance(
        array<f32, 12>(
            r[0].x, r[1].x, r[2].x,
            r[0].y, r[1].y, r[2].y,
            r[0].z, r[1].z, r[2].z,
            r[0].w, r[1].w, r[2].w,
        ),
        0u,
        mask,
        0u,
        slot,
        blas_ref,
    );
}

#else

// `VkAccelerationStructureInstanceKHR`, byte-identical to `D3D12_RAYTRACING_INSTANCE_DESC`,
// 64 bytes.
struct TlasInstance {
    transform: array<vec4<f32>, 3>,
    // Instance slot in the low 24 bits, visibility mask in the high 8.
    custom_data_and_mask: u32,
    // Shader binding table offset in the low 24 bits, instance flags in the high 8. Solari uses
    // neither.
    sbt_offset_and_flags: u32,
    blas_ref: u64,
}

fn make_instance(rows: array<vec4<f32>, 3>, slot: u32, mask: u32, blas_ref: u64) -> TlasInstance {
    return TlasInstance(rows, (slot & 0xFFFFFFu) | (mask << 24u), 0u, blas_ref);
}

#endif

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
    let live = blas_ref != 0lu;

    // A dead slot is masked off as well as pointed away, so it stays unhittable under an API that
    // honours only one of the two.
    let mask = select(0u, 0xFFu, live);
    let resolved = select(dead_blas_ref(), blas_ref, live);

    instances[slot] = make_instance(transforms[slot], slot, mask, resolved);
}
