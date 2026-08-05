enable wgpu_ray_query;

#define_import_path bevy_solari::realtime_bindings

#import bevy_render::view::View
#import bevy_render::utils::{octahedral_decode, octahedral_encode}
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_solari::sampling::{LightSample, NULL_LIGHT_ID}

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(1) var<storage, read_write> light_tile_samples: array<LightSample>;
@group(1) @binding(2) var<storage, read_write> light_tile_resolved_samples: array<ResolvedLightSamplePacked>;
@group(1) @binding(3) var<storage, read_write> reservoirs_a: array<Reservoir>;
@group(1) @binding(4) var<storage, read_write> reservoirs_b: array<Reservoir>;
@group(1) @binding(5) var gbuffer: texture_2d<u32>;
@group(1) @binding(6) var depth_buffer: texture_depth_2d;
@group(1) @binding(7) var motion_vectors: texture_2d<f32>;
@group(1) @binding(8) var previous_gbuffer: texture_2d<u32>;
@group(1) @binding(9) var previous_depth_buffer: texture_depth_2d;
@group(1) @binding(10) var<uniform> view: View;
@group(1) @binding(11) var<uniform> previous_view: PreviousViewUniforms;
@group(1) @binding(12) var<storage, read_write> world_cache: WorldCache;
@group(1) @binding(13) var<uniform> constants: SolariLightingSettings;
@group(1) @binding(14) var<storage, read_write> debug_counters: array<atomic<u32>, #{DEBUG_COUNTER_COUNT}>;
@group(1) @binding(15) var<storage, read_write> debug_flags: array<u32>;
@group(1) @binding(16) var noise_moments_previous: texture_2d<f32>;
@group(1) @binding(17) var noise_moments: texture_storage_2d<rgba32float, write>;

#ifdef DLSS_RR_GUIDE_BUFFERS
@group(2) @binding(0) var diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var normal_roughness: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var specular_motion_vectors: texture_storage_2d<rg16float, write>;
#endif

// User-configurable settings from the `SolariLighting` component, plus per-frame
// state. Field order and types must match `SolariLightingUniforms` in `prepare.rs`.
struct SolariLightingSettings {
    confidence_weight_cap: f32,
    specular_confidence_weight_cap: f32,
    primary_di_samples: u32,
    secondary_di_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_direct_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_cell_updates_soft_target: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
    debug_view: u32,
    debug_counters: u32,
    debug_reset: u32,
}

// Don't adjust the size of this struct without also adjusting `prepare::RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE`.
struct ResolvedLightSamplePacked {
    world_position_x: f32,
    world_position_y: f32,
    world_position_z: f32,
    world_normal: u32,
    radiance: u32,
    inverse_pdf: f32,
}

// Don't adjust the size of this struct without also adjusting `prepare::RESERVOIR_STRUCT_SIZE`.
struct Reservoir {
    sample_point_world_position: vec3<f32>,
    unbiased_contribution_weight: f32,
    radiance: vec3<f32>,
    confidence_weight: f32,
    /// Octahedral normal packed into 16 bits per component by `pack_sample_normal`.
    sample_point_world_normal: u32,
    /// Low 3 bits hold which estimator produced this sample, see `PROVENANCE_*`
    /// in `debug.wgsl`. Bits 3-10 hold the sample's age in frames, see
    /// `reservoir_sample_age`. Remaining bits are unused.
    flags: u32,
    light_sample: LightSample,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        0u,
        0u,
        LightSample(NULL_LIGHT_ID, 0u),
    );
}

/// How many frames the reservoir has carried its current sample, saturating at
/// 255. This is the length of the sample's temporal correlation: a long-lived
/// sample means consecutive frames are not independent estimates, which reads as
/// structure rather than noise and survives denoising.
fn reservoir_sample_age(flags: u32) -> u32 {
    return (flags >> 3u) & 255u;
}

fn reservoir_with_sample_age(flags: u32, age: u32) -> u32 {
    return (flags & ~(255u << 3u)) | (min(age, 255u) << 3u);
}

// 16 bits per octahedral component, which leaves room in the reservoir for the
// provenance flags without growing it past 48 bytes.
fn pack_sample_normal(world_normal: vec3<f32>) -> u32 {
    return pack2x16unorm(octahedral_encode(world_normal));
}

fn unpack_sample_normal(packed: u32) -> vec3<f32> {
    return octahedral_decode(unpack2x16unorm(packed));
}

struct WorldCacheGeometryData {
    world_position: vec3<f32>,
    padding_a: u32,
    world_normal: vec3<f32>,
    padding_b: u32,
}

// Keep in sync with the packed world-cache layout constants in `prepare.rs`.
struct WorldCache {
    checksums: array<atomic<u32>, #{WORLD_CACHE_SIZE}>,
#ifdef WORLD_CACHE_NON_ATOMIC_LIFE_BUFFER
    life: array<u32, #{WORLD_CACHE_SIZE}>,
#else
    life: array<atomic<u32>, #{WORLD_CACHE_SIZE}>,
#endif
    radiance: array<vec4<f32>, #{WORLD_CACHE_SIZE}>,
    geometry_data: array<WorldCacheGeometryData, #{WORLD_CACHE_SIZE}>,
    luminance_deltas: array<f32, #{WORLD_CACHE_SIZE}>,
    active_cells_new_radiance: array<vec3<f32>, #{WORLD_CACHE_SIZE}>,
    a: array<u32, #{WORLD_CACHE_SIZE}>,
    b: array<u32, 1024u>,
    active_cell_indices: array<u32, #{WORLD_CACHE_SIZE}>,
    active_cells_count: u32,
}
