enable wgpu_ray_query;

#define_import_path bevy_solari::realtime_bindings

#import bevy_render::view::View
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_solari::sampling::{LightSample, NULL_LIGHT_ID}
#import bevy_solari::variance::VarianceStats

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

@group(1) @binding(12) var<storage, read_write> world_cache_checksums: array<atomic<u32>, #{WORLD_CACHE_SIZE}>;
#ifdef WORLD_CACHE_NON_ATOMIC_LIFE_BUFFER
@group(1) @binding(13) var<storage, read_write> world_cache_life: array<u32, #{WORLD_CACHE_SIZE}>;
#else
@group(1) @binding(13) var<storage, read_write> world_cache_life: array<atomic<u32>, #{WORLD_CACHE_SIZE}>;
#endif
@group(1) @binding(14) var<storage, read_write> world_cache_radiance: array<vec4<f32>, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(15) var<storage, read_write> world_cache_geometry_data: array<WorldCacheGeometryData, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(16) var<storage, read_write> world_cache_luminance_deltas: array<f32, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(17) var<storage, read_write> world_cache_active_cells_new_radiance: array<vec3<f32>, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(18) var<storage, read_write> world_cache_a: array<u32, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(19) var<storage, read_write> world_cache_b: array<u32, 1024u>;
@group(1) @binding(20) var<storage, read_write> world_cache_active_cell_indices: array<u32, #{WORLD_CACHE_SIZE}>;
@group(1) @binding(21) var<storage, read_write> world_cache_active_cells_count: u32;
@group(1) @binding(22) var<uniform> constants: SolariLightingSettings;

// Debug tooling: per-pixel temporal-variance moments (ping-ponged read/write
// buffers, roles swapped each frame on the CPU) and the global stats reduction.
// See `variance.wgsl`.
@group(1) @binding(23) var<storage, read> variance_moments_read: array<vec4<f32>>;
@group(1) @binding(24) var<storage, read_write> variance_moments_write: array<vec4<f32>>;
@group(1) @binding(25) var<storage, read_write> variance_stats: VarianceStats;

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
    // Debug tooling (see `variance.wgsl`). `variance_debug_mode`: 0 = off,
    // 1 = pre-denoise, 2 = post-denoise. `variance_threshold`: relative-variance
    // value mapped to the top of the heatmap and used for the over-threshold
    // pixel count. `variance_history_length`: capped sample count of the running
    // moments estimate. `path_isolation`: restricts the ReSTIR output to one class
    // of light path for attribution (see the `PATH_ISO_*` constants below).
    variance_debug_mode: u32,
    variance_threshold: f32,
    variance_history_length: f32,
    path_isolation: u32,
    // Bitmask of `DEBUG_FLAG_*` reuse/visibility toggles below, plus padding to a
    // 16-byte multiple (uniform layout rounds the struct size up to 16 anyway).
    debug_flags: u32,
    // Direct-lighting firefly clamp in post-exposure luminance; 0 disables it.
    firefly_clamp: f32,
    _debug_pad1: u32,
    _debug_pad2: u32,
}

// Path-isolation debug modes for `SolariLightingSettings::path_isolation`. Gate
// which candidates contribute during initial sampling so a single class of light
// path can be viewed/measured in isolation. See `initial_path.wgsl`.
const PATH_ISO_NORMAL = 0u;        // everything (no isolation)
const PATH_ISO_DIRECT = 1u;        // bounce-0 direct lighting only (NEE + emissive)
const PATH_ISO_INDIRECT = 2u;      // bounce >= 1 + world cache only (no bounce-0 direct)
const PATH_ISO_WORLD_CACHE = 3u;   // world-cache GI terminations only
const PATH_ISO_NO_WORLD_CACHE = 4u; // everything except the world cache

// `SolariLightingSettings::debug_flags` bits: independently disable a ReSTIR reuse
// stage or the direct-lighting shadow rays, to attribute noise to a stage.
const DEBUG_FLAG_DISABLE_TEMPORAL_REUSE = 1u;
const DEBUG_FLAG_DISABLE_SPATIAL_REUSE = 2u;
const DEBUG_FLAG_FORCE_FULL_VISIBILITY = 4u;
// Disable vector-valued (chroma-marginalized) shading in both the spatial merge and the light RIS,
// falling back to single-selected-sample shading. Lets you A/B the color-noise reduction.
const DEBUG_FLAG_DISABLE_COLOR_NOISE_REDUCTION = 8u;
// Strip the specular lobe from the primary surface (roughness -> 1, metallic -> 0) so direct
// lighting is pure diffuse. If shimmer collapses under this, the specular BRDF is the cause.
const DEBUG_FLAG_FORCE_DIFFUSE = 16u;
// Read this pixel's exact reprojected history instead of a frame-varying permuted neighbor in
// temporal reuse. If shimmer drops, the permutation is reading as temporal instability.
const DEBUG_FLAG_DISABLE_TEMPORAL_PERMUTATION = 32u;
// Two-level light RIS: stage 1 picks a light triangle via RIS with a cheap (diffuse, unshadowed)
// target; stage 2 refines the point on it via RIS with the full BRDF target. Concentrates the
// sample budget on promising lights when uniform light selection wastes candidates.
const DEBUG_FLAG_TWO_LEVEL_LIGHT_RIS = 64u;

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
    sample_point_world_normal: vec2<f32>,
    light_sample: LightSample,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        vec2(0.0),
        LightSample(NULL_LIGHT_ID, 0u),
    );
}

struct WorldCacheGeometryData {
    world_position: vec3<f32>,
    padding_a: u32,
    world_normal: vec3<f32>,
    padding_b: u32,
}
