import bevy_render::view::View;
import bevy_pbr::prepass::bindings::PreviousViewUniforms;
import package::scene::sampling::{LightSample, NULL_LIGHT_ID};

enable wgpu_ray_query;

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(1) var<storage, read_write> light_tile_samples: array<LightSample>;
@group(1) @binding(2) var<storage, read_write> light_tile_resolved_samples: array<ResolvedLightSamplePacked>;
@group(1) @binding(3) var<storage, read_write> reservoirs_a: array<Reservoir>;
@group(1) @binding(4) var<storage, read_write> reservoirs_b: array<Reservoir>;
@group(1) @binding(5) var gbuffer: texture_2d<u32>;
@group(1) @binding(6) var depth_buffer: texture_depth_2d;
@group(1) @binding(7) var motion_vectors: texture_storage_2d<rg16float, read_write>;
@group(1) @binding(8) var previous_gbuffer: texture_2d<u32>;
@group(1) @binding(9) var previous_depth_buffer: texture_depth_2d;
@group(1) @binding(10) var<uniform> view: View;
@group(1) @binding(11) var<uniform> previous_view: PreviousViewUniforms;
@group(1) @binding(12) var<storage, read_write> world_cache: WorldCache;
@group(1) @binding(13) var<uniform> constants: SolariLightingSettings;
@group(1) @binding(14) var<storage, read_write> decorrelation_a: array<u32>;
@group(1) @binding(15) var<storage, read_write> decorrelation_b: array<u32>;
@group(1) @binding(16) var<storage, read_write> unresampled_radiance: array<u32>;

@if(DLSS_RR_GUIDE_BUFFERS) {
@group(2) @binding(0) var diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var normal_roughness: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var dlss_rr_depth: texture_storage_2d<r32float, write>;
@group(2) @binding(4) var specular_motion_vectors: texture_storage_2d<rg16float, write>;
}

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
    decorrelation_mode: u32,
    decorrelation_strength: f32,
    decorrelation_stagnancy_curve: f32,
    decorrelation_stagnancy_smoothing: f32,
    firefly_replacement_strength: f32,
    frame_rng: u32,
    reset: u32,
}

const DECORRELATION_OFF = 0u;
const DECORRELATION_UNIFORM = 1u;
const DECORRELATION_STAGNANCY = 2u;

// The decorrelation buffers are only allocated when one of the two features that needs them is
// enabled, so nothing may touch them otherwise.
fn decorrelation_enabled() -> bool {
    return constants.decorrelation_mode != DECORRELATION_OFF || constants.firefly_replacement_strength > 0.0;
}

// Per-pixel tracking of how stale the currently selected sample is, packed into one u32.
struct DecorrelationState {
    // Frames the current sample has been carried forward, reset when a new sample is generated
    sample_age: u32,
    // Spatially and temporally smoothed 0..1 staleness that drives the fallback probability
    smoothed_stagnancy: f32,
}

fn pack_decorrelation_state(state: DecorrelationState) -> u32 {
    let age = min(state.sample_age, 0xFFFFu);
    let stagnancy = u32(saturate(state.smoothed_stagnancy) * 65535.0 + 0.5);
    return age | (stagnancy << 16u);
}

fn unpack_decorrelation_state(packed: u32) -> DecorrelationState {
    return DecorrelationState(packed & 0xFFFFu, f32(packed >> 16u) / 65535.0);
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

// Keep in sync with the packed world-cache layout constants in `prepare.rs`.
struct WorldCache {
    checksums: array<atomic<u32>, constants::WORLD_CACHE_SIZE>,
    @if(WORLD_CACHE_NON_ATOMIC_LIFE_BUFFER)
    life: array<u32, constants::WORLD_CACHE_SIZE>,
    @if(!WORLD_CACHE_NON_ATOMIC_LIFE_BUFFER)
    life: array<atomic<u32>, constants::WORLD_CACHE_SIZE>,
    radiance: array<vec4<f32>, constants::WORLD_CACHE_SIZE>,
    geometry_data: array<WorldCacheGeometryData, constants::WORLD_CACHE_SIZE>,
    luminance_deltas: array<f32, constants::WORLD_CACHE_SIZE>,
    active_cells_new_radiance: array<vec3<f32>, constants::WORLD_CACHE_SIZE>,
    a: array<u32, constants::WORLD_CACHE_SIZE>,
    b: array<u32, 1024u>,
    active_cell_indices: array<u32, constants::WORLD_CACHE_SIZE>,
    active_cells_count: u32,
}
