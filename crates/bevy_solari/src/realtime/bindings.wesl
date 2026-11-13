import bevy_render::view::View;
import bevy_pbr::atmosphere::types::Atmosphere;
import bevy_pbr::prepass::bindings::PreviousViewUniforms;
import package::scene::sampling::{LightSample, NULL_LIGHT_ID};
import package::scene::bindings::light_sources;

enable wgpu_ray_query;

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(1) var<storage, read_write> light_tile_samples: array<LightSample>;
@group(1) @binding(2) var<storage, read_write> light_tile_resolved_samples: array<ResolvedLightSamplePacked>;
@group(1) @binding(3) var gbuffer: texture_2d<u32>;
@group(1) @binding(4) var depth_buffer: texture_depth_2d;
@if(MOTION_VECTORS_READ_WRITE)
@group(1) @binding(5) var motion_vectors: texture_storage_2d<rg16float, read_write>;
@if(!MOTION_VECTORS_READ_WRITE)
@group(1) @binding(5) var motion_vectors: texture_2d<f32>;
@group(1) @binding(6) var<uniform> view: View;
@group(1) @binding(7) var<uniform> previous_view: PreviousViewUniforms;
@group(1) @binding(8) var<storage, read_write> world_cache: WorldCache;
@group(1) @binding(9) var<uniform> constants: SolariLightingSettings;
// Views without an atmosphere bind fallbacks with a zero `outer_radius`.
@group(1) @binding(10) var atmosphere_transmittance_lut: texture_2d<f32>;
@group(1) @binding(11) var atmosphere_transmittance_sampler: sampler;
@group(1) @binding(12) var<uniform> atmosphere: Atmosphere;
@group(1) @binding(13) var<storage, read_write> light_grid: LightGrid;
/// Every light grid cell's list of slots, see `weighted_cell_slot` in `light_grid.wesl`.
@group(1) @binding(14) var<storage, read_write> light_grid_pool: array<u32>;
@group(1) @binding(15) var<storage, read_write> light_bounds: array<LightBounds>;
@group(1) @binding(16) var<storage, read_write> per_light_samples: PerLightSamples;

@if(RESTIR) {
@group(1) @binding(17) var previous_gbuffer: texture_2d<u32>;
@group(1) @binding(18) var previous_depth_buffer: texture_depth_2d;
@group(1) @binding(19) var<storage, read_write> reservoirs_a: array<Reservoir>;
@group(1) @binding(20) var<storage, read_write> reservoirs_b: array<Reservoir>;
}

@if(DLSS_RR_GUIDE_BUFFERS) {
@group(2) @binding(0) var diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var normal_roughness: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var dlss_rr_depth: texture_storage_2d<r32float, write>;
@group(2) @binding(4) var specular_motion_vectors: texture_storage_2d<rg16float, write>;
}

fn load_motion_vector(pixel_id: vec2<u32>) -> vec2<f32> {
    @if(MOTION_VECTORS_READ_WRITE)
    return textureLoad(motion_vectors, pixel_id).xy;
    @if(!MOTION_VECTORS_READ_WRITE)
    return textureLoad(motion_vectors, pixel_id, 0).xy;
}

/// User-configurable settings from the `SolariLighting` component, plus per-frame
/// state. Field order and types must match `SolariLightingUniforms` in `prepare.rs`.
struct SolariLightingSettings {
    confidence_weight_cap: f32,
    primary_local_light_samples: u32,
    secondary_local_light_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_local_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_cell_updates_soft_target: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
    /// Light ids `0..directional_light_count` are directional lights, and the rest local lights.
    directional_light_count: u32,
    light_grid_base_cell_size: f32,
    light_grid_position_lod_scale: f32,
    light_grid_contribution_threshold: f32,
    light_grid_global_candidate_fraction: f32,
    /// 0 = none, 1 = cells, 2 = lights per cell. See `light_grid_visualization`.
    light_grid_visualization: u32,
    light_grid_weighted_slots: u32,
    light_grid_weight_ratio_cap: f32,
}

/// Directional lights have light ids `0..directional_light_count()`.
fn directional_light_count() -> u32 {
    return constants.directional_light_count;
}

/// Local (point, spot, rect, and emissive mesh) lights have light ids
/// `directional_light_count()..directional_light_count() + local_light_count()`.
fn local_light_count() -> u32 {
    return arrayLength(&light_sources) - constants.directional_light_count;
}

/// Don't adjust the size of this struct without also adjusting `prepare::RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE`.
struct ResolvedLightSamplePacked {
    a: vec4<u32>,
    b: vec4<u32>,
}

/// Don't adjust the size of this struct without also adjusting `prepare::RESERVOIR_STRUCT_SIZE`.
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

/// Keep in sync with the packed world-cache layout constants in `prepare.rs`.
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

/// Keep in sync with the light grid layout constants in `prepare.rs`.
struct LightGrid {
    checksums: array<atomic<u32>, constants::LIGHT_GRID_SIZE>,
    life: array<atomic<u32>, constants::LIGHT_GRID_SIZE>,
    /// Offset into `light_grid_pool` and slot count of each cell's list. A count of 0 means the
    /// cell has no list, and the offset then holds why, see `LIGHT_GRID_NO_LIST_*`.
    lists: array<vec2<u32>, constants::LIGHT_GRID_SIZE>,
    /// Each cell's minimum corner and size.
    bounds: array<vec4<f32>, constants::LIGHT_GRID_SIZE>,
    pool_used: atomic<u32>,
    live_cells: atomic<u32>,
    overflowed_cells: atomic<u32>,
    failed_queries: atomic<u32>,
}

/// Marker value for an empty light grid cell.
const LIGHT_GRID_EMPTY_CELL: u32 = 0u;

/// Why a light grid cell has no list, stored in its list offset when its slot count is 0.
/// The cell was inserted this frame, and gets built next frame.
const LIGHT_GRID_NO_LIST_NOT_BUILT: u32 = 0u;
/// No light's contribution bound reaches the threshold.
const LIGHT_GRID_NO_LIST_NO_LIGHTS: u32 = 1u;
/// The pool ran out of space this frame.
const LIGHT_GRID_NO_LIST_OUT_OF_POOL: u32 = 2u;
/// Not a stored reason: the lookup gave up after too many hash collisions.
const LIGHT_GRID_NO_LIST_TABLE_FULL: u32 = 3u;

/// A sphere bounding a local light, and what its contribution is bounded by,
/// see `light_contribution_bound` in `light_grid_build.wesl`.
struct LightBounds {
    center: vec3<f32>,
    radius: f32,
    /// Lights emit only in front of this plane, or in every direction if zero.
    normal: vec3<f32>,
    /// 0 for point lights.
    area: f32,
    /// Peak radiance for area lights, or intensity for point lights.
    intensity: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

/// Points presampled on every light each frame, `light_samples_per_light()` per light, in light id
/// order. Light grid cell candidates pick a light from the cell's list, then one of its points,
/// rather than resolving the light themselves. The points are shared by the whole screen, like the
/// light tiles, so unlike points presampled per cell they don't make noise line up with cells.
struct PerLightSamples {
    light_samples: array<LightSample, constants::PER_LIGHT_SAMPLES_CAPACITY>,
    resolved_light_samples: array<ResolvedLightSamplePacked, constants::PER_LIGHT_SAMPLES_CAPACITY>,
}

/// How many points each light gets in `per_light_samples`. Every light gets the same amount, as many
/// as fit, up to `MAX_SAMPLES_PER_LIGHT`.
fn light_samples_per_light() -> u32 {
    return min(MAX_SAMPLES_PER_LIGHT, constants::PER_LIGHT_SAMPLES_CAPACITY / arrayLength(&light_sources));
}

const MAX_SAMPLES_PER_LIGHT: u32 = 1024u;
