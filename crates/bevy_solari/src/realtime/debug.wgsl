#define_import_path bevy_solari::debug

#import bevy_solari::realtime_bindings::{constants, debug_counters, debug_flags}

// Debug view ids. Keep in sync with `SolariDebugView` in `mod.rs`.
const DEBUG_VIEW_NONE = 0u;
const DEBUG_VIEW_NOISE_RELATIVE_STD_DEV = 1u;
const DEBUG_VIEW_NOISE_NON_RESAMPLED_SHARE = 2u;
const DEBUG_VIEW_NON_RESAMPLED_SHARE = 3u;
const DEBUG_VIEW_NON_RESAMPLED_ONLY = 4u;
const DEBUG_VIEW_RESAMPLED_ONLY = 5u;
const DEBUG_VIEW_SAMPLE_PROVENANCE = 6u;
const DEBUG_VIEW_CONFIDENCE_WEIGHT = 7u;
const DEBUG_VIEW_TEMPORAL_REJECT_REASON = 8u;
const DEBUG_VIEW_SPATIAL_REUSE_FAILURE = 9u;
const DEBUG_VIEW_JACOBIAN_REJECTION = 10u;
const DEBUG_VIEW_CONTRIBUTION_WEIGHT = 11u;
const DEBUG_VIEW_WORLD_CACHE_SAMPLE_COUNT = 12u;
const DEBUG_VIEW_WORLD_CACHE_PROBE_FAILURE = 13u;
const DEBUG_VIEW_WORLD_CACHE = 14u;
const DEBUG_VIEW_NOISE_RESAMPLED_STD_DEV = 15u;
const DEBUG_VIEW_SAMPLE_AGE = 16u;
const DEBUG_VIEW_SAMPLE_DUPLICATION = 17u;

// Counter slots. Keep in sync with `SOLARI_DEBUG_COUNTERS` in `prepare.rs`.
const DEBUG_COUNTER_PIXELS_SHADED = 0u;
const DEBUG_COUNTER_SPECULAR_PIXELS = 1u;
const DEBUG_COUNTER_TEMPORAL_REPROJECTED_OFFSCREEN = 2u;
const DEBUG_COUNTER_TEMPORAL_REJECTED_DISSIMILAR = 3u;
const DEBUG_COUNTER_TEMPORAL_REJECTED_LIGHT_DESPAWNED = 4u;
const DEBUG_COUNTER_TEMPORAL_NO_HISTORY = 5u;
const DEBUG_COUNTER_X2_NOT_REUSABLE = 6u;
const DEBUG_COUNTER_SPATIAL_NO_NEIGHBOR_FOUND = 7u;
const DEBUG_COUNTER_SPATIAL_CANDIDATES_REJECTED = 8u;
const DEBUG_COUNTER_JACOBIAN_TEMPORAL_DISCARD_NEIGHBOR = 9u;
const DEBUG_COUNTER_JACOBIAN_TEMPORAL_INFLATE_CANONICAL = 10u;
const DEBUG_COUNTER_JACOBIAN_SPATIAL_DISCARD_NEIGHBOR = 11u;
const DEBUG_COUNTER_JACOBIAN_SPATIAL_INFLATE_CANONICAL = 12u;
const DEBUG_COUNTER_WORLD_CACHE_PROBE_EXHAUSTED = 13u;
const DEBUG_COUNTER_WORLD_CACHE_QUERIES = 14u;
const DEBUG_COUNTER_PATH_TERMINATED_INTO_CACHE = 15u;
const DEBUG_COUNTER_PATH_KILLED_BY_RUSSIAN_ROULETTE = 16u;
const DEBUG_COUNTER_NON_RESAMPLED_ENERGY_PERCENT = 17u;
const DEBUG_COUNTER_NOISE_RELATIVE_STD_DEV_PERCENT = 18u;
const DEBUG_COUNTER_NOISE_SPECULAR_PERCENT = 19u;
const DEBUG_COUNTER_NOISE_DIFFUSE_PERCENT = 20u;
const DEBUG_COUNTER_NOISE_RESAMPLED_PERCENT = 21u;
const DEBUG_COUNTER_NOISE_RESAMPLED_SPECULAR_PERCENT = 22u;
const DEBUG_COUNTER_NOISE_NON_RESAMPLED_SHARE_PERCENT = 23u;
const DEBUG_COUNTER_HISTORY_REJECTED_PIXELS = 24u;
const DEBUG_COUNTER_NOISE_HISTORY_REJECTED_PERCENT = 25u;
const DEBUG_COUNTER_NOISE_BYPASS_PERCENT = 26u;
const DEBUG_COUNTER_NOISE_OVER_100PCT_PIXELS = 27u;
const DEBUG_COUNTER_NOISE_OVER_200PCT_PIXELS = 28u;
const DEBUG_COUNTER_CONFIDENCE_WEIGHT_X10 = 29u;
const DEBUG_COUNTER_SAMPLE_AGE_FRAMES = 30u;
const DEBUG_COUNTER_SAMPLE_DUPLICATION_PERCENT = 31u;
const DEBUG_COUNTER_SAMPLE_DUPLICATION_SPECULAR_PERCENT = 32u;
const DEBUG_COUNTER_SAMPLE_DUPLICATION_OVER_25PCT_PIXELS = 33u;

/// Ceiling on a single pixel's recorded relative std dev, as a multiple of the
/// mean. Disoccluded pixels and specular fireflies sit far above 100%, so clamping
/// at 100% (as `saturate` would) erases exactly the tail that dominates what a
/// viewer notices. High enough to keep that tail, low enough that the tally cannot
/// overflow a u32 at 4K.
const NOISE_RECORD_CEILING: f32 = 4.0;

/// Perceptual roughness below which a G-buffer pixel is bucketed as specular, for
/// splitting the noise tallies. Specular pixels are the ones whose lobe rotates
/// with the camera, so their temporal history goes stale under panning.
// Explicitly typed because an abstract-float const does not resolve across a
// naga_oil module import.
const SPECULAR_ROUGHNESS_THRESHOLD: f32 = 0.3;

// Reasons temporal reuse produced nothing, stored in the low bits of `debug_flags`.
const TEMPORAL_STATUS_ACCEPTED = 0u;
const TEMPORAL_STATUS_OFFSCREEN = 1u;
const TEMPORAL_STATUS_DISSIMILAR = 2u;
const TEMPORAL_STATUS_LIGHT_DESPAWNED = 3u;
const TEMPORAL_STATUS_NO_HISTORY = 4u;

// Which estimator produced a reservoir's sample, stored in `Reservoir::flags`.
const PROVENANCE_NONE = 0u;
const PROVENANCE_NEE_DIRECT = 1u;
const PROVENANCE_EMISSIVE_DIRECT = 2u;
const PROVENANCE_RECONNECTED_NEE = 3u;
const PROVENANCE_RECONNECTED_EMISSIVE = 4u;
const PROVENANCE_WORLD_CACHE = 5u;

const PROVENANCE_MASK = 7u;

// Signals gathered deep inside the initial path trace, where threading debug
// parameters through every call would be worse than a private global. Accessed
// through functions so other modules can reach them across imports.
var<private> x2_not_reusable: bool = false;
// Negative until the first query, so an unconverged cell reporting zero samples is
// not mistaken for "nothing recorded yet".
var<private> world_cache_sample_count: f32 = -1.0;
var<private> world_cache_probe_exhausted: bool = false;

fn debug_reset_state() {
    x2_not_reusable = false;
    world_cache_sample_count = -1.0;
    world_cache_probe_exhausted = false;
}

fn debug_mark_x2_not_reusable() { x2_not_reusable = true; }
fn debug_x2_not_reusable() -> bool { return x2_not_reusable; }

/// The world cache stores its temporal sample count per cell, so a pixel that
/// queried several cells reports the least converged one.
fn debug_note_world_cache_sample_count(sample_count: f32) {
    world_cache_sample_count = select(min(world_cache_sample_count, sample_count), sample_count, world_cache_sample_count < 0.0);
}

fn debug_world_cache_sample_count() -> f32 { return max(world_cache_sample_count, 0.0); }

fn debug_mark_world_cache_probe_exhausted() { world_cache_probe_exhausted = true; }
fn debug_world_cache_probe_exhausted() -> bool { return world_cache_probe_exhausted; }

fn debug_enabled() -> bool {
    return constants.debug_view != DEBUG_VIEW_NONE || constants.debug_counters != 0u;
}

fn debug_count(slot: u32, amount: u32) {
    if constants.debug_counters == 0u { return; }
    atomicAdd(&debug_counters[slot], amount);
}

fn debug_count_if(slot: u32, condition: bool) {
    if condition {
        debug_count(slot, 1u);
    }
}

// Per-pixel debug bitfield, written by the initial/temporal pass and read by the
// shading pass that emits the debug view.
//
//   bits 0-2    temporal status
//   bit  3      the path bypassed ReSTIR because x2 was not reuse-safe
//   bit  4      temporal merge could not select the neighbour's sample
//   bit  5      temporal merge zeroed the canonical sample's MIS partner
//   bit  6      a world cache query ran out of probe steps
//   bits 7-14   world cache temporal sample count, clamped to 255
//   bits 15-22  fraction of pixel energy that bypassed ReSTIR, over 0 to 255
fn debug_pack_flags(
    temporal_status: u32,
    x2_not_reusable: bool,
    jacobian_discard_neighbor: bool,
    jacobian_inflate_canonical: bool,
    world_cache_probe_exhausted: bool,
    world_cache_sample_count: f32,
    non_resampled_share: f32,
) -> u32 {
    var flags = temporal_status & 7u;
    flags |= u32(x2_not_reusable) << 3u;
    flags |= u32(jacobian_discard_neighbor) << 4u;
    flags |= u32(jacobian_inflate_canonical) << 5u;
    flags |= u32(world_cache_probe_exhausted) << 6u;
    flags |= u32(clamp(world_cache_sample_count, 0.0, 255.0)) << 7u;
    flags |= u32(saturate(non_resampled_share) * 255.0) << 15u;
    return flags;
}

fn debug_flags_temporal_status(flags: u32) -> u32 { return flags & 7u; }
fn debug_flags_x2_not_reusable(flags: u32) -> bool { return bool((flags >> 3u) & 1u); }
fn debug_flags_jacobian_discard_neighbor(flags: u32) -> bool { return bool((flags >> 4u) & 1u); }
fn debug_flags_jacobian_inflate_canonical(flags: u32) -> bool { return bool((flags >> 5u) & 1u); }
fn debug_flags_world_cache_probe_exhausted(flags: u32) -> bool { return bool((flags >> 6u) & 1u); }
fn debug_flags_world_cache_sample_count(flags: u32) -> f32 { return f32((flags >> 7u) & 255u); }
fn debug_flags_non_resampled_share(flags: u32) -> f32 { return f32((flags >> 15u) & 255u) / 255.0; }

/// Blue -> cyan -> green -> yellow -> red over 0 to 1, for scalar quantities.
fn debug_heatmap(value: f32) -> vec3<f32> {
    let t = saturate(value) * 4.0;
    if t < 1.0 {
        return mix(vec3(0.0, 0.0, 0.35), vec3(0.0, 0.45, 1.0), t);
    } else if t < 2.0 {
        return mix(vec3(0.0, 0.45, 1.0), vec3(0.0, 0.9, 0.35), t - 1.0);
    } else if t < 3.0 {
        return mix(vec3(0.0, 0.9, 0.35), vec3(1.0, 0.9, 0.0), t - 2.0);
    } else {
        return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.0, 0.0), t - 3.0);
    }
}

/// Debug views are written straight to the view target, so radiance needs
/// exposure and a tonemap applied here to stay viewable once the example turns
/// the post-processing chain off.
fn debug_tonemap_radiance(radiance: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = max(radiance * exposure, vec3(0.0));
    return pow(exposed / (exposed + 1.0), vec3(1.0 / 2.2));
}

fn debug_temporal_status_color(status: u32) -> vec3<f32> {
    switch status {
        case TEMPORAL_STATUS_OFFSCREEN: { return vec3(0.0, 0.3, 1.0); }
        case TEMPORAL_STATUS_DISSIMILAR: { return vec3(1.0, 0.0, 0.0); }
        case TEMPORAL_STATUS_LIGHT_DESPAWNED: { return vec3(1.0, 0.0, 1.0); }
        case TEMPORAL_STATUS_NO_HISTORY: { return vec3(0.4, 0.4, 0.4); }
        default: { return vec3(0.0); }
    }
}

fn debug_provenance_color(provenance: u32) -> vec3<f32> {
    switch provenance {
        case PROVENANCE_NEE_DIRECT: { return vec3(1.0, 0.85, 0.1); }
        case PROVENANCE_EMISSIVE_DIRECT: { return vec3(1.0, 0.35, 0.7); }
        case PROVENANCE_RECONNECTED_NEE: { return vec3(0.1, 0.9, 0.3); }
        case PROVENANCE_RECONNECTED_EMISSIVE: { return vec3(1.0, 0.45, 0.0); }
        case PROVENANCE_WORLD_CACHE: { return vec3(0.1, 0.4, 1.0); }
        default: { return vec3(0.15); }
    }
}

/// Relative standard deviation from accumulated first and second moments. This is
/// frame-to-frame fluctuation, not true estimator variance: ReSTIR deliberately
/// correlates samples across frames, so this understates variance while still
/// tracking what actually looks noisy.
fn debug_relative_std_dev(mean: f32, mean_of_squares: f32) -> f32 {
    let variance = max(mean_of_squares - mean * mean, 0.0);
    return sqrt(variance) / max(mean, 0.0001);
}
