//! Builds the light grid, see `light_grid.wesl`.
//!
//! First `compute_light_bounds` (and `compute_big_light_bounds` for meshes with many triangles)
//! bounds every local light, `compact_light_grid_cells` decays every cell and lists the live ones,
//! and `build_light_grid` rebuilds each live cell's list every few frames (carrying it over in
//! between) from a bounded set of candidates: last frame's list of the cell and of a neighbor, and a
//! window of lights that walks over every light. Lights can move, appear and disappear freely.
//! `allocate_light_points` and `presample_lights` then presample points on the lights the lists use,
//! for cell candidates to pick from.

import constants::{LIGHT_GRID_POOL_CAPACITY, PER_LIGHT_SAMPLES_CAPACITY};
import package::realtime::presample_light_tiles::pack_resolved_light_sample;
import package::scene::sampling::generate_light_sample;
import bevy_render::maths::{affine3_to_square, PI};
import constants::LIGHT_GRID_SIZE;
import package::realtime::light_grid::{SPLIT_ACCUM_SCALE, LIGHT_GRID_CELL_LIFETIME, publish_cell_list, cell_entry, cell_light_id, cell_list_size, LightGridCell, find_light_grid_cell, light_grid_frame_parity, load_cell_list, store_cell_list, unpack_cell_list, LIGHT_GRID_PDF_UNITS_MAX};
import package::realtime::bindings::{constants, directional_light_count, light_grid, local_light_count, view, LightBounds, LIGHT_GRID_EMPTY_CELL, LIGHT_GRID_NO_LIST_NO_LIGHTS, LIGHT_GRID_NO_LIST_NOT_BUILT, LIGHT_GRID_NO_LIST_OUT_OF_POOL};
import package::scene::bindings::{emissive_mesh_triangle_count, geometry_ids, light_source_kind, light_sources, load_vertices, material_ids, materials, point_lights, rect_lights, transform_positions, transforms, LIGHT_SOURCE_KIND_EMISSIVE_MESH, LIGHT_SOURCE_KIND_POINT, LIGHT_SOURCE_KIND_RECT};

enable wgpu_ray_query;

/// Threads per light grid cell in `build_light_grid`.
const WORKGROUP_SIZE: u32 = 128u;
/// Threads per workgroup in `compute_light_bounds`, and threads per light.
const LIGHT_BOUNDS_WORKGROUP_SIZE: u32 = 64u;
const LIGHT_BOUNDS_THREADS_PER_LIGHT: u32 = 8u;
/// Enough bits for every possible light id (65535 lights).
const LIGHT_MASK_WORDS: u32 = 2048u;

var<workgroup> bounds_min: array<vec3<f32>, 256>;
var<workgroup> bounds_max: array<vec3<f32>, 256>;
var<workgroup> bounds_area: array<f32, 256>;

/// `LIGHT_BOUNDS_THREADS_PER_LIGHT` threads per light, several lights per workgroup. Emissive meshes
/// with more than `BIG_LIGHT_TRIANGLES` triangles are left to `compute_big_light_bounds`.
@compute @workgroup_size(64, 1, 1)
fn compute_light_bounds(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_index: u32) {
    let lane = thread_index % LIGHT_BOUNDS_THREADS_PER_LIGHT;
    let light_id = workgroup_id.x * (LIGHT_BOUNDS_WORKGROUP_SIZE / LIGHT_BOUNDS_THREADS_PER_LIGHT) + thread_index / LIGHT_BOUNDS_THREADS_PER_LIGHT;
    light_bounds(light_id, light_id < arrayLength(&light_sources), lane, LIGHT_BOUNDS_THREADS_PER_LIGHT, thread_index, true);
}

/// Emissive meshes with more triangles than this get a workgroup of their own.
const BIG_LIGHT_TRIANGLES: u32 = 1024u;
/// At most this many such meshes are bounded per frame (others keep last frame's bounds).
const MAX_BIG_LIGHTS: u32 = 1024u;
/// Keep in sync with `node.rs`.
const BIG_LIGHT_WORKGROUPS: u32 = 16u;
const BIG_LIGHT_LANES: u32 = 256u;
var<workgroup> big_light_count_shared: u32;

/// Bounds the emissive meshes `compute_light_bounds` left over, all of a workgroup's threads per mesh,
/// so that one mesh of 60k triangles doesn't take the pass's whole time on 8 threads.
@compute @workgroup_size(256, 1, 1)
fn compute_big_light_bounds(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_index: u32) {
    if thread_index == 0u {
        big_light_count_shared = min(atomicLoad(&light_grid.big_light_count), MAX_BIG_LIGHTS);
    }
    let count = workgroupUniformLoad(&big_light_count_shared);
    for (var big = workgroup_id.x; big < count; big += BIG_LIGHT_WORKGROUPS) {
        light_bounds(light_grid.big_lights[big], true, thread_index, BIG_LIGHT_LANES, thread_index, false);
        workgroupBarrier();
    }
}

/// One light's bounds, computed by `lanes` threads (`lane` is this thread's, `thread_index` its slot
/// in the workgroup arrays). With `defer_big`, emissive meshes over `BIG_LIGHT_TRIANGLES` are queued
/// for `compute_big_light_bounds` instead.
fn light_bounds(light_id: u32, valid_in: bool, lane: u32, lanes: u32, thread_index: u32, defer_big: bool) {
    var valid = valid_in;
    let light_count = arrayLength(&light_sources);
    let light_source = light_sources[min(light_id, light_count - 1u)];
    let kind = light_source_kind(light_source);

    // Emissive meshes are bounded by the box around their triangles, which needs every thread
    let is_emissive_mesh = valid && kind == LIGHT_SOURCE_KIND_EMISSIVE_MESH;
    var triangle_count = select(0u, emissive_mesh_triangle_count(light_source), is_emissive_mesh);
    if defer_big && triangle_count > BIG_LIGHT_TRIANGLES {
        if lane == 0u {
            let slot = atomicAdd(&light_grid.big_light_count, 1u);
            if slot < MAX_BIG_LIGHTS {
                light_grid.big_lights[slot] = light_id;
            }
        }
        triangle_count = 0u;
        valid = false;
    }
    let instance_id = select(0u, light_source.id, is_emissive_mesh);
    let transform = affine3_to_square(transforms[instance_id]);
    let instance_geometry_ids = geometry_ids[instance_id];

    var aabb_min = vec3(3.40282347e38);
    var aabb_max = vec3(-3.40282347e38);
    var area = 0.0;
    for (var triangle_id = lane; triangle_id < triangle_count; triangle_id += lanes) {
        let local_vertices = load_vertices(instance_geometry_ids, triangle_id);
        let vertices = transform_positions(transform, local_vertices);
        aabb_min = min(aabb_min, min(vertices[0], min(vertices[1], vertices[2])));
        aabb_max = max(aabb_max, max(vertices[0], max(vertices[1], vertices[2])));
        let triangle_area = length(cross(vertices[0] - vertices[1], vertices[0] - vertices[2])) * 0.5;
        area += triangle_area;
    }

    bounds_min[thread_index] = aabb_min;
    bounds_max[thread_index] = aabb_max;
    bounds_area[thread_index] = area;
    workgroupBarrier();
    for (var stride = lanes / 2u; stride > 0u; stride >>= 1u) {
        if lane < stride {
            bounds_min[thread_index] = min(bounds_min[thread_index], bounds_min[thread_index + stride]);
            bounds_max[thread_index] = max(bounds_max[thread_index], bounds_max[thread_index + stride]);
            bounds_area[thread_index] += bounds_area[thread_index + stride];
        }
        workgroupBarrier();
    }

    if lane != 0u || !valid { return; }

    var bounds = LightBounds(vec3(0.0), 0.0, vec3(0.0), 0.0, 0.0, 1.0, 0.0, 0.0);
    if is_emissive_mesh {
        let material = materials[material_ids[instance_id]];
        bounds.center = (bounds_min[thread_index] + bounds_max[thread_index]) * 0.5;
        bounds.radius = distance(bounds_min[thread_index], bounds_max[thread_index]) * 0.5;
        bounds.area = bounds_area[thread_index];
        // Assumes emissive textures are <= 1
        bounds.intensity = max_component(material.emissive);
    } else if kind == LIGHT_SOURCE_KIND_POINT {
        // Includes spot lights, which are bounded as if they were point lights
        let point_light = point_lights[light_source.id];
        bounds.center = point_light.position;
        bounds.radius = point_light.radius;
        bounds.intensity = max_component(point_light.intensity);
    } else if kind == LIGHT_SOURCE_KIND_RECT {
        let rect_light = rect_lights[light_source.id];
        bounds.center = rect_light.center;
        bounds.radius = length(rect_light.edge_x + rect_light.edge_y) * 0.5;
        bounds.normal = normalize(cross(rect_light.edge_y, rect_light.edge_x));
        bounds.cone_cos = 1.0;
        bounds.cone_kind = 1.0;
        bounds.area = rect_light.area;
        bounds.intensity = max_component(rect_light.radiance);
    }
    // Directional lights are never in the grid, and keep zero bounds
    light_grid.light_bounds[light_id] = bounds;
}

/// Decays every cell, frees the ones that have gone unqueried for their whole lifetime, and lists
/// the live ones for `build_light_grid`, so that it runs one workgroup per live cell rather than
/// per hash table slot. One thread per slot.
@compute @workgroup_size(128, 1, 1)
fn compact_light_grid_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    var life = atomicLoad(&light_grid.life[cell_index]);
    if life == 0u { return; }
    // Lighting found the cell through the plain copy last frame
    if light_grid.touched[cell_index] != 0u {
        light_grid.touched[cell_index] = 0u;
        life = LIGHT_GRID_CELL_LIFETIME;
    }
    life -= 1u;
    atomicStore(&light_grid.life[cell_index], life);
    if constants.light_grid_sun_split != 0u {
        update_cell_split(cell_index, life == 0u);
    }
    if life == 0u {
        light_grid.lookup[cell_index] = vec4(0u);
        atomicStore(&light_grid.checksums[cell_index], LIGHT_GRID_EMPTY_CELL);
        store_cell_list(cell_index, 0u, vec2(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u));
        store_cell_list(cell_index, 1u, vec2(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u));
        return;
    }
    light_grid.live_cell_list[atomicAdd(&light_grid.live_cells, 1u)] = cell_index;
}

/// Updates of a cell's learned split before it's used, see `update_cell_split`.
const SPLIT_WARMUP_UPDATES: f32 = 4.0;
/// Weight of the newest frame in the split's running averages.
const SUN_SPLIT_ALPHA: f32 = 0.25;
/// The learned probability that a candidate is the sun is clamped to this range, so both keep
/// candidates (and every strategy keeps a nonzero pdf).
const SUN_SPLIT_MIN: f32 = 0.02;
const SUN_SPLIT_MAX: f32 = 0.75;

/// Folds last frame's NEE records into a cell's learned sun/local split: running averages of the
/// sun's and local lights' shadowed contributions, and from them the probability that a candidate
/// is the sun, clamped so that both keep candidates. A dead cell's split is cleared.
fn update_cell_split(cell_index: u32, dead: bool) {
    let sun = f32(atomicExchange(&light_grid.split_accum[cell_index * 2u], 0u)) / SPLIT_ACCUM_SCALE;
    let local = f32(atomicExchange(&light_grid.split_accum[cell_index * 2u + 1u], 0u)) / SPLIT_ACCUM_SCALE;
    if dead {
        light_grid.split_state[cell_index] = vec4(0.0, 0.0, -1.0, 0.0);
        return;
    }
    var state = light_grid.split_state[cell_index];
    let alpha = SUN_SPLIT_ALPHA;
    state.x = mix(state.x, sun, alpha);
    state.y = mix(state.y, local, alpha);
    state.w += 1.0;
    state.z = -1.0;
    if state.w >= SPLIT_WARMUP_UPDATES && state.x + state.y > 0.0 {
        state.z = clamp(state.x / (state.x + state.y), SUN_SPLIT_MIN, SUN_SPLIT_MAX);
    }
    light_grid.split_state[cell_index] = state;
}

/// The indirect dispatch of `build_light_grid`. Its own buffer, since a buffer can't be both an
/// indirect argument and bound read-write in the same dispatch.
@group(2) @binding(0) var<storage, read_write> light_grid_build_dispatch: array<u32, 3>;

/// Writes the indirect dispatch of `build_light_grid`: one workgroup per live cell.
@compute @workgroup_size(1, 1, 1)
fn write_light_grid_build_dispatch() {
    let count = atomicLoad(&light_grid.live_cells);
    light_grid_build_dispatch[0] = min(count, 256u);
    light_grid_build_dispatch[1] = (count + 255u) / 256u;
    light_grid_build_dispatch[2] = 1u;
}

/// An upper bound on the irradiance a light can contribute to any point in a cell.
fn light_contribution_bound(bounds: LightBounds, cell_min: vec3<f32>, cell_size: f32) -> f32 {
    let half_size = vec3(cell_size * 0.5);

    // One-sided lights contribute nothing to cells fully behind them
    if bounds.cone_kind == 1.0 && bounds.cone_cos >= 1.0 {
        let cell_center = cell_min + half_size;
        if dot(bounds.normal, cell_center - bounds.center) + dot(abs(bounds.normal), half_size) <= 0.0 {
            return 0.0;
        }
    }

    return light_distance_bound(bounds, cell_min, cell_size);
}

/// `light_contribution_bound` without the emitter's orientation.
fn light_distance_bound(bounds: LightBounds, cell_min: vec3<f32>, cell_size: f32) -> f32 {
    let closest_point = clamp(bounds.center, cell_min, cell_min + cell_size);
    let distance_to_light = max(distance(bounds.center, closest_point) - bounds.radius, 0.0);
    let distance_squared = distance_to_light * distance_to_light;

    // Area lights subtend at most area / d^2 steradians, and contribute at most PI * radiance
    if bounds.area > 0.0 {
        return bounds.intensity * min(PI, bounds.area / distance_squared);
    }
    return bounds.intensity / distance_squared;
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

/// Which lights are candidates for this workgroup's cell, indexed by light id.
var<workgroup> light_mask: array<atomic<u32>, LIGHT_MASK_WORDS>;
/// Candidates per half octave of weight, see `weight_bin`.
var<workgroup> weight_histogram: array<atomic<u32>, WEIGHT_BINS>;
/// Per-thread light counts, prefix summed.
var<workgroup> thread_light_counts: array<u32, WORKGROUP_SIZE>;
/// Per-thread units of probability, prefix summed.
var<workgroup> thread_units: array<u32, WORKGROUP_SIZE>;
/// The heaviest candidate's weight, as bits (positive floats sort as u32s).
var<workgroup> max_weight_bits: atomic<u32>;
var<workgroup> candidate_counter: atomic<u32>;
var<workgroup> candidate_count: u32;
var<workgroup> cell_alive: u32;
/// The cell's list offset, and count and units.
var<workgroup> cell_list: vec2<u32>;
/// Last frame's lists of the neighbor cells candidates are taken from.
var<workgroup> neighbor_lists: array<vec2<u32>, MAX_NEIGHBORS>;
/// 1 if the cell carries last frame's list over from `carry_source`, 2 if it carries over having
/// no lights, 0 if it is rebuilt. See `light_grid_rebuild_interval`.
var<workgroup> carry_list: u32;
var<workgroup> carry_source: u32;
var<workgroup> carry_size: u32;

/// Half octaves of weight, see `weight_bin`.
const WEIGHT_BINS: u32 = 512u;
const MAX_NEIGHBORS: u32 = 8u;
/// Neighbors whose last lists each rebuild takes candidates from.
const LIGHT_GRID_NEIGHBORS: u32 = 1u;
/// Lights each rebuild tests besides last frame's lists, a window that walks over every light, so
/// that every light is found within `local light count / LIGHT_GRID_EXPLORATION_COUNT` rebuilds.
const LIGHT_GRID_EXPLORATION_COUNT: u32 = 1024u;
const MAX_LIGHTS_PER_CELL: u32 = 4096u;

/// One workgroup per live cell (an indirect dispatch, see `compact_light_grid_cells`).
///
/// A cell's list is rebuilt every `light_grid_rebuild_interval` frames (staggered between cells,
/// carried over in between) from a bounded set of candidate lights, so the cost per cell doesn't grow
/// with the number of lights in the scene:
/// - the lights in the cell's own list from last frame,
/// - the lights in last frame's lists of `LIGHT_GRID_NEIGHBORS` random neighbor cells, which
///   spreads lights found by one cell to the cells around it,
/// - a window of `LIGHT_GRID_EXPLORATION_COUNT` lights that walks over every local light, a
///   different stretch for every cell, so that every light is eventually tested against every
///   cell. With at least as many as there are local lights, every light is tested every frame.
///
/// Candidates whose contribution bound reaches the threshold are weighted by an estimate of their
/// contribution, and the `light_grid_max_lights_per_cell` heaviest are kept. Each kept light gets
/// units of probability in proportion to its weight, after the ratio cap, see `cell_entry`.
///
/// How lights end up in a list only affects variance: NEE weights the cell's candidates against the
/// global ones with the list's actual probabilities, and the global ones cover every light.
@compute @workgroup_size(128, 1, 1)
fn build_light_grid(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) thread_index: u32) {
    // One workgroup per live cell, see `compact_light_grid_cells`
    let live_index = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    let cell_index = light_grid.live_cell_list[min(live_index, LIGHT_GRID_SIZE - 1u)];
    let parity = light_grid_frame_parity();
    let previous_parity = parity ^ 1u;
    let neighbor_count = LIGHT_GRID_NEIGHBORS;

    let cell_bounds = light_grid.bounds[cell_index];
    let cell_min = cell_bounds.xyz;
    let cell_size = cell_bounds.w;

    if thread_index == 0u {
        // The dispatch rounds the live cell count up
        let alive = select(0u, 1u, live_index < atomicLoad(&light_grid.live_cells));
        cell_alive = alive;

        // Pick random neighbors, any of the 26 around the cell. The neighbor point is looked up at
        // its own LOD, so neighbors across a LOD boundary are found too.
        if alive != 0u {
            var rng = (cell_index * 0x9E3779B9u) ^ (constants.light_grid_frame * 0x85EBCA6Bu);
            for (var n = 0u; n < neighbor_count; n++) {
                rng = rng * 747796405u + 2891336453u;
                var direction = rng % 26u;
                direction += select(0u, 1u, direction >= 13u); // Skip the cell itself
                let offset = vec3(f32(direction % 3u), f32((direction / 3u) % 3u), f32(direction / 9u)) - 1.0;
                let neighbor_point = cell_min + (offset + 0.5) * cell_size;
                let neighbor_key = find_light_grid_cell(neighbor_point);
                var list = vec2(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u);
                if neighbor_key < LIGHT_GRID_SIZE && neighbor_key != cell_index {
                    list = load_cell_list(neighbor_key, previous_parity);
                }
                neighbor_lists[n] = list;
            }
        }

        atomicStore(&max_weight_bits, 0u);
        atomicStore(&candidate_counter, 0u);
    }
    if workgroupUniformLoad(&cell_alive) == 0u { return; }

    // Rebuild each cell every `light_grid_rebuild_interval` frames, staggered between cells, and
    // carry last frame's list over otherwise. Cells without a list from last frame are rebuilt.
    let rebuild_interval = max(constants.light_grid_rebuild_interval, 1u);
    if rebuild_interval > 1u {
        if thread_index == 0u {
            var carry = 0u;
            let previous = load_cell_list(cell_index, previous_parity);
            let previous_count = previous.y & 0xFFFFu;
            let rebuild = (cell_index + constants.light_grid_frame) % rebuild_interval == 0u;
            if !rebuild && previous_count == 0u && previous.x == LIGHT_GRID_NO_LIST_NO_LIGHTS {
                publish_cell_list(cell_index, parity, previous);
                carry = 2u;
            } else if !rebuild && previous_count != 0u {
                let size = cell_list_size(previous_count);
                carry_size = size;
                let list = allocate_light_list(previous_count, size, parity);
                if list.y != 0u {
                    cell_list = vec2(list.x, previous.y);
                    publish_cell_list(cell_index, parity, cell_list);
                    carry_source = previous.x;
                    carry = 1u;
                }
            }
            carry_list = carry;
        }
        let carry = workgroupUniformLoad(&carry_list);
        if carry == 2u { return; }
        if carry == 1u {
            let size = workgroupUniformLoad(&carry_size);
            for (var i = thread_index; i < size; i += WORKGROUP_SIZE) {
                light_grid.pool[cell_list.x + i] = light_grid.pool[carry_source + i];
            }
            // Carried lists' lights need presampled points too
            let count = cell_list.y & 0xFFFFu;
            let total = cell_list.y >> 16u;
            for (var i = thread_index; i < count; i += WORKGROUP_SIZE) {
                let entry = light_grid.pool[carry_source + i];
                let start = select(0u, light_grid.pool[carry_source + max(i, 1u) - 1u] >> 16u, i != 0u);
                mark_light_used(entry & 0xFFFFu, (entry >> 16u) - start, total);
            }
            return;
        }
    }

    let light_count = arrayLength(&light_sources);
    let threshold = constants.light_grid_contribution_threshold / view.exposure;

    let mask_word_count = (light_count + 31u) / 32u;
    for (var word = thread_index; word < mask_word_count; word += WORKGROUP_SIZE) {
        atomicStore(&light_mask[word], 0u);
    }
    for (var bin = thread_index; bin < WEIGHT_BINS; bin += WORKGROUP_SIZE) {
        atomicStore(&weight_histogram[bin], 0u);
    }
    workgroupBarrier();

    // Gather candidates, testing each against a threshold on its contribution after exposure

    // When the window covers every local light, last frame's lists add nothing
    let local_count = local_light_count();
    let window_size = min(LIGHT_GRID_EXPLORATION_COUNT, local_count);
    if window_size < local_count {
        let own_list = unpack_cell_list(load_cell_list(cell_index, previous_parity));
        for (var i = thread_index; i < own_list.count; i += WORKGROUP_SIZE) {
            consider_light(cell_light_id(own_list, i), cell_min, cell_size, threshold, light_count);
        }
        for (var n = 0u; n < neighbor_count; n++) {
            let neighbor_list = unpack_cell_list(neighbor_lists[n]);
            for (var i = thread_index; i < neighbor_list.count; i += WORKGROUP_SIZE) {
                consider_light(cell_light_id(neighbor_list, i), cell_min, cell_size, threshold, light_count);
            }
        }
    }
    if window_size != 0u {
        let window_start = (atomicLoad(&light_grid.checksums[cell_index]) + constants.light_grid_frame * window_size) % local_count;
        for (var i = thread_index; i < window_size; i += WORKGROUP_SIZE) {
            consider_light(directional_light_count() + (window_start + i) % local_count, cell_min, cell_size, threshold, light_count);
        }
    }
    workgroupBarrier();
    if thread_index == 0u {
        candidate_count = atomicLoad(&candidate_counter);
    }
    let found_count = workgroupUniformLoad(&candidate_count);

    // Each thread owns a contiguous range of mask words, so that writing them out in thread order
    // keeps the list sorted by light id
    let words_per_thread = (mask_word_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let word_start = min(thread_index * words_per_thread, mask_word_count);
    let word_end = min(word_start + words_per_thread, mask_word_count);
    let max_weight = bitcast<f32>(atomicLoad(&max_weight_bits));

    // Keep only the heaviest candidates if there are too many: every candidate in the bins at or
    // above the cutoff, the lowest bin that keeps the count within the limit. If the heaviest bin
    // alone is over the limit, all of it is kept.
    let max_lights = clamp(constants.light_grid_max_lights_per_cell, 1u, MAX_LIGHTS_PER_CELL);
    var cutoff = vec2(0u, found_count);
    if found_count > max_lights {
        // The bin where the count goes over always exists here. Keep the bins above it, or that
        // bin alone if it is the heaviest.
        let crossing = find_histogram_crossing(thread_index, max_lights);
        if crossing.z == 0u {
            cutoff = vec2(crossing.y, atomicLoad(&weight_histogram[crossing.y]));
        } else {
            cutoff = vec2(crossing.y + 1u, crossing.z);
        }
    }

    // Lights in a list get at least the heaviest light's weight divided by the ratio cap, and units
    // of probability relative to the heaviest's, so that the list's units fit in 16 bits
    let min_weight = max_weight / max(constants.light_grid_weight_ratio_cap, 1.0);
    let units_per_max_weight = LIGHT_GRID_PDF_UNITS_MAX / max(cutoff.y, 1u);

    var thread_light_count = 0u;
    var thread_unit_sum = 0u;
    for (var word = word_start; word < word_end; word++) {
        var bits = atomicLoad(&light_mask[word]);
        while bits != 0u {
            let light_id = (word << 5u) | firstTrailingBit(bits);
            let units = light_units(light_id, cell_min, cell_size, max_weight, min_weight, units_per_max_weight, cutoff.x);
            thread_light_count += select(0u, 1u, units != 0u);
            thread_unit_sum += units;
            bits &= bits - 1u;
        }
    }

    // Inclusive prefix sums of the per-thread counts and units
    thread_light_counts[thread_index] = thread_light_count;
    thread_units[thread_index] = thread_unit_sum;
    workgroupBarrier();
    for (var stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
        var count = thread_light_counts[thread_index];
        var units = thread_units[thread_index];
        if thread_index >= stride {
            count += thread_light_counts[thread_index - stride];
            units += thread_units[thread_index - stride];
        }
        workgroupBarrier();
        thread_light_counts[thread_index] = count;
        thread_units[thread_index] = units;
        workgroupBarrier();
    }

    if thread_index == 0u {
        let list_count = thread_light_counts[WORKGROUP_SIZE - 1u];
        let total_units = thread_units[WORKGROUP_SIZE - 1u];
        var list = vec2(LIGHT_GRID_NO_LIST_NO_LIGHTS, 0u);
        if list_count != 0u && total_units != 0u {
            list = allocate_light_list(list_count, cell_list_size(list_count), parity);
            if list.y != 0u {
                list.y |= total_units << 16u;
            }
        }
        publish_cell_list(cell_index, parity, list);
        cell_list = list;
    }
    let list = workgroupUniformLoad(&cell_list);
    if (list.y & 0xFFFFu) == 0u { return; }

    // Write the entries, each with the cumulative units up to and including its light
    var index = list.x + thread_light_counts[thread_index] - thread_light_count;
    var cumulative_units = thread_units[thread_index] - thread_unit_sum;
    for (var word = word_start; word < word_end; word++) {
        var bits = atomicLoad(&light_mask[word]);
        while bits != 0u {
            let light_id = (word << 5u) | firstTrailingBit(bits);
            bits &= bits - 1u;
            let units = light_units(light_id, cell_min, cell_size, max_weight, min_weight, units_per_max_weight, cutoff.x);
            if units == 0u { continue; }
            cumulative_units += units;
            light_grid.pool[index] = light_id | (cumulative_units << 16u);
            mark_light_used(light_id, units, list.y >> 16u);
            index += 1u;
        }
    }

    // Build the guide table from the entries just written
    storageBarrier();
    let cell = LightGridCell(list.x, list.y & 0xFFFFu, list.y >> 16u, LIGHT_GRID_SIZE, -1.0);
    for (var pair = thread_index; pair < (cell.count + 1u) / 2u; pair += WORKGROUP_SIZE) {
        var packed = 0u;
        for (var k = 0u; k < 2u; k++) {
            let j = min(pair * 2u + k, cell.count - 1u);
            let target_units = j * cell.total / cell.count;
            // The first entry whose cumulative units exceed the target
            var low = 0u;
            var high = cell.count - 1u;
            while low < high {
                let middle = (low + high) >> 1u;
                if (cell_entry(cell, middle) >> 16u) <= target_units {
                    low = middle + 1u;
                } else {
                    high = middle;
                }
            }
            packed |= low << (k * 16u);
        }
        light_grid.pool[cell.offset + cell.count + pair] = packed;
    }
}

/// Each list's usage, see `LightGrid::light_usage`.
const USAGE_PER_LIST: u32 = 4096u;
/// The fewest points a used light gets.
const MIN_POINTS_PER_USED_LIGHT: u32 = 4u;
/// The most points per used light on average.
const AVERAGE_POINTS_PER_USED_LIGHT: u32 = 64u;
const ALLOCATE_THREADS: u32 = 256u;
var<workgroup> allocate_usage: array<f32, ALLOCATE_THREADS>;
var<workgroup> allocate_points: array<u32, ALLOCATE_THREADS>;

/// Splits `per_light_samples` between the used lights (those in some cell's list), each getting
/// `MIN_POINTS_PER_USED_LIGHT` plus a share of the rest in proportion to its usage, and writes where
/// each one's points start. One workgroup; each thread takes a contiguous range of slots.
@compute @workgroup_size(256, 1, 1)
fn allocate_light_points(@builtin(local_invocation_index) thread_index: u32) {
    let used = min(atomicLoad(&light_grid.used_light_count), 65536u);
    let per_thread = (used + ALLOCATE_THREADS - 1u) / ALLOCATE_THREADS;
    let start = min(thread_index * per_thread, used);
    let end = min(start + per_thread, used);

    var usage = 0.0;
    for (var slot = start; slot < end; slot++) {
        usage += f32(atomicLoad(&light_grid.light_usage[light_grid.used_lights[slot]]));
    }
    allocate_usage[thread_index] = usage;
    workgroupBarrier();
    for (var stride = ALLOCATE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if thread_index < stride {
            allocate_usage[thread_index] += allocate_usage[thread_index + stride];
        }
        workgroupBarrier();
    }
    let total_usage = max(allocate_usage[0], 1.0);
    // At most `AVERAGE_POINTS_PER_USED_LIGHT` per used light in all, so scenes with few lights keep
    // their points few enough to stay in cache
    let budget = min(PER_LIGHT_SAMPLES_CAPACITY, AVERAGE_POINTS_PER_USED_LIGHT * max(used, 1u));
    let spare = f32(budget - min(MIN_POINTS_PER_USED_LIGHT * used, budget));

    var thread_points = 0u;
    for (var slot = start; slot < end; slot++) {
        thread_points += allocated_points(slot, spare, total_usage);
    }
    allocate_points[thread_index] = thread_points;
    workgroupBarrier();
    for (var stride = 1u; stride < ALLOCATE_THREADS; stride <<= 1u) {
        var points = allocate_points[thread_index];
        if thread_index >= stride {
            points += allocate_points[thread_index - stride];
        }
        workgroupBarrier();
        allocate_points[thread_index] = points;
        workgroupBarrier();
    }

    var offset = allocate_points[thread_index] - thread_points;
    for (var slot = start; slot < end; slot++) {
        light_grid.point_offsets[slot] = offset;
        offset += allocated_points(slot, spare, total_usage);
    }
    if thread_index == ALLOCATE_THREADS - 1u {
        light_grid.point_offsets[used] = allocate_points[ALLOCATE_THREADS - 1u];
    }
}

fn allocated_points(slot: u32, spare: f32, total_usage: f32) -> u32 {
    let usage = f32(atomicLoad(&light_grid.light_usage[light_grid.used_lights[slot]]));
    return MIN_POINTS_PER_USED_LIGHT + u32(floor(spare * usage / total_usage));
}


const BINS_PER_THREAD: u32 = WEIGHT_BINS / WORKGROUP_SIZE;
var<workgroup> histogram_crossing: vec3<u32>;

/// Walking `weight_histogram` from the heaviest bin down, the first bin where the count goes over
/// `limit`: `(1, bin, count in the bins above it)`, or `(0, 0, 0)` if the count never does. Each
/// thread sums `BINS_PER_THREAD` bins, and the threads prefix sum them.
fn find_histogram_crossing(thread_index: u32, limit: u32) -> vec3<u32> {
    let top = WEIGHT_BINS - thread_index * BINS_PER_THREAD;
    var sum = 0u;
    for (var k = 0u; k < BINS_PER_THREAD; k++) {
        sum += atomicLoad(&weight_histogram[top - 1u - k]);
    }
    if thread_index == 0u {
        histogram_crossing = vec3(0u);
    }
    thread_units[thread_index] = sum;
    workgroupBarrier();
    for (var stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
        var value = thread_units[thread_index];
        if thread_index >= stride {
            value += thread_units[thread_index - stride];
        }
        workgroupBarrier();
        thread_units[thread_index] = value;
        workgroupBarrier();
    }
    let above = thread_units[thread_index] - sum;
    if above <= limit && above + sum > limit {
        var kept = above;
        for (var k = 0u; k < BINS_PER_THREAD; k++) {
            let bin = top - 1u - k;
            let in_bin = atomicLoad(&weight_histogram[bin]);
            if kept + in_bin > limit {
                histogram_crossing = vec3(1u, bin, kept);
                break;
            }
            kept += in_bin;
        }
    }
    return workgroupUniformLoad(&histogram_crossing);
}

/// Adds a light to the cell's candidates if its contribution bound reaches the threshold.
fn consider_light(light_id: u32, cell_min: vec3<f32>, cell_size: f32, threshold: f32, light_count: u32) {
    // Lists from last frame can hold light ids that are gone or now directional
    if light_id >= light_count || light_id < directional_light_count() { return; }

    let bounds = light_grid.light_bounds[light_id];
    if light_contribution_bound(bounds, cell_min, cell_size) < threshold { return; }

    let bit = 1u << (light_id & 31u);
    let old_bits = atomicOr(&light_mask[light_id >> 5u], bit);
    if (old_bits & bit) == 0u {
        let weight = light_importance_weight(bounds, cell_min, cell_size);
        atomicAdd(&candidate_counter, 1u);
        atomicMax(&max_weight_bits, bitcast<u32>(weight));
        atomicAdd(&weight_histogram[weight_bin(weight)], 1u);
    }
}

/// Which half octave a weight is in, over the whole range of positive floats.
fn weight_bin(weight: f32) -> u32 {
    return u32(clamp(floor(log2(max(weight, 1e-38)) * 2.0) + f32(WEIGHT_BINS / 2u), 0.0, f32(WEIGHT_BINS - 1u)));
}

/// A light's units of probability in a cell's list, at least 1, or 0 if its weight is below the
/// cutoff bin.
fn light_units(light_id: u32, cell_min: vec3<f32>, cell_size: f32, max_weight: f32, min_weight: f32, units_per_max_weight: u32, cutoff_bin: u32) -> u32 {
    let weight = light_importance_weight(light_grid.light_bounds[light_id], cell_min, cell_size);
    if weight_bin(weight) < cutoff_bin { return 0u; }
    return clamp(u32(round(max(weight, min_weight) / max_weight * f32(units_per_max_weight))), 1u, units_per_max_weight);
}

/// Allocates a list of `count` lights, `size` u32s, in this frame's half of the pool. Returns the list, or why there isn't one.
fn allocate_light_list(count: u32, size: u32, parity: u32) -> vec2<u32> {
    let half_capacity = LIGHT_GRID_POOL_CAPACITY / 2u;
    let offset = atomicAdd(&light_grid.pool_used, size);
    if offset + size <= half_capacity {
        return vec2(parity * half_capacity + offset, count);
    }
    atomicAdd(&light_grid.overflowed_cells, 1u);
    return vec2(LIGHT_GRID_NO_LIST_OUT_OF_POOL, 0u);
}

/// How much a light should be favored in a cell. Like `light_contribution_bound`, but measured from
/// the cell's center, with the distance clamped to half a cell, so lights inside or touching the
/// cell don't get unbounded weights.
fn light_importance_weight(bounds: LightBounds, cell_min: vec3<f32>, cell_size: f32) -> f32 {
    let cell_center = cell_min + vec3(cell_size * 0.5);
    let distance_to_light = max(distance(bounds.center, cell_center) - bounds.radius, cell_size * 0.5);
    let distance_squared = distance_to_light * distance_to_light;
    if bounds.area > 0.0 {
        return bounds.intensity * min(PI, bounds.area / distance_squared);
    }
    return bounds.intensity / distance_squared;
}


/// Marks a light as in some cell's list this frame, giving it a slot in `per_light_samples`, and adds
/// its probability in the list to its usage, see `allocate_light_points`.
fn mark_light_used(light_id: u32, units: u32, total: u32) {
    atomicAdd(&light_grid.light_usage[light_id], units * USAGE_PER_LIST / max(total, 1u));
    let bit = 1u << (light_id & 31u);
    if (atomicLoad(&light_grid.light_used_bits[light_id >> 5u]) & bit) != 0u { return; }
    if (atomicOr(&light_grid.light_used_bits[light_id >> 5u], bit) & bit) != 0u { return; }
    let slot = atomicAdd(&light_grid.used_light_count, 1u);
    light_grid.light_slots[light_id] = slot;
    light_grid.used_lights[slot] = light_id;
}

/// Presamples points on the lights in some cell's list, as many as `allocate_light_points` gave
/// each, see `PerLightSamples`. Dispatched with one thread per slot of `per_light_samples`.
///
/// Points on a light are shared by every pixel that picks it this frame, so few points would make
/// everything the light illuminates share the same error (coherent, cell-sized flicker under a
/// denoiser). Splitting a fixed budget between the lights the lists actually use, in proportion to
/// how much they're used, gives the lights that matter many points however many lights the scene has.
@compute @workgroup_size(64, 1, 1)
fn presample_lights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // The slot whose points include this one
    let used = min(atomicLoad(&light_grid.used_light_count), 65536u);
    if used == 0u || global_id.x >= light_grid.point_offsets[used] { return; }
    var low = 0u;
    var high = used - 1u;
    while low < high {
        let middle = (low + high + 1u) >> 1u;
        if light_grid.point_offsets[middle] <= global_id.x {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    let light_id = light_grid.used_lights[low];
    if light_id >= arrayLength(&light_sources) { return; }

    var rng = (global_id.x * 0x9E3779B9u) + constants.frame_rng;
    let sample = generate_light_sample(light_id, &rng);
    light_grid.per_light_samples.light_samples[global_id.x] = sample.light_sample;
    light_grid.per_light_samples.resolved_light_samples[global_id.x] = pack_resolved_light_sample(sample.resolved_light_sample);
}
