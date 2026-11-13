//! Builds the light grid, see `light_grid.wesl`.
//!
//! First `compute_light_bounds` bounds every local light, then `build_light_grid` decays every cell,
//! and rebuilds each live cell's list from scratch. Lists are rebuilt every frame, so lights can
//! move, appear and disappear freely. `presample_lights` then presamples points on every light for
//! cell candidates to use.

import constants::LIGHT_GRID_POOL_CAPACITY;
import package::realtime::presample_light_tiles::pack_resolved_light_sample;
import package::scene::sampling::generate_light_sample;
import bevy_render::maths::{affine3_to_square, PI};
import package::realtime::bindings::{constants, directional_light_count, light_bounds, light_grid, light_grid_pool, light_samples_per_light, local_light_count, per_light_samples, view, LightBounds, LIGHT_GRID_EMPTY_CELL, LIGHT_GRID_NO_LIST_NO_LIGHTS, LIGHT_GRID_NO_LIST_NOT_BUILT, LIGHT_GRID_NO_LIST_OUT_OF_POOL};
import package::scene::bindings::{emissive_mesh_triangle_count, geometry_ids, light_source_kind, light_sources, load_vertices, material_ids, materials, point_lights, rect_lights, transform_positions, transforms, LIGHT_SOURCE_KIND_EMISSIVE_MESH, LIGHT_SOURCE_KIND_POINT, LIGHT_SOURCE_KIND_RECT};

enable wgpu_ray_query;

const WORKGROUP_SIZE: u32 = 64u;
/// Enough bits for every possible light id (65535 lights).
const LIGHT_MASK_WORDS: u32 = 2048u;

var<workgroup> bounds_min: array<vec3<f32>, WORKGROUP_SIZE>;
var<workgroup> bounds_max: array<vec3<f32>, WORKGROUP_SIZE>;
var<workgroup> bounds_area: array<f32, WORKGROUP_SIZE>;

/// One workgroup per light.
@compute @workgroup_size(64, 1, 1)
fn compute_light_bounds(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_index: u32) {
    let light_id = workgroup_id.x;
    let light_source = light_sources[light_id];
    let kind = light_source_kind(light_source);

    // Emissive meshes are bounded by the box around their triangles, which needs every thread
    let is_emissive_mesh = kind == LIGHT_SOURCE_KIND_EMISSIVE_MESH;
    let triangle_count = select(0u, emissive_mesh_triangle_count(light_source), is_emissive_mesh);
    let instance_id = select(0u, light_source.id, is_emissive_mesh);
    let transform = affine3_to_square(transforms[instance_id]);
    let instance_geometry_ids = geometry_ids[instance_id];

    var aabb_min = vec3(3.40282347e38);
    var aabb_max = vec3(-3.40282347e38);
    var area = 0.0;
    for (var triangle_id = thread_index; triangle_id < triangle_count; triangle_id += WORKGROUP_SIZE) {
        let vertices = transform_positions(transform, load_vertices(instance_geometry_ids, triangle_id));
        aabb_min = min(aabb_min, min(vertices[0], min(vertices[1], vertices[2])));
        aabb_max = max(aabb_max, max(vertices[0], max(vertices[1], vertices[2])));
        area += length(cross(vertices[0] - vertices[1], vertices[0] - vertices[2])) * 0.5;
    }

    bounds_min[thread_index] = aabb_min;
    bounds_max[thread_index] = aabb_max;
    bounds_area[thread_index] = area;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if thread_index < stride {
            bounds_min[thread_index] = min(bounds_min[thread_index], bounds_min[thread_index + stride]);
            bounds_max[thread_index] = max(bounds_max[thread_index], bounds_max[thread_index + stride]);
            bounds_area[thread_index] += bounds_area[thread_index + stride];
        }
        workgroupBarrier();
    }

    if thread_index != 0u { return; }

    var bounds = LightBounds(vec3(0.0), 0.0, vec3(0.0), 0.0, 0.0, 0.0, 0.0, 0.0);
    if is_emissive_mesh {
        let material = materials[material_ids[instance_id]];
        bounds.center = (bounds_min[0] + bounds_max[0]) * 0.5;
        bounds.radius = distance(bounds_min[0], bounds_max[0]) * 0.5;
        bounds.area = bounds_area[0];
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
        bounds.area = rect_light.area;
        bounds.intensity = max_component(rect_light.radiance);
    }
    // Directional lights are never in the grid, and keep zero bounds
    light_bounds[light_id] = bounds;
}

/// An upper bound on the irradiance a light can contribute to any point in a cell.
fn light_contribution_bound(bounds: LightBounds, cell_min: vec3<f32>, cell_size: f32) -> f32 {
    let half_size = vec3(cell_size * 0.5);

    // One-sided lights contribute nothing to cells fully behind them
    if any(bounds.normal != vec3(0.0)) {
        let cell_center = cell_min + half_size;
        if dot(bounds.normal, cell_center - bounds.center) + dot(abs(bounds.normal), half_size) <= 0.0 {
            return 0.0;
        }
    }

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

/// Which lights are in this workgroup's cell, indexed by light id.
var<workgroup> light_mask: array<atomic<u32>, LIGHT_MASK_WORDS>;
/// Per-thread light counts, prefix summed.
var<workgroup> thread_light_counts: array<u32, WORKGROUP_SIZE>;
/// Per-thread weight sums, prefix summed.
var<workgroup> thread_weight_sums: array<f32, WORKGROUP_SIZE>;
/// The heaviest light's weight, as bits (positive floats sort as u32s).
var<workgroup> max_weight_bits: atomic<u32>;
var<workgroup> cell_alive: u32;
/// The cell's list offset and slot count.
var<workgroup> cell_list: vec2<u32>;

const MAX_WEIGHTED_SLOTS: u32 = 512u;

/// One workgroup per cell, dispatched as `(256, LIGHT_GRID_SIZE / 256)`.
///
/// Fills the cell's list with `light_grid_weighted_slots` slots by systematic resampling over the
/// weights of the lights that pass the threshold, in light id order. Every light then holds the
/// floor or ceiling of its share of the slots, and the slots come out sorted by light id. Each slot
/// packs the light id with its number of slots, so a slot pick reads its pdf directly, and other
/// strategies find a light's pdf by binary search, see `weighted_cell_pdf`.
@compute @workgroup_size(64, 1, 1)
fn build_light_grid(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_index: u32) {
    let cell_index = workgroup_id.x + workgroup_id.y * 256u;

    // Decay the cell, and free it once it has gone unqueried for its whole lifetime
    if thread_index == 0u {
        var alive = 0u;
        var life = atomicLoad(&light_grid.life[cell_index]);
        if life > 0u {
            life -= 1u;
            atomicStore(&light_grid.life[cell_index], life);
            if life == 0u {
                atomicStore(&light_grid.checksums[cell_index], LIGHT_GRID_EMPTY_CELL);
                light_grid.lists[cell_index] = vec2(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u);
            } else {
                alive = 1u;
            }
        }
        cell_alive = alive;
        atomicStore(&max_weight_bits, 0u);
    }
    if workgroupUniformLoad(&cell_alive) == 0u { return; }

    let cell_bounds = light_grid.bounds[cell_index];
    let cell_min = cell_bounds.xyz;
    let cell_size = cell_bounds.w;

    let mask_word_count = (arrayLength(&light_sources) + 31u) / 32u;
    for (var word = thread_index; word < mask_word_count; word += WORKGROUP_SIZE) {
        atomicStore(&light_mask[word], 0u);
    }
    workgroupBarrier();

    // Test each local light once, against a threshold on its contribution after exposure
    let threshold = constants.light_grid_contribution_threshold / view.exposure;
    let directional_count = directional_light_count();
    let local_count = local_light_count();
    for (var i = thread_index; i < local_count; i += WORKGROUP_SIZE) {
        let light_id = directional_count + i;
        let bounds = light_bounds[light_id];
        if light_contribution_bound(bounds, cell_min, cell_size) >= threshold {
            atomicOr(&light_mask[light_id >> 5u], 1u << (light_id & 31u));
            atomicMax(&max_weight_bits, bitcast<u32>(light_importance_weight(bounds, cell_min, cell_size)));
        }
    }
    workgroupBarrier();

    // Each thread owns a contiguous range of mask words, so that writing them out in thread order
    // keeps the list sorted by light id
    let words_per_thread = (mask_word_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let word_start = min(thread_index * words_per_thread, mask_word_count);
    let word_end = min(word_start + words_per_thread, mask_word_count);

    // Lights in a list get at least the heaviest light's weight divided by the ratio cap
    let max_weight = bitcast<f32>(atomicLoad(&max_weight_bits));
    let min_weight = max_weight / max(constants.light_grid_weight_ratio_cap, 1.0);

    // Per-thread light counts and weight sums over the thread's own lights
    var thread_light_count = 0u;
    var thread_weight_sum = 0.0;
    for (var word = word_start; word < word_end; word++) {
        var bits = atomicLoad(&light_mask[word]);
        thread_light_count += countOneBits(bits);
        while bits != 0u {
            let light_id = (word << 5u) | firstTrailingBit(bits);
            thread_weight_sum += light_list_weight(light_id, cell_min, cell_size, min_weight);
            bits &= bits - 1u;
        }
    }

    // Inclusive prefix sums of the per-thread counts and weight sums
    thread_light_counts[thread_index] = thread_light_count;
    thread_weight_sums[thread_index] = thread_weight_sum;
    workgroupBarrier();
    for (var stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
        var count = thread_light_counts[thread_index];
        var sum = thread_weight_sums[thread_index];
        if thread_index >= stride {
            count += thread_light_counts[thread_index - stride];
            sum += thread_weight_sums[thread_index - stride];
        }
        workgroupBarrier();
        thread_light_counts[thread_index] = count;
        thread_weight_sums[thread_index] = sum;
        workgroupBarrier();
    }
    let total_weight = thread_weight_sums[WORKGROUP_SIZE - 1u];

    let slot_count = clamp(constants.light_grid_weighted_slots, 1u, MAX_WEIGHTED_SLOTS);
    if thread_index == 0u {
        var list = vec2(LIGHT_GRID_NO_LIST_NO_LIGHTS, 0u);
        if thread_light_counts[WORKGROUP_SIZE - 1u] != 0u && total_weight > 0.0 {
            list = allocate_light_list(slot_count);
        }
        atomicAdd(&light_grid.live_cells, 1u);
        light_grid.lists[cell_index] = list;
        cell_list = list;
    }
    let list = workgroupUniformLoad(&cell_list);
    if list.y == 0u { return; }

    // Slot k takes the light whose cumulative weight interval contains (k + 0.5) / slots of the total.
    // Each thread's range starts and ends at the scanned sums its neighbors use, so the lights' slot
    // ranges tile the slots without gaps. Each slot is a whole u32, so threads write their own
    // lights' slots directly.
    let slots_per_weight = f32(slot_count) / total_weight;
    var cumulative = select(0.0, thread_weight_sums[max(thread_index, 1u) - 1u], thread_index != 0u);
    let thread_end = thread_weight_sums[thread_index];
    let thread_list_end = thread_light_counts[thread_index];
    // Lights before this thread's, counting up to its last light's index + 1
    var light_index = select(0u, thread_light_counts[max(thread_index, 1u) - 1u], thread_index != 0u);
    var first_slot = u32(clamp(ceil(cumulative * slots_per_weight - 0.5), 0.0, f32(slot_count)));
    for (var word = word_start; word < word_end; word++) {
        var bits = atomicLoad(&light_mask[word]);
        while bits != 0u {
            let light_id = (word << 5u) | firstTrailingBit(bits);
            light_index += 1u;
            cumulative += light_list_weight(light_id, cell_min, cell_size, min_weight);
            if light_index == thread_list_end {
                cumulative = thread_end;
            }
            let end_slot = u32(clamp(ceil(cumulative * slots_per_weight - 0.5), 0.0, f32(slot_count)));
            let packed = light_id | ((end_slot - first_slot) << 16u);
            for (var slot = first_slot; slot < end_slot; slot++) {
                light_grid_pool[list.x + slot] = packed;
            }
            first_slot = end_slot;
            bits &= bits - 1u;
        }
    }
}

/// Allocates `slot_count` slots in the pool. Returns the list, or why there isn't one.
fn allocate_light_list(slot_count: u32) -> vec2<u32> {
    let offset = atomicAdd(&light_grid.pool_used, slot_count);
    if offset + slot_count <= LIGHT_GRID_POOL_CAPACITY {
        return vec2(offset, slot_count);
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

/// A light's weight in a cell's list, after the ratio cap.
fn light_list_weight(light_id: u32, cell_min: vec3<f32>, cell_size: f32, min_weight: f32) -> f32 {
    return max(light_importance_weight(light_bounds[light_id], cell_min, cell_size), min_weight);
}

/// Presamples `light_samples_per_light()` points on every light, see `PerLightSamples`.
/// Dispatched with one thread per slot of `per_light_samples`.
@compute @workgroup_size(64, 1, 1)
fn presample_lights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let light_id = global_id.x / light_samples_per_light();
    if light_id >= arrayLength(&light_sources) { return; }

    var rng = (global_id.x * 0x9E3779B9u) + constants.frame_rng;
    let sample = generate_light_sample(light_id, &rng);
    per_light_samples.light_samples[global_id.x] = sample.light_sample;
    per_light_samples.resolved_light_samples[global_id.x] = pack_resolved_light_sample(sample.resolved_light_sample);
}
