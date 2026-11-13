//! A camera-relative world-space hash grid, where each cell holds a list of the local lights that
//! can contribute significantly to it, weighted by importance. See `light_grid_build.wesl` for how
//! lists are built, and `light_candidates.wesl` for how NEE uses them.

import constants::LIGHT_GRID_SIZE;
import package::realtime::bindings::{constants, light_grid, light_grid_pool, local_light_count, view, LIGHT_GRID_EMPTY_CELL, LIGHT_GRID_NO_LIST_NO_LIGHTS, LIGHT_GRID_NO_LIST_NOT_BUILT, LIGHT_GRID_NO_LIST_OUT_OF_POOL, LIGHT_GRID_NO_LIST_TABLE_FULL};
import package::realtime::world_cache_query::{compute_checksum, compute_key};

enable wgpu_ray_query;

/// Frames a cell lives for without being queried. Must be at least 2, since cells queried during
/// lighting are built at the start of the next frame, after decaying once.
const LIGHT_GRID_CELL_LIFETIME: u32 = 4u;
/// How many slots past a cell's home slot it can end up in, after hash collisions.
const LIGHT_GRID_MAX_SEARCH_STEPS: u32 = 16u;

/// A cell's list of slots in `light_grid_pool`, see `weighted_cell_slot`. A count of 0 means there
/// is no list, and the offset then holds why, see `LIGHT_GRID_NO_LIST_*`.
struct LightGridCell {
    offset: u32,
    count: u32,
}

fn empty_light_grid_cell() -> LightGridCell {
    return LightGridCell(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u);
}

/// The cell NEE uses at `world_position`, or an empty cell without the light grid, in which case
/// every local light candidate is a global one.
fn light_grid_cell_at(world_position: vec3<f32>) -> LightGridCell {
    @if(LIGHT_GRID) {
        return query_light_grid(world_position);
    } @else {
        return empty_light_grid_cell();
    }
}

/// Finds the cell containing `world_position`, inserting it to be built next frame if it's missing.
///
/// This is deterministic for a given position and frame (no jitter, no stochastic LOD), so that
/// NEE and the MIS weights computed elsewhere for the same shading point agree on the cell.
fn query_light_grid(world_position: vec3<f32>) -> LightGridCell {
    let lod = light_grid_lod(world_position);
    let cell_size = light_grid_cell_size(lod);
    let cell_coordinates = floor(world_position / cell_size);
    let position_key = bitcast<vec3<u32>>(vec3<i32>(cell_coordinates));
    let lod_key = vec3(lod, 0u, 0u);
    let home_key = compute_key(position_key, lod_key);
    let checksum = compute_checksum(position_key, lod_key);

    // Search the whole probe window before inserting. Cells that die leave empty slots in the
    // middle of probe sequences, and stopping at the first empty slot would insert a duplicate of
    // a cell stored past it, which then has no list until the next frame.
    for (var i = 0u; i < LIGHT_GRID_MAX_SEARCH_STEPS; i++) {
        let key = (home_key + i) & (LIGHT_GRID_SIZE - 1u);
        if atomicLoad(&light_grid.checksums[key]) == checksum {
            return light_grid_cell_hit(key);
        }
    }

    // Not found - insert into the first empty slot
    for (var i = 0u; i < LIGHT_GRID_MAX_SEARCH_STEPS; i++) {
        let key = (home_key + i) & (LIGHT_GRID_SIZE - 1u);
        if atomicLoad(&light_grid.checksums[key]) != LIGHT_GRID_EMPTY_CELL { continue; }

        let cas = atomicCompareExchangeWeak(&light_grid.checksums[key], LIGHT_GRID_EMPTY_CELL, checksum);
        if cas.exchanged {
            // New cell, it gets built next frame
            atomicStore(&light_grid.life[key], LIGHT_GRID_CELL_LIFETIME);
            light_grid.bounds[key] = vec4(cell_coordinates * cell_size, cell_size);
            return LightGridCell(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u);
        }
        if cas.old_value == checksum {
            // Another thread inserted the same cell first
            return light_grid_cell_hit(key);
        }
    }

    atomicAdd(&light_grid.failed_queries, 1u);
    return LightGridCell(LIGHT_GRID_NO_LIST_TABLE_FULL, 0u);
}

/// Keeps a found cell alive, and returns its list.
fn light_grid_cell_hit(key: u32) -> LightGridCell {
    if atomicLoad(&light_grid.life[key]) != LIGHT_GRID_CELL_LIFETIME {
        atomicStore(&light_grid.life[key], LIGHT_GRID_CELL_LIFETIME);
    }
    let list = light_grid.lists[key];
    return LightGridCell(list.x, list.y);
}

fn light_grid_lod(world_position: vec3<f32>) -> u32 {
    let camera_distance = distance(view.world_position, world_position) / constants.light_grid_position_lod_scale;
    return u32(floor(log2(1.0 + camera_distance)));
}

fn light_grid_cell_size(lod: u32) -> f32 {
    return constants.light_grid_base_cell_size * exp2(f32(lod));
}

/// A cell's `i`th slot. The light id is in the low 16 bits, and how many of the cell's slots hold
/// that light in the high 16 bits, so a slot pick knows its own pdf without searching.
///
/// Slots are sorted by light id, and each light holds a number of slots proportional to its
/// importance, so a uniformly random slot is an importance sampled light.
fn weighted_cell_slot(cell: LightGridCell, i: u32) -> u32 {
    return light_grid_pool[cell.offset + i];
}

/// The light id in a cell's `i`th slot.
fn cell_light_id(cell: LightGridCell, i: u32) -> u32 {
    return weighted_cell_slot(cell, i) & 0xFFFFu;
}

/// The index of the first slot of a cell's list whose light id is not below `light_id`.
fn cell_lower_bound(cell: LightGridCell, light_id: u32) -> u32 {
    var low = 0u;
    var high = cell.count;
    while low < high {
        let middle = (low + high) >> 1u;
        if cell_light_id(cell, middle) < light_id {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}

/// The probability that a uniformly random slot of a cell holds `light_id`.
fn weighted_cell_pdf(cell: LightGridCell, light_id: u32) -> f32 {
    let i = cell_lower_bound(cell, light_id);
    if i >= cell.count { return 0.0; }
    let slot = weighted_cell_slot(cell, i);
    return select(0.0, f32(slot >> 16u) / f32(cell.count), (slot & 0xFFFFu) == light_id);
}

/// A debug color for the cell containing `world_position`, in display range.
/// See `SolariLighting::light_grid_visualization`.
fn light_grid_visualization(world_position: vec3<f32>, world_normal: vec3<f32>, mode: u32) -> vec3<f32> {
    let cell = query_light_grid(world_position);

    var color = vec3(0.0);
    if cell.count == 0u {
        switch cell.offset {
            case LIGHT_GRID_NO_LIST_NOT_BUILT: { color = vec3(0.0); }
            case LIGHT_GRID_NO_LIST_NO_LIGHTS: { color = vec3(0.08); }
            case LIGHT_GRID_NO_LIST_OUT_OF_POOL: { color = vec3(1.0); }
            case LIGHT_GRID_NO_LIST_TABLE_FULL: { color = vec3(1.0, 0.0, 0.0); }
            default: {}
        }
    } else if mode == 1u {
        let lod = light_grid_lod(world_position);
        let position_key = bitcast<vec3<u32>>(vec3<i32>(floor(world_position / light_grid_cell_size(lod))));
        let hash = compute_checksum(position_key, vec3(lod, 0u, 0u));
        color = mix(vec3(0.15), vec3(1.0), vec3(f32(hash & 0xFFu), f32((hash >> 8u) & 0xFFu), f32((hash >> 16u) & 0xFFu)) / 255.0);
    } else {
        // Count the distinct lights in the list, which repeats lights across their slots
        var light_count = 0u;
        var i = 0u;
        while i < cell.count {
            i += max(weighted_cell_slot(cell, i) >> 16u, 1u);
            light_count += 1u;
        }
        let t = saturate(f32(light_count) / f32(local_light_count()));
        color = select(mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t * 2.0 - 1.0), mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t * 2.0), t < 0.5);
    }

    // Shade by the surface's facing so geometry stays readable
    let facing = saturate(dot(world_normal, normalize(view.world_position - world_position)));
    return color * (0.3 + 0.7 * facing);
}
