//! A camera-relative world-space hash grid, where each cell holds a list of the local lights that
//! can contribute significantly to it, weighted by importance. See `light_grid_build.wesl` for how
//! lists are built, and `light_candidates.wesl` for how NEE uses them.
//!
//! Lists are double buffered: the build reads every cell's list from last frame (its own, and its
//! neighbors') while writing this frame's, see `light_grid_frame_parity`.

import constants::LIGHT_GRID_SIZE;
import package::realtime::bindings::{constants, light_grid, local_light_count, view, LIGHT_GRID_EMPTY_CELL, LIGHT_GRID_NO_LIST_NO_LIGHTS, LIGHT_GRID_NO_LIST_NOT_BUILT, LIGHT_GRID_NO_LIST_OUT_OF_POOL, LIGHT_GRID_NO_LIST_TABLE_FULL};
import package::realtime::world_cache_query::{compute_checksum, compute_key};
import bevy_pbr::render::utils::rand_f;

enable wgpu_ray_query;

/// Frames a cell lives for without being queried. Must be at least 2, since cells queried during
/// lighting are built at the start of the next frame, after decaying once.
const LIGHT_GRID_CELL_LIFETIME: u32 = 4u;
/// How many slots past a cell's home slot it can end up in, after hash collisions.
const LIGHT_GRID_MAX_SEARCH_STEPS: u32 = 16u;
/// Each cell's list of lights splits this many units of probability between its lights.
/// See `cell_entry`.
const LIGHT_GRID_PDF_UNITS_MAX: u32 = 65535u;

/// A cell's list in `light_grid_pool`, see `cell_entry`. A count of 0 means there is no list, and
/// the offset then holds why, see `LIGHT_GRID_NO_LIST_*`.
struct LightGridCell {
    offset: u32,
    /// Number of distinct lights in the list.
    count: u32,
    /// Units of probability over the whole list, the last entry's cumulative units.
    total: u32,
    /// The cell's hash table slot, or `LIGHT_GRID_SIZE` if it has none.
    key: u32,
    /// The learned probability that an NEE candidate is the sun, or negative if there is none yet,
    /// see `SolariLighting::light_grid_sun_split`.
    sun_probability: f32,
}

fn empty_light_grid_cell() -> LightGridCell {
    return LightGridCell(LIGHT_GRID_NO_LIST_NOT_BUILT, 0u, 0u, LIGHT_GRID_SIZE, -1.0);
}

/// Which of the two list buffers this frame's lists are in.
fn light_grid_frame_parity() -> u32 {
    return constants.light_grid_frame & 1u;
}

/// The stored list of cell `key` in buffer `parity`, packed as `(offset, count | total << 16)`.
fn load_cell_list(key: u32, parity: u32) -> vec2<u32> {
    if parity == 0u {
        return light_grid.lists_a[key];
    }
    return light_grid.lists_b[key];
}

/// Stores this frame's list of a live cell, and its plain copy for lighting, see `query_light_grid`.
fn publish_cell_list(key: u32, parity: u32, list: vec2<u32>) {
    store_cell_list(key, parity, list);
    light_grid.lookup[key] = vec4(atomicLoad(&light_grid.checksums[key]), list, 0u);
}

fn store_cell_list(key: u32, parity: u32, list: vec2<u32>) {
    if parity == 0u {
        light_grid.lists_a[key] = list;
    } else {
        light_grid.lists_b[key] = list;
    }
}

fn unpack_cell_list(list: vec2<u32>) -> LightGridCell {
    return LightGridCell(list.x, list.y & 0xFFFFu, list.y >> 16u, LIGHT_GRID_SIZE, -1.0);
}

/// Like `light_grid_cell_at`, but only finds cells that exist (built this frame), and inserts none.
fn light_grid_existing_cell_at(world_position: vec3<f32>) -> LightGridCell {
    @if(LIGHT_GRID) {
        let cell_key = light_grid_cell_key(world_position);
        for (var i = 0u; i < LIGHT_GRID_MAX_SEARCH_STEPS; i++) {
            let key = (cell_key.home_key + i) & (LIGHT_GRID_SIZE - 1u);
            let entry = light_grid.lookup[key];
            if entry.x == cell_key.checksum {
                return with_cell_split(unpack_cell_list(entry.yz), key);
            }
        }
    }
    return empty_light_grid_cell();
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

struct LightGridCellKey {
    home_key: u32,
    checksum: u32,
    cell_min: vec3<f32>,
    cell_size: f32,
}

fn light_grid_cell_key(world_position: vec3<f32>) -> LightGridCellKey {
    let lod = light_grid_lod(world_position);
    let cell_size = light_grid_cell_size(lod);
    let cell_coordinates = floor(world_position / cell_size);
    let position_key = bitcast<vec3<u32>>(vec3<i32>(cell_coordinates));
    let lod_key = vec3(lod, 0u, 0u);
    return LightGridCellKey(
        compute_key(position_key, lod_key),
        compute_checksum(position_key, lod_key),
        cell_coordinates * cell_size,
        cell_size,
    );
}

/// Finds the cell containing `world_position`, inserting it to be built next frame if it's missing.
///
/// This is deterministic for a given position and frame (no jitter, no stochastic LOD), so that
/// NEE and the MIS weights computed elsewhere for the same shading point agree on the cell.
fn query_light_grid(world_position: vec3<f32>) -> LightGridCell {
    let cell_key = light_grid_cell_key(world_position);
    let home_key = cell_key.home_key;
    let checksum = cell_key.checksum;

    // Cells built this frame are in the plain copy, which plain loads can cache. Marking the cell
    // used is a plain store too. Cells it doesn't have go through the atomic path below.
    for (var i = 0u; i < LIGHT_GRID_MAX_SEARCH_STEPS; i++) {
        let key = (home_key + i) & (LIGHT_GRID_SIZE - 1u);
        let entry = light_grid.lookup[key];
        if entry.x == checksum {
            // Store only once per cell and frame: scattered stores are what cost
            if light_grid.touched[key] == 0u {
                light_grid.touched[key] = 1u;
            }
            return with_cell_split(unpack_cell_list(entry.yz), key);
        }
    }

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
            // New cell, it gets built next frame. Dead cells have both lists cleared, so a new
            // cell never sees a previous occupant's list.
            atomicStore(&light_grid.life[key], LIGHT_GRID_CELL_LIFETIME);
            light_grid.bounds[key] = vec4(cell_key.cell_min, cell_key.cell_size);
            return empty_light_grid_cell();
        }
        if cas.old_value == checksum {
            // Another thread inserted the same cell first
            return light_grid_cell_hit(key);
        }
    }

    atomicAdd(&light_grid.failed_queries, 1u);
    return LightGridCell(LIGHT_GRID_NO_LIST_TABLE_FULL, 0u, 0u, LIGHT_GRID_SIZE, -1.0);
}

/// The key of the cell containing `world_position`, without inserting it or keeping it alive, or
/// `LIGHT_GRID_SIZE` if there is no such cell.
fn find_light_grid_cell(world_position: vec3<f32>) -> u32 {
    let cell_key = light_grid_cell_key(world_position);
    for (var i = 0u; i < LIGHT_GRID_MAX_SEARCH_STEPS; i++) {
        let key = (cell_key.home_key + i) & (LIGHT_GRID_SIZE - 1u);
        if atomicLoad(&light_grid.checksums[key]) == cell_key.checksum {
            return key;
        }
    }
    return LIGHT_GRID_SIZE;
}

/// Keeps a found cell alive, and returns its list.
fn light_grid_cell_hit(key: u32) -> LightGridCell {
    if atomicLoad(&light_grid.life[key]) != LIGHT_GRID_CELL_LIFETIME {
        atomicStore(&light_grid.life[key], LIGHT_GRID_CELL_LIFETIME);
    }
    return with_cell_split(unpack_cell_list(load_cell_list(key, light_grid_frame_parity())), key);
}

/// Fills in a cell's key and learned sun/local split.
fn with_cell_split(cell_in: LightGridCell, key: u32) -> LightGridCell {
    var cell = cell_in;
    cell.key = key;
    if constants.light_grid_sun_split != 0u {
        cell.sun_probability = light_grid.split_state[key].z;
    }
    return cell;
}

/// Fixed point scale of `LightGrid::split_accum`, and the most one record adds (in exposed units).
const SPLIT_ACCUM_SCALE: f32 = 256.0;
const SPLIT_RECORD_MAX: f32 = 64.0;
/// The fraction of NEE winners that record, see `light_grid_record_split`.
const SPLIT_RECORD_RATE: f32 = 0.25;

/// Records an NEE winner's shadowed contribution (luminance, unexposed) for its cell's learned
/// sun/local split. A random `light_grid_sun_split_record_rate` of the winners record, which scales
/// both sums alike.
fn light_grid_record_split(cell: LightGridCell, sun_winner: bool, shadowed: f32, rng: ptr<function, u32>) {
    if constants.light_grid_sun_split == 0u || cell.key >= LIGHT_GRID_SIZE { return; }
    if rand_f(rng) >= SPLIT_RECORD_RATE { return; }
    let value = min(shadowed * view.exposure, SPLIT_RECORD_MAX);
    if !(value > 0.0) { return; }
    atomicAdd(&light_grid.split_accum[cell.key * 2u + select(1u, 0u, sun_winner)], u32(value * SPLIT_ACCUM_SCALE));
}

fn light_grid_lod(world_position: vec3<f32>) -> u32 {
    let camera_distance = distance(view.world_position, world_position) / constants.light_grid_position_lod_scale;
    return u32(floor(log2(1.0 + camera_distance)));
}

fn light_grid_cell_size(lod: u32) -> f32 {
    return constants.light_grid_base_cell_size * exp2(f32(lod));
}

/// A cell's `i`th entry. Entries are sorted by light id, one per distinct light, with the light id
/// in the low 16 bits and the cumulative units of probability up to and including this light in
/// the high 16 bits. A light's probability is its own units over `cell.total`.
///
/// Sorted by light id, the cumulative units also increase, so the same list is binary searched by
/// units to sample a light, and by light id to find a light's pdf.
fn cell_entry(cell: LightGridCell, i: u32) -> u32 {
    return light_grid.pool[cell.offset + i];
}

/// The light id in a cell's `i`th entry.
fn cell_light_id(cell: LightGridCell, i: u32) -> u32 {
    return cell_entry(cell, i) & 0xFFFFu;
}

/// The units of probability of a cell's `i`th entry.
fn cell_entry_units(cell: LightGridCell, i: u32) -> u32 {
    let end = cell_entry(cell, i) >> 16u;
    let start = select(0u, cell_entry(cell, max(i, 1u) - 1u) >> 16u, i != 0u);
    return end - start;
}

struct CellLightSample {
    light_id: u32,
    pdf: f32,
}

/// The `j`th of a cell's guide table, which follows its entries, two u16 per u32: the first entry
/// whose cumulative units exceed `j * total / count`. Sampling starts its search there, so it takes
/// one or two steps on average rather than a binary search (Chen and Asau's guide table).
fn cell_guide(cell: LightGridCell, j: u32) -> u32 {
    return (light_grid.pool[cell.offset + cell.count + (j >> 1u)] >> ((j & 1u) * 16u)) & 0xFFFFu;
}

/// How many u32s a list of `count` lights takes in the pool, entries and guide table.
fn cell_list_size(count: u32) -> u32 {
    return count + (count + 1u) / 2u;
}

/// Picks a light from a cell's list with the list's probabilities. `rand_units` is uniform in
/// `[0, cell.total)`.
fn sample_cell_light(cell: LightGridCell, rand_units: u32) -> CellLightSample {
    // The first entry whose cumulative units exceed `rand_units`
    var i = cell_guide(cell, rand_units * cell.count / cell.total);
    while i < cell.count - 1u && (cell_entry(cell, i) >> 16u) <= rand_units {
        i += 1u;
    }
    return CellLightSample(cell_light_id(cell, i), f32(cell_entry_units(cell, i)) / f32(cell.total));
}

/// The index of the first entry of a cell's list whose light id is not below `light_id`.
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

/// The probability that `sample_cell_light` picks `light_id`.
fn cell_light_pdf(cell: LightGridCell, light_id: u32) -> f32 {
    let i = cell_lower_bound(cell, light_id);
    if i >= cell.count { return 0.0; }
    if cell_light_id(cell, i) != light_id { return 0.0; }
    return f32(cell_entry_units(cell, i)) / f32(cell.total);
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
        let cell_key = light_grid_cell_key(world_position);
        let hash = cell_key.checksum;
        color = mix(vec3(0.15), vec3(1.0), vec3(f32(hash & 0xFFu), f32((hash >> 8u) & 0xFFu), f32((hash >> 16u) & 0xFFu)) / 255.0);
    } else {
        // Lights in the list, relative to the most a list can hold
        let t = saturate(f32(cell.count) / f32(max(constants.light_grid_max_lights_per_cell, 1u)));
        color = select(mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t * 2.0 - 1.0), mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t * 2.0), t < 0.5);
    }

    // Shade by the surface's facing so geometry stays readable
    let facing = saturate(dot(world_normal, normalize(view.world_position - world_position)));
    return color * (0.3 + 0.7 * facing);
}
