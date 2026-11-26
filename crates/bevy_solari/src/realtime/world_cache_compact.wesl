import constants::WORLD_CACHE_SIZE;
import package::realtime::world_cache_query::{WORLD_CACHE_EMPTY_CELL, WORLD_CACHE_CELL_LIFETIME};
import package::realtime::bindings::{
    constants,
    world_cache,
};

enable wgpu_ray_query;

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

@group(2) @binding(0) var<storage, read_write> world_cache_active_cells_dispatch: DispatchIndirectArgs;

// x = active cells, y = hot cells
var<workgroup> w1: array<vec2<u32>, 1024u>;
var<workgroup> w2: array<vec2<u32>, 1024u>;

/// A hot cell was read by a camera path last frame: its life was reset to WORLD_CACHE_CELL_LIFETIME, and has since been decayed once.
/// Cells only read by world cache updates inherit the reading cell's (already decayed) life, so they end up lower.
fn cell_flags(cell_index: u32) -> vec2<u32> {
    let life = world_cache.life[cell_index];
    let is_active = life != 0u;
    let is_hot = is_active && life >= WORLD_CACHE_CELL_LIFETIME - 1u;
    return vec2(u32(is_active), u32(is_hot));
}

@compute @workgroup_size(1024, 1, 1)
fn decay_world_cache(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var life = world_cache.life[global_id.x];
    if life > 0u {
        life = select(life - 1u, 0u, bool(constants.reset));
        world_cache.life[global_id.x] = life;

        if life == 0u {
            world_cache.checksums[global_id.x] = WORLD_CACHE_EMPTY_CELL;
            world_cache.radiance[global_id.x] = vec4(0.0);
            world_cache.radiance_indirect[global_id.x] = vec4(0.0);
            world_cache.luminance_deltas[global_id.x] = 0.0;
        }
    }
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_single_block(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    if t == 0u { w1[0u] = vec2(0u); } else { w1[t] = cell_flags(cell_id.x - 1u); }; workgroupBarrier();
    if t < 1u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 1u]; } workgroupBarrier();
    if t < 2u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 2u]; } workgroupBarrier();
    if t < 4u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 4u]; } workgroupBarrier();
    if t < 8u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 8u]; } workgroupBarrier();
    if t < 16u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 16u]; } workgroupBarrier();
    if t < 32u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 32u]; } workgroupBarrier();
    if t < 64u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 64u]; } workgroupBarrier();
    if t < 128u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 128u]; } workgroupBarrier();
    if t < 256u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 256u]; } workgroupBarrier();
    var r: vec2<u32>;
    if t < 512u { r = w2[t]; } else { r = w2[t] + w2[t - 512u]; }
    // Per-block exclusive prefix sums are at most 1023, so pack both into one u32
    world_cache.a[cell_id.x] = r.x | (r.y << 16u);
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_blocks(@builtin(local_invocation_index) t: u32) {
    let block_count = WORLD_CACHE_SIZE / 1024u;
    if t == 0u || t >= block_count {
        w1[t] = vec2(0u);
    } else {
        let last = t * 1024u - 1u;
        let packed = world_cache.a[last];
        w1[t] = vec2(packed & 0xFFFFu, packed >> 16u) + cell_flags(last);
    }; workgroupBarrier();
    if t < 1u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 1u]; } workgroupBarrier();
    if t < 2u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 2u]; } workgroupBarrier();
    if t < 4u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 4u]; } workgroupBarrier();
    if t < 8u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 8u]; } workgroupBarrier();
    if t < 16u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 16u]; } workgroupBarrier();
    if t < 32u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 32u]; } workgroupBarrier();
    if t < 64u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 64u]; } workgroupBarrier();
    if t < 128u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 128u]; } workgroupBarrier();
    if t < 256u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 256u]; } workgroupBarrier();
    if t >= block_count { return; }
    var r: vec2<u32>;
    if t < 512u { r = w2[t]; } else { r = w2[t] + w2[t - 512u]; }
    world_cache.b[t] = r.x;
    world_cache.b_hot[t] = r.y;
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_write_active_cells(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let packed = world_cache.a[cell_id.x];
    let active_prefix = (packed & 0xFFFFu) + world_cache.b[workgroup_id.x];
    let hot_prefix = (packed >> 16u) + world_cache.b_hot[workgroup_id.x];
    let flags = cell_flags(cell_id.x);

    // Totals, from the last cell
    let last = WORLD_CACHE_SIZE - 1u;
    let last_packed = world_cache.a[last];
    let last_flags = cell_flags(last);
    let last_block = last / 1024u;
    let active_cell_count = (last_packed & 0xFFFFu) + world_cache.b[last_block] + last_flags.x;
    let hot_cell_count = (last_packed >> 16u) + world_cache.b_hot[last_block] + last_flags.y;

    // Hot cells first, then cold cells
    if flags.y != 0u {
        world_cache.active_cell_indices[hot_prefix] = cell_id.x;
    } else if flags.x != 0u {
        world_cache.active_cell_indices[hot_cell_count + (active_prefix - hot_prefix)] = cell_id.x;
    }

    if cell_id.x == last {
        let cold_cell_count = active_cell_count - hot_cell_count;
        let previous_cold_cell_count = world_cache.active_cells_count - world_cache.hot_cells_count;
        world_cache.active_cells_count = active_cell_count;
        world_cache.hot_cells_count = hot_cell_count;

        let max_updates = constants.world_cache_max_cell_updates_per_frame;
        let update_count = hot_cell_count + min(cold_cell_count, max_updates);
        world_cache_active_cells_dispatch = DispatchIndirectArgs((update_count + 63u) / 64u, 1u, 1u);

        // Rolling window over the cold cells
        var window_start = 0u;
        if cold_cell_count > max_updates {
            let rescaled_start = u32(f32(world_cache.update_window_start) * f32(cold_cell_count) / max(f32(previous_cold_cell_count), 1.0));
            window_start = (rescaled_start + max_updates) % cold_cell_count;
        }
        world_cache.update_window_start = window_start;
    }
}
