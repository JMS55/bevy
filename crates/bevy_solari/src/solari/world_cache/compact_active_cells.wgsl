#import bevy_solari::world_cache::bindings
#import bevy_solari::world_cache::utils

var<workgroup> w1: array<u32, 1024u>;
var<workgroup> w2: array<u32, 1024u>;
var<workgroup> w_b2: u32;

@compute @workgroup_size(1024, 1, 1)
fn decay_world_cache_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var life = world_cache_life[global_id.x];
    if life > 0u {
        life -= 1u;
        world_cache_life[global_id.x] = life;

        if life == 0u {
            world_cache_checksums[global_id.x] = WORLD_CACHE_EMPTY_CELL;
            world_cache_irradiance[global_id.x] = vec4(0.0);
        }
    }
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_single_block(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    if t == 0u { w1[0u] = 0u; } else { w1[t] = u32(world_cache_life[cell_id.x - 1u] != 0u); }; workgroupBarrier();
    if t < 1u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 1u]; } workgroupBarrier();
    if t < 2u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 2u]; } workgroupBarrier();
    if t < 4u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 4u]; } workgroupBarrier();
    if t < 8u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 8u]; } workgroupBarrier();
    if t < 16u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 16u]; } workgroupBarrier();
    if t < 32u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 32u]; } workgroupBarrier();
    if t < 64u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 64u]; } workgroupBarrier();
    if t < 128u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 128u]; } workgroupBarrier();
    if t < 256u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 256u]; } workgroupBarrier();
    if t < 512u { world_cache_b1[t] = w2[t]; } else { world_cache_b1[t] = w2[t] + w2[t - 512u]; }
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_blocks(@builtin(local_invocation_index) t: u32) {
    if t == 0u { w1[0u] = 0u; } else { w1[t] = world_cache_b1[t * 1024u - 1u]; }; workgroupBarrier();
    if t < 1u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 1u]; } workgroupBarrier();
    if t < 2u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 2u]; } workgroupBarrier();
    if t < 4u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 4u]; } workgroupBarrier();
    if t < 8u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 8u]; } workgroupBarrier();
    if t < 16u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 16u]; } workgroupBarrier();
    if t < 32u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 32u]; } workgroupBarrier();
    if t < 64u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 64u]; } workgroupBarrier();
    if t < 128u { w1[t] = w2[t]; } else { w1[t] = w2[t] + w2[t - 128u]; } workgroupBarrier();
    if t < 256u { w2[t] = w1[t]; } else { w2[t] = w1[t] + w1[t - 256u]; } workgroupBarrier();
    if t < 512u { world_cache_b2[t] = w2[t]; } else {world_cache_b2[t] = w2[t] + w2[t - 512u]; }
}

@compute @workgroup_size(1024, 1, 1)
fn compact_world_cache_write_active_cells(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) thread_index: u32,
) {
    if thread_index == 1023u {
        w_b2 = world_cache_b2[workgroup_id.x];
    }
    workgroupBarrier();

    let compacted_index = world_cache_b1[cell_id.x] + w_b2;
    if world_cache_life[cell_id.x] != 0u {
        world_cache_active_cell_indices[compacted_index] = cell_id.x;
    }

    if thread_index == 1023u && workgroup_id.x == 1023u {
        world_cache_active_cells_count = compacted_index + 1u;
        world_cache_active_cells_dispatch = DispatchIndirect((world_cache_active_cells_count + 1023u) / 1024u, 1u, 1u);
    }
}
