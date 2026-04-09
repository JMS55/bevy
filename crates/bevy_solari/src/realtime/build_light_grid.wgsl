enable wgpu_ray_query;

#import bevy_solari::raytracing_scene_bindings::local_lights
#import bevy_solari::realtime_bindings::{light_grid_cells, view, constants}

// TODO: Load local_lights into workgroup shared memory to reduce global memory traffic

@compute @workgroup_size(64, 1, 1)
fn build_light_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1D cell index for the thread
    let cells_per_axis = vec3<u32>(constants.light_grid_cells_per_axis_x, constants.light_grid_cells_per_axis_y, constants.light_grid_cells_per_axis_z);
    let total_cells = cells_per_axis.x * cells_per_axis.y * cells_per_axis.z;
    let cell_index = global_id.x;
    if cell_index >= total_cells { return; }

    // 3D cell index for the thread
    let cell_z = cell_index / (cells_per_axis.x * cells_per_axis.y);
    let remaining = cell_index % (cells_per_axis.x * cells_per_axis.y);
    let cell_y = remaining / cells_per_axis.x;
    let cell_x = remaining % cells_per_axis.x;
    let cell_coordinates = vec3<u32>(cell_x, cell_y, cell_z);

    // World-space center of the cell
    let cell_size = constants.light_grid_cell_size;
    let half_cell = cell_size * 0.5;
    let grid_half_extent = vec3<f32>(cells_per_axis) * half_cell;
    let cell_center = view.world_position.xyz
        + (vec3<f32>(cell_coordinates) + 0.5) * cell_size
        - grid_half_extent;

    // Loop over every local light
    let emissive_light_count = arrayLength(&local_lights);
    let max_lights = constants.light_grid_max_lights_per_cell;
    let cell_base_index = cell_index * max_lights;
    let null_light_id = 0xFFFFu;
    let null_u32 = null_light_id | (null_light_id << 16u);
    var cell_light_count = 0u;
    var pending_u32 = null_u32;
    for (var i = 0u; i < emissive_light_count; i++) {
        // Shortest distance from cell center to the light AABB
        let light = local_lights[i];
        let closest = clamp(cell_center, light.aabb_min, light.aabb_max);
        let offset = closest - cell_center;
        let distance_squared = max(dot(offset, offset), half_cell * half_cell);

        // Light contribution (discounting emissive_texture)
        let contribution = light.luminance * view.exposure / distance_squared;

        // Write lights with enough contribution into the cell
        // TODO: There's probably something better to write out than just the light ID
        if contribution >= constants.light_grid_contribution_threshold && cell_light_count < max_lights {
            // Pack u16 light ID into a pending u32, and flush every other write
            let light_id_u16 = light.light_id & 0xFFFFu;
            if cell_light_count % 2u == 0u {
                pending_u32 = light_id_u16;
            } else {
                pending_u32 |= light_id_u16 << 16u;
                light_grid_cells[(cell_base_index + cell_light_count) / 2u] = pending_u32;
                pending_u32 = null_u32;
            }
            cell_light_count += 1u;
        }
    }

    // Flush the last u32 if we ended on an odd count
    if cell_light_count % 2u == 1u {
        light_grid_cells[(cell_base_index + cell_light_count - 1u) / 2u] = pending_u32 | (null_light_id << 16u);
    }

    // Fill remaining slots with null light IDs
    let remaining_u32s = (max_lights - cell_light_count + 1u) / 2u;
    let start_u32 = (cell_base_index + cell_light_count + 1u) / 2u;
    for (var i = 0u; i < remaining_u32s; i++) {
        light_grid_cells[start_u32 + i] = null_u32;
    }
}
