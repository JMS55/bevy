//! Inserts the light grid cell of every primary surface before the light grid is built, so that
//! cells that just came into view get a list the same frame, rather than the next.
//! See `SolariLighting::light_grid_primary_cell_prepass`.

import package::realtime::gbuffer_utils::gpixel_resolve;
import package::realtime::bindings::{depth_buffer, gbuffer, view};
import package::realtime::light_grid::query_light_grid;
import package::scene::bindings::RAY_T_MIN;

enable wgpu_ray_query;

@compute @workgroup_size(8, 8, 1)
fn insert_primary_light_grid_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 { return; }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip, view.world_position);

    // The same point initial sampling looks the cell up at
    _ = query_light_grid(surface.world_position + surface.world_normal * RAY_T_MIN);
}
