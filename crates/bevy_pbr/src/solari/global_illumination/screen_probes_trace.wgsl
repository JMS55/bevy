#import bevy_solari::scene_bindings uniforms, map_ray_hit
#import bevy_solari::global_illumination::view_bindings view, depth_buffer, screen_probes_a, FIRST_RADIANCE_CASCADE_INTERVAL
#import bevy_solari::world_cache::query query_world_cache
#import bevy_solari::utils rand_f, rand_vec2f, trace_ray, depth_to_world_position
#import bevy_pbr::utils octahedral_decode

@compute @workgroup_size(8, 8, 1)
fn trace_screen_probes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe_size = u32(pow(2.0, f32(global_id.z) + 3.0));
    let probe_pixel_id = ((global_id.xy / probe_size) * probe_size) + (probe_size / 2u);

    let probe_pixel_depth = textureLoad(depth_buffer, probe_pixel_id, 0i);
    if probe_pixel_depth == 0.0 {
        textureStore(screen_probes_a, global_id.xy, global_id.z, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let probe_pixel_uv = (vec2<f32>(probe_pixel_id) + 0.5) / view.viewport.zw;
    let probe_pixel_world_position = depth_to_world_position(probe_pixel_depth, probe_pixel_uv);

    let octahedral_pixel_center = vec2<f32>(global_id.xy % probe_size) + 0.5;
    let octahedral_pixel_uv = octahedral_pixel_center / f32(probe_size);
    let octahedral_normal = octahedral_decode(octahedral_pixel_uv);

    var radiance_interval_min = FIRST_RADIANCE_CASCADE_INTERVAL * pow(2.0, f32(global_id.z) - 1.0);
    if global_id.z == 0u {
        radiance_interval_min = 0.001;
    }
    let radiance_interval_max = FIRST_RADIANCE_CASCADE_INTERVAL * pow(2.0, f32(global_id.z));

    var color = vec4(0.0, 0.0, 0.0, 1.0);
    let ray_hit = trace_ray(probe_pixel_world_position, octahedral_normal, radiance_interval_min, radiance_interval_max);
    if ray_hit.kind != RAY_QUERY_INTERSECTION_NONE {
        let ray_hit = map_ray_hit(ray_hit);
        let hit_color = ray_hit.material.base_color * query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal);
        color = vec4(hit_color, 0.0);
    }

    textureStore(screen_probes_a, global_id.xy, global_id.z, color);
}
