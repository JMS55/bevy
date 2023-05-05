#import bevy_solari::scene_bindings
#import bevy_solari::view_bindings
#import bevy_solari::utils

@compute @workgroup_size(8, 8, 1)
fn filter_screen_probes(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_count: vec3<u32>,
) {
    let probe_index = workgroup_id.x + workgroup_id.y * workgroup_count.x;
    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    let frame_index = globals.frame_count * 5782582u;
    var rng = pixel_index + frame_index;

    let pixel_uv = (vec2<f32>(global_id.xy) + rand_vec2(&rng)) / view.viewport.zw;
    let g_buffer_pixel = textureLoad(g_buffer, global_id.xy);
    if decode_g_buffer_depth(g_buffer_pixel) < 0.0 {
        textureStore(view_target, global_id.xy, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let pixel_world_normal = decode_g_buffer_world_normal(g_buffer_pixel);
    let material = decode_m_buffer(textureLoad(m_buffer, global_id.xy), pixel_uv);

    let c1 = 0.429043;
    let c2 = 0.511664;
    let c3 = 0.743125;
    let c4 = 0.886227;
    let c5 = 0.247708;
    let x = pixel_world_normal.x;
    let y = pixel_world_normal.y;
    let z = pixel_world_normal.z;
    let xz = x * z;
    let yz = y * z;
    let xy = x * y;
    let zz = z * z;
    let xx_yy = x * x - y * y;
    let sh = screen_probe_spherical_harmonics[probe_index];
    let L00 = sh.b0.xyz;
    let L11 = vec3(sh.b0.w, sh.b1.xy);
    let L10 = vec3(sh.b1.zw, sh.b2.x);
    let L1_1 = sh.b2.yzw;
    let L21 = sh.b3.xyz;
    let L2_1 = vec3(sh.b3.w, sh.b4.xy);
    let L2_2 = vec3(sh.b4.zw, sh.b5.x);
    let L20 = sh.b5.yzw;
    let L22 = sh.b6;
    let irradiance = (c1 * L22 * xx_yy) + (c3 * L20 * zz) + (c4 * L00) - (c5 * L20) + (2.0 * c1 * ((L2_2 * xy) + (L21 * xz) + (L2_1 * yz))) + (2.0 * c2 * ((L11 * x) + (L1_1 * y) + (L10 * z)));

    let final_color = (material.base_color * irradiance) + material.emission;
    textureStore(view_target, global_id.xy, vec4(final_color, 1.0));
}
