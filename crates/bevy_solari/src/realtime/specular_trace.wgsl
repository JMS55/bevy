#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::utils::octahedral_decode
#import bevy_render::maths::orthonormalize
#import bevy_render::view::View
#import bevy_solari::brdf::specular_brdf
#import bevy_solari::sampling::{sample_ggx_vndf, ggx_vndf_pdf}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::world_cache::query_world_cache

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(7) var gbuffer: texture_2d<u32>;
@group(1) @binding(8) var depth_buffer: texture_depth_2d;
@group(1) @binding(12) var<uniform> view: View;
struct PushConstants { frame_index: u32, reset: u32 }
var<push_constant> constants: PushConstants;

@compute @workgroup_size(8, 8, 1)
fn specular_trace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        return;
    }
    let gpixel = textureLoad(gbuffer, global_id.xy, 0);
    let world_position = reconstruct_world_position(global_id.xy, depth);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));
    let base_rough = unpack4x8unorm(gpixel.r);
    let base_color = pow(base_rough.rgb, vec3(2.2));
    let perceptual_roughness = base_rough.a;
    let roughness = clamp(perceptual_roughness * perceptual_roughness, 0.001, 1.0);
    let props = unpack4x8unorm(gpixel.b);
    let reflectance = vec3(props.r);
    let metallic = props.g;
    if roughness > 0.04 { return; }

    let TBN = orthonormalize(world_normal);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo = normalize(view.world_position - world_position);
    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
    let wi_tangent = sample_ggx_vndf(wo_tangent, roughness, &rng);
    let wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

    let ray_hit = trace_ray(world_position, -wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
    if ray_hit.kind == RAY_QUERY_INTERSECTION_NONE { return; }
    let sample_point = resolve_ray_hit_full(ray_hit);

    let radiance = query_world_cache(sample_point.world_position, sample_point.geometric_world_normal, view.world_position);
    let inverse_pdf = 1.0 / ggx_vndf_pdf(wo_tangent, wi_tangent, roughness);
    let brdf = specular_brdf(world_normal, wo, wi, base_color, metallic, reflectance, perceptual_roughness, roughness);

    var pixel_color = textureLoad(view_output, global_id.xy);
    pixel_color += vec4(radiance * inverse_pdf * brdf * view.exposure, 0.0);
    textureStore(view_output, global_id.xy, pixel_color);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view.main_pass_viewport.zw;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = view.world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}
