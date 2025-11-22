#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_render::maths::{orthonormalize, PI}
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_brdf, evaluate_specular_brdf}
#import bevy_solari::gbuffer_utils::gpixel_resolve
#import bevy_solari::sampling::{sample_random_light, sample_ggx_vndf, ggx_vndf_pdf}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::world_cache::query_world_cache

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var<storage, read_write> gi_reservoirs_a: array<Reservoir>;
@group(1) @binding(7) var gbuffer: texture_2d<u32>;
@group(1) @binding(8) var depth_buffer: texture_depth_2d;
@group(1) @binding(12) var<uniform> view: View;
@group(1) @binding(13) var<uniform> previous_view: PreviousViewUniforms;
#ifdef SPECULAR_MOTION_VECTORS
@group(2) @binding(3) var specular_motion_vectors: texture_storage_2d<rg16float, write>;
#endif
struct PushConstants { frame_index: u32, reset: u32 }
var<push_constant> constants: PushConstants;

@compute @workgroup_size(8, 8, 1)
fn specular_gi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let wo = normalize(view.world_position - surface.world_position);

    var radiance: vec3<f32>;
    var wi: vec3<f32>;
    if surface.material.roughness > 0.1 {
        // Surface is very rough, reuse the ReSTIR GI reservoir
        let gi_reservoir = gi_reservoirs_a[pixel_index];
        wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);
        radiance = gi_reservoir.radiance * gi_reservoir.unbiased_contribution_weight;
    } else {
        // Surface is glossy or mirror-like, trace a new path
        let TBN = orthonormalize(surface.world_normal);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
        let wi_tangent = sample_ggx_vndf(wo_tangent, surface.material.roughness, &rng);
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;
        let pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, surface.material.roughness);

        radiance = trace_glossy_path(global_id.xy, surface.world_position, wi, &rng) / pdf;
    }

    let brdf = evaluate_specular_brdf(surface.world_normal, wo, wi, surface.material.base_color, surface.material.metallic,
        surface.material.reflectance, surface.material.perceptual_roughness, surface.material.roughness);
    let cos_theta = saturate(dot(wi, surface.world_normal));
    radiance *= brdf * cos_theta * view.exposure;

    var pixel_color = textureLoad(view_output, global_id.xy);
    pixel_color += vec4(radiance, 0.0);
    textureStore(view_output, global_id.xy, pixel_color);

#ifdef VISUALIZE_WORLD_CACHE
    textureStore(view_output, global_id.xy, vec4(query_world_cache(surface.world_position, surface.world_normal, view.world_position, &rng) * view.exposure, 1.0));
#endif
}

fn trace_glossy_path(pixel_id: vec2<u32>, initial_ray_origin: vec3<f32>, initial_wi: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var ray_origin = initial_ray_origin;
    var wi = initial_wi;
    var end_point_previous_frame_world_position = ray_origin;

    // Trace up to three bounces, getting the net throughput from them
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    for (var i = 0u; i < 3u; i += 1u) {
        // Trace ray
        let ray = trace_ray(ray_origin, wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { break; }
        let ray_hit = resolve_ray_hit_full(ray);

        ray_origin = ray_hit.world_position;
        end_point_previous_frame_world_position = ray_hit.previous_frame_world_position;
        let wo = -wi;

        if ray_hit.material.roughness > 0.1 && i != 0u {
            // Surface is very rough, terminate path in the world cache
            let diffuse_brdf = ray_hit.material.base_color / PI;
            radiance += throughput * diffuse_brdf * query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, rng);
            break;
        } else {
            // Sample direct lighting
            let direct_lighting = sample_random_light(ray_hit.world_position, ray_hit.world_normal, rng);
            let direct_lighting_brdf = evaluate_brdf(ray_hit.world_normal, wo, direct_lighting.wi, ray_hit.material);
            radiance += throughput * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_lighting_brdf;
        }

        // Sample new ray direction from the GGX BRDF for next bounce
        let TBN = calculate_tbn_mikktspace(ray_hit.world_normal, ray_hit.world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
        let wi_tangent = sample_ggx_vndf(wo_tangent, ray_hit.material.roughness, rng);
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

        // Update throughput for next bounce
        let pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, ray_hit.material.roughness);
        let brdf = evaluate_brdf(N, wo, wi, ray_hit.material);
        let cos_theta = saturate(dot(wi, N));
        throughput *= (brdf * cos_theta) / pdf;
    }

#ifdef SPECULAR_MOTION_VECTORS
    let motion_vector = calculate_motion_vector(ray_origin, end_point_previous_frame_world_position);
    textureStore(specular_motion_vectors, pixel_id, vec4(motion_vector, 0.0, 0.0));
#endif

    return radiance;
}

fn calculate_motion_vector(world_position: vec3<f32>, previous_world_position: vec3<f32>) -> vec2<f32> {
    let clip_position_t = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    let clip_position = clip_position_t.xy / clip_position_t.w;
    let previous_clip_position_t = previous_view.clip_from_world * vec4(previous_world_position, 1.0);
    let previous_clip_position = previous_clip_position_t.xy / previous_clip_position_t.w;
    // These motion vectors are used as offsets to UV positions and are stored
    // in the range -1,1 to allow offsetting from the one corner to the
    // diagonally-opposite corner in UV coordinates, in either direction.
    // A difference between diagonally-opposite corners of clip space is in the
    // range -2,2, so this needs to be scaled by 0.5. And the V direction goes
    // down where clip space y goes up, so y needs to be flipped.
    return (clip_position - previous_clip_position) * vec2(0.5, -0.5);
}

// Don't adjust the size of this struct without also adjusting GI_RESERVOIR_STRUCT_SIZE.
struct Reservoir {
    sample_point_world_position: vec3<f32>,
    weight_sum: f32,
    radiance: vec3<f32>,
    confidence_weight: f32,
    sample_point_world_normal: vec3<f32>,
    unbiased_contribution_weight: f32,
}
