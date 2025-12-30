#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_pbr::utils::rand_f
#import bevy_render::maths::{orthonormalize, PI}
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_brdf, evaluate_specular_brdf}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, pixel_dissimilar}
#import bevy_solari::sampling::{sample_random_light, random_emissive_light_pdf, sample_ggx_vndf, ggx_vndf_pdf, balance_heuristic, power_heuristic}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, ResolvedRayHitFull, ResolvedMaterial, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::world_cache::{query_world_cache, get_cell_size, WORLD_CACHE_CELL_LIFETIME}

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var<storage, read_write> gi_reservoirs_a: array<Reservoir>;
@group(1) @binding(7) var<storage, read> specular_reservoirs_a: array<Reservoir>;
@group(1) @binding(8) var<storage, read_write> specular_reservoirs_b: array<Reservoir>;
@group(1) @binding(9) var gbuffer: texture_2d<u32>;
@group(1) @binding(10) var depth_buffer: texture_depth_2d;
@group(1) @binding(11) var motion_vectors: texture_2d<f32>;
@group(1) @binding(12) var previous_gbuffer: texture_2d<u32>;
@group(1) @binding(13) var previous_depth_buffer: texture_depth_2d;
@group(1) @binding(14) var<uniform> view: View;
@group(1) @binding(15) var<uniform> previous_view: PreviousViewUniforms;
struct PushConstants { frame_index: u32, reset: u32 }
var<push_constant> constants: PushConstants;

const DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD: f32 = 0.4;
const TERMINATE_IN_WORLD_CACHE_THRESHOLD: f32 = 0.03;
const CONFIDENCE_WEIGHT_CAP = 8.0;

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

    let wo_unnormalized = view.world_position - surface.world_position;
    let wo = normalize(wo_unnormalized);

    var radiance: vec3<f32>;
    var wi: vec3<f32>;
    if surface.material.roughness > DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD {
        // Surface is very rough, reuse the ReSTIR GI reservoir
        let gi_reservoir = gi_reservoirs_a[pixel_index];
        wi = normalize(gi_reservoir.sample_point_world_position - surface.world_position);
        radiance = gi_reservoir.radiance * gi_reservoir.unbiased_contribution_weight;

        let brdf = evaluate_specular_brdf(surface.world_normal, wo, wi, surface.material.base_color, surface.material.metallic,
            surface.material.reflectance, surface.material.perceptual_roughness, surface.material.roughness);
        let cos_theta = saturate(dot(wi, surface.world_normal));
        radiance *= brdf * cos_theta;
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

        // https://d1qx31qr3h6wln.cloudfront.net/publications/mueller21realtime.pdf#subsection.3.4, equation (4)
        let cos_theta = saturate(dot(wo, surface.world_normal));
        var a0 = dot(wo_unnormalized, wo_unnormalized) / (4.0 * PI * cos_theta);
        a0 *= TERMINATE_IN_WORLD_CACHE_THRESHOLD;

        var initial_reservoir = trace_glossy_path(surface.world_position, wi, pdf, a0, &rng);
        initial_reservoir.unbiased_contribution_weight = 1.0 / pdf;

        let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal);

        let merge_result = merge_reservoirs(initial_reservoir, surface.world_position, surface.world_normal, surface.material,
            temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.material, wo, &rng);
        var combined_reservoir = merge_result.merged_reservoir;

        specular_reservoirs_b[pixel_index] = combined_reservoir;

        radiance = combined_reservoir.radiance * combined_reservoir.unbiased_contribution_weight;
    }

    let brdf = evaluate_specular_brdf(surface.world_normal, wo, wi, surface.material.base_color, surface.material.metallic,
        surface.material.reflectance, surface.material.perceptual_roughness, surface.material.roughness);
    let cos_theta = saturate(dot(wi, surface.world_normal));
    radiance *= brdf * cos_theta * view.exposure;

    var pixel_color = textureLoad(view_output, global_id.xy);
    pixel_color += vec4(radiance * view.exposure, 0.0);
    textureStore(view_output, global_id.xy, pixel_color);

#ifdef VISUALIZE_WORLD_CACHE
    textureStore(view_output, global_id.xy, vec4(query_world_cache(surface.world_position, surface.world_normal, view.world_position, WORLD_CACHE_CELL_LIFETIME, &rng) * view.exposure, 1.0));
#endif
}

fn trace_glossy_path(initial_ray_origin: vec3<f32>, initial_wi: vec3<f32>, initial_p_bounce: f32, a0: f32, rng: ptr<function, u32>) -> Reservoir {
    var ray_origin = initial_ray_origin;
    var wi = initial_wi;
    var p_bounce = initial_p_bounce;
    var surface_perfectly_specular = false;
    var path_spread = 0.0;

    var reservoir = empty_reservoir();
    reservoir.confidence_weight = 1.0;

    // Trace up to three bounces, getting the net throughput from them
    var throughput = vec3(1.0);
    for (var i = 0u; i < 3u; i += 1u) {
        // Trace ray
        let ray = trace_ray(ray_origin, wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { break; }
        let ray_hit = resolve_ray_hit_full(ray);

        if i == 0u {
            reservoir.sample_point_world_position = ray_hit.world_position;
            reservoir.sample_point_world_normal = ray_hit.world_normal;
        }

        let TBN = calculate_tbn_mikktspace(ray_hit.world_normal, ray_hit.world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];

        let wo = -wi;
        let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));

        // Add emissive contribution (but not on the first bounce, since ReSTIR DI handles that)
        if i != 0u {
            reservoir.radiance += throughput * emissive_mis_weight(p_bounce, ray_hit, surface_perfectly_specular) * ray_hit.material.emissive;
        }

        // Should not perform NEE for mirror-like surfaces
        surface_perfectly_specular = ray_hit.material.roughness <= 0.001 && ray_hit.material.metallic > 0.9999;

        // https://d1qx31qr3h6wln.cloudfront.net/publications/mueller21realtime.pdf#subsection.3.4, equation (3)
        path_spread += sqrt((ray.t * ray.t) / (p_bounce * wo_tangent.z));

        if path_spread * path_spread > a0 * get_cell_size(ray_hit.world_position, view.world_position) {
            // Path spread is wide enough, terminate path in the world cache
            let diffuse_brdf = ray_hit.material.base_color / PI;
            reservoir.radiance += throughput * diffuse_brdf * query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, WORLD_CACHE_CELL_LIFETIME, rng);
            break;
        } else if !surface_perfectly_specular {
            // Sample direct lighting (NEE)
            let direct_lighting = sample_random_light(ray_hit.world_position, ray_hit.world_normal, rng);
            let direct_lighting_brdf = evaluate_brdf(ray_hit.world_normal, wo, direct_lighting.wi, ray_hit.material);
            let mis_weight = nee_mis_weight(direct_lighting.inverse_pdf, direct_lighting.brdf_rays_can_hit, wo_tangent, direct_lighting.wi, ray_hit, TBN);
            reservoir.radiance += throughput * mis_weight * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_lighting_brdf;
        }

        // Sample new ray direction from the GGX BRDF for next bounce
        let wi_tangent = sample_ggx_vndf(wo_tangent, ray_hit.material.roughness, rng);
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;
        ray_origin = ray_hit.world_position;

        // Update throughput for next bounce
        p_bounce = ggx_vndf_pdf(wo_tangent, wi_tangent, ray_hit.material.roughness);
        let brdf = evaluate_brdf(N, wo, wi, ray_hit.material);
        let cos_theta = saturate(dot(wi, N));
        throughput *= (brdf * cos_theta) / p_bounce;
    }

    return reservoir;
}

fn emissive_mis_weight(p_bounce: f32, ray_hit: ResolvedRayHitFull, previous_surface_perfectly_specular: bool) -> f32 {
    if previous_surface_perfectly_specular { return 1.0; }

    let p_light = random_emissive_light_pdf(ray_hit);
    return power_heuristic(p_bounce, p_light);
}

fn nee_mis_weight(inverse_p_light: f32, brdf_rays_can_hit: bool, wo_tangent: vec3<f32>, wi: vec3<f32>, ray_hit: ResolvedRayHitFull, TBN: mat3x3<f32>) -> f32 {
    if !brdf_rays_can_hit {
        return 1.0;
    }

    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];
    let wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));

    let p_light = 1.0 / inverse_p_light;
    let p_bounce = ggx_vndf_pdf(wo_tangent, wi_tangent, ray_hit.material.roughness);
    return power_heuristic(p_light, p_bounce);
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));

    // Check if the current pixel was off screen during the previous frame (current pixel is newly visible),
    // or if all temporal history should assumed to be invalid
    if any(temporal_pixel_id_float < vec2(0.0)) || any(temporal_pixel_id_float >= view.main_pass_viewport.zw) || bool(constants.reset) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), ResolvedMaterial(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0));
    }

    let temporal_pixel_id = vec2<u32>(temporal_pixel_id_float);

    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, temporal_pixel_id, 0), temporal_depth, temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), ResolvedMaterial(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0));
    }

    let temporal_pixel_index = temporal_pixel_id.x + temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    var temporal_reservoir = specular_reservoirs_a[temporal_pixel_index];

    temporal_reservoir.confidence_weight = min(temporal_reservoir.confidence_weight, CONFIDENCE_WEIGHT_CAP);

    return NeighborInfo(temporal_reservoir, temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material);
}

struct NeighborInfo {
    reservoir: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn jacobian(
    new_world_position: vec3<f32>,
    original_world_position: vec3<f32>,
    sample_point_world_position: vec3<f32>,
    sample_point_world_normal: vec3<f32>,
) -> f32 {
    let r = new_world_position - sample_point_world_position;
    let q = original_world_position - sample_point_world_position;
    let rl = length(r);
    let ql = length(q);
    let phi_r = saturate(dot(r / rl, sample_point_world_normal));
    let phi_q = saturate(dot(q / ql, sample_point_world_normal));
    let jacobian = (phi_r * ql * ql) / (phi_q * rl * rl);
    return select(jacobian, 0.0, isinf(jacobian) || isnan(jacobian));
}

fn isinf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

fn isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
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

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
    );
}

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_radiance: vec3<f32>,
}

fn merge_reservoirs(
    canonical_reservoir: Reservoir,
    canonical_world_position: vec3<f32>,
    canonical_world_normal: vec3<f32>,
    canonical_material: ResolvedMaterial,
    other_reservoir: Reservoir,
    other_world_position: vec3<f32>,
    other_world_normal: vec3<f32>,
    other_material: ResolvedMaterial,
    wo: vec3<f32>,
    rng: ptr<function, u32>,
) -> ReservoirMergeResult {
    // Target functions for resampling and MIS
    var wi = normalize(canonical_reservoir.sample_point_world_position - canonical_world_position);
    let canonical_sample_radiance = canonical_reservoir.radiance
        * saturate(dot(wi, canonical_world_normal))
        * evaluate_specular_brdf(canonical_world_normal, wo, wi, canonical_material.base_color, canonical_material.metallic,
            canonical_material.reflectance, canonical_material.perceptual_roughness, canonical_material.roughness);
    let canonical_target_function_canonical_sample = luminance(canonical_sample_radiance);

    wi = normalize(other_reservoir.sample_point_world_position - canonical_world_position);
    let other_sample_radiance = other_reservoir.radiance
        * saturate(dot(wi, canonical_world_normal))
        * evaluate_specular_brdf(canonical_world_normal, wo, wi, canonical_material.base_color, canonical_material.metallic,
            canonical_material.reflectance, canonical_material.perceptual_roughness, canonical_material.roughness);
    let canonical_target_function_other_sample = luminance(other_sample_radiance);

    // Extra target functions for MIS
    wi = normalize(canonical_reservoir.sample_point_world_position - other_world_position);
    let other_target_function_canonical_sample = luminance(
        canonical_reservoir.radiance
        * saturate(dot(wi, other_world_normal))
        * evaluate_specular_brdf(other_world_normal, wo, wi, other_material.base_color, other_material.metallic,
            other_material.reflectance, other_material.perceptual_roughness, other_material.roughness)
    );

    wi = normalize(other_reservoir.sample_point_world_position - other_world_position);
    let other_target_function_other_sample = luminance(
        other_reservoir.radiance
        * saturate(dot(wi, other_world_normal))
        * evaluate_specular_brdf(other_world_normal, wo, wi, other_material.base_color, other_material.metallic,
            other_material.reflectance, other_material.perceptual_roughness, other_material.roughness)
    );

    // Jacobians for resampling and MIS
    let canonical_target_function_other_sample_jacobian = jacobian(
        canonical_world_position,
        other_world_position,
        other_reservoir.sample_point_world_position,
        other_reservoir.sample_point_world_normal
    );
    let other_target_function_canonical_sample_jacobian = jacobian(
        other_world_position,
        canonical_world_position,
        canonical_reservoir.sample_point_world_position,
        canonical_reservoir.sample_point_world_normal
    );

    // Don't merge samples with huge jacobians, as it explodes the variance
    if canonical_target_function_other_sample_jacobian > 1.2 {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_radiance);
    }

    // Resampling weight for canonical sample
    let canonical_sample_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_target_function_canonical_sample,
        other_reservoir.confidence_weight * other_target_function_canonical_sample * other_target_function_canonical_sample_jacobian,
    );
    let canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_target_function_canonical_sample * canonical_reservoir.unbiased_contribution_weight;

    // Resampling weight for other sample
    let other_sample_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_target_function_other_sample,
        canonical_reservoir.confidence_weight * canonical_target_function_other_sample * canonical_target_function_other_sample_jacobian,
    );
    let other_sample_resampling_weight = other_sample_mis_weight * canonical_target_function_other_sample * other_reservoir.unbiased_contribution_weight * canonical_target_function_other_sample_jacobian;

    // Perform resampling
    var combined_reservoir = empty_reservoir();
    combined_reservoir.confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    combined_reservoir.weight_sum = canonical_sample_resampling_weight + other_sample_resampling_weight;

    if rand_f(rng) < other_sample_resampling_weight / combined_reservoir.weight_sum {
        combined_reservoir.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = other_reservoir.radiance;

        let inverse_target_function = select(0.0, 1.0 / canonical_target_function_other_sample, canonical_target_function_other_sample > 0.0);
        combined_reservoir.unbiased_contribution_weight = combined_reservoir.weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_sample_radiance);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;

        let inverse_target_function = select(0.0, 1.0 / canonical_target_function_canonical_sample, canonical_target_function_canonical_sample > 0.0);
        combined_reservoir.unbiased_contribution_weight = combined_reservoir.weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_radiance);
    }
}
