enable wgpu_ray_query;

#define_import_path bevy_solari::initial_path

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_range_u}
#import bevy_render::maths::PI
#import bevy_render::utils::octahedral_encode
#import bevy_solari::brdf::{brdf_pdf, diffuse_brdf_pdf, evaluate_and_sample_brdf, evaluate_and_sample_diffuse_brdf, evaluate_brdf, evaluate_diffuse_brdf, F_AB}
#import bevy_solari::presample_light_tiles::unpack_resolved_light_sample
#import bevy_solari::brdf::EvaluateAndSampleBrdfResult
#import bevy_solari::realtime_bindings::{empty_reservoir, light_tile_resolved_samples, light_tile_samples, Reservoir, constants, view}
#import bevy_solari::sampling::{calculate_resolved_light_contribution, LightSample, NULL_LIGHT_ID, power_heuristic, trace_visibility}
#import bevy_solari::scene_bindings::{light_sources, RAY_T_MAX, RAY_T_MIN, resolve_ray_hit_full, ResolvedMaterial, ResolvedRayHitFull, trace_ray}
#import bevy_solari::world_cache::{get_cell_size, query_world_cache, WORLD_CACHE_CELL_LIFETIME}

const RECONNECTION_FOOTPRINT_KAPPA = 0.02;
const RECONNECTION_ROUGHNESS_MIN = 0.6;
const RECONNECTION_RELAX_DISTANCE = 1.0;

const CACHE_TERMINATION_MIN_SOLID_ANGLE = PI;

struct InitialSamplingResult {
    reservoir: Reservoir,
    non_resampled_radiance: vec3<f32>,
}

// Path vertices use the following convention: x0 = camera, x1 = primary ray hit (the G-buffer
// surface), x2 = first BRDF-sampled hit (the reconnection vertex).
struct PathState {
    ray_origin: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    material: ResolvedMaterial,
    // Throughput past x1, excluding brdf*cos at x1
    throughput_past_first_hit: vec3<f32>,
    // Reconnection vertex x2, the first BRDF-sampled hit shared by every length >= 2 candidate
    x2_position: vec3<f32>,
    x2_normal: vec3<f32>,
    // If false, candidates built on x2 are shaded directly into non_resampled_radiance instead of
    // published to the reservoir
    x2_reusable: bool,
    // brdf*cos at x1 for the direction toward x2
    x1_brdf: vec3<f32>,
    // False when this path's x1 is not the pixel's own G-buffer surface, so nothing about it may be
    // published to the reservoir. See `generate_initial_reservoir`.
    resampling_allowed: bool,
    // Sample only the diffuse lobe at x1. A property of the primary vertex alone — every deeper vertex
    // samples normally, or the path would lose all specular global illumination.
    force_diffuse_at_x1: bool,
}

// `wo` is passed in rather than derived from the camera, and `path_throughput` scales everything this
// path produces.
//
// Both exist for the mirror handoff. When the guide pass has already walked a pixel's reflection chain,
// ReSTIR starts here at the chain's end instead of tracing the same bounces again — so x1 is a surface
// somewhere else in the world, looked at from the last mirror rather than from the camera, and
// everything it contributes has to be multiplied by the chain's reflectance on the way back to the eye.
//
// `resampling_allowed` must be false in that case. A reservoir is keyed to the pixel's G-buffer surface
// — temporal reprojection, the spatial neighbour's similarity test and the shading in
// `spatial_and_shade` all assume it — and a sample generated at a surface metres away would be
// resampled as though it belonged to the mirror. Costing nothing today: a delta lobe at x1 already
// fails `reconnection_reusable`, so mirror pixels have always shaded directly instead of publishing.
fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, geometric_normal: vec3<f32>, material: ResolvedMaterial, wo: vec3<f32>, path_throughput: vec3<f32>, resampling_allowed: bool, force_diffuse_at_x1: bool, bounce_offset: u32, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> InitialSamplingResult {
    var reservoir = empty_reservoir();
    reservoir.confidence_weight = 1.0;

    var non_resampled_radiance = vec3(0.0);
    var weight_sum = 0.0;
    var selected_target_function = 0.0;

    let primary_NdotV = max(dot(world_normal, wo), 0.0001);
    let primary_F_ab = F_AB(material.perceptual_roughness, primary_NdotV);

    var path: PathState;
    // Offset along the *geometric* normal, not the shading one. They are the same for a G-buffer
    // surface, which has only one normal, but the chain's terminus is a traced hit where a normal map
    // can tilt the shading normal far enough off the triangle to push the origin below it — and a
    // self-intersecting shadow ray reads as a black speckle that looks like noise rather than a bug.
    // Every deeper vertex already offsets this way.
    path.ray_origin = world_position + (geometric_normal * RAY_T_MIN);
    path.normal = world_normal;
    path.wo = wo;
    path.material = material;
    path.throughput_past_first_hit = vec3(1.0);
    path.x2_position = vec3(0.0);
    path.x2_normal = vec3(0.0);
    path.x2_reusable = false;
    path.x1_brdf = vec3(0.0);
    path.resampling_allowed = resampling_allowed;
    path.force_diffuse_at_x1 = force_diffuse_at_x1;

    // Indexed from zero so that `bounce == 0u` keeps meaning "this path's own first vertex", which is
    // what the reconnection bookkeeping is built around. `bounce_offset` shortens the path instead:
    // a reflected path picks up where the chain left off, so the chain's bounces come out of the same
    // budget rather than being free.
    let max_bounces = max(constants.max_bounces, 1u);
    let bounce_budget = max_bounces - min(bounce_offset, max_bounces - 1u);
    for (var bounce = 0u; bounce < bounce_budget; bounce++) {
        let NdotV = max(dot(path.normal, path.wo), 0.0001);
        let F_ab = F_AB(path.material.perceptual_roughness, NdotV);

        // Stochastic NEE, with probability proportional to how diffuse the vertex is. Mirror-like
        // metals have too narrow a lobe for NEE to help, so mostly skip it there and let
        // BRDF-sampled emissive do the work. Pure dielectrics always run NEE.
        // A forced-diffuse vertex always wants next-event estimation: the lobe being sampled is the
        // diffuse one, whatever the surface's metalness would otherwise imply about its specular.
        var p_nee = mix(1.0, path.material.perceptual_roughness, path.material.metallic);
        if bounce == 0u && force_diffuse_at_x1 { p_nee = 1.0; }
        // Only a path that actually starts at the camera's surface gets the primary sample count.
        let di_samples = select(constants.secondary_di_samples, constants.primary_di_samples, bounce == 0u && bounce_offset == 0u);
        generate_nee_candidate(&reservoir, &weight_sum, &selected_target_function, &non_resampled_radiance,
            path, F_ab, p_nee, di_samples, workgroup_id, bounce, rng);

        // Sample the BRDF and trace the next ray
        var next_bounce: EvaluateAndSampleBrdfResult;
        if bounce == 0u && force_diffuse_at_x1 {
            next_bounce = evaluate_and_sample_diffuse_brdf(path.wo, path.normal, path.material, F_ab, rng);
        } else {
            next_bounce = evaluate_and_sample_brdf(path.wo, path.normal, path.material, F_ab, rng);
        }
        if next_bounce.pdf == 0.0 { break; }
        let ray = trace_ray(path.ray_origin, next_bounce.wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { break; }
        let ray_hit = resolve_ray_hit_full(ray);
        let p_brdf = next_bounce.pdf;

        // Capture x2, the first BRDF-sampled hit
        if bounce == 0u {
            path.x2_position = ray_hit.world_position;
            path.x2_normal = ray_hit.world_normal;

            // Diffuse-only when the lobe was forced, or the specular half would be counted once here
            // and again in the chain's contribution.
            if force_diffuse_at_x1 {
                path.x1_brdf = evaluate_diffuse_brdf(wo, next_bounce.wi, world_normal, material, primary_F_ab);
            } else {
                path.x1_brdf = evaluate_brdf(wo, next_bounce.wi, world_normal, material, primary_F_ab);
            }

            path.x2_reusable = resampling_allowed && reconnection_reusable(ray.t, p_brdf, next_bounce.wi, next_bounce.diffuse_selected, ray_hit, world_position, material.perceptual_roughness, primary_NdotV);

            // The primary brdf*cos is applied at shade time, so divide it out of next_bounce.throughput
            // to leave 1/pdf (or 1/specular_weight for mirrors, avoiding the 1/INF = 0 that would kill
            // mirror GI).
            path.throughput_past_first_hit *= next_bounce.throughput / max(path.x1_brdf, vec3(0.0001));
        } else {
            // Later bounces keep the full brdf*cos/pdf for L_at_reconnection.
            path.throughput_past_first_hit *= next_bounce.throughput;
        }

        // Resample emissive hits
        if any(ray_hit.material.emissive > vec3(0.0)) && dot(ray_hit.world_normal, -next_bounce.wi) > 0.0 {
            generate_emissive_candidate(&reservoir, &weight_sum, &selected_target_function, &non_resampled_radiance,
                path, ray_hit, next_bounce.wi, p_brdf, ray.t, p_nee, di_samples, bounce, rng);
        }

        // Try terminating into the world cache
        if terminate_into_cache(&reservoir, &weight_sum, &selected_target_function, &non_resampled_radiance, path, ray_hit, ray.t, p_brdf, bounce, bounce_budget, rng) {
            break;
        }

        // Advance to the next vertex
        path.ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
        path.normal = ray_hit.world_normal;
        path.wo = -next_bounce.wi;
        path.material = ray_hit.material;

        // Russian roulette for early termination
        if bounce > 0u {
            // throughput_past_first_hit has the primary brdf*cos divided out (so it can be re-applied at shade
            // time), which inflates it. Multiply x1_brdf back in to get the true energy-bounded path
            // throughput, which is the correct quantity for the RR survival probability.
            let full_throughput = path.throughput_past_first_hit * max(path.x1_brdf, vec3(0.0001));
            let rr = saturate(luminance(full_throughput));
            if rand_f(rng) >= rr { break; }
            path.throughput_past_first_hit /= rr;
        }
    }

    if selected_target_function > 0.0 {
        reservoir.unbiased_contribution_weight = weight_sum / selected_target_function;
    }

    // x1's own emissive. Normally `spatial_and_shade` adds it, reading it from the G-buffer — but under
    // the handoff x1 is not the G-buffer surface, so a mirror pointed at a light would otherwise show
    // everything about that light except the light. It goes into the non-resampled radiance, which is
    // inside the exposure multiply where emissive belongs.
    if !resampling_allowed {
        non_resampled_radiance += material.emissive;
    }

    // Everything above was computed at the chain's end, where the light actually is. This is the walk
    // back to the eye: one factor per mirror the chain bounced off.
    return InitialSamplingResult(reservoir, non_resampled_radiance * path_throughput);
}

fn generate_nee_candidate(
    reservoir: ptr<function, Reservoir>,
    weight_sum: ptr<function, f32>,
    selected_target_function: ptr<function, f32>,
    non_resampled_radiance: ptr<function, vec3<f32>>,
    path: PathState,
    F_ab: vec2<f32>,
    p_nee: f32,
    di_samples: u32,
    workgroup_id: vec2<u32>,
    bounce: u32,
    rng: ptr<function, u32>,
) {
    if rand_f(rng) >= p_nee { return; }

    let di = sample_light_ris(path.ray_origin, path.normal, path.wo, path.material, F_ab, di_samples, workgroup_id, bounce, rng);
    let di_target_function = luminance(di.brdf_radiance);
    if di_target_function <= 0.0 { return; }

    // MIS against the BRDF strategy. RIS over N candidates makes the effective NEE pdf at the
    // winner roughly N * light_pdf(winner), so scale by p_nee for the stochastic gate.
    var nee_mis_weight = 1.0;
    if di.brdf_rays_can_hit && di.inverse_solid_angle_pdf > 0.0 {
        let p_nee_strategy = f32(di_samples) * (1.0 / di.inverse_solid_angle_pdf) * p_nee;
        // The BRDF strategy this is weighed against picks diffuse with probability one at a forced
        // vertex, so the mixture pdf `brdf_pdf` returns would be the wrong denominator.
        var p_brdf_at_nee = brdf_pdf(path.wo, di.wi, path.normal, path.material, F_ab);
        if bounce == 0u && path.force_diffuse_at_x1 {
            p_brdf_at_nee = diffuse_brdf_pdf(di.wi, path.normal);
        }
        nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
    }

    if bounce == 0u && !path.resampling_allowed {
        // The one candidate that publishes regardless of `x2_reusable`, because at bounce 0 the sample
        // lives at x1 rather than at x2. Under the handoff x1 is not this pixel's surface, so the
        // reservoir is the wrong home for it and it is shaded straight in instead. `brdf_radiance`
        // already carries brdf*cos at x1.
        *non_resampled_radiance += di.brdf_radiance * di.unbiased_contribution_weight * nee_mis_weight / p_nee;
    } else if bounce == 0u {
        // Bounce 0: Candidate is the light sample, stored by reference and re-resolved each frame
        // nee_mis_weight goes into the target function since it gets recomputed per-pixel during reuse
        let target_function = di_target_function * nee_mis_weight;
        let resampling_weight = target_function * di.unbiased_contribution_weight / p_nee;

        *weight_sum += resampling_weight;
        if rand_f(rng) * (*weight_sum) < resampling_weight {
            (*reservoir).light_sample = di.light_sample;
            *selected_target_function = target_function;
        }
    } else {
        // Deeper bounces: Candidate is the reconnection radiance at x2
        let L_at_reconnection = path.throughput_past_first_hit * di.brdf_radiance * di.unbiased_contribution_weight * nee_mis_weight / p_nee;
        if !path.x2_reusable {
            // x1 -> x2 not reuse-safe: shade directly at this pixel instead of publishing.
            *non_resampled_radiance += path.x1_brdf * L_at_reconnection;
        } else {
            let target_function = luminance(path.x1_brdf * L_at_reconnection);
            let resampling_weight = target_function;

            *weight_sum += resampling_weight;
            if rand_f(rng) * (*weight_sum) < resampling_weight {
                (*reservoir).light_sample = LightSample(NULL_LIGHT_ID, 0u);
                (*reservoir).sample_point_world_position = path.x2_position;
                (*reservoir).sample_point_world_normal = octahedral_encode(path.x2_normal);
                (*reservoir).radiance = L_at_reconnection;
                *selected_target_function = target_function;
            }
        }
    }
}

struct DiSample {
    unbiased_contribution_weight: f32,
    light_sample: LightSample,
    wi: vec3<f32>,
    brdf_radiance: vec3<f32>,
    inverse_solid_angle_pdf: f32,
    brdf_rays_can_hit: bool,
}

fn sample_light_ris(ray_origin: vec3<f32>, normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>, di_samples: u32, workgroup_id: vec2<u32>, bounce: u32, rng: ptr<function, u32>) -> DiSample {
    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y + bounce;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var weight_sum = 0.0;
    var selected_target_function = 0.0;
    var selected_tile_sample = 0u;
    var selected_world_position = vec4(0.0);
    var selected_wi = vec3(0.0);
    var selected_brdf_radiance = vec3(0.0);
    var selected_inverse_solid_angle_pdf = 0.0;
    var selected_brdf_rays_can_hit = false;
    let mis_weight = 1.0 / f32(di_samples);
    for (var i = 0u; i < di_samples; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, normal);
        let brdf_current = evaluate_brdf(wo, light_contribution.wi, normal, material, F_ab);
        let brdf_radiance = brdf_current * light_contribution.radiance;

        let target_function = luminance(brdf_radiance);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if rand_f(rng) * weight_sum < resampling_weight {
            selected_target_function = target_function;
            selected_tile_sample = tile_sample;
            selected_world_position = resolved_light_sample.world_position;
            selected_wi = light_contribution.wi;
            selected_inverse_solid_angle_pdf = light_contribution.inverse_solid_angle_pdf;
            selected_brdf_rays_can_hit = light_contribution.brdf_rays_can_hit;
            selected_brdf_radiance = brdf_radiance;
        }
    }

    var unbiased_contribution_weight = 0.0;
    if selected_target_function > 0.0 {
        unbiased_contribution_weight = weight_sum / selected_target_function;
        unbiased_contribution_weight *= trace_visibility(ray_origin, selected_world_position);
    }

    return DiSample(unbiased_contribution_weight, light_tile_samples[selected_tile_sample], selected_wi, selected_brdf_radiance, selected_inverse_solid_angle_pdf, selected_brdf_rays_can_hit);
}

fn generate_emissive_candidate(
    reservoir: ptr<function, Reservoir>,
    weight_sum: ptr<function, f32>,
    selected_target_function: ptr<function, f32>,
    non_resampled_radiance: ptr<function, vec3<f32>>,
    path: PathState,
    ray_hit: ResolvedRayHitFull,
    wi: vec3<f32>,
    p_brdf: f32,
    ray_t: f32,
    p_nee: f32,
    di_samples: u32,
    bounce: u32,
    rng: ptr<function, u32>,
) {
    let NdotV_hit = max(dot(ray_hit.world_normal, -wi), 0.0001);
    let light_count = arrayLength(&light_sources);
    let area_pdf = 1.0 / (f32(light_count) * f32(ray_hit.triangle_count) * ray_hit.triangle_area);
    let p_light = area_pdf * ray_t * ray_t / NdotV_hit;
    let emissive_mis_weight = power_heuristic(p_brdf, p_light * p_nee * f32(di_samples));

    if !path.x2_reusable {
        // x1 -> x2 not reuse-safe (mirror/sharp lobe or failed gate): shade directly at this pixel
        // instead of publishing, since a reuse shift would waste it or make a firefly. Mirror lobes
        // always land here (p_brdf = INF, footprint 0), where emissive_mis_weight is 1.
        *non_resampled_radiance += path.x1_brdf * path.throughput_past_first_hit * ray_hit.material.emissive * emissive_mis_weight;
        return;
    }

    if bounce == 0u {
        // Bounce 0: Candidate is the emissive hit
        let target_function = luminance(path.x1_brdf * ray_hit.material.emissive) * emissive_mis_weight;
        let resampling_weight = luminance(path.x1_brdf * path.throughput_past_first_hit * ray_hit.material.emissive) * emissive_mis_weight;

        *weight_sum += resampling_weight;
        if rand_f(rng) * (*weight_sum) < resampling_weight {
            (*reservoir).light_sample = LightSample(NULL_LIGHT_ID, bitcast<u32>(area_pdf));
            (*reservoir).sample_point_world_position = path.x2_position;
            (*reservoir).sample_point_world_normal = octahedral_encode(path.x2_normal);
            (*reservoir).radiance = ray_hit.material.emissive;
            *selected_target_function = target_function;
        }
    } else {
        // Deeper bounces: Candidate is the reconnection radiance at x2
        let emissive_L_at_reconnection = path.throughput_past_first_hit * ray_hit.material.emissive * emissive_mis_weight;
        let target_function = luminance(path.x1_brdf * emissive_L_at_reconnection);
        let resampling_weight = target_function;

        *weight_sum += resampling_weight;
        if rand_f(rng) * (*weight_sum) < resampling_weight {
            (*reservoir).light_sample = LightSample(NULL_LIGHT_ID, 0u);
            (*reservoir).sample_point_world_position = path.x2_position;
            (*reservoir).sample_point_world_normal = octahedral_encode(path.x2_normal);
            (*reservoir).radiance = emissive_L_at_reconnection;
            *selected_target_function = target_function;
        }
    }
}

fn terminate_into_cache(
    reservoir: ptr<function, Reservoir>,
    weight_sum: ptr<function, f32>,
    selected_target_function: ptr<function, f32>,
    non_resampled_radiance: ptr<function, vec3<f32>>,
    path: PathState,
    ray_hit: ResolvedRayHitFull,
    ray_t: f32,
    p_brdf: f32,
    bounce: u32,
    bounce_budget: u32,
    rng: ptr<function, u32>,
) -> bool {
    // Only terminate into the world cache when the bounce was from a wide-enough BRDF sample
    // because the cache is less noisy than continuing the path for rough surfaces,
    // but less accurate for smooth surfaces
    let lobe_solid_angle = 1.0 / p_brdf;
    let broad_enough_to_terminate = lobe_solid_angle >= CACHE_TERMINATION_MIN_SOLID_ANGLE;
    // This path's own budget, not the global cap. A reflected path starts partway through the bounce
    // allowance, so its last vertex is not `max_bounces - 1`; comparing against the cap meant such a
    // path never force-terminated and silently dropped the cached indirect light it should collect.
    let forced_terminate = bounce == bounce_budget - 1u;
    if !(broad_enough_to_terminate || forced_terminate) { return false; }

    // Only use the cache when the ray cleared the cache cell (diagonal = sqrt(3) * cell_size). Short
    // rays land in a cell that may straddle occluders and leak light through corners.
    var rng_copy = *rng;
    let world_cache_cell_size = get_cell_size(ray_hit.world_position, view.world_position, ray_t, &rng_copy);
    if ray_t <= sqrt(3.0) * world_cache_cell_size { return false; }

    let cached_radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, ray_t, WORLD_CACHE_CELL_LIFETIME, rng);

    let cache_outgoing = (ray_hit.material.base_color / PI) * cached_radiance;
    let cache_L_at_reconnection = path.throughput_past_first_hit * cache_outgoing;
    if !path.x2_reusable {
        *non_resampled_radiance += path.x1_brdf * cache_L_at_reconnection;
        return true;
    }

    let target_function = luminance(path.x1_brdf * cache_L_at_reconnection);
    let resampling_weight = target_function;
    *weight_sum += resampling_weight;
    if rand_f(rng) * (*weight_sum) < resampling_weight {
        (*reservoir).light_sample = LightSample(NULL_LIGHT_ID, 0u);
        (*reservoir).sample_point_world_position = path.x2_position;
        (*reservoir).sample_point_world_normal = octahedral_encode(path.x2_normal);
        (*reservoir).radiance = cache_L_at_reconnection;
        *selected_target_function = target_function;
    }

    return true;
}

// ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing
// Section 4 (sorta)
// https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced.pdf
fn reconnection_reusable(ray_t: f32, p_brdf: f32, wi: vec3<f32>, diffuse_selected: bool, ray_hit: ResolvedRayHitFull, world_position: vec3<f32>, x1_perceptual_roughness: f32, primary_NdotV: f32) -> bool {
    // ray_footprint = t^2 / (p_brdf * cos_x2) is the area a sample represents at x2. It goes to 0 for
    // mirror lobes (p_brdf = INF) and shrinks for sharp lobes or short segments. Compared against a
    // uniform 1/(4*PI) primary footprint, so the test trades roughness against distance.
    let cos_x2 = max(dot(ray_hit.world_normal, -wi), 0.0001);
    let ray_footprint = (ray_t * ray_t) / (p_brdf * cos_x2);
    let primary_dist = length(view.world_position - world_position);
    let primary_footprint = 4.0 * PI * primary_dist * primary_dist / primary_NdotV;
    let footprint_ok = ray_footprint >= (RECONNECTION_FOOTPRINT_KAPPA / 100.0) * primary_footprint;

    // Roughness floor at x1, only for specular lobes (a diffuse bounce is always rough). Guards
    // low-roughness specular lobes that resample with poorly-conditioned MIS/jacobian. The footprint
    // test alone is too permissive here.
    let x1_lobe_ok = diffuse_selected || x1_perceptual_roughness >= RECONNECTION_ROUGHNESS_MIN;

    // Guard at x2. A sharp reflector there makes the stored radiance view-dependent and wrong to
    // reuse from a neighbor's direction. The roughness floor relaxes with segment length: a distant
    // glossy x2 is seen by neighbors from nearly the same direction, so the view-dependence washes out.
    // Diffuse, rough, and emissive vertices are always reuse-safe.
    let x2_is_light = any(ray_hit.material.emissive > vec3(0.0));
    let x2_roughness = mix(1.0, ray_hit.material.perceptual_roughness, ray_hit.material.metallic);
    let x2_roughness_floor = RECONNECTION_ROUGHNESS_MIN * saturate(RECONNECTION_RELAX_DISTANCE / ray_t);
    let x2_end_ok = x2_is_light || x2_roughness >= x2_roughness_floor;

    return footprint_ok && x1_lobe_ok && x2_end_ok;
}
