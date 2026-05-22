// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{octahedral_decode, octahedral_encode, rand_f, rand_range_u, sample_disk}
#import bevy_render::maths::PI
#import bevy_solari::brdf::{evaluate_and_sample_brdf_uniform_diffuse, evaluate_brdf, EvaluateAndSampleBrdfResult, F_AB}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, permute_pixel, pixel_dissimilar}
#import bevy_solari::presample_light_tiles::unpack_resolved_light_sample
#import bevy_solari::realtime_bindings::{constants, depth_buffer, gbuffer, light_tile_resolved_samples, light_tile_samples, motion_vectors, previous_depth_buffer, previous_gbuffer, previous_view, reservoirs_a, reservoirs_b, Reservoir, view, view_output}
#import bevy_solari::sampling::{balance_heuristic, calculate_resolved_light_contribution, isnan, LightSample, NULL_LIGHT_ID, resolve_light_sample, ResolvedLightSample, trace_light_visibility}
#import bevy_solari::scene_bindings::{light_sources, LIGHT_NOT_PRESENT_THIS_FRAME, previous_frame_light_id_translations, RAY_T_MAX, RAY_T_MIN, resolve_ray_hit_full, ResolvedMaterial, trace_ray}
#import bevy_solari::world_cache::{query_world_cache, WORLD_CACHE_CELL_LIFETIME}

const INITIAL_DI_SAMPLES = 8u;
const GI_MAX_BOUNCES = 3u;
const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 8.0;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_b[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let wo = normalize(view.world_position - surface.world_position);
    let initial_reservoir = generate_initial_reservoir(surface.world_position, surface.world_normal, wo, surface.material, workgroup_id.xy, &rng);

    let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal);
    let merge_result = merge_reservoirs(initial_reservoir, surface.world_position, surface.world_normal, surface.material,
        temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.material, &rng);

    reservoirs_b[pixel_index] = merge_result.merged_reservoir;
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_a[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let spatial = load_spatial_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, &rng);

    let input_reservoir = reservoirs_b[pixel_index];
    let merge_result = merge_reservoirs(input_reservoir, surface.world_position, surface.world_normal, surface.material,
        spatial.reservoir, spatial.world_position, spatial.world_normal, spatial.material, &rng);
    var combined_reservoir = merge_result.merged_reservoir;

    reservoirs_a[pixel_index] = combined_reservoir;

    combined_reservoir.unbiased_contribution_weight *= trace_light_visibility(surface.world_position + (surface.world_normal * RAY_T_MIN), merge_result.selected_sample_world_position);

    let wo = normalize(view.world_position - surface.world_position);
    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);
    let brdf = evaluate_brdf(wo, merge_result.wi, surface.world_normal, surface.material, F_ab);

    var pixel_color = merge_result.selected_sample_radiance * combined_reservoir.unbiased_contribution_weight;
    pixel_color *= brdf;
    pixel_color += surface.material.emissive;
    pixel_color *= view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}

fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> Reservoir {
    let NdotV = max(dot(world_normal, wo), 0.0001);
    let F_ab = F_AB(material.perceptual_roughness, NdotV);
    let brdf_sample = evaluate_and_sample_brdf_uniform_diffuse(wo, world_normal, material, F_ab, rng);
    let initial_di_reservoir = generate_initial_di_reservoir(world_position, world_normal, wo, material, F_ab, workgroup_id, rng);
    let initial_gi_reservoir = generate_initial_gi_reservoir(world_position, world_normal, wo, material, F_ab, brdf_sample, workgroup_id, rng);
    return merge_initial_reservoirs(initial_di_reservoir, initial_gi_reservoir, rng);
}

struct InitialReservoirResult {
    reservoir: Reservoir,
    target_function: f32,
}

fn generate_initial_gi_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>, brdf_sample: EvaluateAndSampleBrdfResult, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> InitialReservoirResult {
    var reservoir = empty_reservoir();

    var ray_origin = world_position;
    var ray_origin_normal = world_normal;
    var current_brdf_sample = brdf_sample;
    // Accumulated brdf*cos/pdf from bounces after the first. The first bounce's
    // 1/pdf is stored in unbiased_contribution_weight and its brdf is re-evaluated
    // at resampling time against the world_position→sample_point direction.
    var throughput = vec3(1.0);
    // Radiance leaving the first hit toward world_position, summed across the chain.
    var accumulated_radiance = vec3(0.0);

    for (var bounce = 0u; bounce < GI_MAX_BOUNCES; bounce++) {
        let ray = trace_ray(ray_origin + (ray_origin_normal * RAY_T_MIN), current_brdf_sample.wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            return InitialReservoirResult(reservoir, 0.0);
        }
        let hit = resolve_ray_hit_full(ray);

        // The first hit is the reservoir's sample_point — the geometric reference
        // used by the spatial/temporal jacobian. Subsequent bounces only refine
        // the radiance leaving this point toward world_position.
        if bounce == 0u {
            reservoir.sample_point_world_position = hit.world_position;
            reservoir.sample_point_world_normal = octahedral_encode(hit.world_normal);
            reservoir.confidence_weight = 1.0;
        }

        if any(hit.material.emissive != vec3(0.0)) {
            accumulated_radiance += throughput * hit.material.emissive;
            break;
        }

        let is_last_bounce = bounce == GI_MAX_BOUNCES - 1u;
        if current_brdf_sample.diffuse_selected || is_last_bounce {
            let world_cache_radiance = query_world_cache(hit.world_position, hit.geometric_world_normal, view.world_position, ray.t, WORLD_CACHE_CELL_LIFETIME, rng);
            accumulated_radiance += throughput * world_cache_radiance * (hit.material.base_color / PI);
            break;
        }

        // Specular ray: continue the chain instead of terminating in the world cache.
        // Sample direct lighting at this intermediate hit to compensate for not
        // using the world cache (which already includes direct lighting).
        let next_wo = -current_brdf_sample.wi;
        let next_NdotV = max(dot(hit.world_normal, next_wo), 0.0001);
        let next_F_ab = F_AB(hit.material.perceptual_roughness, next_NdotV);

        let di = generate_initial_di_reservoir(hit.world_position, hit.world_normal, next_wo, hit.material, next_F_ab, workgroup_id, rng);
        if di.reservoir.light_sample.light_id != NULL_LIGHT_ID {
            let resolved = resolve_light_sample(di.reservoir.light_sample, light_sources[di.reservoir.light_sample.light_id >> 16u]);
            let light_contribution = calculate_resolved_light_contribution(resolved, hit.world_position, hit.world_normal);
            let di_brdf = evaluate_brdf(next_wo, light_contribution.wi, hit.world_normal, hit.material, next_F_ab);
            accumulated_radiance += throughput * light_contribution.radiance * di_brdf * di.reservoir.unbiased_contribution_weight;
        }

        current_brdf_sample = evaluate_and_sample_brdf_uniform_diffuse(next_wo, hit.world_normal, hit.material, next_F_ab, rng);
        if current_brdf_sample.pdf <= 0.0 {
            break;
        }
        throughput *= current_brdf_sample.throughput;
        ray_origin = hit.world_position;
        ray_origin_normal = hit.world_normal;

        // Russian roulette for early termination
        let p = luminance(throughput);
        if rand_f(rng) > p { break; }
        throughput /= p;
    }

    reservoir.radiance = accumulated_radiance;
    reservoir.unbiased_contribution_weight = 1.0 / brdf_sample.pdf;

    let wi = normalize(reservoir.sample_point_world_position - world_position);
    let target_function = luminance(reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab));
    return InitialReservoirResult(reservoir, target_function);
}

fn generate_initial_di_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> InitialReservoirResult {
    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var reservoir = empty_reservoir();
    var weight_sum = 0.0;
    let mis_weight = 1.0 / f32(INITIAL_DI_SAMPLES);

    var reservoir_target_function = 0.0;
    var light_sample_world_position = vec4(0.0);
    var selected_tile_sample = 0u;
    for (var i = 0u; i < INITIAL_DI_SAMPLES; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, world_position, world_normal);

        let target_function = luminance(light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab));
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if rand_f(rng) < resampling_weight / weight_sum {
            reservoir_target_function = target_function;
            light_sample_world_position = resolved_light_sample.world_position;
            selected_tile_sample = tile_sample;
        }
    }

    if reservoir_target_function != 0.0 {
        reservoir.light_sample = light_tile_samples[selected_tile_sample];
    }

    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let inverse_target_function = select(0.0, 1.0 / reservoir_target_function, reservoir_target_function > 0.0);
        reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        reservoir.unbiased_contribution_weight *= trace_light_visibility(world_position + (world_normal * RAY_T_MIN), light_sample_world_position);
    }

    reservoir.confidence_weight = 1.0;
    return InitialReservoirResult(reservoir, reservoir_target_function);
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));
    var point_temporal_pixel_id = vec2<u32>(temporal_pixel_id_float);

    if bool(constants.reset) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    if any(temporal_pixel_id_float < vec2(0.0)) || any(temporal_pixel_id_float >= view.main_pass_viewport.zw) {
        point_temporal_pixel_id = pixel_id;
    }

    let permuted_temporal_pixel_id = permute_pixel(point_temporal_pixel_id, constants.frame_index, view.main_pass_viewport.zw);
    var temporal = load_temporal_reservoir_inner(permuted_temporal_pixel_id, depth, world_position, world_normal);

    // Check if the light selected in the previous frame no longer exists in the current frame (e.g. entity despawned)
    if temporal.reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let previous_light_id = temporal.reservoir.light_sample.light_id >> 16u;
        let triangle_id = temporal.reservoir.light_sample.light_id & 0xFFFFu;
        let light_id = previous_frame_light_id_translations[previous_light_id];
        if light_id == LIGHT_NOT_PRESENT_THIS_FRAME {
            return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
        }
        temporal.reservoir.light_sample.light_id = (light_id << 16u) | triangle_id;
    }

    temporal.reservoir.confidence_weight = min(temporal.reservoir.confidence_weight, CONFIDENCE_WEIGHT_CAP);

    return temporal;
}

fn load_temporal_reservoir_inner(temporal_pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, temporal_pixel_id, 0), temporal_depth, temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    let temporal_pixel_index = temporal_pixel_id.x + temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    let temporal_reservoir = reservoirs_a[temporal_pixel_index];

    return NeighborInfo(temporal_reservoir, temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material);
}

fn load_spatial_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> NeighborInfo {
    for (var i = 0u; i < 5u; i++) {
        let spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view) {
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        let spatial_reservoir = reservoirs_b[spatial_pixel_index];
        return NeighborInfo(spatial_reservoir, spatial_surface.world_position, spatial_surface.world_normal, spatial_surface.material);
    }

    return NeighborInfo(empty_reservoir(), world_position, world_normal, empty_material());
}

fn get_neighbor_pixel_id(center_pixel_id: vec2<u32>, search_radius: f32, rng: ptr<function, u32>) -> vec2<u32> {
    var spatial_id = vec2<f32>(center_pixel_id) + sample_disk(search_radius, rng);
    spatial_id = clamp(spatial_id, vec2(0.0), view.main_pass_viewport.zw - 1.0);
    return vec2<u32>(spatial_id);
}

struct NeighborInfo {
    reservoir: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn empty_material() -> ResolvedMaterial {
    return ResolvedMaterial(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
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

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        vec2(0.0),
        LightSample(NULL_LIGHT_ID, 0u),
    );
}

fn merge_initial_reservoirs(
    di: InitialReservoirResult,
    gi: InitialReservoirResult,
    rng: ptr<function, u32>,
) -> Reservoir {
    let di_weight = di.target_function * di.reservoir.unbiased_contribution_weight;
    let gi_weight = gi.target_function * gi.reservoir.unbiased_contribution_weight;
    let weight_sum = di_weight + gi_weight;

    var merged = empty_reservoir();
    merged.confidence_weight = di.reservoir.confidence_weight + gi.reservoir.confidence_weight;

    if rand_f(rng) < gi_weight / weight_sum {
        merged.sample_point_world_position = gi.reservoir.sample_point_world_position;
        merged.sample_point_world_normal = gi.reservoir.sample_point_world_normal;
        merged.radiance = gi.reservoir.radiance;
        merged.light_sample = gi.reservoir.light_sample;
        merged.unbiased_contribution_weight = weight_sum * select(0.0, 1.0 / gi.target_function, gi.target_function > 0.0);
    } else {
        merged.sample_point_world_position = di.reservoir.sample_point_world_position;
        merged.sample_point_world_normal = di.reservoir.sample_point_world_normal;
        merged.radiance = di.reservoir.radiance;
        merged.light_sample = di.reservoir.light_sample;
        merged.unbiased_contribution_weight = weight_sum * select(0.0, 1.0 / di.target_function, di.target_function > 0.0);
    }

    return merged;
}

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_radiance: vec3<f32>,
    wi: vec3<f32>,
    // Resolved sample world position (vec4 with w=1 for surface points/area lights, w=0 for directional).
    selected_sample_world_position: vec4<f32>,
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
    rng: ptr<function, u32>,
) -> ReservoirMergeResult {
    var canonical_resolved: ResolvedLightSample;
    if canonical_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        canonical_resolved = resolve_light_sample(canonical_reservoir.light_sample, light_sources[canonical_reservoir.light_sample.light_id >> 16u]);
    }
    var other_resolved: ResolvedLightSample;
    if other_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        other_resolved = resolve_light_sample(other_reservoir.light_sample, light_sources[other_reservoir.light_sample.light_id >> 16u]);
    }

    let canonical_wo = normalize(view.world_position - canonical_world_position);
    let canonical_NdotV = max(dot(canonical_world_normal, canonical_wo), 0.0001);
    let canonical_F_ab = F_AB(canonical_material.perceptual_roughness, canonical_NdotV);
    let other_wo = normalize(view.world_position - other_world_position);
    let other_NdotV = max(dot(other_world_normal, other_wo), 0.0001);
    let other_F_ab = F_AB(other_material.perceptual_roughness, other_NdotV);

    // Contributions for resampling and MIS
    let canonical_sample_at_canonical = reservoir_contribution(canonical_reservoir, canonical_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    let other_sample_at_canonical = reservoir_contribution(other_reservoir, other_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    let canonical_sample_at_other = reservoir_contribution(canonical_reservoir, canonical_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);
    let other_sample_at_other = reservoir_contribution(other_reservoir, other_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);

    // Jacobians for resampling and MIS. Light samples don't need a reprojection jacobian,
    // since resolve_and_calculate_light_contribution already accounts for the shading point's geometry.
    var other_sample_at_canonical_jacobian = 1.0;
    if other_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        other_sample_at_canonical_jacobian = jacobian(
            canonical_world_position,
            other_world_position,
            other_reservoir.sample_point_world_position,
            octahedral_decode(other_reservoir.sample_point_world_normal)
        );
    }
    var canonical_sample_at_other_jacobian = 1.0;
    if canonical_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        canonical_sample_at_other_jacobian = jacobian(
            other_world_position,
            canonical_world_position,
            canonical_reservoir.sample_point_world_position,
            octahedral_decode(canonical_reservoir.sample_point_world_normal)
        );
    }

    // Don't merge samples with huge jacobians, as it explodes the variance
    if other_sample_at_canonical_jacobian > 1.2 || canonical_sample_at_other_jacobian > 1.2 {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_at_canonical.radiance, canonical_sample_at_canonical.wi, canonical_sample_at_canonical.sample_world_position);
    }

    // Resampling weight for canonical sample
    let canonical_sample_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_sample_at_canonical.target_function,
        other_reservoir.confidence_weight * canonical_sample_at_other.target_function * canonical_sample_at_other_jacobian,
    );
    let canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_sample_at_canonical.target_function * canonical_reservoir.unbiased_contribution_weight;

    // Resampling weight for other sample
    let other_sample_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_sample_at_other.target_function,
        canonical_reservoir.confidence_weight * other_sample_at_canonical.target_function * other_sample_at_canonical_jacobian,
    );
    let other_sample_resampling_weight = other_sample_mis_weight * other_sample_at_canonical.target_function * other_reservoir.unbiased_contribution_weight * other_sample_at_canonical_jacobian;

    // Perform resampling
    var combined_reservoir = empty_reservoir();
    combined_reservoir.confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    let weight_sum = canonical_sample_resampling_weight + other_sample_resampling_weight;

    if rand_f(rng) < other_sample_resampling_weight / weight_sum {
        combined_reservoir.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = other_reservoir.radiance;
        combined_reservoir.light_sample = other_reservoir.light_sample;

        let inverse_target_function = select(0.0, 1.0 / other_sample_at_canonical.target_function, other_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_sample_at_canonical.radiance, other_sample_at_canonical.wi, other_sample_at_canonical.sample_world_position);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;
        combined_reservoir.light_sample = canonical_reservoir.light_sample;

        let inverse_target_function = select(0.0, 1.0 / canonical_sample_at_canonical.target_function, canonical_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_at_canonical.radiance, canonical_sample_at_canonical.wi, canonical_sample_at_canonical.sample_world_position);
    }
}

struct ReservoirContribution {
    radiance: vec3<f32>,
    target_function: f32,
    wi: vec3<f32>,
    sample_world_position: vec4<f32>,
}

fn reservoir_contribution(reservoir: Reservoir, resolved: ResolvedLightSample, world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> ReservoirContribution {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
        let target_function = luminance(light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab));
        return ReservoirContribution(light_contribution.radiance, target_function, light_contribution.wi, resolved.world_position);
    } else if any(reservoir.radiance != vec3(0.0)) {
        let wi = normalize(reservoir.sample_point_world_position - world_position);
        let target_function = luminance(reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab));
        return ReservoirContribution(reservoir.radiance, target_function, wi, vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(reservoir.sample_point_world_position, 1.0));
    }
}
