// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_pbr::utils::{rand_f, rand_range_u, sample_uniform_hemisphere, uniform_hemisphere_inverse_pdf, sample_disk, octahedral_encode, octahedral_decode}
#import bevy_render::maths::PI
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_diffuse_brdf, evaluate_specular_brdf}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, pixel_dissimilar, permute_pixel}
#import bevy_solari::sampling::{sample_random_light, trace_light_visibility, balance_heuristic, calculate_resolved_light_contribution, resolve_light_sample, LightSample, ResolvedLightSample, NULL_LIGHT_ID, isnan}
#import bevy_solari::scene_bindings::{light_sources, previous_frame_light_id_translations, LIGHT_NOT_PRESENT_THIS_FRAME, trace_ray, resolve_ray_hit_full, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::world_cache::{query_world_cache, WORLD_CACHE_CELL_LIFETIME}
#import bevy_solari::realtime_bindings::{view_output, light_tile_samples, light_tile_resolved_samples, gi_reservoirs_a, gi_reservoirs_b, gbuffer, depth_buffer, motion_vectors, previous_gbuffer, previous_depth_buffer, view, previous_view, constants, Reservoir}
#import bevy_solari::specular_gi::DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD
#import bevy_solari::presample_light_tiles::unpack_resolved_light_sample
#import bevy_solari::specular_gi::SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD

const INITIAL_DI_SAMPLES = 8u;
const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 8.0;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        gi_reservoirs_b[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    if surface.material.metallic > 0.9999 && surface.material.roughness <= DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD {
        gi_reservoirs_b[pixel_index] = empty_reservoir();
        return;
    }
    let diffuse_brdf = surface.material.base_color / PI;

    let initial_di_reservoir = generate_initial_di_reservoir(surface.world_position, surface.world_normal, diffuse_brdf, workgroup_id.xy, &rng);
    let initial_gi_reservoir = generate_initial_gi_reservoir(surface.world_position, surface.world_normal, diffuse_brdf, &rng);
    let initial_reservoir = merge_initial_reservoirs(initial_di_reservoir, initial_gi_reservoir, &rng);

    let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal);
    let merge_result = merge_reservoirs(initial_reservoir, surface.world_position, surface.world_normal, diffuse_brdf,
        temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.diffuse_brdf, &rng);

    gi_reservoirs_b[pixel_index] = merge_result.merged_reservoir;
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        gi_reservoirs_a[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    if surface.material.metallic > 0.9999 && surface.material.roughness <= DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD {
        gi_reservoirs_a[pixel_index] = empty_reservoir();
        return;
    }

    let spatial = load_spatial_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, &rng);

    let input_reservoir = gi_reservoirs_b[pixel_index];
    let merge_result = merge_reservoirs(input_reservoir, surface.world_position, surface.world_normal, surface.material.base_color / PI,
        spatial.reservoir, spatial.world_position, spatial.world_normal, spatial.diffuse_brdf, &rng);
    var combined_reservoir = merge_result.merged_reservoir;

    gi_reservoirs_a[pixel_index] = combined_reservoir;

    combined_reservoir.unbiased_contribution_weight *= trace_light_visibility(surface.world_position + (surface.world_normal * RAY_T_MIN), merge_result.selected_sample_world_position);

    let wo = normalize(view.world_position - surface.world_position);
    var brdf = evaluate_diffuse_brdf(wo, merge_result.wi, surface.world_normal, surface.material);
    // Only consider the specular lobe for DI if the surface is not smooth, else leave it for the specular GI pass to handle
    if combined_reservoir.light_sample.light_id != NULL_LIGHT_ID && surface.material.roughness > SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD {
        brdf += evaluate_specular_brdf(wo, merge_result.wi, surface.world_normal, surface.material);
    }

    var pixel_color = merge_result.selected_sample_radiance * combined_reservoir.unbiased_contribution_weight;
    pixel_color *= brdf;
    pixel_color += surface.material.emissive;
    pixel_color *= view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}

struct InitialReservoirResult {
    reservoir: Reservoir,
    target_function: f32,
}

fn generate_initial_gi_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, rng: ptr<function, u32>) -> InitialReservoirResult {
    var reservoir = empty_reservoir();

    let ray_direction = sample_uniform_hemisphere(world_normal, rng);
    let ray = trace_ray(world_position + (world_normal * RAY_T_MIN), ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);

    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        return InitialReservoirResult(reservoir, 0.0);
    }

    let sample_point = resolve_ray_hit_full(ray);

    if any(sample_point.material.emissive != vec3(0.0)) {
        return InitialReservoirResult(reservoir, 0.0); // TODO: Don't return empty reservoir. Instead, return a DI reservoir.
    }

    reservoir.sample_point_world_position = sample_point.world_position;
    reservoir.sample_point_world_normal = octahedral_encode(sample_point.world_normal);
    reservoir.confidence_weight = 1.0;

#ifdef NO_WORLD_CACHE
    let direct_lighting = sample_random_light(sample_point.world_position, sample_point.world_normal, rng);
    reservoir.radiance = direct_lighting.radiance * saturate(dot(direct_lighting.wi, sample_point.world_normal));
    reservoir.unbiased_contribution_weight = direct_lighting.inverse_pdf * uniform_hemisphere_inverse_pdf();
#else
    reservoir.radiance = query_world_cache(sample_point.world_position, sample_point.geometric_world_normal, view.world_position, ray.t, WORLD_CACHE_CELL_LIFETIME, rng);
    reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
#endif

    let sample_point_diffuse_brdf = sample_point.material.base_color / PI;
    reservoir.radiance *= sample_point_diffuse_brdf;

    let wi = normalize(reservoir.sample_point_world_position - world_position);
    let target_function = luminance(reservoir.radiance * diffuse_brdf * saturate(dot(wi, world_normal)));
    return InitialReservoirResult(reservoir, target_function);
}

fn generate_initial_di_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> InitialReservoirResult {
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

        let target_function = luminance(light_contribution.radiance * diffuse_brdf * saturate(dot(light_contribution.wi, world_normal)));
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
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), vec3(0.0));
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
            return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), vec3(0.0));
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
    let temporal_diffuse_brdf = temporal_surface.material.base_color / PI;
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), vec3(0.0));
    }

    let temporal_pixel_index = temporal_pixel_id.x + temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    let temporal_reservoir = gi_reservoirs_a[temporal_pixel_index];

    return NeighborInfo(temporal_reservoir, temporal_surface.world_position, temporal_surface.world_normal, temporal_diffuse_brdf);
}

fn load_spatial_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> NeighborInfo {
    for (var i = 0u; i < 5u; i++) {
        let spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        let spatial_diffuse_brdf = spatial_surface.material.base_color / PI;
        if pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view) {
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        let spatial_reservoir = gi_reservoirs_b[spatial_pixel_index];
        return NeighborInfo(spatial_reservoir, spatial_surface.world_position, spatial_surface.world_normal, spatial_diffuse_brdf);
    }

    return NeighborInfo(empty_reservoir(), world_position, world_normal, vec3(0.0));
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
    diffuse_brdf: vec3<f32>,
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
    canonical_diffuse_brdf: vec3<f32>,
    other_reservoir: Reservoir,
    other_world_position: vec3<f32>,
    other_world_normal: vec3<f32>,
    other_diffuse_brdf: vec3<f32>,
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

    // Contributions for resampling and MIS
    let canonical_sample_at_canonical = reservoir_contribution(canonical_reservoir, canonical_resolved, canonical_world_position, canonical_world_normal, canonical_diffuse_brdf);
    let other_sample_at_canonical = reservoir_contribution(other_reservoir, other_resolved, canonical_world_position, canonical_world_normal, canonical_diffuse_brdf);
    let canonical_sample_at_other = reservoir_contribution(canonical_reservoir, canonical_resolved, other_world_position, other_world_normal, other_diffuse_brdf);
    let other_sample_at_other = reservoir_contribution(other_reservoir, other_resolved, other_world_position, other_world_normal, other_diffuse_brdf);

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

fn reservoir_contribution(reservoir: Reservoir, resolved: ResolvedLightSample, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>) -> ReservoirContribution {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
        let target_function = luminance(light_contribution.radiance * diffuse_brdf * saturate(dot(light_contribution.wi, world_normal)));
        return ReservoirContribution(light_contribution.radiance, target_function, light_contribution.wi, resolved.world_position);
    } else if any(reservoir.radiance != vec3(0.0)) {
        let wi = normalize(reservoir.sample_point_world_position - world_position);
        let target_function = luminance(reservoir.radiance * diffuse_brdf * saturate(dot(wi, world_normal)));
        return ReservoirContribution(reservoir.radiance, target_function, wi, vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(reservoir.sample_point_world_position, 1.0));
    }
}
