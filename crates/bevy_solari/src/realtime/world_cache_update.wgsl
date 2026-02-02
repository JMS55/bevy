enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_range_u, sample_cosine_hemisphere}
#import bevy_render::view::View
#import bevy_solari::presample_light_tiles::{ResolvedLightSamplePacked, unpack_resolved_light_sample}
#import bevy_solari::sampling::{calculate_resolved_light_contribution, trace_light_visibility}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN}
#import bevy_solari::world_cache::{
    WORLD_CACHE_MAX_TEMPORAL_SAMPLES,
    WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT,
    WORLD_CACHE_MAX_GI_RAY_DISTANCE,
    WORLD_CACHE_CELL_UPDATES_SOFT_CAP,
    query_world_cache,
}
#import bevy_solari::realtime_bindings::{
    light_tile_resolved_samples,
    view,
    constants,
    world_cache_active_cells_count,
    world_cache_active_cell_indices,
    world_cache_life,
    world_cache_geometry_data,
    world_cache_radiance,
    world_cache_luminance_deltas,
    world_cache_active_cells_new_radiance,
    world_cache_sampling_light_ids,
}

@compute @workgroup_size(64, 1, 1)
fn sample_di(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache_active_cells_count { return; }

    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    let geometry_data = world_cache_geometry_data[cell_index];
    var rng = cell_index + constants.frame_index;

    var good_lights: array<u32, 8u>;
    var good_light_target_functions: array<f32, 8u>;
    var good_reservoir = empty_reservoir();
    var weight_sum = 0.0;
    for (var i = 0u; i < 8u; i += 1u) {
        var light = world_cache_sampling_light_ids[cell_index * 8u + i];
        let light_id = previous_frame_light_id_translations[light >> 16u];
        let triangle_id = light & 0xFFFFu;

        if light_id == LIGHT_NOT_PRESENT_THIS_FRAME {
            good_lights[i] = NULL_LIGHT_ID;
            continue;
        }

        light = (light_id << 16u) | triangle_id;
        good_lights[i] = light;

        let light_sample = LightSample(light, seed);
        let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_id]);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, geometry_data.world_position, geometry_data.world_normal);
        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, geometry_data.world_normal));
        let target_function = luminance(contribution);
        good_light_target_functions[i] = target_function;
        let mis_weight = 1.0 / 8.0;
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);
        weight_sum += resampling_weight;
        if rand_f(rng) < resampling_weight / weight_sum {
            good_reservoir.sample = light_sample;
            good_reservoir.sample_contribution = contribution;
            good_reservoir.sample_target_function = target_function;
            good_reservoir.sample_world_position = resolved_light_sample.world_position;
            good_reservoir.good_light_index = i;
        }

    }
    good_reservoir.unbiased_contribution_weight = weight_sum * select(0.0, 1.0 / good_reservoir.sample_target_function, good_reservoir.sample_target_function > 0.0);

    var bad_reservoir = empty_reservoir();
    weight_sum = 0.0;

    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;
    var selected_tile_sample = 0u;
    for (var i = 0u; i < 4u; i += 1u) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, geometry_data.world_position, geometry_data.world_normal);
        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, geometry_data.world_normal));
        let target_function = luminance(contribution);
        let mis_weight = 1.0 / 4.0;
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);
        weight_sum += resampling_weight;
        if rand_f(rng) < resampling_weight / weight_sum {
            selected_tile_sample = tile_sample;
            bad_reservoir.sample_contribution = contribution;
            bad_reservoir.sample_target_function = target_function;
            bad_reservoir.sample_world_position = resolved_light_sample.world_position;
        }
    }
    bad_reservoir.sample = light_tile_samples[selected_tile_sample];
    bad_reservoir.unbiased_contribution_weight = weight_sum * select(0.0, 1.0 / bad_reservoir.sample_target_function, bad_reservoir.sample_target_function > 0.0);

    var combined_reservoir = empty_reservoir();

    let mis_weight = 1.0 / 2.0;
    let good_resampling_weight = 0.9 * mis_weight * (good_reservoir.sample_target_function * good_reservoir.unbiased_contribution_weight);
    let bad_resampling_weight = (1.0 - 0.9) * mis_weight * (bad_reservoir.sample_target_function * bad_reservoir.unbiased_contribution_weight);
    weight_sum = good_resampling_weight + bad_resampling_weight;
    if rand_f(rng) < bad_resampling_weight / weight_sum {
        combined_reservoir = bad_reservoir;
    } else {
        combined_reservoir = good_reservoir;
    }
    combined_reservoir.unbiased_contribution_weight = weight_sum * select(0.0, 1.0 / combined_reservoir.sample_target_function, combined_reservoir.sample_target_function > 0.0);
    if combined_reservoir.unbiased_contribution_weight > 0.0 {
        unbiased_contribution_weight *= trace_light_visibility(geometry_data.world_position, combined_reservoir.sample_world_position);
    }

    if combined_reservoir.good_light_index < 8u {
        // TODO: Update sample count + 1, visibility + result
    } else if combined_reservoir.unbiased_contribution_weight {
        // TODO: Can probably do this while doing RIS over the 8 lights
        var worst_index = 8u;
        var worst_target_function = 3.402823466e38;
        for (var i = 0u; i < 8u; i += 1u) {
            let t = good_light_target_functions[i];
            if combined_reservoir.target_function > t && t < worst_target_function {
                worst_index = i;
                worst_target_function = t;
            }
        }
        if worst_index != 8u {
            good_lights[worst_index] = combined_reservoir.sample.light_id;
        }
    }

    for (var i = 0u; i < 8u; i += 1u) {
        world_cache_sampling_light_ids[cell_index * 8u + i] = good_lights[i];
    }

    // if rand_f(&rng) >= f32(WORLD_CACHE_CELL_UPDATES_SOFT_CAP) / f32(world_cache_active_cells_count) { return; }

    world_cache_active_cells_new_radiance[active_cell_id.x] = combined_reservoir.sample_contribution;
}

@compute @workgroup_size(64, 1, 1)
fn sample_gi(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache_active_cells_count { return; }

    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    let geometry_data = world_cache_geometry_data[cell_index];
    var rng = cell_index + constants.frame_index;

    if rand_f(&rng) >= f32(WORLD_CACHE_CELL_UPDATES_SOFT_CAP) / f32(world_cache_active_cells_count) { return; }

    let ray_direction = sample_cosine_hemisphere(geometry_data.world_normal, &rng);
    let ray = trace_ray(geometry_data.world_position, ray_direction, RAY_T_MIN, WORLD_CACHE_MAX_GI_RAY_DISTANCE, RAY_FLAG_NONE);
    if ray.kind != RAY_QUERY_INTERSECTION_NONE {
        let ray_hit = resolve_ray_hit_full(ray);
        let cell_life = atomicLoad(&world_cache_life[cell_index]);
        let radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, ray.t, cell_life, &rng);
        world_cache_active_cells_new_radiance[active_cell_id.x] += ray_hit.material.base_color * radiance;
    }
}

@compute @workgroup_size(64, 1, 1)
fn blend_new_samples(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache_active_cells_count { return; }

    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    var rng = cell_index + constants.frame_index;

    if rand_f(&rng) >= f32(WORLD_CACHE_CELL_UPDATES_SOFT_CAP) / f32(world_cache_active_cells_count) { return; }

    let old_radiance = world_cache_radiance[cell_index];
    let new_radiance = world_cache_active_cells_new_radiance[active_cell_id.x];
    let luminance_delta = world_cache_luminance_deltas[cell_index];

    // https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a
    let sample_count = min(old_radiance.a + 1.0, WORLD_CACHE_MAX_TEMPORAL_SAMPLES);
    let alpha = abs(luminance_delta) / max(luminance(old_radiance.rgb), 0.001);
    let max_sample_count = mix(WORLD_CACHE_MAX_TEMPORAL_SAMPLES, 1.0, pow(saturate(alpha), 1.0 / 8.0));
    let blend_amount = 1.0 / min(sample_count, max_sample_count);

    let blended_radiance = mix(old_radiance.rgb, new_radiance, blend_amount);
    let blended_luminance_delta = mix(luminance_delta, luminance(blended_radiance) - luminance(old_radiance.rgb), 1.0 / 8.0);

    world_cache_radiance[cell_index] = vec4(blended_radiance, sample_count);
    world_cache_luminance_deltas[cell_index] = blended_luminance_delta;
}

fn sample_random_light_ris(world_position: vec3<f32>, world_normal: vec3<f32>, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> vec3<f32> {
    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var weight_sum = 0.0;
    var selected_sample_radiance = vec3(0.0);
    var selected_sample_target_function = 0.0;
    var selected_sample_world_position = vec4(0.0);
    let mis_weight = 1.0 / f32(WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT);
    for (var i = 0u; i < WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, world_position, world_normal);

        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, world_normal));
        let target_function = luminance(contribution);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if rand_f(rng) < resampling_weight / weight_sum {
            selected_sample_radiance = contribution;
            selected_sample_target_function = target_function;
            selected_sample_world_position = resolved_light_sample.world_position;
        }
    }

    var unbiased_contribution_weight = 0.0;
    if all(selected_sample_radiance != vec3(0.0)) {
        let inverse_target_function = select(0.0, 1.0 / selected_sample_target_function, selected_sample_target_function > 0.0);
        unbiased_contribution_weight = weight_sum * inverse_target_function;

        unbiased_contribution_weight *= trace_light_visibility(world_position, selected_sample_world_position);
    }

    return selected_sample_radiance * unbiased_contribution_weight;
}

struct Reservoir {
    sample: LightSample,
    sample_contribution: vec3<f32>,
    sample_target_function: f32,
    sample_world_position: vec3<f32>,
    unbiased_contribution_weight: f32,
    good_light_index: u32,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        LightSample(NULL_LIGHT_ID, 0u),
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        8u,
    );
}
