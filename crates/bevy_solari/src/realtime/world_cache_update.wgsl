enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_u, rand_range_u, sample_cosine_hemisphere}
#import bevy_render::view::View
#import bevy_solari::presample_light_tiles::{ResolvedLightSamplePacked, unpack_resolved_light_sample}
#import bevy_solari::sampling::{resolve_light_sample, calculate_resolved_light_contribution, trace_light_visibility, LightSample}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN, LIGHT_SOURCE_KIND_DIRECTIONAL, light_sources, directional_lights, transforms, material_ids, materials}
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
    world_cache_good_lights,
    world_cache_good_light_weights,
    world_cache_active_cells_new_radiance,
}

@compute @workgroup_size(64, 1, 1)
fn sample_di(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache_active_cells_count { return; }

    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    let geometry_data = world_cache_geometry_data[cell_index];
    var rng = cell_index + constants.frame_index;

    // Build list of lights and their weights
    let base_index = active_cell_id.x * 128u;
    let light_count = arrayLength(&light_sources);
    var total_weight = 0.0;
    for (var i = base_index; i < base_index + 128u; i++) {
        let light_id = rand_range_u(light_count, &rng);
        let light_source = light_sources[light_id];

        var weight: f32;
        if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
            let directional_light = directional_lights[light_source.id];
            weight = luminance(directional_light.luminance) * 0.01;
        } else {
            let light_world_position = transforms[light_source.id][3u].xyz;
            let material_id = material_ids[light_source.id];
            let emission = luminance(materials[material_id].emissive.rgb); // TODO: Handle emissive texture
            let d = geometry_data.world_position - light_world_position;
            let distance2 = max(dot(d, d), 0.00001);
            weight = emission / distance2;
        }

        total_weight += weight;
        world_cache_good_lights[i] = light_id;
        world_cache_good_light_weights[i] = weight;
    }

    // Build CDF
    world_cache_good_light_weights[base_index] /= total_weight;
    for (var i = base_index + 1u; i < base_index + 128u; i++) {
        world_cache_good_light_weights[i] = world_cache_good_light_weights[i - 1u] + (world_cache_good_light_weights[i] / total_weight);
    }

    // Sample CDF
    let r = rand_f(&rng);
    var chosen_i = base_index + 127u;
    var light_weight = world_cache_good_light_weights[chosen_i];
    for (var i = base_index; i < base_index + 127u; i++) {
        let weight = world_cache_good_light_weights[i];
        if r < weight {
            chosen_i = i;
            light_weight = weight;
            break;
        }
    }
    let light_id = world_cache_good_lights[chosen_i];
    if chosen_i > base_index {
        light_weight -= world_cache_good_light_weights[chosen_i - 1u];
    }

    // Pick random point on light
    let light_source = light_sources[light_id];
    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = rand_range_u(triangle_count, &rng);
    }
    let seed = rand_u(&rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    // Compute light contribution
    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf /= 128.0 * light_weight;
    var light_contribution = calculate_resolved_light_contribution(resolved_light_sample, geometry_data.world_position, geometry_data.world_normal);
    light_contribution.radiance *= resolved_light_sample.inverse_pdf;
    light_contribution.radiance *= trace_light_visibility(geometry_data.world_position, resolved_light_sample.world_position);

    world_cache_active_cells_new_radiance[active_cell_id.x] = light_contribution.radiance;
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
