import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::render::utils::{rand_f, rand_range_u, sample_cosine_hemisphere};
import bevy_render::maths::PI;
import package::realtime::presample_light_tiles::unpack_resolved_light_sample;
import package::realtime::atmosphere::calculate_light_contribution;
import package::scene::sampling::{trace_visibility, sample_environment_map_light};
import package::scene::bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN};
import package::realtime::world_cache_query::query_world_cache;
import package::realtime::bindings::{
    light_tile_resolved_samples,
    view,
    constants,
    world_cache,
};

enable wgpu_ray_query;

@if(WORLD_CACHE_FORCE_OPAQUE)
const WORLD_CACHE_RAY_FLAG = RAY_FLAG_FORCE_OPAQUE;
@if(!WORLD_CACHE_FORCE_OPAQUE)
const WORLD_CACHE_RAY_FLAG = RAY_FLAG_NONE;

@compute @workgroup_size(64, 1, 1)
fn update_world_cache(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) thread_id: vec3<u32>) {
    let active_cell_id = get_active_cell(thread_id.x);
    if active_cell_id == NO_ACTIVE_CELL { return; }

    let cell_index = world_cache.active_cell_indices[active_cell_id];
    let geometry_data = world_cache.geometry_data[cell_index];
    var rng = cell_index + constants.frame_rng;

    // Hot cells (read by camera paths last frame) come first in the active cell list, and are updated every frame.
    // Their direct light is recomputed from scratch every frame with several samples, so it never lags behind light changes.
    // Their indirect light is only traced as often as if every active cell took turns, which keeps the indirect sample rate (and noise) of the rolling update.
    // Cold cells take turns in a rolling window, with a single sample.
    let hot_cells_count = world_cache.hot_cells_count;
    let cold_cells_count = world_cache.active_cells_count - hot_cells_count;
    let is_hot = active_cell_id < hot_cells_count;

    var sample_count = 1u;
    var trace_indirect = true;
    var frames_between_indirect_updates = ceil(f32(cold_cells_count) / f32(constants.world_cache_max_cell_updates_per_frame));
    if is_hot {
        let budget_samples = constants.world_cache_hot_cell_direct_light_sample_budget / max(hot_cells_count, 1u);
        sample_count = clamp(budget_samples, 1u, max(constants.world_cache_hot_cell_direct_light_samples, 1u));

        let indirect_period = max(u32(ceil(f32(world_cache.active_cells_count) / f32(constants.world_cache_max_cell_updates_per_frame))), 1u);
        trace_indirect = ((cell_index + constants.frame_count) % indirect_period) == 0u;
        frames_between_indirect_updates = f32(indirect_period);
    }

    var direct = vec3(0.0);
    var indirect = vec3(0.0);
    for (var sample_index = 0u; sample_index < sample_count; sample_index++) {
        // Direct light
        direct += sample_random_light_ris(geometry_data.world_position, geometry_data.world_normal, workgroup_id.xy, sample_index, &rng);
        if !trace_indirect { continue; }

        // Indirect light
        let ray_direction = sample_cosine_hemisphere(geometry_data.world_normal, &rng);
        let ray = trace_ray(geometry_data.world_position + (geometry_data.world_normal * RAY_T_MIN), ray_direction, RAY_T_MIN, constants.world_cache_max_gi_ray_distance, WORLD_CACHE_RAY_FLAG);
        if ray.kind != RAY_QUERY_INTERSECTION_NONE {
            let ray_hit = resolve_ray_hit_full(ray, -ray_direction);
            if ray.front_face || ray_hit.double_sided {
                let cell_life = atomicLoad(&world_cache.life[cell_index]);
                let radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, cell_life, &rng);
                indirect += ray_hit.material.base_color * radiance;
            }
        } else {
            // The ray is cut short at world_cache_max_gi_ray_distance to save on traversal costs.
            // A miss here may actually have been a surface further away if we kept tracing, and not actually have been a miss.
            // We could be leaking sky light due to this if world_cache_max_gi_ray_distance is too small.
            indirect += PI * sample_environment_map_light(ray_direction);
        }
    }
    let inverse_sample_count = 1.0 / f32(sample_count);
    direct *= inverse_sample_count;
    indirect *= inverse_sample_count;

    // Only the indirect light is blended over time. The direct light is this update's estimate.
    var indirect_radiance = world_cache.radiance_indirect[cell_index];
    if trace_indirect {
        indirect_radiance = blend_indirect(indirect_radiance, indirect, cell_index, frames_between_indirect_updates);
        world_cache.radiance_indirect[cell_index] = indirect_radiance;
    }
    world_cache.radiance[cell_index] = vec4(direct + indirect_radiance.rgb, 1.0);
}

/// Blends a new indirect light sample into the cell's history (rgb = radiance, a = sample count), and updates the cell's luminance trend.
fn blend_indirect(old_radiance: vec4<f32>, new_radiance: vec3<f32>, cell_index: u32, frames_between_updates: f32) -> vec4<f32> {
    let luminance_delta = world_cache.luminance_deltas[cell_index];

    // https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a
    let sample_count = min(old_radiance.a + 1.0, constants.world_cache_max_temporal_samples);
    let alpha = abs(luminance_delta) / max(luminance(old_radiance.rgb), 0.001);
    let max_sample_count = mix(constants.world_cache_max_temporal_samples, 1.0, pow(saturate(alpha), 1.0 / 8.0));
    let blend_amount = max(1.0 / min(sample_count, max_sample_count), saturate(frames_between_updates / constants.world_cache_max_temporal_samples));

    let blended_radiance = mix(old_radiance.rgb, new_radiance, blend_amount);
    let blended_luminance_delta = mix(luminance_delta, luminance(blended_radiance) - luminance(old_radiance.rgb), 1.0 / 8.0);

    world_cache.luminance_deltas[cell_index] = blended_luminance_delta;
    return vec4(blended_radiance, sample_count);
}

const NO_ACTIVE_CELL = 0xFFFFFFFFu;

/// Every hot cell, followed by a rolling window over the cold cells.
fn get_active_cell(thread_id: u32) -> u32 {
    let hot_cells_count = world_cache.hot_cells_count;
    if thread_id < hot_cells_count { return thread_id; }

    let cold_cells_count = world_cache.active_cells_count - hot_cells_count;
    let cold_id = thread_id - hot_cells_count;
    if cold_id >= min(cold_cells_count, constants.world_cache_max_cell_updates_per_frame) { return NO_ACTIVE_CELL; }

    return hot_cells_count + (world_cache.update_window_start + cold_id) % cold_cells_count;
}

fn sample_random_light_ris(world_position: vec3<f32>, world_normal: vec3<f32>, workgroup_id: vec2<u32>, sample_index: u32, rng: ptr<function, u32>) -> vec3<f32> {
    // Each of a cell's samples draws from a different light tile
    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y + (sample_index * 7919u);
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var weight_sum = 0.0;
    var selected_sample_radiance = vec3(0.0);
    var selected_sample_target_function = 0.0;
    var selected_sample_world_position = vec4(0.0);
    let mis_weight = 1.0 / f32(constants.world_cache_direct_light_sample_count);
    for (var i = 0u; i < constants.world_cache_direct_light_sample_count; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_light_contribution(resolved_light_sample, world_position, world_normal);

        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, world_normal));
        let target_function = luminance(contribution);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if rand_f(rng) * weight_sum < resampling_weight {
            selected_sample_radiance = contribution;
            selected_sample_target_function = target_function;
            selected_sample_world_position = resolved_light_sample.world_position;
        }
    }

    var unbiased_contribution_weight = 0.0;
    if any(selected_sample_radiance != vec3(0.0)) {
        let inverse_target_function = select(0.0, 1.0 / selected_sample_target_function, selected_sample_target_function > 0.0);
        unbiased_contribution_weight = weight_sum * inverse_target_function;

        unbiased_contribution_weight *= trace_visibility(world_position + (world_normal * RAY_T_MIN), selected_sample_world_position, WORLD_CACHE_RAY_FLAG);
    }

    return selected_sample_radiance * unbiased_contribution_weight;
}
