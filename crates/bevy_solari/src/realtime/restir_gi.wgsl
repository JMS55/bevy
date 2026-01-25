// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
// https://d1qx31qr3h6wln.cloudfront.net/publications/ReSTIR%20GI.pdf

enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{sample_uniform_hemisphere, uniform_hemisphere_inverse_pdf, sample_disk}
#import bevy_render::maths::PI
#import bevy_solari::brdf::evaluate_diffuse_brdf
#import bevy_solari::gbuffer_utils::{gpixel_resolve, pixel_dissimilar, permute_pixel, ResolvedGPixel}
#import bevy_solari::gi_reservoir::{GIReservoir, empty_gi_reservoir, new_resampling_state, add_noncanonical_sample, add_canonical_sample, finish_resampling, jacobian}
#import bevy_solari::realtime_bindings::{view_output, gi_reservoirs_a, gi_reservoirs_b, gbuffer, depth_buffer, motion_vectors, previous_gbuffer, previous_depth_buffer, view, previous_view, constants}
#import bevy_solari::sampling::{sample_random_light, trace_point_visibility}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, RAY_T_MIN, RAY_T_MAX}
#import bevy_solari::specular_gi::DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD
#import bevy_solari::world_cache::{query_world_cache, WORLD_CACHE_CELL_LIFETIME}

const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 8.0;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        gi_reservoirs_b[pixel_index] = empty_gi_reservoir();
        return;
    }
    let primary_surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    if primary_surface.material.metallic > 0.9999 && primary_surface.material.roughness <= DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD {
        gi_reservoirs_b[pixel_index] = empty_gi_reservoir();
        return;
    }

    let initial_reservoir = generate_initial_reservoir(primary_surface, &rng);
    let temporal = load_temporal_reservoir(global_id.xy, depth, primary_surface);

    var resampling_state = new_resampling_state(primary_surface, initial_reservoir.confidence_weight, temporal.reservoir.confidence_weight);
    add_noncanonical_sample(temporal.reservoir, temporal.world_position, &resampling_state, &rng);
    add_canonical_sample(initial_reservoir, &resampling_state, &rng);
    let combined_reservoir = finish_resampling(&resampling_state);

    gi_reservoirs_b[pixel_index] = combined_reservoir;
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        gi_reservoirs_a[pixel_index] = empty_gi_reservoir();
        return;
    }
    let primary_surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    if primary_surface.material.metallic > 0.9999 && primary_surface.material.roughness <= DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD {
        gi_reservoirs_a[pixel_index] = empty_gi_reservoir();
        return;
    }

    let input_reservoir = gi_reservoirs_b[pixel_index];

    let spatial_count = select(1u, 8u, input_reservoir.confidence_weight < CONFIDENCE_WEIGHT_CAP);
    var spatial_confidence_weight_sum = 0.0;
    // TODO: Cache valid neighbors

    var rng_copy = rng;
    for (var i = 0u; i < spatial_count; i += 1u) {
        let spatial_pixel_id = get_neighbor_pixel_id(global_id.xy, SPATIAL_REUSE_RADIUS_PIXELS, &rng_copy);

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, primary_surface.world_position, spatial_surface.world_position, primary_surface.world_normal, spatial_surface.world_normal, view) {
            // search_radius /= 2.0; // TODO
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        let spatial_reservoir = gi_reservoirs_b[spatial_pixel_index];

        let jacobian = jacobian(spatial_surface.world_position, primary_surface.world_position, spatial_reservoir.sample_point_world_position, spatial_reservoir.sample_point_world_normal);
        if jacobian < 1.0 / 8.0 || jacobian > 8.0 {
            continue;
        }

        spatial_confidence_weight_sum += spatial_reservoir.confidence_weight;
    }

    var resampling_state = new_resampling_state(primary_surface, input_reservoir.confidence_weight, spatial_confidence_weight_sum);

    rng_copy = rng;
    for (var i = 0u; i < spatial_count; i += 1u) {
        let spatial_pixel_id = get_neighbor_pixel_id(global_id.xy, SPATIAL_REUSE_RADIUS_PIXELS, &rng_copy);

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, primary_surface.world_position, spatial_surface.world_position, primary_surface.world_normal, spatial_surface.world_normal, view) {
            // search_radius /= 2.0;
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        let spatial_reservoir = gi_reservoirs_b[spatial_pixel_index];

        let jacobian = jacobian(spatial_surface.world_position, primary_surface.world_position, spatial_reservoir.sample_point_world_position, spatial_reservoir.sample_point_world_normal);
        if jacobian < 1.0 / 8.0 || jacobian > 8.0 {
            continue;
        }

        add_noncanonical_sample(spatial_reservoir, spatial_surface.world_position, &resampling_state, &rng);
    }

    add_canonical_sample(input_reservoir, &resampling_state, &rng);
    var combined_reservoir = finish_resampling(&resampling_state);

    gi_reservoirs_a[pixel_index] = combined_reservoir;

    combined_reservoir.unbiased_contribution_weight *= trace_point_visibility(primary_surface.world_position, combined_reservoir.sample_point_world_position);

    let wi = normalize(input_reservoir.sample_point_world_position - primary_surface.world_position);
    let brdf = evaluate_diffuse_brdf(primary_surface.world_normal, wi, primary_surface.material.base_color, primary_surface.material.metallic);

    var pixel_color = textureLoad(view_output, global_id.xy);
    pixel_color += vec4(input_reservoir.radiance * brdf * input_reservoir.unbiased_contribution_weight * view.exposure, 0.0);
    textureStore(view_output, global_id.xy, pixel_color);
}

fn generate_initial_reservoir(primary_surface: ResolvedGPixel, rng: ptr<function, u32>) -> GIReservoir {
    var reservoir = empty_gi_reservoir();

    let ray_direction = sample_uniform_hemisphere(primary_surface.world_normal, rng);
    let ray = trace_ray(primary_surface.world_position, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);

    if ray.kind == RAY_QUERY_INTERSECTION_NONE {
        return reservoir;
    }

    let sample_point = resolve_ray_hit_full(ray);

    if all(sample_point.material.emissive != vec3(0.0)) {
        return reservoir;
    }

    reservoir.sample_point_world_position = sample_point.world_position;
    reservoir.sample_point_world_normal = sample_point.world_normal;
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

    let cos_theta = saturate(dot(ray_direction, primary_surface.world_normal));
    reservoir.target_function = luminance(reservoir.radiance * (primary_surface.material.base_color / PI) * cos_theta);

    return reservoir;
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, primary_surface: ResolvedGPixel) -> NeighborInfo {
    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));

    // Check if the current pixel was off screen during the previous frame (current pixel is newly visible),
    // or if all temporal history should assumed to be invalid
    if any(temporal_pixel_id_float < vec2(0.0)) || any(temporal_pixel_id_float >= view.main_pass_viewport.zw) || bool(constants.reset) {
        return NeighborInfo(empty_gi_reservoir(), vec3(0.0));
    }

    let permuted_temporal_pixel_id = permute_pixel(vec2<u32>(temporal_pixel_id_float), constants.frame_index, view.main_pass_viewport.zw);
    var temporal = load_temporal_reservoir_inner(permuted_temporal_pixel_id, depth, primary_surface);

    // If permuted reprojection failed (tends to happen on object edges), try point reprojection
    if temporal.reservoir.confidence_weight == 0.0 {
        temporal = load_temporal_reservoir_inner(vec2<u32>(temporal_pixel_id_float), depth, primary_surface);
    }

    temporal.reservoir.confidence_weight = min(temporal.reservoir.confidence_weight, CONFIDENCE_WEIGHT_CAP);

    return temporal;
}

fn load_temporal_reservoir_inner(temporal_pixel_id: vec2<u32>, depth: f32, primary_surface: ResolvedGPixel) -> NeighborInfo {
    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, temporal_pixel_id, 0), temporal_depth, temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, primary_surface.world_position, temporal_surface.world_position, primary_surface.world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_gi_reservoir(), vec3(0.0));
    }

    let temporal_pixel_index = temporal_pixel_id.x + temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    let temporal_reservoir = gi_reservoirs_a[temporal_pixel_index];

    let jacobian = jacobian(temporal_surface.world_position, primary_surface.world_position, temporal_reservoir.sample_point_world_position, temporal_reservoir.sample_point_world_normal);
    if jacobian < 1.0 / 8.0 || jacobian > 8.0 {
        return NeighborInfo(empty_gi_reservoir(), vec3(0.0));
    }

    return NeighborInfo(temporal_reservoir, temporal_surface.world_position);
}

fn get_neighbor_pixel_id(center_pixel_id: vec2<u32>, search_radius: f32, rng: ptr<function, u32>) -> vec2<u32> {
    var spatial_id = vec2<f32>(center_pixel_id) + sample_disk(search_radius, rng);
    spatial_id = clamp(spatial_id, vec2(0.0), view.main_pass_viewport.zw - 1.0);
    return vec2<u32>(spatial_id);
}

struct NeighborInfo {
    reservoir: GIReservoir,
    world_position: vec3<f32>,
}
