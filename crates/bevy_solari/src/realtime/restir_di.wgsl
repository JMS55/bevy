#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_pbr::rgb9e5::rgb9e5_to_vec3_
#import bevy_pbr::utils::{rand_f, octahedral_decode}
#import bevy_render::maths::PI
#import bevy_render::view::View
#import bevy_solari::reservoir::{Reservoir, empty_reservoir, reservoir_valid}
#import bevy_solari::sampling::{generate_random_light_sample, calculate_light_contribution, trace_light_visibility, sample_disk}

@group(1) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<storage, read> previous_reservoirs: array<Reservoir>;
@group(1) @binding(2) var<storage, read_write> reservoirs: array<Reservoir>;
@group(1) @binding(3) var gbuffer: texture_2d<u32>;
@group(1) @binding(4) var depth_buffer: texture_depth_2d;
@group(1) @binding(5) var motion_vectors: texture_2d<f32>;
@group(1) @binding(6) var previous_gbuffer: texture_2d<u32>;
@group(1) @binding(7) var previous_depth_buffer: texture_depth_2d;
@group(1) @binding(8) var<uniform> view: View;
@group(1) @binding(9) var<uniform> previous_view: PreviousViewUniforms;
@group(1) @binding(10) var accumulation_texture: texture_storage_2d<rgba32float, read_write>;
struct PushConstants { frame_index: u32, reset: u32 }
var<push_constant> constants: PushConstants;

const INITIAL_SAMPLES = 32u;
const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 20.0 * f32(INITIAL_SAMPLES);

@compute @workgroup_size(8, 8, 1)
fn restir_di(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs[pixel_index] = empty_reservoir();
        textureStore(view_output, global_id.xy, vec4(vec3(0.0), 1.0));
        return;
    }
    let gpixel = textureLoad(gbuffer, global_id.xy, 0);
    let world_position = reconstruct_world_position(global_id.xy, depth);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));
    let base_color = pow(unpack4x8unorm(gpixel.r).rgb, vec3(2.2));
    let diffuse_brdf = base_color / PI;
    let emissive = rgb9e5_to_vec3_(gpixel.g);

    let canonical_reservoir = generate_initial_samples(world_position, world_normal, diffuse_brdf, &rng);
    let spatiotemporal_reservoir = load_spatiotemporal_reservoir(global_id.xy, depth, world_normal, &rng);
    let temporal_reservoir = load_temporal_reservoir(global_id.xy, world_normal);

    let mis_weight_denominator = 1.0 / (canonical_reservoir.confidence_weight
        + spatiotemporal_reservoir.confidence_weight
        + temporal_reservoir.confidence_weight);

    var reservoir = empty_reservoir();
    var reservoir_target_function = 0.0;

    let canonical_mis_weight = canonical_reservoir.confidence_weight * mis_weight_denominator;
    let canonical_reservoir_radiance = select(
        vec3(0.0),
        calculate_light_contribution(canonical_reservoir.sample, world_position, world_normal).radiance,
        reservoir_valid(canonical_reservoir),
    );
    let canonical_target_function = luminance(canonical_reservoir_radiance * diffuse_brdf);
    let canonical_resampling_weight = canonical_mis_weight * (canonical_target_function * canonical_reservoir.unbiased_contribution_weight);
    reservoir.weight_sum += canonical_resampling_weight;
    reservoir.confidence_weight += canonical_reservoir.confidence_weight;
    if rand_f(&rng) < canonical_resampling_weight / reservoir.weight_sum {
        reservoir.sample = canonical_reservoir.sample;
        reservoir_target_function = canonical_target_function;
    }

    let spatiotemporal_mis_weight = spatiotemporal_reservoir.confidence_weight * mis_weight_denominator;
    var spatiotemporal_reservoir_radiance = select(
        vec3(0.0),
        calculate_light_contribution(spatiotemporal_reservoir.sample, world_position, world_normal).radiance,
        reservoir_valid(spatiotemporal_reservoir),
    );
    let spatiotemporal_target_function = luminance(spatiotemporal_reservoir_radiance * diffuse_brdf);
    let spatiotemporal_resampling_weight = spatiotemporal_mis_weight * (spatiotemporal_target_function * spatiotemporal_reservoir.unbiased_contribution_weight);
    reservoir.weight_sum += spatiotemporal_resampling_weight;
    reservoir.confidence_weight += spatiotemporal_reservoir.confidence_weight;
    if rand_f(&rng) < spatiotemporal_resampling_weight / reservoir.weight_sum {
        reservoir.sample = spatiotemporal_reservoir.sample;
        reservoir_target_function = spatiotemporal_target_function;
    }

    let temporal_mis_weight = temporal_reservoir.confidence_weight * mis_weight_denominator;
    let temporal_reservoir_radiance = select(
        vec3(0.0),
        calculate_light_contribution(temporal_reservoir.sample, world_position, world_normal).radiance,
        reservoir_valid(temporal_reservoir),
    );
    let temporal_target_function = luminance(temporal_reservoir_radiance * diffuse_brdf);
    let temporal_resampling_weight = temporal_mis_weight * (temporal_target_function * temporal_reservoir.unbiased_contribution_weight);
    reservoir.weight_sum += temporal_resampling_weight;
    reservoir.confidence_weight += temporal_reservoir.confidence_weight;
    if rand_f(&rng) < temporal_resampling_weight / reservoir.weight_sum {
        reservoir.sample = temporal_reservoir.sample;
        reservoir_target_function = temporal_target_function;
    }

    if reservoir_valid(reservoir) {
        let inverse_target_function = select(0.0, 1.0 / reservoir_target_function, reservoir_target_function > 0.0);
        reservoir.unbiased_contribution_weight = reservoir.weight_sum * inverse_target_function;
    }

    reservoir.confidence_weight = min(reservoir.confidence_weight, CONFIDENCE_WEIGHT_CAP);
    reservoirs[pixel_index] = reservoir;

    if reservoir_valid(spatiotemporal_reservoir) {
        spatiotemporal_reservoir_radiance *= trace_light_visibility(spatiotemporal_reservoir.sample, world_position);
    }

    let inverse_weight_sum = 1.0 / reservoir.weight_sum;
    var radiance = (canonical_reservoir_radiance * canonical_reservoir.unbiased_contribution_weight)
        * (canonical_resampling_weight * inverse_weight_sum);
    radiance += (spatiotemporal_reservoir_radiance * spatiotemporal_reservoir.unbiased_contribution_weight)
        * (spatiotemporal_resampling_weight * inverse_weight_sum);
    radiance += (temporal_reservoir_radiance * temporal_reservoir.unbiased_contribution_weight)
        * (temporal_resampling_weight * inverse_weight_sum);

    let pixel_color = emissive + (radiance * view.exposure * diffuse_brdf);
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}

fn generate_initial_samples(world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    var reservoir_target_function = 0.0;
    for (var i = 0u; i < INITIAL_SAMPLES; i++) {
        let light_sample = generate_random_light_sample(rng);

        let mis_weight = 1.0 / f32(INITIAL_SAMPLES);
        let light_contribution = calculate_light_contribution(light_sample, world_position, world_normal);
        let target_function = luminance(light_contribution.radiance * diffuse_brdf);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        reservoir.weight_sum += resampling_weight;

        if rand_f(rng) < resampling_weight / reservoir.weight_sum {
            reservoir.sample = light_sample;
            reservoir_target_function = target_function;
        }
    }

    if reservoir_valid(reservoir) {
        let inverse_target_function = select(0.0, 1.0 / reservoir_target_function, reservoir_target_function > 0.0);
        reservoir.unbiased_contribution_weight = reservoir.weight_sum * inverse_target_function;
        reservoir.unbiased_contribution_weight *= trace_light_visibility(reservoir.sample, world_position);
    }

    reservoir.confidence_weight = f32(INITIAL_SAMPLES);

    return reservoir;
}

fn load_spatiotemporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_normal: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    let neighbor_pixel_id = get_neighbor_pixel_id(pixel_id, rng);

    let neighbor_depth = textureLoad(previous_depth_buffer, neighbor_pixel_id, 0);
    let neighbor_gpixel = textureLoad(previous_gbuffer, neighbor_pixel_id, 0);
    let neighbor_world_position = reconstruct_previous_world_position(neighbor_pixel_id, neighbor_depth);
    let neighbor_world_normal = octahedral_decode(unpack_24bit_normal(neighbor_gpixel.a));
    if is_neighbor_invalid(depth, neighbor_depth, world_normal, neighbor_world_normal) || bool(constants.reset) {
        return empty_reservoir();
    }

    let neighbor_pixel_index = neighbor_pixel_id.x + neighbor_pixel_id.y * u32(view.viewport.z);
    return previous_reservoirs[neighbor_pixel_index];
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, world_normal: vec3<f32>) -> Reservoir {
    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let previous_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.viewport.zw));
    let previous_pixel_id = vec2<u32>(previous_pixel_id_float);
    if any(previous_pixel_id_float < vec2(0.0)) || any(previous_pixel_id_float >= view.viewport.zw) || bool(constants.reset) {
        return empty_reservoir();
    }

    let previous_gpixel = textureLoad(previous_gbuffer, previous_pixel_id, 0);
    let previous_world_normal = octahedral_decode(unpack_24bit_normal(previous_gpixel.a));
    if is_previous_invalid(world_normal, previous_world_normal) {
        return empty_reservoir();
    }

    let previous_pixel_index = previous_pixel_id.x + previous_pixel_id.y * u32(view.viewport.z);
    return previous_reservoirs[previous_pixel_index];
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view.viewport.zw;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = view.world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn reconstruct_previous_world_position(pixel_id: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view.viewport.zw;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = previous_view.world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn get_neighbor_pixel_id(center_pixel_id: vec2<u32>, rng: ptr<function, u32>) -> vec2<u32> {
    var neighbor_id = vec2<i32>(center_pixel_id) + vec2<i32>(sample_disk(SPATIAL_REUSE_RADIUS_PIXELS, rng));
    neighbor_id = clamp(neighbor_id, vec2(0i), vec2<i32>(view.viewport.zw) - 1i);
    return vec2<u32>(neighbor_id);
}

// TODO: Plane distance instead of depth
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf#page=45
fn is_neighbor_invalid(depth: f32, neighbor_depth: f32, normal: vec3<f32>, neighbor_normal: vec3<f32>) -> bool {
    let linear_depth = -depth_ndc_to_view_z(depth);
    let linear_neighbor_depth = -depth_ndc_to_view_z(neighbor_depth);

    // Reject if depth difference more than 10% or angle between normals more than 25 degrees
    return linear_neighbor_depth > 1.1 * linear_depth || linear_neighbor_depth < 0.9 * linear_depth ||
        dot(normal, neighbor_normal) < 0.906;
}

fn is_previous_invalid(normal: vec3<f32>, previous_normal: vec3<f32>) -> bool {
    // Reject if angle between normals more than 25 degrees
    return dot(normal, previous_normal) < 0.906;
}

fn depth_ndc_to_view_z(ndc_depth: f32) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -previous_view.clip_from_view[3][2]() / ndc_depth;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return -(previous_view.clip_from_view[3][2] - ndc_depth) / previous_view.clip_from_view[2][2];
#else
    let view_pos = previous_view.view_from_clip * vec4(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
#endif
}
