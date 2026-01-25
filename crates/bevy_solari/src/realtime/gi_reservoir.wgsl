#define_import_path bevy_solari::gi_reservoir

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::rand_f
#import bevy_render::maths::PI
#import bevy_solari::gbuffer_utils::ResolvedGPixel

// Don't adjust the size of this struct without also adjusting `prepare::GI_RESERVOIR_STRUCT_SIZE`.
struct GIReservoir {
    sample_point_world_position: vec3<f32>,
    target_function: f32,
    radiance: vec3<f32>,
    confidence_weight: f32,
    sample_point_world_normal: vec3<f32>,
    unbiased_contribution_weight: f32,
}

fn empty_gi_reservoir() -> GIReservoir {
    return GIReservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
    );
}

struct GIResamplingState {
    reservoir: GIReservoir,
    canonical_world_position: vec3<f32>,
    canonical_world_normal: vec3<f32>,
    canonical_diffuse_brdf: vec3<f32>,
    weight_sum: f32,
    canonical_confidence_weight: f32,
    noncanonical_confidence_weight_sum: f32,
    inverse_confidence_weight_sum: f32,
    canonical_partial_mis_weight_sum: f32,
}

fn new_resampling_state(canonical_surface: ResolvedGPixel, canonical_confidence_weight: f32, noncanonical_confidence_weight_sum: f32) -> GIResamplingState {
    return GIResamplingState(
        empty_gi_reservoir(),
        canonical_surface.world_position,
        canonical_surface.world_normal,
        canonical_surface.material.base_color / PI,
        0.0,
        canonical_confidence_weight,
        noncanonical_confidence_weight_sum,
        1.0 / (canonical_confidence_weight + noncanonical_confidence_weight_sum),
        0.0,
    );
}

fn add_noncanonical_sample(sample: GIReservoir, sample_world_position: vec3<f32>, state: ptr<function, GIResamplingState>, rng: ptr<function, u32>) {
    if sample.confidence_weight == 0.0 { return; }

    let jacobian = jacobian(sample_world_position, state.canonical_world_position, sample.sample_point_world_position, sample.sample_point_world_normal);
    let cos_theta = saturate(dot(normalize(sample.sample_point_world_position - state.canonical_world_position), state.canonical_world_normal));
    let target_function = luminance(sample.radiance * state.canonical_diffuse_brdf * cos_theta);
    let mis_weight = mis_weight_defensive_pairwise(sample, *state, target_function, jacobian);
    let resampling_weight = mis_weight.x * target_function * sample.unbiased_contribution_weight * jacobian;

    state.weight_sum += resampling_weight;
    state.reservoir.confidence_weight += sample.confidence_weight;
    state.canonical_partial_mis_weight_sum += mis_weight.y;

    if rand_f(rng) < resampling_weight / state.weight_sum {
        state.reservoir.sample_point_world_position = sample.sample_point_world_position;
        state.reservoir.sample_point_world_normal = sample.sample_point_world_normal;
        state.reservoir.radiance = sample.radiance;
        state.reservoir.target_function = target_function;
    }
}

fn add_canonical_sample(sample: GIReservoir, state: ptr<function, GIResamplingState>, rng: ptr<function, u32>) {
    if sample.confidence_weight == 0.0 { return; }

    let mis_weight = (state.canonical_confidence_weight * state.inverse_confidence_weight_sum) + state.canonical_partial_mis_weight_sum;
    let resampling_weight = mis_weight * sample.target_function * sample.unbiased_contribution_weight;

    state.weight_sum += resampling_weight;
    state.reservoir.confidence_weight += sample.confidence_weight;

    if rand_f(rng) < resampling_weight / state.weight_sum {
        state.reservoir.sample_point_world_position = sample.sample_point_world_position;
        state.reservoir.sample_point_world_normal = sample.sample_point_world_normal;
        state.reservoir.radiance = sample.radiance;
        state.reservoir.target_function = sample.target_function;
    }
}

fn finish_resampling(state: ptr<function, GIResamplingState>) -> GIReservoir {
    let inverse_target_function = select(0.0, 1.0 / state.reservoir.target_function, state.reservoir.target_function > 0.0);
    state.reservoir.unbiased_contribution_weight = state.weight_sum * inverse_target_function;
    return state.reservoir;
}

// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf#subsection.7.1.3
// Algorithm 7.8, returning vec2(m_i(y), partial_m_c(y))
fn mis_weight_defensive_pairwise(sample: GIReservoir, state: GIResamplingState, canonical_target_function: f32, jacobian: f32) -> vec2<f32> {
    let target_function_from_sample = sample.target_function / jacobian;
    let numerator = sample.confidence_weight * target_function_from_sample;
    let denominator_left = state.noncanonical_confidence_weight_sum * target_function_from_sample;
    let denominator_right = state.canonical_confidence_weight * canonical_target_function;
    let inverse_denominator = 1.0 / (denominator_left + denominator_right);
    let defense_ratio = state.noncanonical_confidence_weight_sum * state.inverse_confidence_weight_sum;
    let noncanonical_weight = max(0.0, defense_ratio * numerator * inverse_denominator);

    let canonical_defense_ratio = sample.confidence_weight * state.inverse_confidence_weight_sum;
    let partial_canonical_weight = max(0.0, canonical_defense_ratio * denominator_right * inverse_denominator);

    return vec2(noncanonical_weight, partial_canonical_weight);
}

fn jacobian(
    source_world_position: vec3<f32>,
    target_world_position: vec3<f32>,
    sample_point_world_position: vec3<f32>,
    sample_point_world_normal: vec3<f32>,
) -> f32 {
    let r = target_world_position - sample_point_world_position;
    let q = source_world_position - sample_point_world_position;
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
