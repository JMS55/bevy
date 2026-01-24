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
    rng: u32,
}

fn new_resampling_state(canonical_surface: ResolvedGPixel, canonical_confidence_weight: f32, noncanonical_confidence_weight_sum: f32, rng: u32) -> GIResamplingState {
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
        rng,
    );
}

fn add_noncanonical_sample(sample: GIReservoir, sample_world_position: vec3<f32>, state: GIResamplingState) -> GIResamplingState {
    let jacobian = jacobian(sample_world_position, state.canonical_world_position, sample.sample_point_world_position, sample.sample_point_world_normal);
    let cos_theta = saturate(dot(normalize(sample.sample_point_world_position - state.canonical_world_position), state.canonical_world_normal));
    let target_function = luminance(sample.radiance * state.canonical_diffuse_brdf * cos_theta);
    let mis_weight = mis_weight_defensive_pairwise_noncanonical(sample, state, target_function, jacobian);
    let resampling_weight = mis_weight.x * target_function * sample.unbiased_contribution_weight * jacobian;

    state.weight_sum += resampling_weight;
    state.reservoir.confidence_weight += sample.confidence_weight;
    state.canonical_partial_mis_weight_sum += mis_weight.y;

    if rand_f(&state.rng) < resampling_weight / state.weight_sum {
        state.reservoir.sample_point_world_position = sample.sample_point_world_position;
        state.reservoir.sample_point_world_normal = sample.sample_point_world_normal;
        state.reservoir.radiance = sample.radiance;
        state.target_function = target_function;
    }

    return state;
}

fn add_canonical_sample(sample: GIReservoir, state: GIResamplingState) -> GIResamplingState {
    let mis_weight = (state.canonical_confidence_weight * state.inverse_confidence_weight_sum) + state.canonical_partial_mis_weight_sum;
    let resampling_weight = mis_weight * sample.target_function * sample.unbiased_contribution_weight;

    state.weight_sum += resampling_weight;
    state.reservoir.confidence_weight += sample.confidence_weight;

    if rand_f(&state.rng) < resampling_weight / state.weight_sum {
        state.reservoir.sample_point_world_position = sample.sample_point_world_position;
        state.reservoir.sample_point_world_normal = sample.sample_point_world_normal;
        state.reservoir.radiance = sample.radiance;
        state.target_function = sample.target_function;
    }

    return state;
}

fn finish_resampling(state: GIResamplingState) -> GIReservoir {
    let inverse_target_function = select(0.0, 1.0 / sample.target_function, sample.target_function > 0.0);
    state.reservoir.unbiased_contribution_weight = state.weight_sum * inverse_target_function;
    reutrn state.reservoir;
}

fn mis_weight_defensive_pairwise(sample: GIReservoir, state: GIResamplingState, canonical_target_function: f32, jacobian: f32) -> vec2<f32> {
    let numerator = sample.confidence_weight * sample.target_function;
    let denominator1 = state.noncanonical_confidence_weight_sum * sample.target_function;
    let denominator2 = state.canonical_confidence_weight * (canonical_target_function / jacobian);
    let defense_ratio = state.noncanonical_confidence_weight_sum * state.inverse_confidence_weight_sum;
    let noncanonical_weight = (numerator * defense_ratio) / (denominator1 + denominator2);

    let defense_ratio2 = sample.confidence_weight * state.inverse_confidence_weight_sum;
    let partial_canonical_weight = (denominator2 * defense_ratio2) / (denominator1 + denominator2);

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
