// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf

#define_import_path bevy_solari::reservoir

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::rand_f
#import bevy_solari::sampling::{LightSample, calculate_light_contribution, trace_light_visibility}

const NULL_RESERVOIR_SAMPLE = 0xFFFFFFFFu;

struct Reservoir {
    sample: LightSample,
    weight_sum: f32,
    confidence_weight: f32,
    unbiased_contribution_weight: f32,
    _padding: f32,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        LightSample(vec2(NULL_RESERVOIR_SAMPLE, 0u), vec2(0.0)),
        0.0,
        0.0,
        0.0,
        0.0
    );
}

fn reservoir_valid(reservoir: Reservoir) -> bool {
    return reservoir.sample.light_id.x != NULL_RESERVOIR_SAMPLE;
}

struct ReservoirContext {
    reservoir: Reservoir,
    target_function: f32,
    radiance: vec3<f32>,
}

fn empty_reservoir_context() -> ReservoirContext {
    return ReservoirContext(empty_reservoir(), 0.0, vec3(0.0));
}

fn reservoir_add_sample(
    context: ptr<function, ReservoirContext>,
    sample: LightSample,
    mis_weight: f32,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    rng: ptr<function, u32>,
) {
    let light_contribution = calculate_light_contribution(sample, world_position, world_normal);
    let target_function = luminance(light_contribution.radiance);
    let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

    (*context).reservoir.weight_sum += resampling_weight;

    if rand_f(rng) < resampling_weight / (*context).reservoir.weight_sum {
        (*context).reservoir.sample = sample;
        (*context).target_function = target_function;
        (*context).radiance = light_contribution.radiance;
    }
}

fn reservoir_calculate_unbiased_contribution_weight_with_visibility(context: ptr<function, ReservoirContext>, world_position: vec3<f32>) {
    if reservoir_valid((*context).reservoir) {
        let inverse_target_function = select(0.0, 1.0 / (*context).target_function, (*context).target_function > 0.0);
        (*context).reservoir.unbiased_contribution_weight = (*context).reservoir.weight_sum * inverse_target_function;
        (*context).reservoir.unbiased_contribution_weight *= trace_light_visibility((*context).reservoir.sample, world_position);
    }
}

fn reservoir_calculate_unbiased_contribution_weight(context: ptr<function, ReservoirContext>, world_position: vec3<f32>) {
    if reservoir_valid((*context).reservoir) {
        let inverse_target_function = select(0.0, 1.0 / (*context).target_function, (*context).target_function > 0.0);
        (*context).reservoir.unbiased_contribution_weight = (*context).reservoir.weight_sum * inverse_target_function;
    }
}

struct MisWeights {
    canonical_mis_weight: f32,
    canonical_target_function: f32,
    other_mis_weight: f32,
    other_target_function: f32,
}

fn calculate_mis_weights(
    canonical_reservoir: Reservoir,
    canonical_world_position: vec3<f32>,
    canonical_world_normal: vec3<f32>,
    other_reservoir: Reservoir,
    other_world_position: vec3<f32>,
    other_world_normal: vec3<f32>,
) -> MisWeights {
    let tf_cc = reservoir_target_function(canonical_reservoir, canonical_world_position, canonical_world_normal);
    let tf_oc = reservoir_target_function(other_reservoir, canonical_world_position, canonical_world_normal);

#ifdef BIASED
        let inverse_confidence_sum = 1.0 / (canonical_reservoir.confidence_weight + other_reservoir.confidence_weight);
        let canonical_mis_weight = canonical_reservoir.confidence_weight * inverse_confidence_sum;
        let other_mis_weight = other_reservoir.confidence_weight * inverse_confidence_sum;
#else
        let tf_co = reservoir_target_function(canonical_reservoir, other_world_position, other_world_normal) * canonical_reservoir.confidence_weight;
        let tf_oo = reservoir_target_function(other_reservoir, other_world_position, other_world_normal) * other_reservoir.confidence_weight;

        let canonical_mis_weight = max(0.0,
            (tf_cc * canonical_reservoir.confidence_weight) / ((tf_cc * canonical_reservoir.confidence_weight) + tf_co)
        );
        let other_mis_weight = max(0.0,
            tf_oo / (tf_oo + (tf_oc * other_reservoir.confidence_weight))
        );
#endif

    return MisWeights(canonical_mis_weight, tf_cc, other_mis_weight, tf_oc);
}

fn reservoir_target_function(reservoir: Reservoir, world_position: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if !reservoir_valid(reservoir) { return 0.0; }
    return luminance(calculate_light_contribution(reservoir.sample, world_position, world_normal).radiance);
}
