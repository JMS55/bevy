// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf

#define_import_path bevy_solari::reservoir

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::rand_f
#import bevy_solari::sampling::{LightSample, calculate_light_contribution}

const NULL_RESERVOIR_SAMPLE = 0xFFFFFFFFu;

const SAMPLE_COUNT_CAP = 20.0;

struct Reservoir {
    sample: LightSample,
    sample_count: f32,
    unbiased_contribution_weight: f32,
    visibility: f32,
}

// Don't adjust the size of this struct without also adjusting PACKED_RESERVOIR_STRUCT_SIZE.
struct PackedReservoir {
    sample: LightSample,
    packed: u32,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        LightSample(NULL_RESERVOIR_SAMPLE, 0u),
        0.0,
        0.0,
        0.0
    );
}

fn pack_reservoir(reservoir: Reservoir) -> PackedReservoir {
    let packed = (u32(reservoir.sample_count) << 24u) |
        (u32(saturate(reservoir.visibility) * 255.0 + 0.5) << 16u) |
        pack2x16float(vec2(reservoir.unbiased_contribution_weight, 0.0));
    return PackedReservoir(reservoir.sample, packed);
}

fn unpack_reservoir(packed_reservoir: PackedReservoir) -> Reservoir {
    return Reservoir(
        packed_reservoir.sample,
        f32(packed_reservoir.packed >> 24u),
        unpack2x16float(packed_reservoir.packed).x,
        f32((packed_reservoir.packed >> 16u) & 0xFFu) / 255.0,
    );
}

fn reservoir_valid(reservoir: Reservoir) -> bool {
    return reservoir.sample.light_id != NULL_RESERVOIR_SAMPLE;
}

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_radiance: vec3<f32>,
}

fn merge_reservoirs(
    canonical_reservoir: Reservoir,
    other_reservoir: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
    sample_count_weight: f32,
) -> ReservoirMergeResult {
    // TODO: Balance heuristic MIS weights
    let canonical_confidence_weight = canonical_reservoir.sample_count * sample_count_weight;
    let other_confidence_weight = other_reservoir.sample_count * sample_count_weight;
    let mis_weight_denominator = 1.0 / (canonical_confidence_weight + other_confidence_weight);

    let canonical_mis_weight = canonical_confidence_weight * mis_weight_denominator;
    let canonical_target_function = reservoir_target_function(canonical_reservoir, world_position, world_normal, diffuse_brdf);
    let canonical_resampling_weight = canonical_mis_weight * (canonical_target_function.a * canonical_reservoir.unbiased_contribution_weight);

    let other_mis_weight = canonical_confidence_weight * mis_weight_denominator;
    let other_target_function = reservoir_target_function(other_reservoir, world_position, world_normal, diffuse_brdf);
    let other_resampling_weight = other_mis_weight * (other_target_function.a * other_reservoir.unbiased_contribution_weight);

    var combined_reservoir = empty_reservoir();
    var combined_reservoir_weight_sum = canonical_resampling_weight + other_resampling_weight;
    combined_reservoir.sample_count = min(canonical_reservoir.sample_count + other_reservoir.sample_count, SAMPLE_COUNT_CAP);

    // https://yusuketokuyoshi.com/papers/2024/Efficient_Visibility_Reuse_for_Real-time_ReSTIR_(Supplementary_Document).pdf
    combined_reservoir.visibility = max(0.0, (canonical_reservoir.visibility * canonical_resampling_weight
        + other_reservoir.visibility * other_resampling_weight) / combined_reservoir_weight_sum);

    if rand_f(rng) < other_resampling_weight / combined_reservoir_weight_sum {
        combined_reservoir.sample = other_reservoir.sample;

        let inverse_target_function = select(0.0, 1.0 / other_target_function.a, other_target_function.a > 0.0);
        combined_reservoir.unbiased_contribution_weight = combined_reservoir_weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_target_function.rgb);
    } else {
        combined_reservoir.sample = canonical_reservoir.sample;

        let inverse_target_function = select(0.0, 1.0 / canonical_target_function.a, canonical_target_function.a > 0.0);
        combined_reservoir.unbiased_contribution_weight = combined_reservoir_weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_target_function.rgb);
    }
}

fn reservoir_target_function(reservoir: Reservoir, world_position: vec3<f32>, world_normal: vec3<f32>, diffuse_brdf: vec3<f32>) -> vec4<f32> {
    if !reservoir_valid(reservoir) { return vec4(0.0); }
    let light_contribution = calculate_light_contribution(reservoir.sample, world_position, world_normal).radiance;
    let target_function = luminance(light_contribution * diffuse_brdf);
    return vec4(light_contribution, target_function);
}
