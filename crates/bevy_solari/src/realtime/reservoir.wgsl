#define_import_path bevy_solari::reservoir

#import bevy_solari::sampling::LightSample

const NULL_RESERVOIR_SAMPLE = 0xFFFFFFFFu;

struct Reservoir {
    sample: LightSample,
    weight_sum: f32,
    unbiased_contribution_weight: f32,
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(LightSample(vec2(NULL_RESERVOIR_SAMPLE, 0u), vec2(0.0)), 0.0, 0.0);
}
