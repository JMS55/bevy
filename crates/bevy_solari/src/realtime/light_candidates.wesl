//! NEE candidates for directional and local (point, spot, rect, and emissive mesh) lights.
//!
//! NEE draws candidates from two strategies: directional lights picked uniformly, and local lights
//! from the presampled light tiles. Callers run one RIS pick, with one shadow ray, over the
//! candidates of both. That's the same as merging a reservoir of local light candidates with a
//! reservoir of directional light candidates: the two sets don't overlap, and each candidate's pdf
//! only involves its own strategy, so the pick stays unbiased.
//!
//! The light tiles sample every light uniformly, so before the split, the sun was one light among
//! hundreds of emissives in Bistro and rarely a candidate at all. Now the sun is always a candidate,
//! so sunlit surfaces converge quickly. In shadow, the occluded sun's unshadowed contribution still
//! wins most picks and its shadow ray fails, which leaves local lights noisier there than a separate
//! pick per strategy would, but that would cost a second shadow ray.

import bevy_pbr::render::utils::rand_range_u;
import package::realtime::bindings::{directional_light_count, light_tile_resolved_samples, light_tile_samples, local_light_count, view};
import package::realtime::presample_light_tiles::unpack_resolved_light_sample;
import package::scene::sampling::{generate_light_sample, LightSample, ResolvedLightSample};

enable wgpu_ray_query;

/// Directional light candidates on top of the local light candidates, when there are both.
///
/// Scenes rarely have more than one directional light, so more candidates would mostly pick the same
/// light, differing only in the point on its disk, which RIS gains little from.
const DIRECTIONAL_CANDIDATES: u32 = 1u;

/// How many of a RIS loop's candidates come from each strategy.
///
/// Candidates `[0, directional)` are directional lights, and `[directional, directional + local)`
/// local lights.
struct LightCandidateSplit {
    directional: u32,
    local: u32,
}

fn light_candidate_split(candidate_count: u32) -> LightCandidateSplit {
    let directional_count = directional_light_count();
    let local_count = local_light_count();

    if directional_count == 0u {
        return LightCandidateSplit(0u, select(candidate_count, 0u, local_count == 0u));
    }

    // Without local lights, every candidate goes to directional lights, as without the split,
    // but they always keep at least their own candidates
    if local_count == 0u {
        return LightCandidateSplit(max(candidate_count, DIRECTIONAL_CANDIDATES), 0u);
    }

    // Local lights get every candidate asked for, and directional lights get extra ones of their own
    return LightCandidateSplit(DIRECTIONAL_CANDIDATES, candidate_count);
}

fn light_candidate_count(split: LightCandidateSplit) -> u32 {
    return split.directional + split.local;
}

/// The sum over every candidate of the probability that it picked a given local light, without the
/// pdf of the point on the light. The same for every local light.
fn local_light_selection_pdf(split: LightCandidateSplit) -> f32 {
    return f32(split.local) / f32(max(local_light_count(), 1u));
}

struct LightCandidate {
    light_sample: LightSample,
    /// `inverse_pdf` includes the probability of picking the light, summed over the candidates of
    /// its strategy, so the candidate's RIS weight is just `target_function * inverse_pdf`.
    resolved_light_sample: ResolvedLightSample,
}

/// Generates the `i`th of a RIS loop's candidates, see `LightCandidateSplit`.
fn generate_light_candidate(i: u32, split: LightCandidateSplit, light_tile_start: u32, rng: ptr<function, u32>) -> LightCandidate {
    if i < split.directional {
        let directional_count = directional_light_count();
        // Directional lights come first in the light array
        let light_id = rand_range_u(directional_count, rng);
        var sample = generate_light_sample(light_id, rng);
        sample.resolved_light_sample.inverse_pdf *= f32(directional_count) / f32(split.directional);
        return LightCandidate(sample.light_sample, sample.resolved_light_sample);
    }

    let tile_sample = light_tile_start + rand_range_u(1024u, rng);
    var resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
    resolved_light_sample.inverse_pdf /= local_light_selection_pdf(split);
    return LightCandidate(light_tile_samples[tile_sample], resolved_light_sample);
}
