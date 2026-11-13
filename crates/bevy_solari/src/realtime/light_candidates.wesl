//! NEE candidates for directional and local (point, spot, rect, and emissive mesh) lights.
//!
//! NEE draws candidates from several strategies: directional lights picked uniformly, and local
//! lights from the shading point's light grid cell, or from the global presampled light tiles.
//! Callers run a separate RIS pick, and a separate shadow ray, for directional and for local
//! light candidates. In one shared pick, an occluded sun's unshadowed contribution dwarfs any local
//! light's, so in shadow the sun wins nearly every pick, and local lights only get through as rare
//! fireflies.
//!
//! Local light candidates are weighted by the balance heuristic over the cell and global
//! strategies, so a light missing from a cell can still be picked by the global candidates, and the
//! estimate stays unbiased. Without the light grid, every local light candidate is a global one.

import bevy_pbr::render::utils::rand_range_u;
import package::realtime::bindings::{constants, directional_light_count, light_samples_per_light, light_tile_resolved_samples, light_tile_samples, local_light_count, per_light_samples, view};
import package::realtime::light_grid::{weighted_cell_pdf, weighted_cell_slot, LightGridCell};
import package::realtime::presample_light_tiles::unpack_resolved_light_sample;
import package::scene::sampling::{generate_light_sample, LightSample, NULL_LIGHT_ID, ResolvedLightSample};

enable wgpu_ray_query;

/// Directional light candidates on top of the local light candidates, when there are both.
///
/// Scenes rarely have more than one directional light, so more candidates would mostly pick the same
/// light, differing only in the point on its disk, which RIS gains little from.
const DIRECTIONAL_CANDIDATES: u32 = 1u;

/// How many of a RIS loop's candidates come from each strategy.
///
/// Candidates `[0, directional)` are directional lights, then `cell` from the light grid cell's
/// list, and the rest (`global`) from the global light tiles.
struct LightCandidateSplit {
    directional: u32,
    cell: u32,
    global: u32,
}

/// Local lights get `local_candidate_count` candidates, and directional lights get extra ones of
/// their own.
fn light_candidate_split(local_candidate_count: u32, cell: LightGridCell) -> LightCandidateSplit {
    let directional_count = directional_light_count();
    let local_count = local_light_count();

    // Without local lights, every candidate goes to directional lights, as without the split
    if local_candidate_count == 0u || local_count == 0u {
        return LightCandidateSplit(select(local_candidate_count, 0u, directional_count == 0u), 0u, 0u);
    }

    let directional = select(DIRECTIONAL_CANDIDATES, 0u, directional_count == 0u);

    // Cells without a list send their candidates to the global strategy
    if cell.count == 0u {
        return LightCandidateSplit(directional, 0u, local_candidate_count);
    }
    let global = clamp(u32(round(f32(local_candidate_count) * constants.light_grid_global_candidate_fraction)), 1u, local_candidate_count);
    return LightCandidateSplit(directional, local_candidate_count - global, global);
}

fn light_candidate_count(split: LightCandidateSplit) -> u32 {
    return split.directional + split.cell + split.global;
}

/// The sum over every local light candidate of the probability that it picked `light_id`, i.e. the
/// balance heuristic's denominator, without the pdf of the point on the light.
fn local_light_selection_pdf_sum(light_id: u32, cell: LightGridCell, split: LightCandidateSplit) -> f32 {
    var pdf_sum = f32(split.global) / f32(max(local_light_count(), 1u));
    if split.cell != 0u {
        pdf_sum += f32(split.cell) * weighted_cell_pdf(cell, light_id);
    }
    return pdf_sum;
}

struct LightCandidate {
    light_sample: LightSample,
    /// `inverse_pdf` is divided by the light's selection pdf summed over the candidates of its
    /// strategies, so the candidate's RIS weight is just `target_function * inverse_pdf`.
    resolved_light_sample: ResolvedLightSample,
}

/// Generates the `i`th of a RIS loop's candidates, see `LightCandidateSplit`.
fn generate_light_candidate(i: u32, split: LightCandidateSplit, cell: LightGridCell, light_tile_start: u32, rng: ptr<function, u32>) -> LightCandidate {
    if i < split.directional {
        let directional_count = directional_light_count();
        // Directional lights come first in the light array
        let light_id = rand_range_u(directional_count, rng);
        var sample = generate_light_sample(light_id, rng);
        sample.resolved_light_sample.inverse_pdf *= f32(directional_count) / f32(split.directional);
        return LightCandidate(sample.light_sample, sample.resolved_light_sample);
    }

    var candidate: LightCandidate;
    var selection_pdf_sum: f32;
    if i < split.directional + split.cell {
        // A presampled point on a light picked from the cell's list. The slot holds the light's
        // slot count, so there's no need to search for its pdf.
        let slot = weighted_cell_slot(cell, rand_range_u(cell.count, rng));
        let light_id = slot & 0xFFFFu;
        candidate = per_light_sample(light_id, rng);
        selection_pdf_sum = f32(split.global) / f32(local_light_count()) + f32(split.cell) * f32(slot >> 16u) / f32(cell.count);
    } else {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        candidate = LightCandidate(light_tile_samples[tile_sample], unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure));
        selection_pdf_sum = local_light_selection_pdf_sum(candidate.light_sample.light_id >> 16u, cell, split);
    }
    candidate.resolved_light_sample.inverse_pdf /= selection_pdf_sum;
    return candidate;
}

/// A presampled point on a light, see `PerLightSamples`.
fn per_light_sample(light_id: u32, rng: ptr<function, u32>) -> LightCandidate {
    let samples_per_light = light_samples_per_light();
    let light_point = light_id * samples_per_light + rand_range_u(samples_per_light, rng);
    let light_sample = per_light_samples.light_samples[light_point];
    let resolved_light_sample = unpack_resolved_light_sample(per_light_samples.resolved_light_samples[light_point], view.exposure);
    return LightCandidate(light_sample, resolved_light_sample);
}

/// A ReSTIR reservoir holding a BRDF-sampled emissive hit stores the hit light's id with this
/// triangle id, so that each domain it is reused in can work out its own NEE pdf for the MIS
/// weight. The seed holds the area pdf of the point given the light, `1 / (triangle_count * area)`.
///
/// Emissive meshes have at most 65535 triangles, so this is never a real triangle id.
const EMISSIVE_HIT_TRIANGLE_ID: u32 = 0xFFFFu;

fn emissive_hit_light_sample(light_id: u32, area_pdf: f32) -> LightSample {
    return LightSample((light_id << 16u) | EMISSIVE_HIT_TRIANGLE_ID, bitcast<u32>(area_pdf));
}

fn is_emissive_hit_light_sample(light_sample: LightSample) -> bool {
    return light_sample.light_id != NULL_LIGHT_ID && (light_sample.light_id & 0xFFFFu) == EMISSIVE_HIT_TRIANGLE_ID;
}
