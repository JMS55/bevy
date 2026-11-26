//! NEE candidates for directional and local (point, spot, rect, and emissive mesh) lights.
//!
//! NEE draws candidates from several strategies: directional lights picked uniformly, and local
//! lights from the shading point's light grid cell, or from the global presampled light tiles.
//! Callers run one RIS pick, with one shadow ray, over the candidates of all of them. That's the
//! same as merging a reservoir per strategy: the directional and local sets don't overlap, and each
//! candidate's pdf is the balance heuristic over the strategies that can pick it, so the pick stays
//! unbiased.
//!
//! Local light candidates are weighted by the balance heuristic over the cell and global
//! strategies, so a light missing from a cell can still be picked by the global candidates, and the
//! estimate stays unbiased. Without the light grid, every local light candidate is a global one.
//!
//! With the light grid, each cell also learns how much of its NEE contribution comes from the sun,
//! and gives directional lights that share of the candidates (clamped so both keep some), see
//! `light_candidate_split`.

import bevy_pbr::render::utils::{rand_f, rand_range_u};
import package::realtime::bindings::{constants, directional_light_count, light_grid, light_tile_resolved_samples, light_tile_samples, local_light_count, view};
import package::realtime::light_grid::{cell_light_pdf, sample_cell_light, LightGridCell};
import package::realtime::presample_light_tiles::unpack_resolved_light_sample;
import package::scene::sampling::{generate_light_sample, LightSample, NULL_LIGHT_ID, ResolvedLightSample};

enable wgpu_ray_query;

/// Directional light candidates on top of the local light candidates, when there are both.
///
/// Scenes rarely have more than one directional light, so more candidates would mostly pick the same
/// light, differing only in the point on its disk, which RIS gains little from.
const DIRECTIONAL_CANDIDATES: u32 = 1u;

/// How many of a RIS loop's candidates come from each strategy: `directional` directional lights,
/// `cell` local lights from the light grid cell's list, and `global` local lights from the global
/// light tiles. Candidates `[0, directional)` are directional, then `cell` cell candidates, then the
/// global ones.
struct LightCandidateSplit {
    directional: u32,
    cell: u32,
    global: u32,
    /// The expected number of candidates of each strategy, which divide the candidates' pdfs. Equal
    /// to the counts, unless the cell has a learned sun/local split (`sun_probability >= 0`), with
    /// which the counts are drawn per RIS loop, see `realize_light_candidate_split`.
    directional_weight: f32,
    cell_weight: f32,
    global_weight: f32,
    sun_probability: f32,
}

/// Local lights get `candidate_count` candidates, split between the cell and global strategies,
/// and directional lights get extra ones of their own (or, with a learned split, their share of
/// the same total).
fn light_candidate_split(candidate_count: u32, cell: LightGridCell) -> LightCandidateSplit {
    let directional_count = directional_light_count();
    let local_count = local_light_count();

    if directional_count == 0u {
        let local = select(candidate_count, 0u, local_count == 0u);
        return local_only_split(local, cell);
    }

    // Without local lights, every candidate goes to directional lights, as without the split,
    // but they always keep at least their own candidates
    if local_count == 0u {
        let directional = max(candidate_count, DIRECTIONAL_CANDIDATES);
        return LightCandidateSplit(directional, 0u, 0u, f32(directional), 0.0, 0.0, -1.0);
    }

    var split = local_only_split(candidate_count, cell);
    split.directional = DIRECTIONAL_CANDIDATES;
    split.directional_weight = f32(DIRECTIONAL_CANDIDATES);

    // The cell's learned sun/local split: the same total, a `sun_probability` share for the sun
    if cell.sun_probability >= 0.0 && candidate_count != 0u {
        let total = f32(candidate_count);
        let local_weight = total * (1.0 - cell.sun_probability);
        split.sun_probability = cell.sun_probability;
        split.directional_weight = total * cell.sun_probability;
        split.cell_weight = local_weight * f32(split.cell) / total;
        split.global_weight = local_weight * f32(split.global) / total;
    }
    return split;
}

/// `local` local light candidates, split between the cell and the global strategies.
fn local_only_split(local: u32, cell: LightGridCell) -> LightCandidateSplit {
    // Cells without a list send their candidates to the global strategy
    if cell.count == 0u || local == 0u {
        return LightCandidateSplit(0u, 0u, local, 0.0, 0.0, f32(local), -1.0);
    }
    let global = clamp(u32(round(f32(local) * constants.light_grid_global_candidate_fraction)), 1u, local);
    return LightCandidateSplit(0u, local - global, global, 0.0, f32(local - global), f32(global), -1.0);
}

fn light_candidate_count(split: LightCandidateSplit) -> u32 {
    return split.directional + split.cell + split.global;
}

/// `x` rounded down or up, up with probability `x - floor(x)`, so that its expectation is `x`.
fn randomized_round(x: f32, rng: ptr<function, u32>) -> u32 {
    let f = floor(x);
    return u32(f) + select(0u, 1u, rand_f(rng) < x - f);
}

/// With a learned sun/local split, draws this RIS loop's candidate counts, whose expectations are
/// the split's weights (randomized rounding, first the sun's share, then the cell's share of the
/// rest), so that the counts barely vary between loops. Drawing each candidate's strategy on its own
/// instead gives binomial counts, and many loops with no local candidate at all.
///
/// The weights stay as they are: they're what the pdfs are divided by, here and wherever MIS
/// evaluates the strategies, so a strategy this loop drew no candidate of is still counted.
fn realize_light_candidate_split(split_in: LightCandidateSplit, rng: ptr<function, u32>) -> LightCandidateSplit {
    var split = split_in;
    if split.sun_probability < 0.0 { return split; }
    let total = split.cell + split.global;
    let directional = min(randomized_round(f32(total) * split.sun_probability, rng), total);
    let local_count = total - directional;
    let cell = min(randomized_round(f32(local_count) * f32(split.cell) / f32(total), rng), local_count);
    split.directional = directional;
    split.cell = cell;
    split.global = local_count - cell;
    return split;
}

/// The sum over every local light candidate of the probability that it picked `light_id`, i.e. the
/// balance heuristic's denominator, without the pdf of the point on the light.
fn local_light_selection_pdf_sum(light_id: u32, cell: LightGridCell, split: LightCandidateSplit) -> f32 {
    var pdf_sum = split.global_weight / f32(max(local_light_count(), 1u));
    if split.cell_weight > 0.0 {
        pdf_sum += split.cell_weight * cell_light_pdf(cell, light_id);
    }
    return pdf_sum;
}

/// `LightCandidate::light_tile_slot` of candidates that carry their own `LightSample`.
const NO_LIGHT_TILE_SLOT: u32 = 0xFFFFFFFFu;
/// Marks a candidate's `light_tile_slot` as a point in `PerLightSamples` rather than a light tile slot.
const PER_LIGHT_POINT_SLOT: u32 = 0x80000000u;

struct LightCandidate {
    /// Only set for directional light candidates. Local light candidates set `light_tile_slot`
    /// instead (a light tile slot, or a presampled point with `PER_LIGHT_POINT_SLOT`), so that RIS
    /// loops only load the `LightSample` of the candidate they pick, see `candidate_light_sample`.
    light_sample: LightSample,
    light_tile_slot: u32,
    /// `inverse_pdf` is divided by the light's selection pdf summed over the candidates of its
    /// strategies, so the candidate's RIS weight is just `target_function * inverse_pdf`.
    resolved_light_sample: ResolvedLightSample,
}

/// Generates the `i`th of a RIS loop's candidates, see `LightCandidateSplit`. With a learned split,
/// call `realize_light_candidate_split` first.
fn generate_light_candidate(i: u32, split: LightCandidateSplit, cell: LightGridCell, light_tile_start: u32, rng: ptr<function, u32>) -> LightCandidate {
    if i < split.directional {
        let directional_count = directional_light_count();
        // Directional lights come first in the light array
        let light_id = rand_range_u(directional_count, rng);
        var sample = generate_light_sample(light_id, rng);
        sample.resolved_light_sample.inverse_pdf *= f32(directional_count) / split.directional_weight;
        return LightCandidate(sample.light_sample, NO_LIGHT_TILE_SLOT, sample.resolved_light_sample);
    }

    if i < split.directional + split.cell {
        // A presampled point on a light picked from the cell's list, see `PerLightSamples`
        let cell_sample = sample_cell_light(cell, rand_range_u(cell.total, rng));
        let slot = light_grid.light_slots[cell_sample.light_id];
        let offset = light_grid.point_offsets[slot];
        let light_point = offset + rand_range_u(light_grid.point_offsets[slot + 1u] - offset, rng);
        var resolved_light_sample = unpack_resolved_light_sample(light_grid.per_light_samples.resolved_light_samples[light_point], view.exposure);
        resolved_light_sample.inverse_pdf /= split.global_weight / f32(local_light_count()) + split.cell_weight * cell_sample.pdf;
        return LightCandidate(LightSample(cell_sample.light_id << 16u, 0u), PER_LIGHT_POINT_SLOT | light_point, resolved_light_sample);
    }

    let tile_sample = light_tile_start + rand_range_u(1024u, rng);
    var resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
    // The strategies' weights, not this loop's realized counts, which can be 0 for a strategy MIS
    // still has to count (see `realize_light_candidate_split`)
    if split.cell_weight == 0.0 {
        // Only the global strategy, whose pdf is the same for every light
        resolved_light_sample.inverse_pdf *= f32(local_light_count()) / split.global_weight;
    } else {
        let light_id = light_tile_samples[tile_sample].light_id >> 16u;
        resolved_light_sample.inverse_pdf /= local_light_selection_pdf_sum(light_id, cell, split);
    }
    return LightCandidate(LightSample(NULL_LIGHT_ID, 0u), tile_sample, resolved_light_sample);
}

/// The `LightSample` of a candidate: its own, or loaded now from its light tile slot or presampled
/// point.
fn candidate_light_sample(light_sample: LightSample, light_tile_slot: u32) -> LightSample {
    if light_tile_slot == NO_LIGHT_TILE_SLOT { return light_sample; }
    if (light_tile_slot & PER_LIGHT_POINT_SLOT) != 0u {
        return light_grid.per_light_samples.light_samples[light_tile_slot & ~PER_LIGHT_POINT_SLOT];
    }
    return light_tile_samples[light_tile_slot];
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
