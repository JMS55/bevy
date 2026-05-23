// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{octahedral_decode, octahedral_encode, rand_f, rand_range_u, sample_disk}
#import bevy_render::maths::PI
#import bevy_solari::brdf::{brdf_pdf, evaluate_and_sample_brdf, evaluate_and_sample_brdf_uniform_diffuse, evaluate_brdf, EvaluateAndSampleBrdfResult, F_AB}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, permute_pixel, pixel_dissimilar}
#import bevy_solari::presample_light_tiles::unpack_resolved_light_sample
#import bevy_solari::realtime_bindings::{accumulation_texture, constants, depth_buffer, gbuffer, light_tile_resolved_samples, light_tile_samples, motion_vectors, previous_depth_buffer, previous_gbuffer, previous_view, reservoirs_a, reservoirs_b, Reservoir, view, view_output}
#import bevy_solari::sampling::{balance_heuristic, calculate_resolved_light_contribution, generate_random_light_sample, isnan, LightSample, NULL_LIGHT_ID, power_heuristic, resolve_light_sample, ResolvedLightSample, sample_random_light, trace_light_visibility}
#import bevy_solari::scene_bindings::{light_sources, LIGHT_NOT_PRESENT_THIS_FRAME, previous_frame_light_id_translations, RAY_T_MAX, RAY_T_MIN, resolve_ray_hit_full, ResolvedMaterial, trace_ray}
#import bevy_solari::world_cache::{get_cell_size, query_world_cache, WORLD_CACHE_CELL_LIFETIME}

const INITIAL_DI_SAMPLES = 8u;
const GI_MAX_BOUNCES = 3u;
const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 8.0;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_b[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let wo = normalize(view.world_position - surface.world_position);
    let initial_reservoir = generate_initial_reservoir(surface.world_position, surface.world_normal, wo, surface.material, workgroup_id.xy, &rng);

    let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal);
    let merge_result = merge_reservoirs(initial_reservoir, surface.world_position, surface.world_normal, surface.material,
        temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.material, false, &rng);

    reservoirs_b[pixel_index] = merge_result.merged_reservoir;
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_index;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_a[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let spatial = load_spatial_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, &rng);

    let input_reservoir = reservoirs_b[pixel_index];
    let merge_result = merge_reservoirs(input_reservoir, surface.world_position, surface.world_normal, surface.material,
        spatial.reservoir, spatial.world_position, spatial.world_normal, spatial.material, true, &rng);
    var combined_reservoir = merge_result.merged_reservoir;

    reservoirs_a[pixel_index] = combined_reservoir;

    // Visibility is now folded into the merge's resampling weights (one ray per merge,
    // tracing the "other sample at canonical" pair), so the deferred check is redundant.
    // combined_reservoir.unbiased_contribution_weight *= trace_light_visibility(surface.world_position + (surface.world_normal * RAY_T_MIN), merge_result.selected_sample_world_position);

    let wo = normalize(view.world_position - surface.world_position);
    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);
    let brdf = evaluate_brdf(wo, merge_result.wi, surface.world_normal, surface.material, F_ab);

    var pixel_color = merge_result.selected_sample_radiance * combined_reservoir.unbiased_contribution_weight;
    pixel_color *= brdf;
    pixel_color += surface.material.emissive;
    pixel_color *= view.exposure;

    // Accumulate over frames (like pathtracer.wgsl) — sample count stashed in alpha
    // var sample_count = 0.0;
    // if !bool(constants.reset) {
    //     let old = textureLoad(accumulation_texture, global_id.xy);
    //     sample_count = old.a;
    //     pixel_color = mix(old.rgb, pixel_color, 1.0 / (sample_count + 1.0));
    // }
    // textureStore(accumulation_texture, global_id.xy, vec4(pixel_color, sample_count + 1.0));
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}

// Unified-reservoir ReSTIR PT: every candidate is a complete path described by a
// reconnection vertex x_rc and the radiance L_at_rc leaving x_rc toward x1.
//   - Length-1 paths (bounce-0 NEE): x_rc = the chosen light vertex.
//   - Length >= 2 paths: x_rc = x2 (the first BRDF-sampled hit), regardless of how
//     many more bounces follow. Deeper NEE/emissive contributions are folded into
//     L_at_x_rc via the throughput_past_x1 factor.
// At shade time: pixel = brdf(x1, x1->rc) * L_at_rc * visibility(x1, rc) * W.
// The primary BRDF*cos is *not* baked into L_at_rc; it is applied externally.
fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    var w_sum = 0.0;
    var selected_target_function = 0.0;

    let primary_NdotV = max(dot(world_normal, wo), 0.0001);
    let primary_F_ab = F_AB(material.perceptual_roughness, primary_NdotV);

    var ray_origin = world_position + (world_normal * RAY_T_MIN);
    var n = world_normal;
    var v = wo;
    var m = material;

    // Throughput along the path past x1, EXCLUDING brdf*cos at x1. At bounce >= 1
    // this carries (1/pdf_brdf_0) and the brdf*cos/pdf factors of any deeper jumps.
    var throughput_past_x1 = vec3(1.0);
    // Pathtracer-style full throughput (brdf*cos/pdf accumulated at every bounce).
    // Used ONLY for Russian roulette; it is bounded by albedo at each step, whereas
    // throughput_past_x1 = 1/pdf at bounce 0 can be tiny for sharp specular lobes.
    var full_throughput = vec3(1.0);

    // First BRDF-sampled hit (the reconnection vertex x2 shared by every length >= 2 candidate).
    var x2_position = vec3(0.0);
    var x2_normal = vec3(0.0);
    var x2_set = false;
    // Computed once when x2 is captured; reused by every bounce >= 1 candidate plus the
    // bounce-0 emissive/cache candidates (all of which apply the primary BRDF at the
    // x1 -> x2 direction). Also reused as `primary_brdf_at_next` in the bounce-0
    // throughput step (where next_bounce.wi IS the x1 -> x2 direction).
    var wi_to_x2 = vec3(0.0);
    var primary_brdf_at_x2 = vec3(0.0);

    for (var bounce = 0u; bounce < 3u; bounce++) {
        let NdotV = max(dot(n, v), 0.0001);
        let F_ab = F_AB(m.perceptual_roughness, NdotV);

        // === NEE candidate at the current vertex ===
        // Stochastic NEE — probability proportional to how "diffuse" the vertex is.
        // Mirror-like metals have such a narrow BRDF lobe that NEE almost never
        // contributes; skip it most of the time there and let BRDF-sampled emissive
        // do the work. Pure dielectrics always run NEE.
        let p_nee = mix(1.0, m.perceptual_roughness, m.metallic);
        if rand_f(rng) < p_nee {
            let nee_sample = generate_random_light_sample(rng);
            var nee = calculate_resolved_light_contribution(nee_sample.resolved_light_sample, ray_origin, n);
            nee.radiance *= trace_light_visibility(ray_origin, nee_sample.resolved_light_sample.world_position);
            // Compensate for the stochastic skip: dividing the source-pdf-inverses by
            // p_nee folds the (1/p_nee) scaling into every downstream contribution
            // and into the NEE leg of the MIS pdf below.
            nee.inverse_pdf /= p_nee;
            nee.inverse_solid_angle_pdf /= p_nee;

            var nee_mis_weight = 1.0;
            if nee.brdf_rays_can_hit {
                let p_nee_strategy = 1.0 / nee.inverse_solid_angle_pdf;
                let p_brdf_at_nee = brdf_pdf(v, nee.wi, n, m, F_ab);
                nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
            }

            // Build the candidate. Bounce 0 NEE is a length-1 path whose reconnection vertex
            // is the chosen light — store the LightSample so the light can be re-resolved
            // (and directional lights handled correctly) at shade time. Longer paths use
            // x_rc = x2 with the radiance baked into L_at_rc.
            //
            // The target uses the vis-aware radiance so occluded candidates get w_i = 0 and
            // are never selected. Mathematically identical to the unshadowed convention
            // since any surviving sample necessarily has vis = 1 — just less plumbing.
            var nee_target: f32;
            if bounce == 0u {
                let primary_brdf_at_nee = evaluate_brdf(wo, nee.wi, world_normal, material, primary_F_ab);
                nee_target = luminance(primary_brdf_at_nee * nee.radiance);
                let nee_w = nee_target * nee.inverse_pdf * nee_mis_weight;
                w_sum += nee_w;
                reservoir.confidence_weight += 1.0;
                if w_sum > 0.0 && rand_f(rng) * w_sum < nee_w {
                    reservoir.light_sample = nee_sample.light_sample;
                    // sample_point / radiance fields are unused when light_sample is set —
                    // reservoir_contribution re-resolves the light freshly each time.
                    selected_target_function = nee_target;
                }
            } else {
                let nee_brdf_current = evaluate_brdf(v, nee.wi, n, m, F_ab);
                let L_at_rc = throughput_past_x1 * nee_brdf_current * nee.radiance * nee.inverse_pdf * nee_mis_weight;
                nee_target = luminance(primary_brdf_at_x2 * L_at_rc);
                w_sum += nee_target;
                reservoir.confidence_weight += 1.0;
                if w_sum > 0.0 && rand_f(rng) * w_sum < nee_target {
                    reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                    reservoir.sample_point_world_position = x2_position;
                    reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                    reservoir.radiance = L_at_rc;
                    selected_target_function = nee_target;
                }
            }
        }

        // === Sample BRDF and trace next ray ===
        let next_bounce = evaluate_and_sample_brdf(v, n, m, F_ab, rng);
        if next_bounce.pdf == 0.0 { break; }

        let ray = trace_ray(ray_origin, next_bounce.wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { break; }
        let ray_hit = resolve_ray_hit_full(ray);
        let p_brdf = next_bounce.pdf;

        // Capture x2 on the first BRDF jump, and compute the primary BRDF at x2 once —
        // it's reused by every downstream candidate (emissive, cache, every bounce >= 1
        // NEE) since x2 never changes after this.
        if !x2_set {
            x2_position = ray_hit.world_position;
            x2_normal = ray_hit.world_normal;
            wi_to_x2 = normalize(x2_position - world_position);
            primary_brdf_at_x2 = evaluate_brdf(wo, wi_to_x2, world_normal, material, primary_F_ab);
            x2_set = true;
        }

        // At bounce 0 the primary brdf*cos is applied externally at shade time, so
        // throughput_past_x1 must exclude it. Dividing next_bounce.throughput by the
        // BRDF at the sampled direction (= primary_brdf_at_x2 at bounce 0, since that
        // IS the x1 -> x2 direction) extracts the remaining factor:
        //  - non-mirror GGX/diffuse: throughput = brdf*cos/pdf -> result = 1/pdf
        //  - mirror specular: throughput = brdf_reflectance/specular_weight, pdf = INF
        //    -> result = 1/specular_weight (avoids 1/INF = 0 which would kill mirror GI)
        // At later bounces include the full brdf*cos/pdf — these are post-x2 and belong in L_at_rc.
        var throughput_step = next_bounce.throughput;
        if bounce == 0u {
            throughput_step = next_bounce.throughput / max(primary_brdf_at_x2, vec3(0.0001));
        }
        throughput_past_x1 *= throughput_step;
        full_throughput *= next_bounce.throughput;

        // === BRDF-sampled emissive candidate (x_rc = x2) ===
        if any(ray_hit.material.emissive > vec3(0.0)) {
            let NdotV_hit = max(dot(ray_hit.world_normal, -next_bounce.wi), 0.0001);
            let light_count = arrayLength(&light_sources);
            let area_pdf = 1.0 / (f32(light_count) * f32(ray_hit.triangle_count) * ray_hit.triangle_area);
            let p_light = area_pdf * ray.t * ray.t / NdotV_hit;
            // Stochastic NEE: NEE only fires at this vertex with probability p_nee,
            // so the effective competing NEE strategy pdf for this specific light is
            // p_light * p_nee (zero for pure mirror metals, full for dielectrics).
            let emissive_mis_weight = power_heuristic(p_brdf, p_light * p_nee);

            let emissive_L_at_rc = throughput_past_x1 * ray_hit.material.emissive * emissive_mis_weight;
            let emissive_target = luminance(primary_brdf_at_x2 * emissive_L_at_rc);
            w_sum += emissive_target;
            reservoir.confidence_weight += 1.0;
            if w_sum > 0.0 && rand_f(rng) * w_sum < emissive_target {
                reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                reservoir.sample_point_world_position = x2_position;
                reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                reservoir.radiance = emissive_L_at_rc;
                selected_target_function = emissive_target;
            }
        }

        // === Terminate into the world cache (diffuse bounces, or the last bounce) ===
        // The cache stores diffuse-ish outgoing radiance at (position, normal) cells, so
        // it's a good approximation when the bounce was diffuse (radiance is roughly
        // isotropic) or when we're out of bounce budget and need *something* to close
        // the path.
        if next_bounce.diffuse_selected || bounce == GI_MAX_BOUNCES - 1u {
            // Only terminate into the cache when the BRDF ray was long enough to clear
            // the cache cell (cell diagonal = sqrt(3) * cell_size). Short rays land in a
            // cell that may straddle nearby occluding geometry and leak light through
            // corners.
            var rng_copy = *rng;
            let world_cache_cell_size = get_cell_size(ray_hit.world_position, view.world_position, ray.t, &rng_copy);
            if ray.t > sqrt(3.0) * world_cache_cell_size {
                let cached_radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, ray.t, WORLD_CACHE_CELL_LIFETIME, rng);
                // The cache stores irradiance; apply the Lambertian diffuse BRDF
                // (base_color / PI) at ray_hit to get outgoing radiance toward the
                // previous vertex (matches the old restir_gi.wgsl convention).
                let cache_outgoing = (ray_hit.material.base_color / PI) * cached_radiance;
                let cache_L_at_rc = throughput_past_x1 * cache_outgoing;
                let cache_target = luminance(primary_brdf_at_x2 * cache_L_at_rc);
                w_sum += cache_target;
                reservoir.confidence_weight += 1.0;
                if w_sum > 0.0 && rand_f(rng) * w_sum < cache_target {
                    reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                    reservoir.sample_point_world_position = x2_position;
                    reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                    reservoir.radiance = cache_L_at_rc;
                    selected_target_function = cache_target;
                }
                break;
            }
        }

        // === Update state for next iteration ===
        ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
        n = ray_hit.world_normal;
        v = -next_bounce.wi;
        m = ray_hit.material;

        // Russian roulette on the pathtracer-style full throughput (which is bounded by
        // albedo at each step); scale BOTH throughput trackers to keep them unbiased.
        let rr = saturate(luminance(full_throughput));
        if rand_f(rng) > rr { break; }
        throughput_past_x1 /= rr;
        full_throughput /= rr;
    }

    if selected_target_function > 0.0 {
        reservoir.unbiased_contribution_weight = w_sum / selected_target_function;
    }
    return reservoir;
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));
    var point_temporal_pixel_id = vec2<u32>(temporal_pixel_id_float);

    if bool(constants.reset) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    if any(temporal_pixel_id_float < vec2(0.0)) || any(temporal_pixel_id_float >= view.main_pass_viewport.zw) {
        point_temporal_pixel_id = pixel_id;
    }

    let permuted_temporal_pixel_id = permute_pixel(point_temporal_pixel_id, constants.frame_index, view.main_pass_viewport.zw);
    var temporal = load_temporal_reservoir_inner(permuted_temporal_pixel_id, depth, world_position, world_normal);

    // Check if the light selected in the previous frame no longer exists in the current frame (e.g. entity despawned)
    if temporal.reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let previous_light_id = temporal.reservoir.light_sample.light_id >> 16u;
        let triangle_id = temporal.reservoir.light_sample.light_id & 0xFFFFu;
        let light_id = previous_frame_light_id_translations[previous_light_id];
        if light_id == LIGHT_NOT_PRESENT_THIS_FRAME {
            return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
        }
        temporal.reservoir.light_sample.light_id = (light_id << 16u) | triangle_id;
    }

    temporal.reservoir.confidence_weight = min(temporal.reservoir.confidence_weight, CONFIDENCE_WEIGHT_CAP);

    return temporal;
}

fn load_temporal_reservoir_inner(temporal_pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, temporal_pixel_id, 0), temporal_depth, temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    let temporal_pixel_index = temporal_pixel_id.x + temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    let temporal_reservoir = reservoirs_a[temporal_pixel_index];

    return NeighborInfo(temporal_reservoir, temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material);
}

fn load_spatial_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> NeighborInfo {
    for (var i = 0u; i < 5u; i++) {
        let spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view) {
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        let spatial_reservoir = reservoirs_b[spatial_pixel_index];
        return NeighborInfo(spatial_reservoir, spatial_surface.world_position, spatial_surface.world_normal, spatial_surface.material);
    }

    return NeighborInfo(empty_reservoir(), world_position, world_normal, empty_material());
}

fn get_neighbor_pixel_id(center_pixel_id: vec2<u32>, search_radius: f32, rng: ptr<function, u32>) -> vec2<u32> {
    var spatial_id = vec2<f32>(center_pixel_id) + sample_disk(search_radius, rng);
    spatial_id = clamp(spatial_id, vec2(0.0), view.main_pass_viewport.zw - 1.0);
    return vec2<u32>(spatial_id);
}

struct NeighborInfo {
    reservoir: Reservoir,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn empty_material() -> ResolvedMaterial {
    return ResolvedMaterial(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
}

fn jacobian(
    new_world_position: vec3<f32>,
    original_world_position: vec3<f32>,
    sample_point_world_position: vec3<f32>,
    sample_point_world_normal: vec3<f32>,
) -> f32 {
    let r = new_world_position - sample_point_world_position;
    let q = original_world_position - sample_point_world_position;
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

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3(0.0),
        0.0,
        vec3(0.0),
        0.0,
        vec2(0.0),
        LightSample(NULL_LIGHT_ID, 0u),
    );
}

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_radiance: vec3<f32>,
    wi: vec3<f32>,
    // Resolved sample world position (vec4 with w=1 for surface points/area lights, w=0 for directional).
    selected_sample_world_position: vec4<f32>,
}

fn merge_reservoirs(
    canonical_reservoir: Reservoir,
    canonical_world_position: vec3<f32>,
    canonical_world_normal: vec3<f32>,
    canonical_material: ResolvedMaterial,
    other_reservoir: Reservoir,
    other_world_position: vec3<f32>,
    other_world_normal: vec3<f32>,
    other_material: ResolvedMaterial,
    // True for spatial merge (neighbor pixel — no baked visibility we can trust at our
    // shading point, must trace fresh). False for temporal merge (same pixel's history —
    // visibility is already baked into the stored radiance for GI samples; only trace
    // when the temporal reservoir is a bounce-0 NEE light_sample, since those are
    // re-resolved fresh each frame and carry no baked visibility).
    is_spatial: bool,
    rng: ptr<function, u32>,
) -> ReservoirMergeResult {
    var canonical_resolved: ResolvedLightSample;
    if canonical_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        canonical_resolved = resolve_light_sample(canonical_reservoir.light_sample, light_sources[canonical_reservoir.light_sample.light_id >> 16u]);
    }
    var other_resolved: ResolvedLightSample;
    if other_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        other_resolved = resolve_light_sample(other_reservoir.light_sample, light_sources[other_reservoir.light_sample.light_id >> 16u]);
    }

    let canonical_wo = normalize(view.world_position - canonical_world_position);
    let canonical_NdotV = max(dot(canonical_world_normal, canonical_wo), 0.0001);
    let canonical_F_ab = F_AB(canonical_material.perceptual_roughness, canonical_NdotV);
    let other_wo = normalize(view.world_position - other_world_position);
    let other_NdotV = max(dot(other_world_normal, other_wo), 0.0001);
    let other_F_ab = F_AB(other_material.perceptual_roughness, other_NdotV);

    // Contributions for resampling and MIS
    let canonical_sample_at_canonical = reservoir_contribution(canonical_reservoir, canonical_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    var other_sample_at_canonical = reservoir_contribution(other_reservoir, other_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    let canonical_sample_at_other = reservoir_contribution(canonical_reservoir, canonical_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);
    let other_sample_at_other = reservoir_contribution(other_reservoir, other_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);

    // Shadow the resampling weight for the "other sample at canonical" pair — that's
    // the one that decides whether other's sample is allowed to win the merge.
    //
    // Skip the trace for temporal merges of GI-style reservoirs: their stored radiance
    // already has the visibility baked in from when this same trace ran in a previous
    // frame's merge, so re-tracing is redundant (and would double-multiply visibility).
    // Bounce-0 NEE (light_sample) reservoirs are re-resolved fresh each frame and carry
    // no baked visibility, so they still need the trace.
    let other_is_light_sample = other_reservoir.light_sample.light_id != NULL_LIGHT_ID;
    let need_visibility_trace = is_spatial || other_is_light_sample;
    if need_visibility_trace && other_sample_at_canonical.target_function > 0.0 {
        let vis = trace_light_visibility(canonical_world_position + canonical_world_normal * RAY_T_MIN, other_sample_at_canonical.sample_world_position);
        other_sample_at_canonical.target_function *= vis;
        other_sample_at_canonical.radiance *= vis;
    }

    // Jacobians for resampling and MIS. Light samples don't need a reprojection jacobian,
    // since resolve_and_calculate_light_contribution already accounts for the shading point's geometry.
    var other_sample_at_canonical_jacobian = 1.0;
    if other_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        other_sample_at_canonical_jacobian = jacobian(
            canonical_world_position,
            other_world_position,
            other_reservoir.sample_point_world_position,
            octahedral_decode(other_reservoir.sample_point_world_normal)
        );
    }
    var canonical_sample_at_other_jacobian = 1.0;
    if canonical_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        canonical_sample_at_other_jacobian = jacobian(
            other_world_position,
            canonical_world_position,
            canonical_reservoir.sample_point_world_position,
            octahedral_decode(canonical_reservoir.sample_point_world_normal)
        );
    }

    // Don't merge samples with huge jacobians, as it explodes the variance
    if other_sample_at_canonical_jacobian > 8.0 || canonical_sample_at_other_jacobian > 8.0 {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_at_canonical.radiance, canonical_sample_at_canonical.wi, canonical_sample_at_canonical.sample_world_position);
    }

    // Resampling weight for canonical sample
    let canonical_sample_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_sample_at_canonical.target_function,
        other_reservoir.confidence_weight * canonical_sample_at_other.target_function * canonical_sample_at_other_jacobian,
    );
    let canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_sample_at_canonical.target_function * canonical_reservoir.unbiased_contribution_weight;

    // Resampling weight for other sample
    let other_sample_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_sample_at_other.target_function,
        canonical_reservoir.confidence_weight * other_sample_at_canonical.target_function * other_sample_at_canonical_jacobian,
    );
    let other_sample_resampling_weight = other_sample_mis_weight * other_sample_at_canonical.target_function * other_reservoir.unbiased_contribution_weight * other_sample_at_canonical_jacobian;

    // Perform resampling
    var combined_reservoir = empty_reservoir();
    combined_reservoir.confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    let weight_sum = canonical_sample_resampling_weight + other_sample_resampling_weight;

    if rand_f(rng) < other_sample_resampling_weight / weight_sum {
        combined_reservoir.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = other_reservoir.radiance;
        combined_reservoir.light_sample = other_reservoir.light_sample;

        let inverse_target_function = select(0.0, 1.0 / other_sample_at_canonical.target_function, other_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_sample_at_canonical.radiance, other_sample_at_canonical.wi, other_sample_at_canonical.sample_world_position);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;
        combined_reservoir.light_sample = canonical_reservoir.light_sample;

        let inverse_target_function = select(0.0, 1.0 / canonical_sample_at_canonical.target_function, canonical_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_at_canonical.radiance, canonical_sample_at_canonical.wi, canonical_sample_at_canonical.sample_world_position);
    }
}

struct ReservoirContribution {
    radiance: vec3<f32>,
    target_function: f32,
    wi: vec3<f32>,
    sample_world_position: vec4<f32>,
}

fn reservoir_contribution(reservoir: Reservoir, resolved: ResolvedLightSample, world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> ReservoirContribution {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);
        let target_function = luminance(light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab));
        return ReservoirContribution(light_contribution.radiance, target_function, light_contribution.wi, resolved.world_position);
    } else if any(reservoir.radiance != vec3(0.0)) {
        let wi = normalize(reservoir.sample_point_world_position - world_position);
        let target_function = luminance(reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab));
        return ReservoirContribution(reservoir.radiance, target_function, wi, vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec3(0.0), vec4(reservoir.sample_point_world_position, 1.0));
    }
}
