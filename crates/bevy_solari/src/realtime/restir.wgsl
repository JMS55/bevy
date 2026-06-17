// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_range_u, rand_u, sample_disk}
#import bevy_render::maths::PI
#import bevy_render::utils::{octahedral_decode, octahedral_encode}
#import bevy_solari::brdf::{brdf_pdf, evaluate_and_sample_brdf, evaluate_brdf, F_AB}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, permute_pixel, pixel_dissimilar}
#import bevy_solari::presample_light_tiles::unpack_resolved_light_sample
#import bevy_solari::realtime_bindings::{constants, depth_buffer, gbuffer, light_tile_resolved_samples, light_tile_samples, motion_vectors, previous_depth_buffer, previous_gbuffer, previous_view, reservoirs_a, reservoirs_b, Reservoir, view, view_output}
#import bevy_solari::sampling::{balance_heuristic, calculate_resolved_light_contribution, isinf, isnan, LightSample, NULL_LIGHT_ID, power_heuristic, resolve_light_sample, ResolvedLightSample, trace_light_visibility}
#import bevy_solari::scene_bindings::{light_sources, LIGHT_NOT_PRESENT_THIS_FRAME, MIRROR_ROUGHNESS_THRESHOLD, previous_frame_light_id_translations, RAY_T_MAX, RAY_T_MIN, resolve_ray_hit_full, ResolvedMaterial, ResolvedRayHitFull, trace_ray}
#import bevy_solari::world_cache::{get_cell_size, query_world_cache, WORLD_CACHE_CELL_LIFETIME}
#ifdef DLSS_RR_GUIDE_BUFFERS
#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_solari::realtime_bindings::{diffuse_albedo, specular_albedo, normal_roughness, specular_motion_vectors}
#import bevy_solari::resolve_dlss_rr_textures::env_brdf_approx2
#endif

const INITIAL_DI_SAMPLES = 8u;
// NEE RIS candidates at bounce >= 1 vertices. Their result is frozen into the reconnection
// payload (L_at_rc) as sub-path noise, where extra candidates are far less visible than at
// bounce 0 (whose candidate directly becomes the resampled target). Quality/perf knob.
const SECONDARY_DI_SAMPLES = 4u;
const MAX_BOUNCES = 3u;
const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const CONFIDENCE_WEIGHT_CAP = 8.0;
// Below this value of mix(1, perceptual_roughness, metallic) the specular lobe dominates
// and temporal/spatial neighbors rarely share the lobe direction — resampling
// from them adds variance without quality gain. Pure dielectrics always equal
// 1.0 here regardless of roughness, so they are never skipped.
const SPECULAR_DOMINANCE_SKIP_RESAMPLING_THRESHOLD = 0.3;
// === Footprint-based reconnection criteria (ReSTIR PT Enhanced 2026, Section 4) ===
// A path's x1 -> x2 reconnection segment is only entered into the reservoir (i.e. shared with
// other pixels) when reusing it stays low variance. Otherwise the path is shaded directly into
// this pixel and never published, which avoids the correlated specular fireflies a reconnection
// shift through a sharp BSDF lobe produces (see generate_initial_reservoir).
//
// Ray-footprint multiplier: the area a sample represents at x2 (when traced from x1) must be at
// least (KAPPA / 100) of the primary ray footprint. Smaller footprints — a sharp lobe and/or a
// short x1 -> x2 segment — fail the test. Larger KAPPA is more conservative (excludes more).
// ReSTIR PT Enhanced uses 0.02; calibrate by eye, raising it if specular blobs persist.
const RECONNECTION_FOOTPRINT_KAPPA = 0.02;
// Roughness guard (Enhanced Section 4.2). Used as a single-vertex floor at x1 for specular-lobe
// samples (footprint bounds get unreliable at very low roughness / high curvature), and as the
// specular-dominance cutoff at x2 (a sharper reflector there makes the stored radiance
// view-dependent and unsafe to reuse from a neighbor's connection direction).
const RECONNECTION_ROUGHNESS_MIN = 0.3;

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
    let initial = generate_initial_reservoir(surface.world_position, surface.world_normal, wo, surface.material, workgroup_id.xy, global_id.xy, &rng);
    let initial_reservoir = initial.reservoir;

    // Stage the directly-accumulated non-resampled radiance (mirror / sharp-specular paths) for
    // spatial_and_shade, which reads it back out of view_output and adds it to the final pixel
    // color. Written unconditionally (it is usually zero) so the read never sees a stale value.
    textureStore(view_output, global_id.xy, vec4(initial.non_resampled_radiance, 0.0));

    // Skip resampling for specular-dominated surfaces — temporal/spatial neighbors rarely
    // share the lobe direction, so resampling adds variance without quality gain.
    if mix(1.0, surface.material.perceptual_roughness, surface.material.metallic) < SPECULAR_DOMINANCE_SKIP_RESAMPLING_THRESHOLD {
        reservoirs_b[pixel_index] = initial_reservoir;
        return;
    }

    let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal);
    // PreviousViewUniforms doesn't expose world_position — derive it as the world-space image of the
    // view-space origin: world_from_view * (0,0,0,1), with world_from_view = world_from_clip * clip_from_view.
    let prev_camera_homog = previous_view.world_from_clip * (previous_view.clip_from_view * vec4(0.0, 0.0, 0.0, 1.0));
    let prev_camera_world_position = prev_camera_homog.xyz / prev_camera_homog.w;
    let merge_result = merge_reservoirs(initial_reservoir, surface.world_position, surface.world_normal, surface.material,
        temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.material, prev_camera_world_position, false, &rng);

    reservoirs_b[pixel_index] = merge_result.merged_reservoir;
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    // Constant offset gives this pass a distinct RNG stream: initial_and_temporal seeds
    // with the same pixel_index + frame_index and would otherwise replay the identical
    // rand sequence in this pass.
    var rng = pixel_index + constants.frame_index + 0x6A09E667u;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_a[pixel_index] = empty_reservoir();
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    let input_reservoir = reservoirs_b[pixel_index];
    let wo = normalize(view.world_position - surface.world_position);
    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let F_ab = F_AB(surface.material.perceptual_roughness, NdotV);

    var combined_reservoir: Reservoir;
    var shade_brdf_radiance: vec3<f32>;
    if mix(1.0, surface.material.perceptual_roughness, surface.material.metallic) < SPECULAR_DOMINANCE_SKIP_RESAMPLING_THRESHOLD {
        // Specular-dominated: shade the input directly without spatial reuse (see
        // initial_and_temporal).
        combined_reservoir = input_reservoir;
        var resolved: ResolvedLightSample;
        if input_reservoir.light_sample.light_id != NULL_LIGHT_ID {
            resolved = resolve_light_sample(input_reservoir.light_sample, light_sources[input_reservoir.light_sample.light_id >> 16u]);
        }
        shade_brdf_radiance = reservoir_contribution(input_reservoir, resolved, surface.world_position, surface.world_normal, wo, surface.material, F_ab).brdf_radiance;
    } else {
        let spatial = load_spatial_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, &rng);
        let merge_result = merge_reservoirs(input_reservoir, surface.world_position, surface.world_normal, surface.material,
            spatial.reservoir, spatial.world_position, spatial.world_normal, spatial.material, view.world_position, true, &rng);
        combined_reservoir = merge_result.merged_reservoir;
        shade_brdf_radiance = merge_result.selected_sample_brdf_radiance;
    }

    reservoirs_a[pixel_index] = combined_reservoir;

    var pixel_color = shade_brdf_radiance * combined_reservoir.unbiased_contribution_weight;
    pixel_color += surface.material.emissive;
    // Mirror-emissive contribution staged in view_output by initial_and_temporal.
    pixel_color += textureLoad(view_output, global_id.xy).rgb;
    pixel_color *= view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));

#ifdef VISUALIZE_WORLD_CACHE
    textureStore(view_output, global_id.xy, vec4(query_world_cache(surface.world_position, surface.world_normal, view.world_position, RAY_T_MAX, WORLD_CACHE_CELL_LIFETIME, &rng) * view.exposure, 1.0));
#endif
}

// Unified-reservoir ReSTIR PT: every candidate is a complete path described by a
// reconnection vertex x_rc and the radiance L_at_rc leaving x_rc toward x1.
//   - Length-1 paths (bounce-0 NEE): x_rc = the chosen light vertex.
//   - Length >= 2 paths: x_rc = x2 (the first BRDF-sampled hit), regardless of how
//     many more bounces follow. Deeper NEE/emissive contributions are folded into
//     L_at_x_rc via the throughput_past_x1 factor.
// At shade time: pixel = brdf(x1, x1->rc) * L_at_rc * visibility(x1, rc) * W.
// The primary BRDF*cos is *not* baked into L_at_rc; it is applied externally.
struct InitialSamplingResult {
    reservoir: Reservoir,
    // Radiance from path candidates that are not reuse-safe — mirror/sharp-lobe reconnections
    // that fail the footprint criteria, and directly-visible mirror emitters. Accumulated
    // straight into the pixel instead of being published into the reservoir for reuse.
    non_resampled_radiance: vec3<f32>,
}

fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, workgroup_id: vec2<u32>, pixel_id: vec2<u32>, rng: ptr<function, u32>) -> InitialSamplingResult {
    var reservoir = empty_reservoir();
    reservoir.confidence_weight = 1.0;
    var non_resampled_radiance = vec3(0.0);
    var w_sum = 0.0;
    var selected_target_function = 0.0;

#ifdef DLSS_RR_GUIDE_BUFFERS
    // Primary surface replacement (PSR) for perfect mirrors: follow the delta reflection
    // chain to the first non-mirror hit and write that surface's attributes — reflected
    // into the mirror's virtual space — over the DLSS RR guide buffers, so the denoiser
    // treats this pixel as directly seeing the reflected surface.
    // https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/#primary_surface_replacement
    var mirror_rotations = reflection_matrix(world_normal);
    var psr_finished = material.roughness > MIRROR_ROUGHNESS_THRESHOLD || material.metallic <= 0.9999;
#endif

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
    // Whether the x1 -> x2 reconnection is reuse-safe (decided once in the x2-capture block via
    // the footprint criteria). When false, every reconnection candidate built on x2 is shaded
    // directly into non_resampled_radiance instead of being published into the reservoir.
    var x2_reusable = false;
    // Computed once when x2 is captured; reused by every bounce >= 1 candidate plus the
    // bounce-0 emissive/cache candidates (all of which apply the primary BRDF at the
    // x1 -> x2 direction). Also reused in the bounce-0 throughput step (where the
    // sampled wi IS the x1 -> x2 direction).
    var primary_brdf_at_x2 = vec3(0.0);

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let NdotV = max(dot(n, v), 0.0001);
        let F_ab = F_AB(m.perceptual_roughness, NdotV);

        // === NEE candidate at the current vertex ===
        // Stochastic NEE — probability proportional to how "diffuse" the vertex is.
        // Mirror-like metals have such a narrow BRDF lobe that NEE almost never
        // contributes; skip it most of the time there and let BRDF-sampled emissive
        // do the work. Pure dielectrics always run NEE.
        let p_nee = mix(1.0, m.perceptual_roughness, m.metallic);
        // Must be INITIAL_DI_SAMPLES at bounce 0: reservoir_contribution rebuilds the
        // bounce-0 MIS weights with that constant at every reuse. Bounce >= 1 MIS weights
        // are frozen into L_at_rc, so the count there only has to be consistent within
        // this loop iteration (the emissive candidate below uses the same di_samples).
        let di_samples = select(SECONDARY_DI_SAMPLES, INITIAL_DI_SAMPLES, bounce == 0u);
        if rand_f(rng) < p_nee {
            // di_samples of streaming RIS over a workgroup-shared light tile,
            // followed by a single visibility trace for the winning sample (matches the
            // pre-unified restir_di.wgsl structure). Used at every bounce; the per-bounce
            // workgroup_rng init ensures different bounces select different tiles.
            var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y + bounce;
            let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

            var di_weight_sum = 0.0;
            var di_selected_target = 0.0;
            var di_selected_light_sample = LightSample(NULL_LIGHT_ID, 0u);
            var di_selected_world_position = vec4(0.0);
            var di_selected_wi = vec3(0.0);
            // Only used by the bounce >= 1 path (where they're folded into L_at_rc).
            // At bounce 0 we only need the LightSample identity since the merge re-
            // resolves the light fresh each frame.
            var di_selected_radiance = vec3(0.0);
            var di_selected_brdf_current = vec3(0.0);
            var di_selected_inverse_solid_angle_pdf = 0.0;
            var di_selected_brdf_rays_can_hit = false;
            let internal_mis = 1.0 / f32(di_samples);
            let need_gi_fields = bounce > 0u;
            for (var i = 0u; i < di_samples; i++) {
                let tile_sample = light_tile_start + rand_range_u(1024u, rng);
                let resolved = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
                let lc = calculate_resolved_light_contribution(resolved, ray_origin, n);
                let brdf_current = evaluate_brdf(v, lc.wi, n, m, F_ab);
                let target_function = luminance(brdf_current * lc.radiance);
                let w = internal_mis * target_function * lc.inverse_pdf;
                di_weight_sum += w;
                if di_weight_sum > 0.0 && rand_f(rng) * di_weight_sum < w {
                    di_selected_target = target_function;
                    di_selected_light_sample = light_tile_samples[tile_sample];
                    di_selected_world_position = resolved.world_position;
                    di_selected_wi = lc.wi;
                    di_selected_inverse_solid_angle_pdf = lc.inverse_solid_angle_pdf;
                    di_selected_brdf_rays_can_hit = lc.brdf_rays_can_hit;
                    if need_gi_fields {
                        di_selected_radiance = lc.radiance;
                        di_selected_brdf_current = brdf_current;
                    }
                }
            }

            if di_selected_target > 0.0 {
                // Single visibility trace for the surviving DI sample.
                let vis = trace_light_visibility(ray_origin, di_selected_world_position);

                // MIS against the BRDF strategy. With RIS over N candidates the effective
                // NEE pdf at the winner is roughly N * light_pdf(winner). Scale by p_nee
                // for the stochastic-NEE gate.
                var nee_mis_weight = 1.0;
                if di_selected_brdf_rays_can_hit && di_selected_inverse_solid_angle_pdf > 0.0 {
                    let p_nee_strategy = f32(di_samples) * (1.0 / di_selected_inverse_solid_angle_pdf) * p_nee;
                    let p_brdf_at_nee = brdf_pdf(v, di_selected_wi, n, m, F_ab);
                    nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
                }

                if bounce == 0u {
                    // Bounce 0: store the LightSample identity so reservoir_contribution
                    // can re-resolve the light each frame (moving lights, directional
                    // soft-shadow re-sampling). Main-reservoir w_i = di_weight_sum * vis
                    // * mis / p_nee.
                    //
                    // The target function includes nee_mis_weight, matching
                    // reservoir_contribution which recomputes it from the local surface.
                    // This keeps W = w_sum / target free of this pixel's BRDF pdf: the MIS
                    // weight pairs with the bounce-0 emissive strategy of whichever pixel
                    // evaluates the sample, and p_brdf/p_nee are surface- and
                    // view-dependent. Baking it into W would freeze this pixel's partition
                    // into reservoirs reused by other pixels (spatial, across material
                    // variation) and other frames (temporal, under camera motion), so the
                    // NEE + emissive strategies would no longer sum to 1 at the receiver.
                    let nee_w = di_weight_sum * vis * nee_mis_weight / p_nee;
                    w_sum += nee_w;
                    if w_sum > 0.0 && rand_f(rng) * w_sum < nee_w {
                        reservoir.light_sample = di_selected_light_sample;
                        // sample_point / radiance fields are unused when light_sample is
                        // set — reservoir_contribution re-resolves the light freshly each
                        // time.
                        selected_target_function = di_selected_target * nee_mis_weight;
                    }
                } else {
                    // Bounce >= 1: bake the path through this vertex into L_at_rc and
                    // store as a GI candidate. di_W (the sub-reservoir's effective
                    // inverse-pdf = di_weight_sum / di_target, incl. the 1/p_nee stochastic
                    // NEE compensation) replaces the single-sample inverse_pdf; brdf_current
                    // is the BRDF at this vertex toward the chosen light.
                    let di_W = di_weight_sum / di_selected_target;
                    let L_at_rc = throughput_past_x1 * di_selected_brdf_current * di_selected_radiance * vis * di_W * nee_mis_weight / p_nee;
                    if !x2_reusable {
                        // x1 -> x2 not reuse-safe: shade directly instead of publishing (see
                        // the footprint criterion in the x2-capture block).
                        non_resampled_radiance += primary_brdf_at_x2 * L_at_rc;
                    } else {
                        let nee_target = luminance(primary_brdf_at_x2 * L_at_rc);
                        w_sum += nee_target;
                        if w_sum > 0.0 && rand_f(rng) * w_sum < nee_target {
                            reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                            reservoir.sample_point_world_position = x2_position;
                            reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                            reservoir.radiance = L_at_rc;
                            selected_target_function = nee_target;
                        }
                    }
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

#ifdef DLSS_RR_GUIDE_BUFFERS
        if !psr_finished {
            if !isinf(p_brdf) {
                // The lobe sampler took the residual non-delta lobe (metallic can be up to
                // 0.9999 short of pure), so this ray isn't the mirror reflection — keep the
                // resolve pass's guide-buffer defaults this frame.
                psr_finished = true;
            } else if ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999 {
                // Still in the mirror chain; fold this mirror's reflection into the chain.
                mirror_rotations = mirror_rotations * reflection_matrix(ray_hit.world_normal);
            } else {
                psr_finished = true;
                replace_primary_surface(pixel_id, ray_hit, mirror_rotations, world_position);
            }
        }
#endif

        // Capture x2 on the first BRDF jump, and compute the primary BRDF at x2 once —
        // it's reused by every downstream candidate (emissive, cache, every bounce >= 1
        // NEE) since x2 never changes after this.
        if !x2_set {
            x2_position = ray_hit.world_position;
            x2_normal = ray_hit.world_normal;
            // Evaluate at the sampled direction, not at normalize(x2 - x1). The
            // position-reconstructed direction has tiny floating-point error from the
            // ray origin offset (RAY_T_MIN along n) and hit-position rounding, which
            // is enough to push NdotH below the strict 1 - 0.0001 mirror threshold
            // in evaluate_specular_brdf and zero out the BRDF for mirror metals.
            primary_brdf_at_x2 = evaluate_brdf(wo, next_bounce.wi, world_normal, material, primary_F_ab);

            // === Footprint-based reconnection criterion (ReSTIR PT Enhanced 2026, Section 4) ===
            // Decided once for the x1 -> x2 segment that every reconnection candidate shares.
            // ray_footprint = 1 / (p_sigma(x1->x2) * G(x1->x2)) = t^2 / (p_brdf * cos_x2) is the
            // area a sample represents at x2; it goes to 0 for mirror lobes (p_brdf = INF) and
            // shrinks for sharp lobes or short segments. The primary footprint uses a uniform
            // 1/(4*PI) reference density, so the test trades roughness against distance — a sharp
            // lobe is reusable only at long range.
            let cos_x2 = max(dot(ray_hit.world_normal, -next_bounce.wi), 0.0001);
            let ray_footprint = (ray.t * ray.t) / (next_bounce.pdf * cos_x2);
            let primary_dist = length(view.world_position - world_position);
            let primary_footprint = 4.0 * PI * primary_dist * primary_dist / primary_NdotV;
            let footprint_ok = ray_footprint >= (RECONNECTION_FOOTPRINT_KAPPA / 100.0) * primary_footprint;

            // Single-vertex roughness floor at x1, applied only to specular-lobe samples (a
            // diffuse-lobe bounce is always "rough"); guards low-roughness / high-curvature
            // cases where the footprint bound is unreliable (Enhanced Section 4.2).
            let x1_lobe_ok = next_bounce.diffuse_selected || material.perceptual_roughness >= RECONNECTION_ROUGHNESS_MIN;

            // Inverse-footprint guard at the reconnection vertex x2: a sharp specular reflector
            // there makes the stored outgoing radiance view-dependent and wrong when reused from
            // a neighbor's connection direction. mix(1, roughness, metallic) is low only for
            // low-roughness metals; diffuse-dominated dielectrics, rough surfaces, and emissive
            // light vertices are reuse-safe. (Conservative proxy for the exact inverse ray
            // footprint, which needs x2's continuation-lobe pdf a bounce later.)
            let x2_is_light = any(ray_hit.material.emissive > vec3(0.0));
            let x2_end_ok = x2_is_light || mix(1.0, ray_hit.material.perceptual_roughness, ray_hit.material.metallic) >= RECONNECTION_ROUGHNESS_MIN;

            x2_reusable = footprint_ok && x1_lobe_ok && x2_end_ok;
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
            // Stochastic multi-sample NEE: the effective competing NEE strategy pdf
            // for this specific light is p_light * p_nee * di_samples
            // (drawing N RIS candidates concentrates the marginal around any specific
            // direction by ~N; gated by the p_nee stochastic skip).
            let emissive_mis_weight = power_heuristic(p_brdf, p_light * p_nee * f32(di_samples));

            if !x2_reusable {
                // x1 -> x2 reconnection is not reuse-safe (mirror/sharp lobe, or footprint /
                // roughness gate failed). The contribution is fully determined by the traced
                // path and only valid at this pixel, so accumulate it directly rather than
                // publishing it into the reservoir — a shift to a neighbor would otherwise
                // either waste it or inflate it into a correlated firefly. This generalizes the
                // original mirror-emissive special case: mirror lobes always land here (p_brdf =
                // INF -> ray_footprint = 0), where emissive_mis_weight is exactly 1.
                non_resampled_radiance += primary_brdf_at_x2 * throughput_past_x1 * ray_hit.material.emissive * emissive_mis_weight;
            } else if bounce == 0u {
                // At bounce 0 the reconnection vertex IS the directly-visible light, so the
                // radiance leaving x2 toward x1 is exactly the material emission — no frozen
                // sub-path factors. For the sample to be shareable across pixels, its payload
                // must be a pure function of the path, so two generator-specific factors that
                // used to be baked into `radiance` move out of it:
                //  - 1/p_brdf (carried by throughput_past_x1) goes into the candidate weight,
                //    where inverse pdfs belong — it ends up in W like any sampling density.
                //  - emissive_mis_weight stays in the target function only, and is recomputed
                //    from the *evaluating* pixel's surface in reservoir_contribution (exactly
                //    like the bounce-0 NEE candidate above — see that comment). Both strategy
                //    weights are then built from the same local pdfs at every evaluator, so
                //    NEE + emissive partition each light's energy without over-counting.
                // The evaluator needs the light's area pdf to rebuild the MIS weight; it is
                // view-independent, so carry it bitcast in the otherwise-unused seed field.
                // It's always nonzero, which doubles as the sample's tag (bounce >= 1
                // reconnection samples write seed == 0 and are not reweighted).
                let emissive_w = luminance(primary_brdf_at_x2 * throughput_past_x1 * ray_hit.material.emissive) * emissive_mis_weight;
                let emissive_target = luminance(primary_brdf_at_x2 * ray_hit.material.emissive) * emissive_mis_weight;
                w_sum += emissive_w;
                if w_sum > 0.0 && rand_f(rng) * w_sum < emissive_w {
                    reservoir.light_sample = LightSample(NULL_LIGHT_ID, bitcast<u32>(area_pdf));
                    reservoir.sample_point_world_position = x2_position;
                    reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                    reservoir.radiance = ray_hit.material.emissive;
                    selected_target_function = emissive_target;
                }
            } else {
                // Bounce >= 1 emissive is genuine indirect: x_rc = x2 is a non-light surface,
                // and everything past x2 (including this vertex's NEE<->BRDF MIS partition)
                // is sub-path noise frozen into L_at_rc — the standard reconnection-shift
                // approximation, fine to reuse as-is.
                let emissive_L_at_rc = throughput_past_x1 * ray_hit.material.emissive * emissive_mis_weight;
                let emissive_target = luminance(primary_brdf_at_x2 * emissive_L_at_rc);
                w_sum += emissive_target;
                if w_sum > 0.0 && rand_f(rng) * w_sum < emissive_target {
                    reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                    reservoir.sample_point_world_position = x2_position;
                    reservoir.sample_point_world_normal = octahedral_encode(x2_normal);
                    reservoir.radiance = emissive_L_at_rc;
                    selected_target_function = emissive_target;
                }
            }
        }

        // === Terminate into the world cache (non-metallic previous surface, or the last bounce) ===
        // The cache stores diffuse-ish outgoing radiance at (position, normal) cells, so
        // it's a good approximation when the previous surface was non-metallic (radiance is
        // roughly isotropic) or when we're out of bounce budget and need *something* to close
        // the path. Use the same mix(1, perceptual_roughness, metallic) probability as NEE:
        // 1.0 for pure dielectrics, perceptual roughness for pure metals.
        let p_term = mix(1.0, m.perceptual_roughness, m.metallic);
        let stochastic_terminate = rand_f(rng) < p_term;
        let forced_terminate = bounce == MAX_BOUNCES - 1u;
        if stochastic_terminate || forced_terminate {
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
                if !x2_reusable {
                    // x1 -> x2 not reuse-safe: shade directly instead of publishing.
                    non_resampled_radiance += primary_brdf_at_x2 * cache_L_at_rc;
                    break;
                }
                let cache_target = luminance(primary_brdf_at_x2 * cache_L_at_rc);
                w_sum += cache_target;
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
        // >= so that rr == 0 (throughput hits exactly 0 at grazing angles) is a guaranteed
        // break even on the rare rand_f() == 0.0, never a 0/0 = NaN in the divides below.
        let rr = saturate(luminance(full_throughput));
        if rand_f(rng) >= rr { break; }
        throughput_past_x1 /= rr;
        full_throughput /= rr;
    }

    if selected_target_function > 0.0 {
        reservoir.unbiased_contribution_weight = w_sum / selected_target_function;
    }
    return InitialSamplingResult(reservoir, non_resampled_radiance);
}

#ifdef DLSS_RR_GUIDE_BUFFERS
// https://en.wikipedia.org/wiki/Householder_transformation
fn reflection_matrix(plane_normal: vec3<f32>) -> mat3x3<f32> {
    // N times Nᵀ.
    let n_nt = mat3x3<f32>(
        plane_normal * plane_normal.x,
        plane_normal * plane_normal.y,
        plane_normal * plane_normal.z,
    );
    let identity_matrix = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    return identity_matrix - n_nt * 2.0;
}

fn replace_primary_surface(pixel_id: vec2<u32>, ray_hit: ResolvedRayHitFull, mirror_rotations: mat3x3<f32>, primary_surface_world_position: vec3<f32>) {
    // Simplification: apply all rotations in the chain around the first mirror, rather
    // than applying each rotation around its respective mirror.
    let virtual_position = (mirror_rotations * (ray_hit.world_position - primary_surface_world_position)) + primary_surface_world_position;
    // Approximation: reuses this frame's mirror chain for the previous-frame position
    // (a moving mirror's previous orientation isn't tracked).
    let virtual_previous_frame_position = (mirror_rotations * (ray_hit.previous_frame_world_position - primary_surface_world_position)) + primary_surface_world_position;
    let specular_motion_vector = calculate_motion_vector(virtual_position, virtual_previous_frame_position);

    let F0 = calculate_F0(ray_hit.material.base_color, ray_hit.material.metallic, vec3(ray_hit.material.reflectance));
    let wo = normalize(view.world_position - virtual_position);
    let virtual_normal = normalize(mirror_rotations * ray_hit.world_normal);

    textureStore(specular_motion_vectors, pixel_id, vec4(specular_motion_vector, vec2(0.0)));
    textureStore(diffuse_albedo, pixel_id, vec4(calculate_diffuse_color(ray_hit.material.base_color, ray_hit.material.metallic, 0.0, 0.0), 0.0));
    textureStore(specular_albedo, pixel_id, vec4(env_brdf_approx2(F0, ray_hit.material.roughness, virtual_normal, wo), 0.0));
    textureStore(normal_roughness, pixel_id, vec4(virtual_normal, ray_hit.material.perceptual_roughness));
}

fn calculate_motion_vector(world_position: vec3<f32>, previous_world_position: vec3<f32>) -> vec2<f32> {
    let clip_position_t = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    let clip_position = clip_position_t.xy / clip_position_t.w;
    let previous_clip_position_t = previous_view.unjittered_clip_from_world * vec4(previous_world_position, 1.0);
    let previous_clip_position = previous_clip_position_t.xy / previous_clip_position_t.w;
    // These motion vectors are used as offsets to UV positions and are stored
    // in the range -1,1 to allow offsetting from the one corner to the
    // diagonally-opposite corner in UV coordinates, in either direction.
    // A difference between diagonally-opposite corners of clip space is in the
    // range -2,2, so this needs to be scaled by 0.5. And the V direction goes
    // down where clip space y goes up, so y needs to be flipped.
    return (clip_position - previous_clip_position) * vec2(0.5, -0.5);
}
#endif

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>) -> NeighborInfo {
    if bool(constants.reset) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));

    // If reprojection lands off-screen, intentionally fall back to this pixel's own
    // previous-frame reservoir rather than dropping history: the dissimilarity check
    // below still validates the surface, and a same-pixel guess that passes it beats
    // restarting from a confidence-1 initial reservoir at the screen edge.
    var point_temporal_pixel_id = pixel_id;
    if all(temporal_pixel_id_float >= vec2(0.0)) && all(temporal_pixel_id_float < view.main_pass_viewport.zw) {
        point_temporal_pixel_id = vec2<u32>(temporal_pixel_id_float);
    }

    // permute_pixel's 4x4 offset cycle wants a random value that is uniform across all
    // pixels of a frame. constants.frame_index can't be used raw: it's frame_count times
    // an even multiplier (node.rs), so its low 4 bits cycle degenerately (x offset only
    // ever 0 or 2, period 8), which synchronizes temporal-history rejections screen-wide
    // into visible pulsing under camera motion. Hash it once instead.
    var permute_rng = constants.frame_index;
    let permuted_temporal_pixel_id = permute_pixel(point_temporal_pixel_id, rand_u(&permute_rng), view.main_pass_viewport.zw);

    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, permuted_temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, permuted_temporal_pixel_id, 0), temporal_depth, permuted_temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    let temporal_pixel_index = permuted_temporal_pixel_id.x + permuted_temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    var temporal = NeighborInfo(reservoirs_a[temporal_pixel_index], temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material);

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

fn load_spatial_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> NeighborInfo {
    for (var i = 0u; i < 5u; i++) {
        let spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        // The disk sample can land back on the center pixel. Merging a reservoir with
        // itself leaves the UCW unchanged but wastes the merge's visibility rays and
        // double counts confidence — try a different neighbor instead.
        if all(spatial_pixel_id == pixel_id) {
            continue;
        }

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view) {
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        return NeighborInfo(reservoirs_b[spatial_pixel_index], spatial_surface.world_position, spatial_surface.world_normal, spatial_surface.material);
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
    // brdf(wo, wi) * radiance at canonical for the selected sample (already evaluated
    // inside `reservoir_contribution`). Unshadowed, but valid to shade with: visibility is
    // multiplied into the other sample's *target function*, so an occluded other sample
    // has zero resampling weight and can never be the selected sample.
    // Shade time just multiplies by `unbiased_contribution_weight`.
    selected_sample_brdf_radiance: vec3<f32>,
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
    // Camera position at which `other_reservoir` was generated. Equal to view.world_position
    // for spatial reuse (same-frame neighbor) but previous_view.world_position for temporal
    // reuse — using current view there gives a wrong wo at the temporal pixel, biasing
    // p̂_n's BRDF and thus the m_n MIS weight under camera motion (notably zoom).
    other_view_position: vec3<f32>,
    is_spatial: bool,
    rng: ptr<function, u32>,
) -> ReservoirMergeResult {
    var canonical_resolved: ResolvedLightSample;
    if canonical_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        canonical_resolved = resolve_light_sample(canonical_reservoir.light_sample, light_sources[canonical_reservoir.light_sample.light_id >> 16u]);
    }

    let canonical_wo = normalize(view.world_position - canonical_world_position);
    let canonical_NdotV = max(dot(canonical_world_normal, canonical_wo), 0.0001);
    let canonical_F_ab = F_AB(canonical_material.perceptual_roughness, canonical_NdotV);
    let canonical_sample_at_canonical = reservoir_contribution(canonical_reservoir, canonical_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);

    // Empty neighbor (disocclusion, reset, despawned-light rejection, no similar spatial pixel
    // found, or a reservoir left empty because its surface was shadowed for several frames):
    // the merge below degenerates to exactly the canonical reservoir (t_c = 1 and a zero
    // other-sample weight), so skip the neighbor resolve, the three cross contributions, the
    // jacobians, and the visibility traces.
    //
    // The emptiness test (not just confidence_weight == 0) is what fixes moving-occluder shadow
    // lag. A point shadowed for several frames keeps merging a fresh empty reservoir with its
    // also-empty history, so the stored reservoir holds no sample yet its confidence_weight
    // climbs to the cap. Without treating empty-but-confident as no-neighbor, the frame the
    // occluder moves away the fresh, correctly-lit canonical would be MIS-weighted against that
    // stale confidence (via the c_n * canonical_sample_at_other term in the balance denominator)
    // and suppressed to ~c_c / (c_c + c_n) of its true value, ramping back to full only over the
    // confidence-cap window. Returning canonical's own (already unbiased) estimate makes the
    // reveal snap to full strength in one frame.
    //
    // Only *empty* reservoirs qualify. A neighbor that holds a valid sample merely occluded at
    // this pixel keeps its confidence, so its legitimate balance-heuristic share still applies
    // (m_c < 1) — short-circuiting those would over-count the canonical and over-brighten static
    // penumbrae, multi-light regions, and surfaces with occluded-here GI reconnections.
    let other_is_empty = other_reservoir.light_sample.light_id == NULL_LIGHT_ID && all(other_reservoir.radiance == vec3(0.0));
    if other_reservoir.confidence_weight == 0.0 || other_is_empty {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_at_canonical.brdf_radiance);
    }

    var other_resolved: ResolvedLightSample;
    if other_reservoir.light_sample.light_id != NULL_LIGHT_ID {
        other_resolved = resolve_light_sample(other_reservoir.light_sample, light_sources[other_reservoir.light_sample.light_id >> 16u]);
    }
    let other_wo = normalize(other_view_position - other_world_position);
    let other_NdotV = max(dot(other_world_normal, other_wo), 0.0001);
    let other_F_ab = F_AB(other_material.perceptual_roughness, other_NdotV);

    // Contributions for resampling and MIS
    var other_sample_at_canonical = reservoir_contribution(other_reservoir, other_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    var canonical_sample_at_other = reservoir_contribution(canonical_reservoir, canonical_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);
    let other_sample_at_other = reservoir_contribution(other_reservoir, other_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);

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

    // Shifts with jacobians outside [1/8, 8] are inadmissible — a huge jacobian explodes
    // variance by inflating the reused sample's weight (or crushing the canonical's MIS
    // weight via the balance denominator). Each jacobian is zeroed independently so the
    // MIS weights stay a valid per-sample partition: a zeroed other-sample jacobian kills
    // only that sample's resampling weight, and a zeroed canonical jacobian only removes
    // the neighbor term from the canonical's balance denominator (m_c -> 1). The band is
    // reciprocal-symmetric (J and 1/J fail together), so a shift rejected in one direction
    // is also rejected in the other and the weights still sum to 1 at every sample point.
    if other_sample_at_canonical_jacobian < 0.125 || other_sample_at_canonical_jacobian > 8.0 {
        other_sample_at_canonical_jacobian = 0.0;
    }
    if canonical_sample_at_other_jacobian < 0.125 || canonical_sample_at_other_jacobian > 8.0 {
        canonical_sample_at_other_jacobian = 0.0;
    }

    // Visibility for the cross-pixel targets. Skipped when the matching jacobian is zero,
    // since every term the trace would feed is multiplied by that jacobian.
    if other_sample_at_canonical.target_function > 0.0 && other_sample_at_canonical_jacobian > 0.0 {
        let vis = trace_light_visibility(canonical_world_position + canonical_world_normal * RAY_T_MIN, other_sample_at_canonical.sample_world_position);
        other_sample_at_canonical.target_function *= vis;
    }
    if canonical_sample_at_other.target_function > 0.0 && canonical_sample_at_other_jacobian > 0.0 {
        let vis = trace_light_visibility(other_world_position + other_world_normal * RAY_T_MIN, canonical_sample_at_other.sample_world_position);
        canonical_sample_at_other.target_function *= vis;
    }

    // Defensive pairwise MIS (Wyman et al. 2023 ReSTIR course notes, §7.1.3 Eq 7.8 / Algorithm 7
    // — the variant ReSTIR PT adopts by default). The plain balance heuristic lets a neighbor with
    // a large (approximate) p̂ drive the canonical's weight m_c toward 0; when that neighbor is
    // actually a poor estimator here (its reconnection vertex is occluded from / incompatible with
    // this pixel) the reserved weight is lost every frame and the pixel darkens. The defensive
    // term floors the canonical at its confidence share t_c = c_c / (c_c + c_n):
    //   m_c = t_c + (1 - t_c) * balance_c,   m_n = (1 - t_c) * balance_n.
    // For M = 2 this is exactly Algorithm 7. It remains a valid MIS partition (weights sum to 1),
    // so unlike a hard canonical fallback it cannot leak energy / over-brighten. When the neighbor
    // is empty (c_n = 0) it collapses to t_c = 1 -> canonical keeps full weight.
    let total_confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    let defensive_t_c = f32(is_spatial) * select(1.0, canonical_reservoir.confidence_weight / total_confidence_weight, total_confidence_weight > 0.0);

    // Resampling weight for canonical sample
    let canonical_balance_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_sample_at_canonical.target_function,
        other_reservoir.confidence_weight * canonical_sample_at_other.target_function * canonical_sample_at_other_jacobian,
    );
    let canonical_sample_mis_weight = mix(canonical_balance_mis_weight, 1.0, defensive_t_c);
    let canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_sample_at_canonical.target_function * canonical_reservoir.unbiased_contribution_weight;

    // Resampling weight for other sample
    let other_balance_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_sample_at_other.target_function,
        canonical_reservoir.confidence_weight * other_sample_at_canonical.target_function * other_sample_at_canonical_jacobian,
    );
    let other_sample_mis_weight = mix(other_balance_mis_weight, 0.0, defensive_t_c);
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

        return ReservoirMergeResult(combined_reservoir, other_sample_at_canonical.brdf_radiance);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;
        combined_reservoir.light_sample = canonical_reservoir.light_sample;

        let inverse_target_function = select(0.0, 1.0 / canonical_sample_at_canonical.target_function, canonical_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_at_canonical.brdf_radiance);
    }
}

struct ReservoirContribution {
    // brdf(wo, wi) * radiance — the per-sample shading kernel at this vertex.
    // target_function = luminance(brdf_radiance).
    brdf_radiance: vec3<f32>,
    target_function: f32,
    sample_world_position: vec4<f32>,
}

fn reservoir_contribution(reservoir: Reservoir, resolved: ResolvedLightSample, world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> ReservoirContribution {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);

        // MIS weight against the bounce-0 BRDF-emissive strategy, recomputed with THIS
        // surface's BRDF/material rather than baked into the reservoir's W at generation
        // (mirrors the bounce-0 nee_mis_weight in generate_initial_reservoir, which puts
        // the same factor in the stored target function). The NEE and emissive strategies
        // must partition each light's energy per evaluation pixel — w_nee + w_brdf = 1
        // needs both weights built from the local p_brdf/p_nee, while a baked weight
        // would carry the generating pixel's partition into reuse and over/under-count
        // direct light wherever material or view direction differ.
        var nee_mis_weight = 1.0;
        if light_contribution.brdf_rays_can_hit && light_contribution.inverse_solid_angle_pdf > 0.0 {
            // resolve_light_sample's inverse_pdf excludes the 1/light_count light-pick
            // factor that the presampled tiles include (generate_random_light_sample
            // multiplies it in after resolving); add it so the effective NEE pdf here
            // matches the one used at generation and in the emissive candidate's p_light.
            let light_count = arrayLength(&light_sources);
            let inverse_solid_angle_pdf = light_contribution.inverse_solid_angle_pdf * f32(light_count);
            let p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
            let p_nee_strategy = f32(INITIAL_DI_SAMPLES) * (1.0 / inverse_solid_angle_pdf) * p_nee;
            let p_brdf_at_nee = brdf_pdf(wo, light_contribution.wi, world_normal, material, F_ab);
            nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
        }

        let brdf_radiance = light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab) * nee_mis_weight;
        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), resolved.world_position);
    } else if any(reservoir.radiance != vec3(0.0)) {
        // Reconstruct toward the reconnection vertex from the actual ray origin a
        // reconnection ray would use (offset RAY_T_MIN along the normal, matching
        // generate_initial_reservoir's trace and the merge visibility traces).
        // Reconstructing from the un-offset position deviates from the traced
        // direction by ~RAY_T_MIN/distance radians — enough to fail the strict
        // NdotH mirror gate in evaluate_specular_brdf at short reconnection
        // distances or grazing angles, zeroing out mirror reflections at shade.
        let delta = reservoir.sample_point_world_position - (world_position + world_normal * RAY_T_MIN);
        let sample_distance = length(delta);
        let wi = delta / sample_distance;
        var brdf_radiance = reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab);

        // Bounce-0 BRDF-emissive sample (directly-visible light): the seed field carries the
        // light triangle's bitcast area pdf and the stored radiance is the raw emission.
        // Rebuild the MIS weight against THIS surface's NEE strategy — the dual of
        // nee_mis_weight above, mirroring the emissive candidate in
        // generate_initial_reservoir. Both strategy weights are then built from the same
        // local pdfs, so NEE + emissive partition each light's energy at every evaluating
        // pixel and the sample can be reused across pixels without over-counting.
        if reservoir.light_sample.seed != 0u {
            let area_pdf = bitcast<f32>(reservoir.light_sample.seed);
            let light_normal = octahedral_decode(reservoir.sample_point_world_normal);
            let cos_theta_light = max(dot(-wi, light_normal), 0.0001);
            let p_light = area_pdf * sample_distance * sample_distance / cos_theta_light;
            let p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
            let p_brdf = brdf_pdf(wo, wi, world_normal, material, F_ab);
            brdf_radiance *= power_heuristic(p_brdf, p_light * p_nee * f32(INITIAL_DI_SAMPLES));
        }

        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec4(reservoir.sample_point_world_position, 1.0));
    }
}
