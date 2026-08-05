// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf
enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_u, sample_disk}
#import bevy_solari::brdf::{brdf_pdf, evaluate_brdf, F_AB}
#import bevy_solari::debug::{
    debug_count,
    debug_count_if,
    debug_enabled,
    debug_flags_jacobian_discard_neighbor,
    debug_flags_jacobian_inflate_canonical,
    debug_flags_non_resampled_share,
    debug_flags_x2_not_reusable,
    debug_flags_temporal_status,
    debug_flags_world_cache_probe_exhausted,
    debug_flags_world_cache_sample_count,
    debug_heatmap,
    debug_pack_flags,
    debug_provenance_color,
    debug_relative_std_dev,
    debug_reset_state,
    debug_temporal_status_color,
    debug_tonemap_radiance,
    debug_world_cache_probe_exhausted,
    debug_world_cache_sample_count,
    debug_x2_not_reusable,
    DEBUG_COUNTER_CONFIDENCE_WEIGHT_X10,
    DEBUG_COUNTER_HISTORY_REJECTED_PIXELS,
    DEBUG_COUNTER_JACOBIAN_SPATIAL_DISCARD_NEIGHBOR,
    DEBUG_COUNTER_JACOBIAN_SPATIAL_INFLATE_CANONICAL,
    DEBUG_COUNTER_JACOBIAN_TEMPORAL_DISCARD_NEIGHBOR,
    DEBUG_COUNTER_JACOBIAN_TEMPORAL_INFLATE_CANONICAL,
    DEBUG_COUNTER_NOISE_BYPASS_PERCENT,
    DEBUG_COUNTER_NOISE_DIFFUSE_PERCENT,
    DEBUG_COUNTER_NOISE_HISTORY_REJECTED_PERCENT,
    DEBUG_COUNTER_NOISE_NON_RESAMPLED_SHARE_PERCENT,
    DEBUG_COUNTER_NOISE_OVER_100PCT_PIXELS,
    DEBUG_COUNTER_NOISE_OVER_200PCT_PIXELS,
    DEBUG_COUNTER_NOISE_RELATIVE_STD_DEV_PERCENT,
    DEBUG_COUNTER_NOISE_RESAMPLED_PERCENT,
    DEBUG_COUNTER_NOISE_RESAMPLED_SPECULAR_PERCENT,
    DEBUG_COUNTER_NOISE_SPECULAR_PERCENT,
    DEBUG_COUNTER_NON_RESAMPLED_ENERGY_PERCENT,
    DEBUG_COUNTER_PIXELS_SHADED,
    DEBUG_COUNTER_SAMPLE_AGE_FRAMES,
    DEBUG_COUNTER_SAMPLE_DUPLICATION_OVER_25PCT_PIXELS,
    DEBUG_COUNTER_SAMPLE_DUPLICATION_PERCENT,
    DEBUG_COUNTER_SAMPLE_DUPLICATION_SPECULAR_PERCENT,
    DEBUG_COUNTER_SPATIAL_CANDIDATES_REJECTED,
    DEBUG_COUNTER_SPATIAL_NO_NEIGHBOR_FOUND,
    DEBUG_COUNTER_SPECULAR_PIXELS,
    DEBUG_COUNTER_TEMPORAL_NO_HISTORY,
    DEBUG_COUNTER_TEMPORAL_REJECTED_DISSIMILAR,
    DEBUG_COUNTER_TEMPORAL_REJECTED_LIGHT_DESPAWNED,
    DEBUG_COUNTER_TEMPORAL_REPROJECTED_OFFSCREEN,
    DEBUG_COUNTER_X2_NOT_REUSABLE,
    DEBUG_VIEW_CONFIDENCE_WEIGHT,
    DEBUG_VIEW_CONTRIBUTION_WEIGHT,
    DEBUG_VIEW_JACOBIAN_REJECTION,
    DEBUG_VIEW_NOISE_NON_RESAMPLED_SHARE,
    DEBUG_VIEW_NOISE_RELATIVE_STD_DEV,
    DEBUG_VIEW_NOISE_RESAMPLED_STD_DEV,
    DEBUG_VIEW_NONE,
    DEBUG_VIEW_NON_RESAMPLED_ONLY,
    DEBUG_VIEW_NON_RESAMPLED_SHARE,
    DEBUG_VIEW_RESAMPLED_ONLY,
    DEBUG_VIEW_SAMPLE_AGE,
    DEBUG_VIEW_SAMPLE_DUPLICATION,
    DEBUG_VIEW_SAMPLE_PROVENANCE,
    DEBUG_VIEW_SPATIAL_REUSE_FAILURE,
    DEBUG_VIEW_TEMPORAL_REJECT_REASON,
    DEBUG_VIEW_WORLD_CACHE,
    DEBUG_VIEW_WORLD_CACHE_PROBE_FAILURE,
    DEBUG_VIEW_WORLD_CACHE_SAMPLE_COUNT,
    NOISE_RECORD_CEILING,
    PROVENANCE_MASK,
    SPECULAR_ROUGHNESS_THRESHOLD,
    TEMPORAL_STATUS_ACCEPTED,
    TEMPORAL_STATUS_DISSIMILAR,
    TEMPORAL_STATUS_LIGHT_DESPAWNED,
    TEMPORAL_STATUS_NO_HISTORY,
    TEMPORAL_STATUS_OFFSCREEN,
}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, permute_pixel, pixel_dissimilar, ResolvedGPixel}
#import bevy_solari::initial_path::{generate_initial_reservoir, InitialSamplingResult}
#import bevy_solari::realtime_bindings::{debug_flags, depth_buffer, reservoir_sample_age, reservoir_with_sample_age, empty_reservoir, gbuffer, motion_vectors, noise_moments, noise_moments_previous, previous_depth_buffer, previous_gbuffer, previous_view, reservoirs_a, reservoirs_b, unpack_sample_normal, Reservoir, constants, view, view_output}
#import bevy_solari::sampling::{balance_heuristic, calculate_resolved_light_contribution, isinf, isnan, LightSample, NULL_LIGHT_ID, power_heuristic, resolve_light_sample, ResolvedLightSample, trace_visibility, trace_visibility_previous_frame}
#import bevy_solari::scene_bindings::{light_sources, LIGHT_NOT_PRESENT_THIS_FRAME, previous_frame_light_id_translations, RAY_T_MAX, RAY_T_MIN, ResolvedMaterial}
#import bevy_solari::world_cache::{query_world_cache, WORLD_CACHE_CELL_LIFETIME}

const SPATIAL_REUSE_RADIUS_PIXELS = 30.0;

/// Perceptual roughness at which `specular_confidence_weight_cap` has fully
/// blended into `confidence_weight_cap`.
const SPECULAR_CONFIDENCE_BLEND_ROUGHNESS = 0.3;

/// Radius in pixels over which `sample_duplication` looks for copies of a pixel's
/// sample, and the gap between taps. A stride above one keeps the tap count (and so
/// the bandwidth) bounded while still covering the radius evenly.
const DUPLICATION_RADIUS_PIXELS: i32 = 9;
const DUPLICATION_TAP_STRIDE: i32 = 3;

/// Duplication fraction that reads as full red. Well below 1.0 because even a
/// badly correlated region shares its sample with only a modest fraction of a
/// 9-pixel radius, and a 0-to-100% ramp would leave the whole map blue.
const DUPLICATION_HEATMAP_RANGE = 0.25;

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_rng;

    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        reservoirs_b[pixel_index] = empty_reservoir();
        if debug_enabled() {
            debug_flags[pixel_index] = 0u;
        }
        return;
    }
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);

    debug_reset_state();

    let initial = generate_initial_reservoir(surface.world_position, surface.world_normal, surface.material, workgroup_id.xy, global_id.xy, &rng);
    textureStore(view_output, global_id.xy, vec4(initial.non_resampled_radiance, 0.0));

    let temporal = load_temporal_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, surface.material.perceptual_roughness);
    let previous_camera_homogeneous = previous_view.world_from_clip * (previous_view.clip_from_view * vec4(0.0, 0.0, 0.0, 1.0));
    let previous_camera_world_position = previous_camera_homogeneous.xyz / previous_camera_homogeneous.w;
    let merge_result = merge_reservoirs(initial.reservoir, surface.world_position, surface.world_normal, surface.material,
        temporal.reservoir, temporal.world_position, temporal.world_normal, temporal.material, previous_camera_world_position, &rng);

    reservoirs_b[pixel_index] = merge_result.merged_reservoir;

    if debug_enabled() {
        record_initial_and_temporal_debug(pixel_index, temporal.debug_status, merge_result, initial.non_resampled_radiance);
    }
}

fn record_initial_and_temporal_debug(pixel_index: u32, temporal_status: u32, merge_result: ReservoirMergeResult, non_resampled_radiance: vec3<f32>) {
    debug_count(DEBUG_COUNTER_PIXELS_SHADED, 1u);
    switch temporal_status {
        case TEMPORAL_STATUS_OFFSCREEN: { debug_count(DEBUG_COUNTER_TEMPORAL_REPROJECTED_OFFSCREEN, 1u); }
        case TEMPORAL_STATUS_DISSIMILAR: { debug_count(DEBUG_COUNTER_TEMPORAL_REJECTED_DISSIMILAR, 1u); }
        case TEMPORAL_STATUS_LIGHT_DESPAWNED: { debug_count(DEBUG_COUNTER_TEMPORAL_REJECTED_LIGHT_DESPAWNED, 1u); }
        case TEMPORAL_STATUS_NO_HISTORY: { debug_count(DEBUG_COUNTER_TEMPORAL_NO_HISTORY, 1u); }
        default: {}
    }
    debug_count_if(DEBUG_COUNTER_X2_NOT_REUSABLE, debug_x2_not_reusable());
    debug_count_if(DEBUG_COUNTER_JACOBIAN_TEMPORAL_DISCARD_NEIGHBOR, merge_result.debug_jacobian_discard_neighbor);
    debug_count_if(DEBUG_COUNTER_JACOBIAN_TEMPORAL_INFLATE_CANONICAL, merge_result.debug_jacobian_inflate_canonical);

    // The resampled term is not shaded until the next pass, so approximate this
    // pixel's bypassed share against the selected sample's contribution here.
    let non_resampled = luminance(non_resampled_radiance);
    let resampled = luminance(merge_result.selected_sample_brdf_radiance * merge_result.merged_reservoir.unbiased_contribution_weight);
    let non_resampled_share = non_resampled / max(non_resampled + resampled, 0.0001);
    // Summed as whole percent rather than a finer unit so the tally cannot overflow
    // a u32 at high resolutions.
    debug_count(DEBUG_COUNTER_NON_RESAMPLED_ENERGY_PERCENT, u32(saturate(non_resampled_share) * 100.0));

    debug_flags[pixel_index] = debug_pack_flags(
        temporal_status,
        debug_x2_not_reusable(),
        merge_result.debug_jacobian_discard_neighbor,
        merge_result.debug_jacobian_inflate_canonical,
        debug_world_cache_probe_exhausted(),
        debug_world_cache_sample_count(),
        non_resampled_share,
    );
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);
    var rng = pixel_index + constants.frame_rng + 0x6A09E667u;

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

    let spatial = load_spatial_reservoir(global_id.xy, depth, surface.world_position, surface.world_normal, &rng);
    let merge_result = merge_reservoirs(input_reservoir, surface.world_position, surface.world_normal, surface.material,
        spatial.reservoir, spatial.world_position, spatial.world_normal, spatial.material, view.world_position, &rng);

    reservoirs_a[pixel_index] = merge_result.merged_reservoir;

    let resampled_radiance = merge_result.selected_sample_brdf_radiance * merge_result.merged_reservoir.unbiased_contribution_weight;
    let non_resampled_radiance = textureLoad(view_output, global_id.xy).rgb;

    var pixel_color = resampled_radiance;
    pixel_color += surface.material.emissive;
    pixel_color += non_resampled_radiance;
    pixel_color *= view.exposure;
    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));

    if debug_enabled() {
        let moments = accumulate_noise_moments(global_id.xy, depth, surface.world_position, surface.world_normal,
            pixel_color, resampled_radiance * view.exposure);
        emit_debug_view(global_id.xy, pixel_index, spatial.debug_status, merge_result, moments,
            resampled_radiance, non_resampled_radiance, surface, &rng);
    }

#ifdef VISUALIZE_WORLD_CACHE
    textureStore(view_output, global_id.xy, vec4(query_world_cache(surface.world_position, surface.world_normal, view.world_position, RAY_T_MAX, WORLD_CACHE_CELL_LIFETIME, &rng) * view.exposure, 1.0));
#endif
}

/// Exponential moving average of the first and second moments of the shaded
/// luminance, for the total and for the resampled term, reprojected along the
/// motion vectors. An EMA needs no sample count, so freshly allocated (garbage)
/// history simply decays away.
///
/// Tracking the resampled term rather than the bypassing one keeps both the total
/// and the ReSTIR estimator's own noise exact. The bypassing term's share is then
/// the residual, which is what `DEBUG_VIEW_NOISE_NON_RESAMPLED_SHARE` reports.
fn accumulate_noise_moments(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, total: vec3<f32>, resampled: vec3<f32>) -> vec4<f32> {
    let total_luminance = luminance(total);
    let resampled_luminance = luminance(resampled);
    let current = vec4(
        total_luminance,
        total_luminance * total_luminance,
        resampled_luminance,
        resampled_luminance * resampled_luminance,
    );

    var alpha = 1.0 / 16.0;
    var previous = current;
    if bool(constants.reset) || bool(constants.debug_reset) {
        alpha = 1.0;
    } else {
        let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
        let previous_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));
        if all(previous_pixel_id_float >= vec2(0.0)) && all(previous_pixel_id_float < view.main_pass_viewport.zw) {
            let previous_pixel_id = vec2<u32>(previous_pixel_id_float);
            let previous_depth = textureLoad(previous_depth_buffer, previous_pixel_id, 0);
            let previous_surface = gpixel_resolve(textureLoad(previous_gbuffer, previous_pixel_id, 0), previous_depth, previous_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
            if pixel_dissimilar(depth, world_position, previous_surface.world_position, world_normal, previous_surface.world_normal, view) {
                // Disocclusion, so restart rather than dragging the old estimate along
                alpha = 1.0;
            } else {
                previous = textureLoad(noise_moments_previous, previous_pixel_id, 0);
            }
        } else {
            alpha = 1.0;
        }
    }

    var moments = mix(previous, current, alpha);
    if alpha >= 1.0 {
        // A single sample has zero sample variance, which would report a freshly
        // disoccluded pixel as perfectly clean and hide noise during camera motion.
        // Seed the second moment at twice the square of the mean instead, which
        // reads as 100% relative std dev, and let it decay toward the truth.
        moments = vec4(current.x, current.y * 2.0, current.z, current.w * 2.0);
    }
    textureStore(noise_moments, pixel_id, moments);
    return moments;
}

fn emit_debug_view(
    pixel_id: vec2<u32>,
    pixel_index: u32,
    spatial_candidates_rejected: u32,
    merge_result: ReservoirMergeResult,
    moments: vec4<f32>,
    resampled_radiance: vec3<f32>,
    non_resampled_radiance: vec3<f32>,
    surface: ResolvedGPixel,
    rng: ptr<function, u32>,
) {
    let flags = debug_flags[pixel_index];
    let total_relative_std_dev = debug_relative_std_dev(moments.x, moments.y);
    let resampled_relative_std_dev = debug_relative_std_dev(moments.z, moments.w);

    // Variance shares rather than energy shares, so this attributes the flicker to
    // whichever term actually causes it. A converged pixel has no variance to
    // attribute, so report zero rather than dividing by ~nothing. The residual also
    // absorbs the covariance between the two terms, and saturates because an
    // anti-correlated pair can leave the resampled variance above the total.
    let total_variance = max(moments.y - moments.x * moments.x, 0.0);
    let resampled_variance = max(moments.w - moments.z * moments.z, 0.0);
    var non_resampled_share = 0.0;
    if total_variance > 0.0000001 {
        non_resampled_share = saturate(1.0 - resampled_variance / total_variance);
    }

    debug_count_if(DEBUG_COUNTER_SPATIAL_NO_NEIGHBOR_FOUND, spatial_candidates_rejected >= 5u);
    debug_count(DEBUG_COUNTER_SPATIAL_CANDIDATES_REJECTED, spatial_candidates_rejected);
    debug_count_if(DEBUG_COUNTER_JACOBIAN_SPATIAL_DISCARD_NEIGHBOR, merge_result.debug_jacobian_discard_neighbor);
    debug_count_if(DEBUG_COUNTER_JACOBIAN_SPATIAL_INFLATE_CANONICAL, merge_result.debug_jacobian_inflate_canonical);

    // Clamped at NOISE_RECORD_CEILING rather than at 1.0, so the pixels that
    // actually dominate perceived noise are not all flattened to the same value.
    let noise_percent = u32(clamp(total_relative_std_dev, 0.0, NOISE_RECORD_CEILING) * 100.0);
    let resampled_noise_percent = u32(clamp(resampled_relative_std_dev, 0.0, NOISE_RECORD_CEILING) * 100.0);

    // Tail of the distribution, which is what reads as "noisy" even when the mean
    // looks acceptable.
    debug_count_if(DEBUG_COUNTER_NOISE_OVER_100PCT_PIXELS, total_relative_std_dev > 1.0);
    debug_count_if(DEBUG_COUNTER_NOISE_OVER_200PCT_PIXELS, total_relative_std_dev > 2.0);

    // Noise conditioned on the two mechanisms that look worst on screen, so a small
    // population of very bad pixels is visible instead of being averaged away.
    // Denominators: history_rejected_pixels here, and x2_not_reusable for the bypass.
    let temporal_status = debug_flags_temporal_status(flags);
    if temporal_status == TEMPORAL_STATUS_DISSIMILAR || temporal_status == TEMPORAL_STATUS_NO_HISTORY {
        debug_count(DEBUG_COUNTER_HISTORY_REJECTED_PIXELS, 1u);
        debug_count(DEBUG_COUNTER_NOISE_HISTORY_REJECTED_PERCENT, noise_percent);
    }
    if debug_flags_x2_not_reusable(flags) {
        debug_count(DEBUG_COUNTER_NOISE_BYPASS_PERCENT, noise_percent);
    }
    debug_count(DEBUG_COUNTER_NOISE_RELATIVE_STD_DEV_PERCENT, noise_percent);
    debug_count(DEBUG_COUNTER_NOISE_RESAMPLED_PERCENT, resampled_noise_percent);
    debug_count(DEBUG_COUNTER_NOISE_NON_RESAMPLED_SHARE_PERCENT, u32(non_resampled_share * 100.0));

    // Split the noise tally by material so panning can be measured against the
    // surfaces whose lobe actually rotates with the camera. Tracking the resampled
    // term separately here separates stale specular history from the 1-spp bypass.
    if surface.material.perceptual_roughness < SPECULAR_ROUGHNESS_THRESHOLD {
        debug_count(DEBUG_COUNTER_SPECULAR_PIXELS, 1u);
        debug_count(DEBUG_COUNTER_NOISE_SPECULAR_PERCENT, noise_percent);
        debug_count(DEBUG_COUNTER_NOISE_RESAMPLED_SPECULAR_PERCENT, resampled_noise_percent);
    } else {
        debug_count(DEBUG_COUNTER_NOISE_DIFFUSE_PERCENT, noise_percent);
    }

    // Mean effective history length, in tenths of a frame. Absolute rather than a
    // fraction of the cap, because a merged reservoir sits at 1 + cap and so any
    // fraction-of-cap reading just saturates, which makes cap changes invisible.
    // Note it counts a neighbour's confidence even when the jacobian clamp
    // discarded that neighbour's sample.
    debug_count(DEBUG_COUNTER_CONFIDENCE_WEIGHT_X10, u32(merge_result.merged_reservoir.confidence_weight * 10.0));

    // How much independence the reuse has cost. Read these against the noise
    // tallies: variance falling while age and duplication climb means samples are
    // being shared rather than gathered, which trades noise a denoiser could remove
    // for structure it cannot.
    let sample_age = reservoir_sample_age(merge_result.merged_reservoir.flags);
    let duplication = sample_duplication(pixel_id, sample_identity(merge_result.merged_reservoir));
    debug_count(DEBUG_COUNTER_SAMPLE_AGE_FRAMES, sample_age);
    debug_count(DEBUG_COUNTER_SAMPLE_DUPLICATION_PERCENT, u32(duplication * 100.0));
    // Split out and tailed, because a scene-wide mean hides the regions that
    // actually blotch, and scarce-sample regions are exactly where reuse
    // concentrates samples.
    debug_count_if(DEBUG_COUNTER_SAMPLE_DUPLICATION_OVER_25PCT_PIXELS, duplication > 0.25);
    if surface.material.perceptual_roughness < SPECULAR_ROUGHNESS_THRESHOLD {
        debug_count(DEBUG_COUNTER_SAMPLE_DUPLICATION_SPECULAR_PERCENT, u32(duplication * 100.0));
    }

    if constants.debug_view == DEBUG_VIEW_NONE { return; }

    var color = vec3(0.0);
    switch constants.debug_view {
        case DEBUG_VIEW_NOISE_RELATIVE_STD_DEV: {
            color = debug_heatmap(total_relative_std_dev);
        }
        case DEBUG_VIEW_NOISE_RESAMPLED_STD_DEV: {
            color = debug_heatmap(resampled_relative_std_dev);
        }
        case DEBUG_VIEW_NOISE_NON_RESAMPLED_SHARE: {
            color = debug_heatmap(non_resampled_share);
        }
        case DEBUG_VIEW_NON_RESAMPLED_SHARE: {
            color = debug_heatmap(debug_flags_non_resampled_share(flags));
        }
        case DEBUG_VIEW_NON_RESAMPLED_ONLY: {
            color = debug_tonemap_radiance(non_resampled_radiance, view.exposure);
        }
        case DEBUG_VIEW_RESAMPLED_ONLY: {
            color = debug_tonemap_radiance(resampled_radiance, view.exposure);
        }
        case DEBUG_VIEW_SAMPLE_AGE: {
            // Over zero to 32 frames. Red means this pixel's estimate has not been
            // independently resampled in a long time.
            color = debug_heatmap(f32(sample_age) / 32.0);
        }
        case DEBUG_VIEW_SAMPLE_DUPLICATION: {
            color = debug_heatmap(duplication / DUPLICATION_HEATMAP_RANGE);
        }
        case DEBUG_VIEW_SAMPLE_PROVENANCE: {
            color = debug_provenance_color(merge_result.merged_reservoir.flags & PROVENANCE_MASK);
        }
        case DEBUG_VIEW_CONFIDENCE_WEIGHT: {
            color = debug_heatmap(merge_result.merged_reservoir.confidence_weight / max(constants.confidence_weight_cap, 0.0001));
        }
        case DEBUG_VIEW_TEMPORAL_REJECT_REASON: {
            color = debug_temporal_status_color(debug_flags_temporal_status(flags));
        }
        case DEBUG_VIEW_SPATIAL_REUSE_FAILURE: {
            color = debug_heatmap(f32(spatial_candidates_rejected) / 5.0);
        }
        case DEBUG_VIEW_JACOBIAN_REJECTION: {
            // Discards cost variance, so they take precedence over the milder
            // MIS-inflation case in the colouring.
            let temporal_discarded = debug_flags_jacobian_discard_neighbor(flags);
            let spatial_discarded = merge_result.debug_jacobian_discard_neighbor;
            let inflated = debug_flags_jacobian_inflate_canonical(flags) || merge_result.debug_jacobian_inflate_canonical;
            if temporal_discarded && spatial_discarded {
                color = vec3(1.0, 0.0, 0.0);
            } else if temporal_discarded {
                color = vec3(0.1, 0.9, 0.3);
            } else if spatial_discarded {
                color = vec3(0.1, 0.4, 1.0);
            } else if inflated {
                color = vec3(0.35);
            }
        }
        case DEBUG_VIEW_CONTRIBUTION_WEIGHT: {
            // log10 over 1 to 1e4, so only genuine outliers reach the top of the ramp
            let w = merge_result.merged_reservoir.unbiased_contribution_weight;
            color = debug_heatmap(log(max(w, 1.0)) / log(10000.0));
        }
        case DEBUG_VIEW_WORLD_CACHE_SAMPLE_COUNT: {
            color = debug_heatmap(debug_flags_world_cache_sample_count(flags) / max(constants.world_cache_max_temporal_samples, 0.0001));
        }
        case DEBUG_VIEW_WORLD_CACHE_PROBE_FAILURE: {
            color = select(vec3(0.0), vec3(1.0, 0.0, 0.0), debug_flags_world_cache_probe_exhausted(flags));
        }
        case DEBUG_VIEW_WORLD_CACHE: {
            let radiance = query_world_cache(surface.world_position, surface.world_normal, view.world_position, RAY_T_MAX, WORLD_CACHE_CELL_LIFETIME, rng);
            color = debug_tonemap_radiance(radiance, view.exposure);
        }
        default: {}
    }

    textureStore(view_output, pixel_id, vec4(color, 1.0));
}

fn load_temporal_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, perceptual_roughness: f32) -> NeighborInfo {
    if bool(constants.reset) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material(), TEMPORAL_STATUS_NO_HISTORY);
    }

    let motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;
    let temporal_pixel_id_float = round(vec2<f32>(pixel_id) - (motion_vector * view.main_pass_viewport.zw));

    // If reprojection lands off-screen, fall back to this pixel's own previous reservoir rather than
    // dropping history. The dissimilarity check below still validates the surface, and a same-pixel
    // guess that passes it beats restarting from a confidence-1 reservoir at the screen edge.
    var point_temporal_pixel_id = pixel_id;
    var status = TEMPORAL_STATUS_ACCEPTED;
    if all(temporal_pixel_id_float >= vec2(0.0)) && all(temporal_pixel_id_float < view.main_pass_viewport.zw) {
        point_temporal_pixel_id = vec2<u32>(temporal_pixel_id_float);
    } else {
        status = TEMPORAL_STATUS_OFFSCREEN;
    }

    var permute_rng = constants.frame_rng;
    let permuted_temporal_pixel_id = permute_pixel(point_temporal_pixel_id, rand_u(&permute_rng), view.main_pass_viewport.zw);

    // Check if the pixel features have changed heavily between the current and previous frame
    let temporal_depth = textureLoad(previous_depth_buffer, permuted_temporal_pixel_id, 0);
    let temporal_surface = gpixel_resolve(textureLoad(previous_gbuffer, permuted_temporal_pixel_id, 0), temporal_depth, permuted_temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material(), TEMPORAL_STATUS_DISSIMILAR);
    }

    let temporal_pixel_index = permuted_temporal_pixel_id.x + permuted_temporal_pixel_id.y * u32(view.main_pass_viewport.z);
    var temporal = NeighborInfo(reservoirs_a[temporal_pixel_index], temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material, status);

    // Check if the light selected in the previous frame no longer exists in the current frame (e.g. entity despawned)
    if temporal.reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let previous_light_id = temporal.reservoir.light_sample.light_id >> 16u;
        let triangle_id = temporal.reservoir.light_sample.light_id & 0xFFFFu;
        let light_id = previous_frame_light_id_translations[previous_light_id];
        if light_id == LIGHT_NOT_PRESENT_THIS_FRAME {
            return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material(), TEMPORAL_STATUS_LIGHT_DESPAWNED);
        }
        temporal.reservoir.light_sample.light_id = (light_id << 16u) | triangle_id;
    }

    if temporal.reservoir.confidence_weight == 0.0 {
        temporal.debug_status = TEMPORAL_STATUS_NO_HISTORY;
    }

    temporal.reservoir.confidence_weight = min(temporal.reservoir.confidence_weight, confidence_weight_cap(perceptual_roughness));

    return temporal;
}

/// Smooth surfaces optionally get a shorter history than rough ones, because their
/// stored sample is re-evaluated against a view direction that keeps moving. Uses
/// plain perceptual roughness so the effect lines up with the specular bucket the
/// debug counters report.
fn confidence_weight_cap(perceptual_roughness: f32) -> f32 {
    let t = saturate(perceptual_roughness / SPECULAR_CONFIDENCE_BLEND_ROUGHNESS);
    return mix(constants.specular_confidence_weight_cap, constants.confidence_weight_cap, t);
}

fn load_spatial_reservoir(pixel_id: vec2<u32>, depth: f32, world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> NeighborInfo {
    var rejected = 0u;
    for (var i = 0u; i < 5u; i++) {
        let spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        if all(spatial_pixel_id == pixel_id) {
            rejected++;
            continue;
        }

        let spatial_depth = textureLoad(depth_buffer, spatial_pixel_id, 0);
        let spatial_surface = gpixel_resolve(textureLoad(gbuffer, spatial_pixel_id, 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view) {
            rejected++;
            continue;
        }

        let spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * u32(view.main_pass_viewport.z);
        return NeighborInfo(reservoirs_b[spatial_pixel_index], spatial_surface.world_position, spatial_surface.world_normal, spatial_surface.material, rejected);
    }

    return NeighborInfo(empty_reservoir(), world_position, world_normal, empty_material(), rejected);
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
    /// Debug only: a `TEMPORAL_STATUS_*` value from `load_temporal_reservoir`, or
    /// the number of rejected candidates from `load_spatial_reservoir`.
    debug_status: u32,
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

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_brdf_radiance: vec3<f32>,
    /// Debug only: the neighbour-to-canonical jacobian was clamped away, so the
    /// neighbour's sample could not be selected at all. Costs variance: the pixel
    /// silently falls back to this frame's canonical sample while still counting
    /// the neighbour's confidence weight.
    debug_jacobian_discard_neighbor: bool,
    /// Debug only: the canonical-to-neighbour jacobian was clamped away, which
    /// drops the canonical sample's MIS partner and snaps its balance-heuristic
    /// weight to one. Biases rather than adding variance.
    debug_jacobian_inflate_canonical: bool,
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
    other_view_position: vec3<f32>,
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

    // Skip resampling empty reservoirs
    if other_reservoir.confidence_weight == 0.0 {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_at_canonical.brdf_radiance, false, false);
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
    // since calculate_resolved_light_contribution already accounts for the shading point's geometry.
    var other_sample_at_canonical_jacobian = 1.0;
    if other_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        other_sample_at_canonical_jacobian = jacobian(
            canonical_world_position,
            other_world_position,
            other_reservoir.sample_point_world_position,
            unpack_sample_normal(other_reservoir.sample_point_world_normal)
        );
    }
    var canonical_sample_at_other_jacobian = 1.0;
    if canonical_reservoir.light_sample.light_id == NULL_LIGHT_ID {
        canonical_sample_at_other_jacobian = jacobian(
            other_world_position,
            canonical_world_position,
            canonical_reservoir.sample_point_world_position,
            unpack_sample_normal(canonical_reservoir.sample_point_world_normal)
        );
    }

    // Don't merge samples with huge jacobians, as it explodes the variance
    var jacobian_discard_neighbor = false;
    var jacobian_inflate_canonical = false;
    if other_sample_at_canonical_jacobian < 0.125 || other_sample_at_canonical_jacobian > 8.0 {
        other_sample_at_canonical_jacobian = 0.0;
        jacobian_discard_neighbor = true;
    }
    if canonical_sample_at_other_jacobian < 0.125 || canonical_sample_at_other_jacobian > 8.0 {
        canonical_sample_at_other_jacobian = 0.0;
        jacobian_inflate_canonical = true;
    }

    // Visibility for the cross-domain targets
    if other_sample_at_canonical.target_function > 0.0 && other_sample_at_canonical_jacobian > 0.0 {
        let visibility = trace_visibility(canonical_world_position + canonical_world_normal * RAY_T_MIN, other_sample_at_canonical.sample_world_position);
        other_sample_at_canonical.target_function *= visibility;
    }
    if canonical_sample_at_other.target_function > 0.0 && canonical_sample_at_other_jacobian > 0.0 {
#ifdef SPATIAL_MERGE
        let visibility = trace_visibility(other_world_position + other_world_normal * RAY_T_MIN, canonical_sample_at_other.sample_world_position);
#else
        let visibility = trace_visibility_previous_frame(other_world_position + other_world_normal * RAY_T_MIN, canonical_sample_at_other.sample_world_position);
#endif
        canonical_sample_at_other.target_function *= visibility;
    }

    // Defensive balance heuristic MIS (for spatial reuse only)
    let total_confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    var defensive_t_c = 0.0;
#ifdef SPATIAL_MERGE
    defensive_t_c = select(1.0, canonical_reservoir.confidence_weight / total_confidence_weight, total_confidence_weight > 0.0);
#endif

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

    if rand_f(rng) * weight_sum < other_sample_resampling_weight {
        combined_reservoir.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = other_reservoir.radiance;
        combined_reservoir.light_sample = other_reservoir.light_sample;
#ifdef SPATIAL_MERGE
        // Same frame, so taking a neighbour's sample inherits its age unchanged.
        combined_reservoir.flags = other_reservoir.flags;
#else
        // The temporal sample survived another frame, so it is one frame older and
        // this pixel's estimate is that much less independent of the last one.
        combined_reservoir.flags = reservoir_with_sample_age(other_reservoir.flags, reservoir_sample_age(other_reservoir.flags) + 1u);
#endif

        let inverse_target_function = select(0.0, 1.0 / other_sample_at_canonical.target_function, other_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_sample_at_canonical.brdf_radiance, jacobian_discard_neighbor, jacobian_inflate_canonical);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;
        combined_reservoir.light_sample = canonical_reservoir.light_sample;
        combined_reservoir.flags = canonical_reservoir.flags;

        let inverse_target_function = select(0.0, 1.0 / canonical_sample_at_canonical.target_function, canonical_sample_at_canonical.target_function > 0.0);
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_at_canonical.brdf_radiance, jacobian_discard_neighbor, jacobian_inflate_canonical);
    }
}

/// Identity of a reservoir's sample, for spotting how many nearby pixels are
/// carrying the very same one.
fn sample_identity(reservoir: Reservoir) -> u32 {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        return reservoir.light_sample.light_id ^ (reservoir.light_sample.seed * 2654435761u);
    }
    // Quantized to a centimetre so reconnection points that are the same vertex
    // hash together despite float drift through the jacobians.
    let quantized = vec3<i32>(floor(reservoir.sample_point_world_position * 100.0));
    return (u32(quantized.x) * 73856093u) ^ (u32(quantized.y) * 19349663u) ^ (u32(quantized.z) * 83492791u);
}

/// Fraction of the pixels within `DUPLICATION_RADIUS_PIXELS` already carrying this
/// pixel's sample, measured on the post-temporal reservoirs.
///
/// This is the spatial correlation a denoiser cannot remove: duplicated samples are
/// indistinguishable from real shading detail, so a spatial filter preserves them
/// as a blotch instead of averaging them away. Measured out to a radius rather than
/// over the immediate neighbours because permutation sampling reshuffles history
/// within 4x4 tiles, so anything tighter than that sits in its blind spot, and
/// because blobs are typically larger than a few pixels anyway.
fn sample_duplication(pixel_id: vec2<u32>, identity: u32) -> f32 {
    var matches = 0u;
    var valid = 0u;
    for (var dy = -DUPLICATION_RADIUS_PIXELS; dy <= DUPLICATION_RADIUS_PIXELS; dy += DUPLICATION_TAP_STRIDE) {
        for (var dx = -DUPLICATION_RADIUS_PIXELS; dx <= DUPLICATION_RADIUS_PIXELS; dx += DUPLICATION_TAP_STRIDE) {
            if dx == 0 && dy == 0 { continue; }
            // Round footprint, so the radius means what it says
            if dx * dx + dy * dy > DUPLICATION_RADIUS_PIXELS * DUPLICATION_RADIUS_PIXELS { continue; }

            let neighbor = vec2<i32>(pixel_id) + vec2(dx, dy);
            if any(neighbor < vec2<i32>(0)) || any(neighbor >= vec2<i32>(view.main_pass_viewport.zw)) { continue; }
            let neighbor_index = u32(neighbor.x) + u32(neighbor.y) * u32(view.main_pass_viewport.z);
            valid++;
            if sample_identity(reservoirs_b[neighbor_index]) == identity {
                matches++;
            }
        }
    }
    return select(0.0, f32(matches) / f32(valid), valid > 0u);
}

struct ReservoirContribution {
    brdf_radiance: vec3<f32>,
    target_function: f32,
    sample_world_position: vec4<f32>,
}

fn reservoir_contribution(reservoir: Reservoir, resolved: ResolvedLightSample, world_position: vec3<f32>, world_normal: vec3<f32>, wo: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>) -> ReservoirContribution {
    if reservoir.light_sample.light_id != NULL_LIGHT_ID {
        let light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);

        // MIS weight against the bounce-0 BRDF-emissive strategy, recomputed from this surface's
        // brdf and material rather than baked into the unbiased contribution weight at generation. Mirrors the bounce-0
        // nee_mis_weight in generate_nee_candidate and generate_emissive_candidate, which puts the same factor in the target.
        var nee_mis_weight = 1.0;
        if light_contribution.brdf_rays_can_hit && light_contribution.inverse_solid_angle_pdf > 0.0 {
            let light_count = arrayLength(&light_sources);
            let inverse_solid_angle_pdf = light_contribution.inverse_solid_angle_pdf * f32(light_count);
            let p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
            let p_nee_strategy = f32(constants.primary_di_samples) * (1.0 / inverse_solid_angle_pdf) * p_nee;
            let p_brdf_at_nee = brdf_pdf(wo, light_contribution.wi, world_normal, material, F_ab);
            nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
        }

        let brdf_radiance = light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab) * nee_mis_weight;
        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), resolved.world_position);
    } else if any(reservoir.radiance != vec3(0.0)) {
        let delta = reservoir.sample_point_world_position - (world_position + world_normal * RAY_T_MIN);
        let sample_distance = length(delta);
        let wi = delta / sample_distance;
        var brdf_radiance = reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab);

        // Bounce-0 BRDF-emissive sample (directly-visible light). The seed field carries the light
        // triangle's bitcast area pdf, and the stored radiance is the raw emission. Rebuild the MIS
        // weight against this surface's NEE strategy, the dual of nee_mis_weight above and a mirror
        // of the emissive candidate in generate_initial_reservoir.
        if reservoir.light_sample.seed != 0u {
            let area_pdf = bitcast<f32>(reservoir.light_sample.seed);
            let light_normal = unpack_sample_normal(reservoir.sample_point_world_normal);
            let cos_theta_light = dot(-wi, light_normal);
            if cos_theta_light <= 0.0 {
                brdf_radiance = vec3(0.0);
            } else {
                let p_light = area_pdf * sample_distance * sample_distance / cos_theta_light;
                let p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
                let p_brdf = brdf_pdf(wo, wi, world_normal, material, F_ab);
                brdf_radiance *= power_heuristic(p_brdf, p_light * p_nee * f32(constants.primary_di_samples));
            }
        }

        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec4(reservoir.sample_point_world_position, 1.0));
    }
}
