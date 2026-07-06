// Debug tooling: pre-denoise temporal-variance accumulation pass.
//
// Runs right after `spatial_and_shade`, reading the raw ReSTIR radiance still in
// `view_output` (the exact signal DLSS ingests, before any denoising). For each
// pixel it reprojects the previous frame's moments via the motion vectors,
// resets on disocclusion, folds in this frame's luminance, and writes the
// updated moments back. It also accumulates the global stats reduction. It never
// writes `view_output` -- the heatmap is drawn later, at output resolution, by
// the post-process present node -- so this pass is invisible unless the stats or
// the moments buffer are inspected.
enable wgpu_ray_query;

#import bevy_render::view::View
#import bevy_solari::gbuffer_utils::{gpixel_resolve, pixel_dissimilar}
#import bevy_solari::realtime_bindings::{depth_buffer, gbuffer, motion_vectors, previous_depth_buffer, previous_gbuffer, previous_view, variance_moments_read, variance_moments_write, variance_stats, constants, view, view_output}
#import bevy_solari::variance::{accumulate_moments, luminance, relative_variance, variance_sum_fixed}

@compute @workgroup_size(8, 8, 1)
fn variance_accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }

    let pixel_index = global_id.x + global_id.y * u32(view.main_pass_viewport.z);

    // Sky/background pixels have no lit surface to be noisy; skip them and leave
    // an empty (no-history) moments entry so a later disocclusion restarts clean.
    let depth = textureLoad(depth_buffer, global_id.xy, 0);
    if depth == 0.0 {
        variance_moments_write[pixel_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let luma = luminance(textureLoad(view_output, global_id.xy).rgb);

    // Reproject the previous-frame moments through the motion vectors, matching
    // how `load_temporal_reservoir` finds a pixel's history, and validate the
    // reprojected surface so history isn't reused across an occlusion edge.
    var previous = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var disoccluded = true;
    let surface = gpixel_resolve(textureLoad(gbuffer, global_id.xy, 0), depth, global_id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    let motion_vector = textureLoad(motion_vectors, global_id.xy, 0).xy;
    let reprojected = round(vec2<f32>(global_id.xy) - (motion_vector * view.main_pass_viewport.zw));
    if all(reprojected >= vec2(0.0)) && all(reprojected < view.main_pass_viewport.zw) {
        let previous_pixel_id = vec2<u32>(reprojected);
        let previous_depth = textureLoad(previous_depth_buffer, previous_pixel_id, 0);
        let previous_surface = gpixel_resolve(textureLoad(previous_gbuffer, previous_pixel_id, 0), previous_depth, previous_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
        if !pixel_dissimilar(depth, surface.world_position, previous_surface.world_position, surface.world_normal, previous_surface.world_normal, view) {
            let previous_index = previous_pixel_id.x + previous_pixel_id.y * u32(view.main_pass_viewport.z);
            previous = variance_moments_read[previous_index];
            disoccluded = false;
        }
    }

    let reset = bool(constants.reset) || disoccluded;
    let moments = accumulate_moments(previous, luma, constants.variance_history_length, reset);
    variance_moments_write[pixel_index] = moments;

    // Accumulate this pixel into the global stats. The atomics are inlined (rather
    // than in a shared helper) because WGSL forbids passing a `storage` pointer as
    // a function argument.
    let rv = relative_variance(moments);
    atomicAdd(&variance_stats.sum_relative_fixed, variance_sum_fixed(rv));
    atomicMax(&variance_stats.max_relative_bits, bitcast<u32>(rv));
    if rv > constants.variance_threshold {
        atomicAdd(&variance_stats.count_over_threshold, 1u);
    }
    atomicAdd(&variance_stats.valid_count, 1u);
}
