// Debug tooling: post-denoise temporal-variance tap + heatmap present pass.
//
// Runs after DLSS Ray Reconstruction (or, with no denoiser, straight after the
// main pass) and before tonemapping, at *output* resolution. It reads the
// denoised HDR color from the view target, accumulates its temporal variance
// into an output-res moments buffer, and records the post-denoise stats. Then,
// depending on the debug mode, it optionally overwrites the view target with a
// heatmap: of the post-denoise variance it just computed, or of the pre-denoise
// variance the ReSTIR pass produced (sampled from the render-resolution moments
// buffer, nearest-upscaled).
//
// Reprojection uses UV-space motion vectors (resolution independent) sampled from
// the render-resolution prepass, validated with a coarse depth-similarity gate --
// enough for a debug estimate without threading the full g-buffer through here.
#import bevy_solari::variance::{accumulate_moments, luminance, relative_variance, variance_heatmap, variance_sum_fixed, VarianceStats}

// 0 = off, 1 = pre-denoise, 2 = post-denoise. Mirrors `VarianceDebugMode`.
const MODE_OFF = 0u;
const MODE_PRE = 1u;
const MODE_POST = 2u;

struct VariancePostUniforms {
    // Render (internal) resolution: the prepass depth/motion textures and the
    // pre-denoise moments buffer are all at this size.
    render_size: vec2<u32>,
    // Output resolution: the view target and the post-denoise moments buffer.
    output_size: vec2<u32>,
    mode: u32,
    threshold: f32,
    history_length: f32,
    reset: u32,
}

@group(0) @binding(0) var view_target: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(1) var depth_buffer: texture_depth_2d;
@group(0) @binding(2) var motion_vectors: texture_2d<f32>;
@group(0) @binding(3) var previous_depth_buffer: texture_depth_2d;
@group(0) @binding(4) var<storage, read> post_moments_read: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> post_moments_write: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> post_stats: VarianceStats;
// Pre-denoise moments written this frame by the ReSTIR variance pass, at render
// resolution. Read only, and only when displaying the pre-denoise heatmap.
@group(0) @binding(7) var<storage, read> pre_moments: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> uniforms: VariancePostUniforms;

// Nearest-neighbor map from an output pixel to its render-resolution pixel.
fn render_pixel_of(output_pixel: vec2<u32>) -> vec2<u32> {
    let uv = (vec2<f32>(output_pixel) + 0.5) / vec2<f32>(uniforms.output_size);
    let render = vec2<u32>(uv * vec2<f32>(uniforms.render_size));
    return min(render, uniforms.render_size - 1u);
}

@compute @workgroup_size(8, 8, 1)
fn variance_present(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_pixel = global_id.xy;
    if any(output_pixel >= uniforms.output_size) { return; }

    let output_index = output_pixel.x + output_pixel.y * uniforms.output_size.x;
    let render_pixel = render_pixel_of(output_pixel);

    // Sky/background: no lit surface, so no meaningful variance. Clear history.
    let depth = textureLoad(depth_buffer, render_pixel, 0);
    if depth == 0.0 {
        post_moments_write[output_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let color = textureLoad(view_target, output_pixel).rgb;
    let luma = luminance(color);

    // Reproject through UV-space motion vectors and gate on depth similarity.
    let uv = (vec2<f32>(output_pixel) + 0.5) / vec2<f32>(uniforms.output_size);
    let motion = textureLoad(motion_vectors, render_pixel, 0).xy;
    let previous_uv = uv - motion;

    var previous = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var disoccluded = true;
    if all(previous_uv >= vec2(0.0)) && all(previous_uv < vec2(1.0)) {
        let previous_render_pixel = min(vec2<u32>(previous_uv * vec2<f32>(uniforms.render_size)), uniforms.render_size - 1u);
        let previous_depth = textureLoad(previous_depth_buffer, previous_render_pixel, 0);
        // Reverse-z depth; a relative difference gate is coarse but adequate for a
        // debug estimate and rejects most cross-surface reprojections.
        if previous_depth > 0.0 && abs(depth - previous_depth) <= 0.1 * max(depth, previous_depth) {
            let previous_output_pixel = min(vec2<u32>(previous_uv * vec2<f32>(uniforms.output_size)), uniforms.output_size - 1u);
            let previous_index = previous_output_pixel.x + previous_output_pixel.y * uniforms.output_size.x;
            previous = post_moments_read[previous_index];
            disoccluded = false;
        }
    }

    let reset = bool(uniforms.reset) || disoccluded;
    let moments = accumulate_moments(previous, luma, uniforms.history_length, reset);
    post_moments_write[output_index] = moments;

    // Accumulate this pixel into the global stats. The atomics are inlined (rather
    // than in a shared helper) because WGSL forbids passing a `storage` pointer as
    // a function argument.
    let post_relative_variance = relative_variance(moments);
    atomicAdd(&post_stats.sum_relative_fixed, variance_sum_fixed(post_relative_variance));
    atomicMax(&post_stats.max_relative_bits, bitcast<u32>(post_relative_variance));
    if post_relative_variance > uniforms.threshold {
        atomicAdd(&post_stats.count_over_threshold, 1u);
    }
    atomicAdd(&post_stats.valid_count, 1u);

    // Present the selected heatmap (or leave the denoised image untouched).
    if uniforms.mode == MODE_POST {
        textureStore(view_target, output_pixel, vec4(variance_heatmap(post_relative_variance, uniforms.threshold), 1.0));
    } else if uniforms.mode == MODE_PRE {
        let pre_index = render_pixel.x + render_pixel.y * uniforms.render_size.x;
        let pre_relative_variance = relative_variance(pre_moments[pre_index]);
        textureStore(view_target, output_pixel, vec4(variance_heatmap(pre_relative_variance, uniforms.threshold), 1.0));
    }
}
