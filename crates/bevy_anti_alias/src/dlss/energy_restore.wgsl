// Post-DLSS-RR energy restore, see `DlssEnergyRestore`.
//
// DLSS Ray Reconstruction does not preserve the mean brightness of its noisy input. This rescales the output's
// low-frequency luminance to the input's, per coarse screen block:
//   reduce_input:  mean luminance of the noisy input per block (render resolution)
//   reduce_output: mean luminance of DLSS-RR's output per block (output resolution)
//   update_gain:   per-frame ratio input / output, smoothed over time as one whole-frame gain times slowly
//                  varying per-block factors
//   apply:         output *= bilinear(per-block gain)
// Both reductions leave out the same pixels: those whose DLSS-RR output this frame is brighter than
// `emitter_threshold` x last frame's mean output (emitters and their highlights, which DLSS-RR darkens).

struct Params {
    out_size: vec2<u32>,
    render_size: vec2<u32>,
    grid: vec2<u32>,
    block: u32,
    reset: u32,
    global_response: f32,
    block_response: f32,
    min_ratio: f32,
    max_ratio: f32,
    emitter_threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
/// DLSS-RR's noisy input (render resolution, top-left region).
@group(0) @binding(1) var input: texture_2d<f32>;
/// A copy of DLSS-RR's output.
@group(0) @binding(2) var output_copy: texture_2d<f32>;
/// DLSS-RR's output texture, rescaled in place.
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> input_mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_mean: array<f32>;
/// Per-block gain applied to the output.
@group(0) @binding(6) var<storage, read_write> gain: array<f32>;
/// Per-block relative factors, then the whole-frame gain at index `block_count`, then last frame's mean output
/// luminance at `block_count + 1`.
@group(0) @binding(7) var<storage, read_write> state: array<f32>;

var<workgroup> partial: array<f32, 256>;
var<workgroup> frame_ratio: f32;
var<workgroup> frame_gain: f32;

const EPSILON: f32 = 1e-4;

fn luminance(c: vec3<f32>) -> f32 {
    let l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // NaN and inf guard
    if !(l == l && l < 1e30) {
        return 0.0;
    }
    return max(l, 0.0);
}

fn block_count() -> u32 {
    return p.grid.x * p.grid.y;
}

/// Whether a pixel is left out of both reductions, judged by this frame's DLSS-RR output.
fn excluded(output_luminance: f32) -> bool {
    let level = state[block_count() + 1u];
    return p.emitter_threshold > 0.0 && level > 0.0 && output_luminance > p.emitter_threshold * level;
}

/// Output-pixel rectangle [lo, hi) of a block.
fn block_rect(b: vec2<u32>) -> vec4<u32> {
    let lo = b * p.block;
    let hi = min(lo + vec2(p.block), p.out_size);
    return vec4(lo, hi);
}

fn workgroup_sum(tid: u32, v: f32) -> f32 {
    workgroupBarrier();
    partial[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            partial[tid] += partial[tid + s];
        }
        workgroupBarrier();
    }
    return partial[0];
}

@compute @workgroup_size(256, 1, 1)
fn reduce_input(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let r = block_rect(wg.xy);
    // The same screen rectangle at render resolution.
    let scale = vec2<f32>(p.render_size) / vec2<f32>(p.out_size);
    let lo = vec2<u32>(floor(vec2<f32>(r.xy) * scale));
    let hi = max(vec2<u32>(floor(vec2<f32>(r.zw) * scale)), lo + vec2(1u));
    let size = hi - lo;
    let n = size.x * size.y;
    let to_output = vec2<f32>(p.out_size) / vec2<f32>(p.render_size);
    var sum = 0.0;
    var count = 0.0;
    for (var i = tid; i < n; i += 256u) {
        let q = lo + vec2(i % size.x, i / size.x);
        let o = min(vec2<u32>((vec2<f32>(q) + 0.5) * to_output), p.out_size - vec2(1u));
        if excluded(luminance(textureLoad(output_copy, o, 0).rgb)) {
            continue;
        }
        sum += luminance(textureLoad(input, q, 0).rgb);
        count += 1.0;
    }
    let total = workgroup_sum(tid, sum);
    let total_count = workgroup_sum(tid, count);
    if tid == 0u {
        input_mean[wg.y * p.grid.x + wg.x] = total / max(total_count, 1.0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_output(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let r = block_rect(wg.xy);
    let size = r.zw - r.xy;
    let n = size.x * size.y;
    var sum = 0.0;
    var count = 0.0;
    for (var i = tid; i < n; i += 256u) {
        let l = luminance(textureLoad(output_copy, r.xy + vec2(i % size.x, i / size.x), 0).rgb);
        if excluded(l) {
            continue;
        }
        sum += l;
        count += 1.0;
    }
    let total = workgroup_sum(tid, sum);
    let total_count = workgroup_sum(tid, count);
    if tid == 0u {
        output_mean[wg.y * p.grid.x + wg.x] = total / max(total_count, 1.0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn update_gain(@builtin(local_invocation_index) tid: u32) {
    let blocks = block_count();
    var input_sum = 0.0;
    var output_sum = 0.0;
    for (var k = tid; k < blocks; k += 256u) {
        let r = block_rect(vec2(k % p.grid.x, k / p.grid.x));
        let area = f32((r.z - r.x) * (r.w - r.y));
        input_sum += input_mean[k] * area;
        output_sum += output_mean[k] * area;
    }
    let input_total = workgroup_sum(tid, input_sum);
    let output_total = workgroup_sum(tid, output_sum);
    if tid == 0u {
        let ratio = (input_total + EPSILON) / (output_total + EPSILON);
        frame_ratio = ratio;
        let clamped = clamp(ratio, p.min_ratio, p.max_ratio);
        let previous = state[blocks];
        if p.reset != 0u || !(previous > 0.0) {
            frame_gain = clamped;
        } else {
            frame_gain = mix(previous, clamped, p.global_response);
        }
        state[blocks] = frame_gain;
        state[blocks + 1u] = output_total / f32(p.out_size.x * p.out_size.y);
    }
    workgroupBarrier();
    let ratio = frame_ratio;
    let global_gain = frame_gain;
    for (var k = tid; k < blocks; k += 256u) {
        let relative = clamp(
            (input_mean[k] + EPSILON) / (output_mean[k] + EPSILON) / ratio,
            p.min_ratio,
            p.max_ratio,
        );
        let previous = state[k];
        var factor = relative;
        if p.reset == 0u && previous > 0.0 {
            factor = mix(previous, relative, p.block_response);
        }
        state[k] = factor;
        gain[k] = global_gain * factor;
    }
}

fn gain_at(b: vec2<i32>) -> f32 {
    let c = clamp(b, vec2(0), vec2<i32>(p.grid) - vec2(1));
    return gain[u32(c.y) * p.grid.x + u32(c.x)];
}

@compute @workgroup_size(8, 8, 1)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= p.out_size) {
        return;
    }
    let c = textureLoad(output_copy, id.xy, 0);
    // Bilinear between block centers.
    let u = (vec2<f32>(id.xy) + 0.5) / f32(p.block) - 0.5;
    let b0 = vec2<i32>(floor(u));
    let f = u - floor(u);
    let g = mix(
        mix(gain_at(b0), gain_at(b0 + vec2(1, 0)), f.x),
        mix(gain_at(b0 + vec2(0, 1)), gain_at(b0 + vec2(1, 1)), f.x),
        f.y,
    );
    textureStore(output, id.xy, vec4(c.rgb * g, c.a));
}
