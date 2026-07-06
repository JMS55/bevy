// Debug tooling: per-pixel temporal-variance estimation and visualization.
//
// Estimates the Monte Carlo noise of the lit signal by accumulating a running
// mean (`M1 = E[L]`) and mean-of-squares (`M2 = E[L^2]`) of luminance across
// frames, motion-compensated by reprojection and reset on disocclusion. The
// display metric is the *relative* variance `Var / (M1^2 + eps)` (coefficient of
// variation squared), which is scale-invariant: a firefly reads "hot" whether it
// sits on a bright or a dark surface, i.e. exactly the pixels that visually stand
// out. Used both pre-denoise (raw ReSTIR output, the denoiser's input) and
// post-denoise (residual flicker the denoiser failed to remove).
#define_import_path bevy_solari::variance

// One per-pixel moments entry is a `vec4<f32>`: `.x = M1` (running mean
// luminance), `.y = M2` (running mean of luminance squared), `.z = sample count`
// (capped history length), `.w` unused/padding.

// Global per-frame reduction, accumulated with atomics and read back to the CPU
// via the render diagnostics path. All fields are integers so they can be
// atomically accumulated (WGSL has no atomic f32):
//   - `sum_relative_fixed`: sum of per-pixel relative variance in fixed point
//     (see `VARIANCE_SUM_FIXED_SCALE`), each pixel clamped to
//     `VARIANCE_SUM_CLAMP` first to bound both the mean and u32 overflow.
//   - `max_relative_bits`: max relative variance, as raw f32 bits. Relative
//     variance is >= 0, whose bit patterns are monotonic, so `atomicMax` on the
//     bits yields the max float.
//   - `count_over_threshold`: pixels whose relative variance exceeds the
//     user threshold (the "how many pixels visibly stand out" count).
//   - `valid_count`: pixels that contributed (on-surface, non-sky).
struct VarianceStats {
    sum_relative_fixed: atomic<u32>,
    max_relative_bits: atomic<u32>,
    count_over_threshold: atomic<u32>,
    valid_count: atomic<u32>,
}

// Fixed-point scale for `sum_relative_fixed`. Per-pixel relative variance is
// clamped to VARIANCE_SUM_CLAMP before scaling, so the worst case per pixel is
// VARIANCE_SUM_CLAMP * VARIANCE_SUM_FIXED_SCALE; summed over a 4K frame this
// stays well under u32::MAX. Divide the read-back sum by this to recover units.
const VARIANCE_SUM_FIXED_SCALE = 16.0;
const VARIANCE_SUM_CLAMP = 32.0;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Relative variance (coefficient of variation squared) from accumulated moments.
fn relative_variance(moments: vec4<f32>) -> f32 {
    let m1 = moments.x;
    let m2 = moments.y;
    let variance = max(m2 - m1 * m1, 0.0);
    return variance / (m1 * m1 + 1e-4);
}

// Fold a new luminance sample into the running moments. `history_length` caps
// how many samples the running average retains (larger = smoother/slower to
// react). When `reset` is true, or `previous` carries no history (count 0), the
// estimate restarts from this sample alone.
fn accumulate_moments(
    previous: vec4<f32>,
    luma: f32,
    history_length: f32,
    reset: bool,
) -> vec4<f32> {
    if reset || previous.z < 1.0 {
        return vec4<f32>(luma, luma * luma, 1.0, 0.0);
    }
    let count = min(previous.z + 1.0, history_length);
    let alpha = 1.0 / count;
    let m1 = mix(previous.x, luma, alpha);
    let m2 = mix(previous.y, luma * luma, alpha);
    return vec4<f32>(m1, m2, count, 0.0);
}

// Google "Turbo" colormap (polynomial fit), for `x` in [0, 1]. Perceptually
// ordered dark-blue -> cyan -> green -> yellow -> red, so higher variance reads
// as hotter.
fn variance_colormap(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

// Map a relative-variance value to a heatmap color. `threshold` sets the value
// mapped to the top of the colormap, so the display auto-scales to the range
// the user cares about (pixels at/above threshold are the ones "standing out").
fn variance_heatmap(relative_variance: f32, threshold: f32) -> vec3<f32> {
    return variance_colormap(relative_variance / max(threshold, 1e-6));
}

// Fixed-point contribution of one pixel's relative variance to
// `VarianceStats::sum_relative_fixed` (clamped, then scaled). The atomic
// accumulation itself is inlined in each pass, since WGSL doesn't allow passing a
// pointer into the `storage` address space as a function parameter.
fn variance_sum_fixed(relative_variance: f32) -> u32 {
    return u32(min(relative_variance, VARIANCE_SUM_CLAMP) * VARIANCE_SUM_FIXED_SCALE);
}
