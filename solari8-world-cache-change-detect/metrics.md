# World cache change detector (rate-normalized, cap ×8): full metrics

Measured on the lab PC (RTX 5080) on `exp/base` (bevy main + world cache improvements #26626 + split NEE
#26631 + an A/B test harness that isn't included), **before** the world cache energy fix
(solari8-world-cache-energy). Raw output (no denoiser), ReSTIR off, default 40k cell update budget.
References are Solari's pathtracer (2×2 downsampled for 640×360). "before" = the detector in main, "this PR" =
rate-normalized ×16/gap, capped at ×8. Each A/B pair ran in one process (in-process variant sweep with a
lighting reset and full warm-up per variant).

## Reactivity (640×360, frames after the event)

`settle` = per-pixel difference settled to 10% of the step; `tail90` / `tail50` = frames until the
whole-frame mean is within 10% / 50% of the remaining indirect step.

| event | before: settle / tail90 / tail50 | this PR |
|---|---|---|
| indoor1 (back room) `local_off`: local lights switched off | 12 / 32 / 11 | **6 / 9 / 4** |
| Zero-Day `local_dim`: local lights ×0.1 | 65 / 123 / 35 | **43 / 102 / 22** ¹ |
| Bistro `sun_orbit`: lag_gap right after the sun stops moving (noise floor 0.0147, run to run ±0.001) | 0.0087 (repeat 0.0072) | 0.0070 ¹ |

¹ Measured with the uncapped ×16/gap variant. At these scenes' update gaps (Zero-Day 2, Bistro ~8) the boost
is ≤ 8, so the capped version computes exactly the same values.

Moving local lights (`local_orbit`) in indoor0 and Zero-Day, and `many_lights` `local_off`, were below the
noise floor for every variant.

## Static image quality (median of repeats [min..max])

| scene | n (before / PR) | mean vs pathtracer, before | this PR | relbias2, before | this PR | relvar_t999, before | this PR |
|---|---|---|---|---|---|---|---|
| indoor1, 640×360 (gap 1, cap binds) | 7 / 7 | −2.55% [−3.01..−2.23] | +1.21% [+0.85..+2.04] | 0.0033 [0.0021..0.0050] | 0.0028 [0.0018..0.0039] | 6.01 [5.75..6.20] | 5.98 [5.80..6.13] |
| indoor1, 1280×720 (gap 2) | 4 / 4 | −2.16% [−2.56..−1.21] | +1.13% [+0.88..+1.51] | 0.0041 [0.0040..0.0043] | **0.0047 [0.0043..0.0047]** | 6.62 [5.98..7.02] | 6.02 [5.95..6.06] |
| indoor0, 1280×720 (gap 2) | 4 / 4 | −2.29% [−2.53..−0.87] | −0.63% [−0.76..−0.44] | 0.0059 [0.0048..0.0062] | 0.0047 [0.0046..0.0050] | 5.51 [5.31..8.52] | 5.41 [5.35..5.44] |
| Zero-Day, 1280×720 (gap 2) | 14 / 8 | −20.49% [−21.28..−16.54] | −19.22% [−19.66..−18.87] | 0.1175 [0.1071..0.1266] | 0.1216 [0.1140..0.1257] | 7.85 [6.73..14.02] | 7.37 [7.22..7.71] |

- relbias2 = mean squared relative error of the 64-frame mean image vs the reference; relvar_t999 = per-pixel
  relative variance, 99.9th percentile clipped (fireflies excluded).
- Zero-Day 64-frame means vary ±1.2% run to run from fireflies, so its +1.3-point shift is within spread.

## Alternatives tried and rejected

| variant | result |
|---|---|
| uncapped ×16/gap | where cells update every frame or two (boost ×15–16), indoor1 renders 1.3–2.3 points darker, relbias2 1.6–1.9×, noise +7–10%; the ×8 cap removes that |
| ×24/gap, ×32/gap | catch the moving sun faster, but **darken Zero-Day 7.5–12.4%** and indoor0 2–4% (relbias2 +25–60%) |
| noise-normalized trend (subtract k·σ of the samples) | cancels most of the reactivity gain (Bistro sun lag back to ~0.008); static noise −5–6% on indoor1 only; +8 B per cell |
| dual history (fast luminance EMA + clamp) | can't react faster than the history floor at these budgets; +8 B per cell |
| prioritized / stratified cell updates (camera-visible first) | no reaction speed gain: the history floor keeps a ~32-frame window at any update rate |
| continuous rate `active / budget` instead of `ceil` | uncapped it darkens like ×16 where the boost is high; capped it equals `ceil` except above 2× budget, where the boost is already < 8 |
