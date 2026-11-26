# PR notes: Solari realtime energy fix (exp/energy-loss)

Branch `exp/energy-loss` @ `1d51694b5`. All three parts are on by default; `SolariLighting::energy_debug_flags`
(experiment switch, not for upstream) reverts them for A/B. Full evidence: `REPORT.md`.

1. **Directional albedo at cache termination and in the cache's own bounce.** The cache stores irradiance and
   turned it into outgoing radiance with `base_color/pi`, ignoring the specular lobe's reflectance. It now uses
   the BRDF's directional albedo (`lobe_reflectances`: diffuse + specular).
2. **MIS for emission in the cache update.** The cache's cosine GI ray now adds the emission it hits, weighted by
   the power heuristic against the cache's NEE. Unbiased; removes the `1/d^2` blow-ups of cells next to emitters.
3. **⚠ Flagged for review: the cache is read at the last bounce even for very close hits.**

## ⚠ Part 3: always read the cache at the last bounce

**What it does.** When a path reaches its last bounce (`max_bounces - 1`) with a ray shorter than the cache cell's
diagonal (`t <= sqrt(3) * cell`), the path used to end with nothing. That short-ray rule exists to stop light
leaking through corners. At the last bounce the path can't continue anyway, so it now reads the cache there too.

**Why.** On glossy scenes many paths reach the last bounce with a short ray. On Zero-Day this part alone is
most of the gap: -10.5% → -0.8% vs the reference pathtracer, before denoising (lab PC, 1280x720, three repeats
in one process). The recovered energy survives DLSS-RR (+11 pts after denoising).

**Cost.** The recovered energy travels through rare, bright paths, so it arrives as sparse bright speckles.
- **Zero-Day:** ~1.7x raw relvar_t999 and ~1.4x raw p99. After DLSS-RR, ~1.3x per-pixel temporal variance
  (relvar_t999 8.7e-5 → 1.06e-4) and ~1.25x its p99.
- **Indoor and Bistro:** unaffected (brightness and noise within run-to-run spread).
- No measurable GPU cost.
- The alternatives tried (a minimum distance k·cell, a coarser cache LOD, without extra bounces) all trade the same
  energy for the same speckle along one curve.

**How to turn it off.** Set `energy_debug_flags` bit 16 (`EDF_SHORT_RAY_FORCED_KEEP_RULE` in `initial_path.wesl`).
In an upstream version this would be a one-line condition (`!forced_terminate` in the short-ray rule) or a
setting.

**Caveat for anyone checking Zero-Day numbers after DLSS-RR.** Compare variants only inside one process (one
`BENCH_SWEEP`, repeats via `BENCH_SWEEP_REPEATS`). Repeats in one process agree to ≤3.3 pts. The same
configuration in separate processes has read from -3% to +10% after DLSS-RR. Separately, DLSS-RR adds +7..+9%
on Zero-Day for every variant because of its albedo-guide demodulation, which the DLSS-RR guide work is
investigating. Judge this change on pre-DLSS brightness plus post-DLSS noise, not on the post-DLSS overshoot.

## Results with all three parts (reference-only emitter mask; lab PC unless noted)

| scene | before | after | noise |
|---|---|---|---|
| Zero-Day (1280, in-process repeats) | -15.1% | -0.8% | see above |
| indoor0 | -3.5 / -4.0% | ~0% | equal or lower |
| indoor1 | -3.0 / -3.5% | +0.6..+1.0% | equal or lower |
| Bistro | -1.3% | +0.4% | x1.08 |
| many_lights | -2.7% | unchanged (doesn't use the cache) | unchanged |

Known issue found on the way (not fixed here): the reference pathtracer's sun lacks atmosphere transmittance, so
its references are too bright in atmosphere scenes (see `REPORT.md`).
