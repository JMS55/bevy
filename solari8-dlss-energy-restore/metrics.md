# DLSS-RR energy restore: metrics

All runs: lab PC (RTX 5080), Solari realtime, ReSTIR off, DLSS-RR. Measured on the experiment branch
`exp/dlss-guides2` with its A/B bench harness (not included in the PR); the PR is a clean port of the same algorithm
and default settings. References = Solari's pathtracer, 1280×720.
"Restore" = two-level (whole-frame gain, response 0.03 × 128 px block factors, response 0.005), ratio range
0.67–1.5, emitter threshold 16.

## Real camera motion vs DLSS-RR's own noisy input

Camera motion verified on every clip: the harness logs the displacement, and an early frame zoom-matched into a
late one gives a best scale clearly > 1. "Input" = the same path rendered without DLSS (same harness, same
frame-locked camera path), 2 repeats agreeing within 1%, 9-frame smoothed.

Zero-Day, 1920×1080, camera pushes 0.03 m/frame down the hall for 150 frames (4.47 m, zoom 1.4×), then holds
90 frames. Frame-mean luminance / input frame-mean luminance:

| | moving (f20–149) | hold (f200–239) | drift after the stop (f150→f239) |
|---|---|---|---|
| stock DLSS-RR | 0.92 | 0.95 | +7.6% |
| restore (response 0.03) | 1.02 | 1.06 | +2.0% |
| restore (response 0.05) | 1.01 | 1.02 | +0.1% |

Same clip, restore vs stock:
- Motion error: |frame where the camera stops − converged frame at the same viewpoint|. Restore is 0.99× stock
  over the whole frame, 0.88–0.97× by region.
- Sharpness while moving: |Laplacian| energy at the same frames (f15, f60, f100), restore / stock:
  - floor grates 0.96–1.05;
  - floor reflections 1.01–1.05;
  - signage 1.00–1.02;
  - robot arms 1.02–1.03;
  - machinery 1.01–1.02;
  - pallets 0.99.

Indoor (room 0), 1920×1080, camera pushes 0.02 m/frame for 150 frames (2.98 m, zoom 1.25×):
- Restore vs stock: brightness +1.0% while moving, −0.4% in the hold.
- Sharpness while moving 1.02.
- Motion error 0.95× (0.76–1.14× by region: window, boxes, floor, string lights, foreground box).

## Static, 1280×720, 64 frames vs the pathtracer (emitters masked as in the reference)

| scene | stock DLSS-RR | restore | DLSS-RR's noisy input |
|---|---|---|---|
| Zero-Day (3 repeats) | +20.8 / +7.4 / +9.6% | −2.4 / −0.9 / −0.5% | −3.1% |
| many_lights (2) | +2.6 / +2.8% | −3.0 / −2.6% | −2.9% |
| indoor room 1 (2) | +0.8 / +1.3% | +1.1 / +0.6% | |
| Bistro (1) | +0.1% | +0.9..+1.3%¹ | |

¹ Bistro was run with the restore before emitter masking (Bistro is sun-lit).

Zero-Day regions (3 repeats, 64-frame means vs the pathtracer):

| region | stock | restore | noisy input |
|---|---|---|---|
| glossy floor | +6..+20% | −1..−3% | −2% |
| textured wall | −1% | −4% | −1% |
| machinery | +10..+13% | 0..+1% | −6% |
| left pallet | −16..+5% | −17..−26% | −20% |
| right pallet | −13..+1% | −19..−25% | −22% |

The pallets are dark in Solari's own lighting (the input). Stock DLSS-RR's extra brightness happened to cover
part of that; the restore doesn't.

Flicker, static (8 consecutive frames):
- 32 px block temporal std/mean: restore 0.0006–0.0007, stock 0.0005.
- 8 px blocks: 0.0011 vs 0.0010.
- Per-pixel |dL|/L: 0.0017–0.0018 vs 0.0016–0.0017.

Sharpness, static (|Laplacian| of one DLSS-RR frame vs stock): 0.96–1.01 on floor, wall, machinery.

## Reflections (`solari_reflections`, the scene of Solari's reflection denoising work), objects moving

1920×1080. Test material roughness 0.12 metal (glossy) and roughness 0 (mirror, primary surface replacement).
Restore vs stock:

| material | brightness | sharpness of the test panel while moving | motion error |
|---|---|---|---|
| glossy 0.12 | +0.9% | 1.02 | 1.06× (panel 1.04×) |
| mirror | +1.0% | 0.99 | 0.95× (panel 1.07×) |

These two clips were rendered before the final emitter-mask fix (the mask was taken from the previous frame).
That fix only changes results where emitters move across the screen.

## Reactivity (Zero-Day 720p, static camera, all lights off at frame 24 and back on at 120)

Frames to settle within 10% of the final level after the lights come back:
- restore: 8–10 frames (measured with earlier restore variants: whole-frame gain at response 0.05, and 64 px
  blocks; not re-run with the final configuration);
- stock: 13 frames, after which it keeps creeping up for hundreds of frames.

## Cost

Zero-Day 1920×1080, `dlss_ray_reconstruction` GPU span, 4 interleaved repeats, medians:
- stock 2.118 ms;
- restore 2.222 ms (+0.104 ms).

Measured on an earlier revision with the same passes and resolution, without the emitter mask. The final revision
adds the mask test and a pixel count to the two reductions.

## Rejected alternative: changing DLSS-RR's guide buffers

Zero-Day, real camera push (same clip as above). Sharpness vs stock at the same frames, and brightness vs stock
over the move:

| guide change | sharpness while moving | brightness while moving |
|---|---|---|
| diffuse albedo = 1 | 0.87–1.03 | −0.8% |
| neutral (diffuse 1, specular 0) | 0.75–0.94 | −1.3% |
| specular albedo = 0 | 0.79–0.98 | +2.4% |

Static Zero-Day 720p, sharpness and brightness vs the pathtracer (stock is +8..+14%):

| guide change | brightness vs ref | floor sharpness |
|---|---|---|
| neutral guides | +0.8% | 0.71 |
| diffuse lerp toward 1, t 0.5 | +5.4% | 0.93 |
| diffuse lerp toward 1, t 0.75 | 0.0% | 0.91 |
| diffuse guide blurred, σ 1–4 px | +6..+15% | ≈1.0 |
| texture mip bias +1 | +11% | ≈1.0 |
| diffuse floor 0.1 / 0.25 | +8% | ≈1.0 |

- Diffuse lerp 0.5 on the `solari_reflections` glossy test panel, objects moving: sharpness 0.71.
- Guide changes either don't touch the brightening or cost sharpness. In motion they don't change brightness at
  all: the guide-driven part of the brightening only builds up while the camera is still.
