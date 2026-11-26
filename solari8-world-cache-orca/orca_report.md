# Item 6: ORCA applied to Solari's world cache

## TL;DR

**Best ORCA variant: the hybrid ("hyb").** Every world cache cell that camera paths used last frame
recomputes its **direct light from scratch every frame** (no history; up to 4 light samples per cell within a
200k-sample budget). Its **indirect light keeps the existing temporal blend**. This is the remedy the ORCA talk
itself suggests for an in-frame cache too sparse to rebuild per frame: keep the old samples, but relight
their first vertex.
- It removes the lag from direct-light changes at cache cells (the sun, and lights switching or moving) and
  cuts lingering light about 3×.
- Noise and GPU cost are small.
- It carries one small, loudly flagged bias: indoor scenes come out 0.5% (full res) to 1% (640×360) darker.

**Optional on top: r16c8**, item 4's rate-normalized change detector capped at ×8. It makes the remaining
multi-bounce lag settle 2–3× faster at no cost, but adds ~1.6% darkening on Zero-Day. Ship it only if that
trade is wanted.

| vs `exp/base` | **hyb** | hyb + r16c8 |
|---|---|---|
| Lag, moving sun (Bistro `sun_orbit`, lag_gap after the sun stops; base 0.0075–0.0090 and 37–55 frames to catch up) | **below noise** (≤0.0005, 3 runs) | below noise |
| Lag, sun off (Bistro `sun_off`: light still missing in the event frame / frames to settle) | **7.5% → 2.3–2.7%** / 66–73 → 54–67 | 2.6–3.1% / 46–60 |
| Lag, local lights off (indoor1 `local_off`) | 18.6% → **7.2%** / 32 → ~60 frames | 7.1% / **15–16** frames |
| Lag, same per material class (indoor1, forced smooth/rough metal and dielectric) | missing light cut ~3× in every class (e.g. smooth metal 6.6% → 2.2%, rough dielectric 19.2% → 7.3%) | same, and tail 31 → 15–16 frames |
| Lag, dimmed emissives, multi-bounce (Zero-Day `local_dim`, 3 runs: settle / missing light / tail) | 63–151 → **48–50** / 36% → 23% / about the same tail | **25–33** / 23% / **85–90** (base 126–151) |
| Lag, moving local lights / emissives (`local_orbit`) | below noise (base too) | below noise |
| Noise, relvar_t999, no denoiser | Zero-Day full res ×8 **1.07×** (base run spread ±20%); indoor full res 1.05×; 640×360 0.94–1.04×; Bistro 1.01×; many_lights 1.01×; per material class **1.00–1.05×** | the same within noise |
| Mean brightness vs base | Zero-Day ×8 **+0.1%**, Bistro +0.1%, many_lights +0.4%; ⚠️ **indoor0/indoor1 full res ×4: −0.6 / −0.5 pts; 640×360 ×8: −1.1 pts** | ⚠️ **Zero-Day ×8 −1.3 pts (−1.6%)**; indoor −0.6 to −0.9 pts (full res), up to −1.4 pts (640) |
| relbias2 | Zero-Day 0.1180 → 0.1167; indoor0 full 0.0059 → 0.0064; indoor1 full 0.0046 → 0.0052 | Zero-Day 0.1176; indoor1 full 0.0051 |
| GPU, 1080p DLSS-RR, locked clocks, interleaved repeats | **+0.22 ms** on both Bistro (3.00 → 3.22 ms) and Zero-Day (1.78 → 1.99 ms), all in the world cache pass. An earlier run measured +0.18–0.26 ms, ReSTIR off and on. | same (r16c8 is free) |

**What to ship as code** (`exp/orca`, worktree `bevy-exp-orca`; the same knobs exist on `exp/knobs`):
- **Make the hybrid the only world cache update path.** Today that is
  `ORCA_SCHED=1 ORCA_MODE=3 ORCA_COLD_MODE=2 ORCA_SAMPLES=4 ORCA_SAMPLE_BUDGET=200000`. Concretely:
  - Hot/cold compaction (two-channel prefix sum; hot = queried by a camera path last frame). Every hot cell is
    updated every frame. Cold cells keep the 40k rolling window.
  - In-frame direct light: `min(4, 200k / hot_cells)` RIS samples.
  - Temporally blended indirect in `radiance_indirect` (+16 B/cell). Hot cells trace their GI ray on the
    baseline's rolling schedule.
  - Delete modes 0/1, the env knobs, `radiance_sun`, and the pdi / direct-alpha / det-total experiments.
- **Expose the sample budget** as a `SolariLighting` field (e.g. `world_cache_direct_sample_budget`,
  default 200k).
- **Optional r16c8:** item 4's `world_cache_detector = 9`, scale 16.
- **Before merging:** find the indoor darkening's cause (see the flag).

**Rejected** (all measured, details in sections 6–8):
- **Pure history-free ORCA:** reacts in ~3 frames, but +5–70% noise and ~4 ms of cache time.
- **Per-pixel direct light at cache terminations** ("pdi", the user's idea):
  - 1.5–1.7× noise indoors and on Zero-Day (full res ×4), 1.18× on Bistro.
  - +1.0 ms on Bistro (indicative timing).
  - It only beats the hybrid on sun-off (1.8% vs 2.7% missing light) and doesn't help local lights.
- **pdi-sun:** +6–16% noise, about +0.65 ms on Bistro, and the same small sun-off gain.
- **Variants that changed nothing:** total-normalized change detector (`ORCA_DET_TOTAL`), a short direct-light
  history (`ORCA_DIRECT_ALPHA` ½ and ¼, which also gives back lag), and baseline blending for cold cells
  (`ORCA_COLD_MODE=0`).

### ⚠️ Bias flag

- **The hybrid adds a small darkening on the indoor scenes**, measured against base in interleaved runs:
  - Full res, 4 samples each: indoor0 −0.6 points (−4.18% → −4.77% vs the pathtracer), indoor1 −0.5 points
    (−4.06% → −4.57%). Noise 1.03× / 1.05×.
  - 640×360: indoor0 and indoor1 −1.1 points each (8 samples each). relbias2 rises a little.
  - It isn't there on Zero-Day (full res ×8: +0.1%), Bistro (+0.1%) or many_lights (+0.4%).
  - The cause isn't isolated. Ruled out: detector normalization (`ORCA_DET_TOTAL`), the frozen single-sample
    direct light of cold cells (`ORCA_COLD_MODE=0`), and a short direct-light history.
  - It's small next to the existing world-cache bias (indoor −3…−4%, Zero-Day −20% vs the pathtracer), and
    the lag gains are large. **I'd call it small and worth it**, but it's an open item.
- **hyb + r16c8 darkens Zero-Day by 1.6%** (−19.65% → −20.94%, 8 samples each, about 3 standard errors).
  - relbias2 doesn't change.
  - r16c8 alone doesn't darken Zero-Day (item 4: +2.2%), so this is an interaction with the hybrid's
    indirect-only blend.
- **pdi** shifts brightness both ways: +1.9% on Bistro (toward the reference) and −2.3% on Zero-Day. It isn't
  double counting emissives. It's rejected anyway.

### Images (no denoiser unless noted)

- **Static noise**, Zero-Day full res, single frames: base | hyb | hyb+r16c8 | pdi (rejected) | reference.
  Base and the hybrids are visually indistinguishable; pdi is visibly noisier.
  ![](img/final_zero_day_static_crop.png)
- **Lag**, Bistro sun off at full res, base | hyb | hyb+r16c8, 8 frames after (also 1 and 32 frames after:
  `img/final_bistro_sun_off_f025.png`, `_f056.png`). Base's building fronts still glow with the old daylight
  bounce:
  ![](img/final_bistro_sun_off_f032.png)
- **Through DLSS-RR**, Bistro sun off, base | hybrid | pure ORCA, quarter speed:
  [`video/bistro_sun_off_dlss.mp4`](video/bistro_sun_off_dlss.mp4). Still at 17 frames after:
  ![](video/peek.png)

### Commits on `exp/orca`
- Hybrid and knobs: `613ec674a`, `22d0ab4ab`, `7a2060a97`.
- r16c8 port: `8043265d0`, `266d23b13`.
- Rejected or no-effect variants: `a8daf1e42` (pdi), `983126d4a` (total-normalized detector), `417ec6b95`
  (direct-light alpha and cold mode 0).
- Harness (`ORCA_SWEEP` in-process A/B, light restore, `BENCH_MATERIAL`): `6ffb08927`, `30a3c8299`,
  `5525a4c10`.
- Plus `exp/base`'s harness fixes (`2827a780e` … `f5388906c`).

---

# Details

## 1. What ORCA actually is (verified against the talk's slides)

**Sources**
- Course page, SIGGRAPH 2026 Advances in Real-Time Rendering: https://advances.realtimerendering.com/s2026/index.html
  (title, speaker, abstract; the only material is a 447 MB PPTX, no PDF, notes or video).
- Slide deck (with speaker notes), downloaded with the user's OK and read in full (59 slides):
  https://advances.realtimerendering.com/s2026/content/ORCA%20Siggraph%20r3.4_%20Advances.pptx
  Slide numbers below are into this deck.
- ACM DL https://dl.acm.org/doi/10.1145/3799820.3828561 returns 403 to automated fetches (not read).

The previous agent's draft (`REPORT_research_draft.txt`) only had the abstract and labelled the rest as guesses.
Its guess "filled and consumed in the same frame, no temporal history" is **confirmed**. Its guess "trace a cheap
set of training paths and accumulate radiance into a spatial structure with atomics" is **confirmed**. Its guess
"many training samples or coarse cells" is **partly right**: ORCA uses a *multi-scale* hierarchy, not just coarse cells.
Its guess that ORCA is a drop-in history-free replacement for a SHARC-style world cache is **wrong**: see the F1
point at the end.

**What ORCA is, in my words** (summarised from the speaker notes; slide numbers in brackets)
- Scope: it accelerates the *indirect* part of a real-time path tracer. The G-buffer and primary direct light are
  done separately [4]. The goal is far less than 1 path per pixel, for lower-end hardware [4].
- Split the pixels into **sparse** pixels (about 1/16, i.e. 6.25%) and **reconstruction** pixels [27, 38].
  - Sparse pixels trace a full path (4 bounces in F1 [59]). They record the path radiance arriving back at
    their **first indirect hit** [27, 28].
  - Reconstruction pixels trace only the first ray off the G-buffer. They look up that hit in the cache and use the
    cached radiance as the path tail, times their own throughput [30-35, 57].
- **The cache** is a GPU hash map [13-14] in the style of Binder et al. 2018 (path-space filtering by jittered
  spatial hashing). It hashes position, an **incoming direction** (octahedral, 4 bits, so 256 directional slots
  [57]) and a voxel size, with a second hash as the checksum and linear probing [14].
  - Only first hits are inserted, keyed by position *and* direction, to avoid aggregating across thin walls [37, 57].
- **Multi-scale ("downrez") hierarchy** [19-25]. Each sparse first hit is inserted at *every* voxel size in a fixed
  range. Sizes are powers of 8, from about a G-buffer pixel's footprint up to very large [21].
  - Radiance is aggregated into coarser levels with fixed-point atomics, one pass per level, then repacked [24-25].
  - A lookup scans from the smallest voxel to the largest and uses the first one that exists [31].
  - This gives very high hit rates from few paths. The speaker says this is what lets ORCA **drop temporal
    accumulation entirely** [27 notes]. Misses are under 1-2% and are left to the denoiser or traced [36].
- **No history**: the hash map is cleared and rebuilt from scratch every frame [36]. The FAQ says it does not need
  to respond to lighting changes, because it has no history to invalidate [57].
- **Pixel selection**: blue/FAST noise thresholds [38]. Specular lobes get extra paths through a BSDF heuristic
  [40-41]. The per-pixel probabilities are renormalised to a fixed path budget, which keeps cost and memory steady [44-45].
  - An optional, lightly temporal variant drives selection from reprojected per-pixel variance [46-48].
- **Other mitigations** [52-53]:
  - Fireflies: clamp, or weight injected radiance by the selection pdf.
  - Cache too sparse: inject *previous frames'* path hits down-weighted by 1/(1+historyLength), after
    **relighting their first vertex** so the radiance is current. The FAQ adds: limit this to static geometry [57].
  - Voxel artifacts in sharp reflections: dither the lookup position per pixel (Binder 2018).
- **Cost** (Xbox Series X, 1536x864) [39-45]: the PT part goes from 14.2 ms to 5.0 ms (uniform 1/16), 6.5 ms
  (specular heuristics), and 7.3 ms (budget-balanced). The cache build itself is about 0.6-0.9 ms. The
  reconstruction rays are the biggest non-PT cost [45, 50].
  - F1 25 on PS5 Pro: the PT part goes from 38.4 ms to 7.4 ms at 1080p [51]. Not shipped [50].
- **ORCA runs *alongside* SHARC in F1** [50, 59]. SHARC (a persistent, temporally accumulated world-space cache,
  i.e. the same kind of thing as Solari's world cache) supplies the **path tails** of the sparse paths. ORCA is
  "instantaneous and extremely short term" and covers the first indirect hit [59].
  - The speaker also says that keeping temporal state works against fully dynamic games [58], which is why
    ORCA itself has none.

**What this means for Solari**
- Solari's per-pixel path already *is* ORCA's reconstruction path for rough lobes:
  - `initial_path.wesl` does NEE at x1, traces one BRDF ray to x2, and terminates into the world cache at x2
    (`terminate_into_cache`).
  - ORCA's 3x speedup comes from replacing full per-pixel paths with "one ray + cache lookup". Solari already
    has that, via its world cache.
- What ORCA does differently is **where the cache's radiance comes from**:
  - ORCA: this frame's sparse camera paths, aggregated over a multi-scale hierarchy. No history, so no lag, but
    each lookup carries only one frame's worth of samples.
  - Solari: a persistent cache whose cells are updated on a budgeted rolling window (1 RIS NEE + 1 GI ray that
    recursively queries the cache) and blended over up to 32 samples. Low noise, but lighting changes take tens
    of frames to propagate.
- So "apply ORCA to the world cache" means: **rebuild the cache's content within the frame instead of
  accumulating it**. ORCA's own remedy for a too-sparse in-frame cache is to keep old samples but **relight their
  first vertex**. That maps onto Solari as "direct light in-frame, indirect light temporal".
- ORCA's multi-scale hierarchy solves *coverage*: with sparse seeding, most lookups would otherwise miss. Solari's
  cache has no coverage problem:
  - A query inserts its cell, and with hot-cell scheduling every queried cell is updated the next frame.
  - With "use the smallest voxel that exists", the hierarchy would therefore never fall back to a coarser level
    in Solari. Making it fall back by *sample count* instead would be spatial filtering, i.e. new bias (leaks,
    blur). That is not what ORCA does, so I did not build it.
- Direction-keyed slots serve glossy first hits. Solari only terminates into the cache for broad lobes and stores
  diffuse irradiance per quantized normal, so this doesn't apply. Dithered lookup is already in Solari (tangent-
  plane jitter in `query_world_cache`). Budgeted sample counts map to `world_cache_max_cell_updates_per_frame` and
  the `ORCA_SAMPLE_BUDGET` knob.
- The real ORCA (sparse camera paths plus reconstruction, as a *lighting-pass* change rather than a world cache
  change) is out of scope here. It would replace Solari's world cache *update* with camera-path seeding, and the
  F1 integration suggests keeping the persistent cache for tails anyway. See "Not tried" below.

## 2. What was implemented

All variants live in `world_cache_update.wesl` / `world_cache_compact.wesl` behind `ORCA_*` knobs. Since
`7a2060a97` the knobs are read every frame, so one process can A/B them.

- **Hot/cold scheduling (`ORCA_SCHED=1`):**
  - Compaction runs a two-channel prefix sum, putting **hot** cells first. A hot cell is one queried by a
    camera path last frame: life == LIFETIME−1 after decay.
  - Every hot cell is updated every frame. Cold cells keep the baseline's 40k rolling window.
  - Hot cells: about 162k on Bistro, 42k on Zero-Day.
- **Update modes**, per hot/cold (`ORCA_MODE` / `ORCA_COLD_MODE`):
  - 0 = baseline temporal blend.
  - 1 = history-free ("orca"): the cell holds this frame's estimate only.
  - 2 = direct light in-frame plus temporally blended indirect, stored in a new `radiance_indirect` array
    (+16 B/cell).
  - 3 = like 2, but hot cells trace the GI ray only on the baseline's rolling schedule. That keeps the
    indirect sample rate, and so its noise, identical to base.
- **`ORCA_SAMPLES` / `ORCA_SAMPLE_BUDGET`:** light samples per hot cell, capped so that
  hot_cells × samples ≤ budget. Zero-Day gets 4, Bistro 1.
  - With 1 sample, Zero-Day was 1.29× noisier in an earlier single run (`q/zero_day_r0_s1m3n1`). With 4 it
    matches base.
  - Sunlit cells don't need more: a cell is one fixed point, so its sun visibility doesn't flicker.
- **`ORCA_DETECTOR=1`, `ORCA_DETECTOR_SCALE=16`** (`8043265d0`): item 4's rate-normalized change detector.
  The luminance trend is scaled by `scale / frames_between_updates` before the adaptive blend. For hybrid
  hot cells, `frames_between_updates` is the indirect update period. No per-cell state.

**Hybrid = SCHED 1, MODE 3, COLD_MODE 2, 4 samples, 200k budget.**

## 3. Results

Base and the variants always ran in the **same process**, with `SolariLighting::reset` and the lights
restored between configs (`ORCA_SWEEP`), and each got its own full warmup.
- "Base" is the orca binary with no knobs set. That is the `exp/base` algorithm, plus the unused
  `radiance_indirect` storage and the two-channel compaction prefix sum.
- The planned `exp/base`-binary timing cross-check wasn't run (cancelled for GPU budget).
- The earlier agent's single `exp/base`-equivalent probe measured the world cache pass at 0.238 ms on
  Bistro, against 0.224 ms here.

### Timing (1080p, DLSS-RR, locked 1792 MHz, base and hyb interleaved in one process, 3 repeats, medians)

| scene | ReSTIR | variant | world_cache ms | lighting ms | presample ms | total Solari ms | Δ vs base | active cells | hot cells |
|---|---|---|---|---|---|---|---|---|---|
| bistro | off | base | 0.224 | 2.721 | 0.049 | 2.994 | | 318k | 0 |
| bistro | off | hyb | 0.451 | 2.722 | 0.049 | 3.243 | **+0.250** | 327k | 162k |
| bistro | on | base | 0.222 | 5.634 | 0.047 | 5.903 | | 318k | 0 |
| bistro | on | hyb | 0.473 | 5.638 | 0.047 | 6.158 | **+0.255** | 327k | 162k |
| zero_day | off | base | 0.204 | 1.766 | 0.061 | 2.030 | | 75k | 0 |
| zero_day | off | hyb | 0.388 | 1.761 | 0.061 | 2.210 | **+0.179** | 88k | 42k |
| zero_day | on | base | 0.174 | 2.964 | 0.057 | 3.196 | | 75k | 0 |
| zero_day | on | hyb | 0.366 | 2.962 | 0.057 | 3.385 | **+0.189** | 89k | 42k |

The active cell count grows ~3% on Bistro and ~17% on Zero-Day: hot cells' GI rays touch more cells.
r16 was not timed; it is one multiply per updated cell.

### Quality (1280×720, 64 frames, no denoiser, vs pathtracer refs, ReSTIR off)

| scene | variant | runs | relvar_t999 (mean; per run) | vs base | relvar | relbias2 | Δrelbias2 | mean brightness vs base | frame_relmse | sunlit / unlit |
|---|---|---|---|---|---|---|---|---|---|---|
| zero_day | base | 3 | 9.82 (13.9; 8.34; 7.27) | | 25.1 | 0.1215 | | | 11.8 | |
| zero_day | hyb | 3 | 8.80 (9.84; 7.88; 8.69) | median 1.04× | 22.1 | 0.1180 | −0.0035 | −0.4% | 12.0 | |
| indoor0 | base | 1 | 5.34 | | 6.01 | 0.00621 | | | 4.24 | 0.512 / 4.65 |
| indoor0 | hyb | 1 | 5.58 | 1.05× | 6.44 | 0.00651 | +0.0003 | −0.7% | 3.87 | 0.488 / 4.39 |
| indoor1 | base | 1 | 5.96 | | 6.39 | 0.00453 | | | 4.84 | — / 4.84 |
| indoor1 | hyb | 1 | 6.25 | 1.05× | 8.01 | 0.00506 | +0.0005 | −0.3% | 4.97 | — / 4.99 |
| bistro* | base / hyb / orca | 1 | 2.26 / 2.30 / 2.39 | 1.015× / 1.055× | | 1.225 / 1.235 / 1.231 | +0.010 / +0.006 | | 2.50 / 2.50 / 2.53 | |

\*Bistro: the 24 frames before the sun starts moving in the full-res `sun_orbit` screening process
(10 s / 800-frame warmup), `screen/`.

- Zero-Day varies ±20% run to run (base's own runs range 7.3–13.9).
- On indoor0 the hybrid's single-frame error is *lower* (frame_relmse 4.24 → 3.87), both in the sun patches
  and elsewhere.
- indoor1's plain relvar +25% says it has slightly more fireflies; the trimmed metric is +5%.

Undenoised single frames (base | hyb | reference), centre crops; the full width is the same name without
`_crop`:
- Zero-Day: ![](img/raw_zero_day_r0_crop.png)
- indoor1: ![](img/raw_indoor1_r0_crop.png)
- indoor0: ![](img/raw_indoor0_r0_crop.png)
- Bistro (base | hyb | orca | reference): ![](img/raw_bistro_r0_crop.png)

### Reactivity

**Protocol settle/tail** (`solari_metrics.py settle … 24 0.1`, ReSTIR off; 640×360 unless noted).

| event | base: left / tail | hyb: left / tail | notes |
|---|---|---|---|
| Bistro `sun_off`, 1280×720 64-frame smoke | 4.8%, still drifting at +38 frames | 1.6% | orca 1.7%, settles in 3 frames |
| Bistro `sun_off` | 7.5–7.6% / 68–69 | 2.8–2.9% / 50–57 | two runs (f6, f8) |
| Bistro `local_off` | below noise | below noise | local lights are small next to the sun |
| indoor0 `sun_move` | 222-frame drift | below noise | small step; not trustworthy at 640×360 |
| Zero-Day `local_dim` | 35–36% / 124–128 | 23–24% / 132–146 | two runs; settle 63–64 → 48–49 |
| indoor1 `local_off` | 18.6% / 32 | 7.2% / 62 | |

**Moving lights** (`solari_metrics.py motion … 88`).

| event | base lag_gap / catch-up | hyb | notes |
|---|---|---|---|
| Bistro `sun_orbit`, 1280×720 | 0.0090 / 37 fr | 0.0003 (below noise) | orca −0.0001 |
| Bistro `sun_orbit` | 0.0075 / 37 fr | 0.0001 (below noise) | f8 |
| indoor0 `local_orbit` | 0.0053 / 99 fr in f6; below noise in f8 | below noise | not reproducible: no detectable lag |
| Zero-Day `local_orbit` | below noise (0.0065, floor 0.17) | below noise (0.0019) | |

Images:
- Bistro sun off through DLSS-RR, 1, 8 and 32 frames after: `img/event_bistro_sun_off_dlss_f017.png`,
  `_f024.png`, `_f048.png`.
- The video: `video/bistro_sun_off_dlss.mp4`.

## 4. Shipping notes (see the TL;DR for the final list)

- Make mode 3 (hot) / mode 2 (cold) the only update path. Drop modes 0/1 and the env reading.
- Keep the hot/cold compaction (two-channel prefix sum) and `radiance_indirect` (+16 B/cell).
- Fold the sample budget into `SolariLighting` (e.g. `world_cache_direct_sample_budget`, default 200k).
- Remove `ORCA_CELL_SCALE` / `ORCA_MAX_UPDATES`, which were never used for results.
- The adaptive blend (`luminance_deltas`) now sees only indirect changes. That's what it should react to,
  since direct is always current. r16 then applies to that indirect blend only.
- **Before merging:**
  - One full-res reactivity confirmation of hyb+r16 on Bistro `sun_off` and Zero-Day `local_dim`. The
    combined lag numbers above are single 640×360 runs.
  - Shipping with ReSTIR on also wants a ReSTIR-on noise check. Only timing ran with ReSTIR on.

## 5. Dead ends and notes

- **Pure history-free cache:** fastest reaction, but every cell carries one frame of samples. Noise on
  many-light scenes and about 4 ms of cache time.
- **`ORCA_SCHED=1` alone** (hot cells blended every frame): no reactivity gain in an earlier single run
  (Zero-Day `local_dim` settle 32 vs 35 frames). The blend floor (`frames_between_updates / 32`) keeps a
  ~32-update window, so updating more often mostly changes noise. This matches item 4's finding.
- **The rest of ORCA** (multi-scale hierarchy, direction-keyed slots, sparse camera-path seeding) doesn't
  apply to Solari's world cache, which has no coverage problem (see section 1). Building the "real" ORCA
  as a lighting-pass change was out of scope.
- **Harness problems found along the way, all fixed on `exp/base`:**
  - Event timing lagged by up to ~200 frames in Bistro (`5a082b8ed`).
  - Readback staging buffers ran the GPU out of memory in slow configs (`1a79c8726`, `ae6942e43`).
  - Full-res indoor stats crashed after the 640 sun masks were added (fixed by the lead).
- **Earlier agent's numbers:**
  - Its settle numbers predate the event-timing fix and are invalid.
  - Its "1.6–2.3× noisier" used plain relvar, which fireflies dominate.
  - Its captures are kept in `q/`, `e/`, `probe/`. Captures from 18:17 to the 20:21 reboot are in
    `f4_suspect_pre_reboot/` and were not used.
- **Data folders:**
  - `f5/` (full-res timing + quality), `f6/` (640 reactivity), `f8/` (640 combined), `f9/` (full-res
    hyb+r16 Zero-Day check).
  - `screen/` (full-res sun_orbit), `video/`, `img/`.
  - `refs_full/` is a temporary copy of refs and can be deleted.

## 6. Combined test and per-pixel direct light (f11, 640×360, one process per scene)

All nine variants were swept in one process per scene (`ORCA_SWEEP`, a throwaway warm-up config first,
`SolariLighting::reset` between configs). Single runs, ReSTIR off.
- **pdi:** where a camera path terminates into the world cache, direct light at that vertex is sampled per
  pixel (light-tile RIS, `secondary_local_light_samples`, one shadow ray, full BRDF). The cache supplies only
  its indirect part.
  - No MIS is needed: the path ends there, so nothing else samples that vertex's direct light.
  - Emission isn't double counted: a cell's GI rays never add the emission of the surface they hit.
- **pdi-sun:** the same for directional lights only. The cache returns its value minus the cell's
  directional direct light, which costs cells a second RIS + shadow ray.

### Lag

`settle / left / tail`: settle frames, share of the change still missing in the event frame
(`indirect_step_frac`), frames to within 10% of that remainder.

| event | base | hyb | r16c8 | hyb+r16c8 | pdi | pdi-sun | hyb+pdi | hyb+pdi-sun | hyb+pdi+r16c8 |
|---|---|---|---|---|---|---|---|---|---|
| zero_day_local_dim (settle / left / tail) | 151 / 35.7% / 151 | 50 / 23.2% / 139 | 108 / 36.2% / 108 | 33 / 23.6% / 90 | 64 / 25.4% / 148 | 61 / 36.5% / 125 | 52 / 24.8% / 140 | 47 / 23.7% / 143 | 26 / 23.4% / 86 |
| indoor1_local_off (settle / left / tail) | 12 / 18.6% / 32 | 0 / 7.2% / 61 | 6 / 19.7% / 9 | 0 / 7.1% / 15 | 0 / 7.5% / 41 | 12 / 19.9% / 32 | 0 / 6.9% / 61 | 0 / 7.2% / 61 | 0 / 7.0% / 16 |
| bistro_sun_off (settle / left / tail) | 0 / 7.5% / 66 | 0 / 2.7% / 54 | 0 / 7.4% / 48 | 0 / 3.1% / 46 | 0 / 1.8% / 190 | 0 / 1.6% / 155 | 0 / 1.7% / 213 | 0 / 1.7% / 90 | 0 / 1.8% / 54 |
| bistro_sun_orbit (lag_gap / catch-up; floor ~0.015) | 0.0087 / 55 fr | 0.0005 (noise) | 0.0075 / 36 fr | -0.0002 (noise) | 0.0006 (noise) | -0.0000 (noise) | -0.0002 (noise) | -0.0001 (noise) | 0.0001 (noise) |

### Static noise and brightness (32 frames, vs `refs/<scene>_640`)

| scene (640x360, 32 frames) | metric | base | hyb | r16c8 | hyb+r16c8 | pdi | pdi-sun | hyb+pdi | hyb+pdi-sun | hyb+pdi+r16c8 |
|---|---|---|---|---|---|---|---|---|---|---|
| indoor0 | relvar_t999 vs base | 1.00 | 1.02 | 1.02 | 1.03 | 1.50 | 1.06 | 1.51 | 1.06 | 1.52 |
| indoor0 | mean vs ref | -4.7% | -5.3% | +0.5% | -5.3% | -3.7% | -2.9% | -3.6% | -3.8% | -3.7% |
| indoor0 | relbias2 | 0.005354 | 0.005549 | 0.003052 | 0.005798 | 0.003192 | 0.005688 | 0.006525 | 0.004642 | 0.002521 |
| indoor0 | frame_relmse sunlit / unlit | 0.447 / 4.61 | 0.475 / 4.53 | 0.383 / 4.28 | 0.431 / 4.39 | 0.565 / 7.96 | 0.463 / 4.27 | 1.13 / 9.79 | 0.503 / 4.83 | 0.88 / 6.95 |
| indoor1 | relvar_t999 vs base | 1.00 | 0.94 | 0.93 | 0.96 | 1.46 | 0.93 | 1.46 | 0.95 | 1.44 |
| indoor1 | mean vs ref | -1.5% | -5.3% | +1.8% | -4.7% | -3.7% | -3.9% | -3.8% | -5.0% | -4.8% |
| indoor1 | relbias2 | 0.002098 | 0.004839 | 0.002058 | 0.003447 | 0.002487 | 0.00369 | 0.003152 | 0.003457 | 0.004812 |
| zero_day | relvar_t999 vs base | 1.00 | 1.04 | 1.01 | 1.04 | 1.67 | 0.87 | 1.72 | 1.03 | 1.66 |
| zero_day | mean vs ref | -23.3% | -24.0% | -20.4% | -25.3% | -26.1% | -24.9% | -26.1% | -23.3% | -28.9% |
| zero_day | relbias2 | 0.121 | 0.1391 | 0.1352 | 0.1278 | 0.1419 | 0.1399 | 0.1281 | 0.129 | 0.1567 |
| bistro | relvar_t999 vs base | 1.00 | 1.01 | 1.01 | 1.01 | 1.18 | 1.16 | 1.19 | 1.16 | 1.19 |
| bistro | mean vs ref | -1.4% | -1.3% | -1.1% | -1.0% | +0.5% | +0.5% | +0.6% | +0.5% | +0.7% |
| bistro | relbias2 | 0.6545 | 0.6569 | 0.6585 | 0.6589 | 0.6551 | 0.6599 | 0.66 | 0.6593 | 0.6584 |

### Timing, per-pixel direct-light variants (1080p, 2 interleaved repeats)

⚠️ Partly captured after the 23:11 screen lock, which throttles rendering (fixed later by harness
`19970a6c4`), so treat these numbers as **indicative only**. The hybrid's clean timing is in section 3;
the post-fix re-run is in section 7.

| scene | variant | runs | world_cache ms | lighting ms | total Solari ms | vs base |
|---|---|---|---|---|---|---|
| bistro | base | 2 | 0.240 | 2.726 | 3.015 | +0.000 |
| bistro | hyb | 2 | 0.508 | 2.786 | 3.345 | +0.330 |
| bistro | pdi | 2 | 0.265 | 3.698 | 4.018 | +1.003 |
| bistro | pdi-sun | 2 | 0.287 | 3.323 | 3.665 | +0.650 |
| bistro | hyb+pdi | 2 | 0.566 | 3.702 | 4.323 | +1.308 |
| zero_day | base | 2 | 0.181 | 1.681 | 1.921 | +0.000 |
| zero_day | hyb | 2 | 0.392 | 1.610 | 2.063 | +0.141 |
| zero_day | pdi | 2 | 0.200 | 2.134 | 2.394 | +0.473 |
| zero_day | pdi-sun | 2 | 0.214 | 1.803 | 2.078 | +0.156 |
| zero_day | hyb+pdi | 2 | 0.412 | 2.289 | 2.762 | +0.841 |

### Full-res Zero-Day brightness (palindromic order within each process)

| | base | hyb | hyb+total-norm | hyb+r16c8 | hyb+r16c8+total-norm | pdi | hyb+pdi |
|---|---|---|---|---|---|---|---|
| samples | 8 | 8 | 4 | 8 | 4 | 4 | 4 |
| mean vs pathtracer | −19.65% (sd 1.01) | −19.58% (sd 0.55) | −19.60% | **−20.94%** (sd 0.45) | −21.24% | −21.7% | −22.1% |
| relvar_t999 median vs base | 1.00 | 1.07 | 1.06 | 1.06 | 1.06 | 1.73 | 1.71 |
| relbias2 | 0.1180 | 0.1167 | 0.1155 | 0.1176 | 0.1233 | 0.1186 | 0.1161 |

### Detector normalization fix (f14)

`ORCA_DET_TOTAL=1` divides the hybrid's indirect trend by the cell's total luminance. On Zero-Day and indoor1
lag it changes nothing:
- Zero-Day `local_dim` settle/tail: hyb 49/143 vs hybt 49/140; hyb+r16c8 25/85 vs 27/92.
- indoor1 `local_off`: hyb 59 vs hybt 61; hyb+r16c8 16 vs 15.
- Zero-Day brightness: no change (table above).
- Indoor brightness: done. Full-res indoor1 ×4: base −4.06%, hyb −4.57% (**−0.5 pts**), hyb+total-norm −4.58%, hyb+r16c8 −4.65%. At 640×360 ×8: indoor1 −1.1 pts, indoor0 −1.1 pts. `ORCA_DET_TOTAL` doesn't change it.

## 7. Confirmation runs (f13: base / hyb / hyb+r16c8, palindromic order)

| | base | hyb | hyb+r16c8 |
|---|---|---|---|
| Bistro `sun_off` full res (left / tail) | 7.6% / 73 | 2.3% / 67 | 2.6% / 60 |
| indoor0 640×360 ×4: mean vs ref / relvar_t999 / relbias2 | −3.50% (sd 0.67) / 1.00 / 0.0049 | **−4.98%** (sd 0.29) / 1.00 / 0.0057 | **−5.15%** / 1.00 / 0.0066 |
| indoor1 640×360 ×4: mean vs ref / relvar_t999 / relbias2 | −3.02% (sd 0.72) / 1.00 / 0.0031 | **−4.62%** (sd 0.31) / 0.95 / 0.0036 | **−4.85%** / 0.94 / 0.0042 |
| many_lights 640×360 ×2: mean vs ref / relvar_t999 | −5.20% / 1.00 | −4.81% / 1.01 | −4.84% / 1.00 |
| many_lights `local_off` | not informative: `local_off` turns every light off, so every variant goes black at once | | |
| Timing (harness fix 19970a6c4, 3 interleaved repeats) | Bistro: base 3.003 ms, hyb 3.219 (+0.216, world cache 0.216 → 0.451), hyb+r16c8 3.215 (+0.211). Zero-Day: base 1.777, hyb 1.994 (+0.216, world cache 0.180 → 0.394), hyb+r16c8 1.992 (+0.215). Frame counts steady (1500–1650 per window), so valid | | |

### Material classes (`BENCH_MATERIAL`, indoor1 `local_off`, 640×360, base / hyb / hyb+r16c8)

`BENCH_MATERIAL` (`5525a4c10`) forces metallic/roughness on every material: smooth metal (1, 0.1), rough
metal (1, 0.6), smooth dielectric (0, 0.1), rough dielectric (0, 0.7). Base colors and emission are kept.
Lag is from `local_off`. Noise is from the 24 pre-event frames, against an **independent** base capture (the
throwaway warm-up config) as the reference.

| class | base: missing light / tail | hyb | hyb+r16c8 | noise t999 hyb / hyb+r16c8 (median pixel) |
|---|---|---|---|---|
| smooth metal | 6.6% / 31 | 2.2% / 59 | 2.2% / **15** | 1.02× / 1.03× (1.12× / 1.09×) |
| rough metal | 13.7% / 31 | 4.5% / 64 | 4.4% / **16** | 1.00× / 1.01× (1.08× / 1.10×) |
| smooth dielectric | 17.4% / 32 | 6.8% / 64 | 6.6% / **16** | 1.05× / 1.03× (1.16× / 1.16×) |
| rough dielectric | 19.2% / 32 | 7.3% / 60 | 7.2% / **16** | 1.02× / 1.03× (1.08× / 1.10×) |
| default (repeat run) | 20.2% / 32 | 7.2% / 60 | — | 1.03× (1.11×) |

The hybrid behaves the same across material classes. Glossy surfaces terminate into the cache later, but the
fresh direct light at the cells they eventually read still removes ~3× of the lingering light, at no
measurable noise cost.

**Measurement pitfall I hit (corrected above).** Scoring noise against base's *own* mean made base look
3–25× quieter than everything else. An identical second base config scored 2.75× the same way. That's the
self-reference effect the lead documented. The independent reference removes it.

## 8. Dead ends from the last round (640×360, indoor1 `local_off` per material class)

All of these were chasing the noise artifact described in section 7, but they also bear on the indoor
darkening and the lag.
- **`ORCA_DIRECT_ALPHA`** ½ and ¼ (`417ec6b95`): a short history for the hybrid's direct light. Noise
  unchanged. Lag partly given back: rough dielectric missing light 7.3% → 13.5% / 16.9%, smooth metal
  2.2% → 4.5% / 5.7%. Rejected.
- **`ORCA_COLD_MODE=0`** (`417ec6b95`): cold cells blended like the baseline, hot cells hybrid. Same noise
  and lag as the hybrid. Rejected as unnecessary.
- **`ORCA_DET_TOTAL=1`** (`983126d4a`): no effect on lag, noise, or the indoor darkening (section 6).
