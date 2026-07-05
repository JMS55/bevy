# Zero-Day Example

Beeple's "Zero-Day" sci-fi corridor (NVIDIA ORCA), **path-traced with Bevy Solari**.

> **Note:** upstream this example also drives an HDR output pipeline (GT7 tonemapping,
> a Rec.2020 working space, and display calibration). That machinery is not on this
> branch, so it is dropped here — the example tone maps with the default tonemapper plus
> `Bloom::NATURAL` and the Solari path tracer is what remains.

Zero-Day is authored to be lit entirely by ~10,000 emissive triangles with no punctual
lights — the way NVIDIA's original real-time ["Measure 1"](https://www.youtube.com/watch?v=0WE7CgJMuVc)
demo renders it. That needs a path tracer, so this example requires Solari: the
emissive meshes become real area lights with global illumination. It plays the film's
take (~550 animated objects plus the film's camera flythrough) and drives the render
camera from that camera.

The film's lights also pulse and pop as the camera flies through, but that sequencing was
procedural in Octane — it is **not** baked into any ORCA measure's FBX (all their animation
is rigid transform), and Bevy can't import animated material properties through glTF
anyway. So `animate_emissive` fakes it with a wave of light sweeping the corridor's panels
(disable it with `--no-pulse`).

Requires a **ray-tracing capable GPU** (Solari currently needs the Vulkan backend in
wgpu).

## Getting the scene

Download "Zero-Day" [from developer.nvidia.com](https://developer.nvidia.com/orca/beeple-zero-day).

The download ships several "measures" — each an `.fbx` plus a sibling `tex/` folder of
`.dds` textures. This example can load any of them (see `--scene` below):

| `--scene`                      | FBX                                          |
|:-------------------------------|:---------------------------------------------|
| `measure_one` (default)        | `MEASURE_ONE/MEASURE_ONE.fbx`                |
| `measure_seven`                | `MEASURE_SEVEN/MEASURE_SEVEN.fbx`            |
| `measure_seven_colored_lights` | `MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx` |

Bevy can't load FBX, and Blender's FBX importer mis-reads this Octane-exported asset's
material conventions, so [`convert.py`](convert.py) rebuilds each material from the
naming/channel convention documented in the download's README, bakes the animation into
one scene-length clip, and exports a single self-contained `.glb`:

| Texture          | Channels                                        |
|:-----------------|:------------------------------------------------|
| `_BaseColor.dds` | RGB = base color (alpha = opacity, kept opaque) |
| `_Specular.dds`  | R = occlusion, **G = roughness, B = metallic**  |
| `_Normal.dds`    | DirectX normal (green flipped in the example)   |
| `_Emissive.dds`  | RGB = emissive color                            |

Convert it with the headless Blender helper (Blender 4.x/5.x), dropping the result in
this example's `assets/` folder (which is `.gitignore`d and never committed). Run it once
per measure you want; the example loads by the filenames below:

```console
# measure_one (the default)
blender --background --python-exit-code 1 --python convert.py -- \
  "MEASURE_ONE/MEASURE_ONE.fbx" \
  "examples/large_scenes/zero_day/assets/zero_day_measure_one.glb"

# measure_seven
blender --background --python-exit-code 1 --python convert.py -- \
  "MEASURE_SEVEN/MEASURE_SEVEN.fbx" \
  "examples/large_scenes/zero_day/assets/zero_day_measure_seven.glb"

# measure_seven_colored_lights (same geometry/animation as measure_seven, recolored emissives)
blender --background --python-exit-code 1 --python convert.py -- \
  "MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx" \
  "examples/large_scenes/zero_day/assets/zero_day_measure_seven_colored_lights.glb"
```

## Running

```console
cargo run -p zero_day --release
# a different measure (must be converted first, see above):
cargo run -p zero_day --release -- --scene measure_seven_colored_lights
# with DLSS Ray Reconstruction (needs an NVIDIA RTX GPU + the DLSS SDK):
cargo run -p zero_day --release --features dlss
# trade sharpness for framerate on the heavy measures (DLSS only):
cargo run -p zero_day --release --features dlss -- --scene measure_seven --dlss-quality performance
# reference path tracer for ground truth (pause with Space to let it converge):
cargo run -p zero_day --release -- --pathtracer
```

Controls:

- **C** — toggle the film flythrough vs. free-fly (WASD + mouse).
- **Space** — pause/resume all animation (the take, every object, and the emissive pulse). Free-fly still works while paused.
- **P** — swap realtime ReSTIR lighting vs. the reference path tracer (see `--pathtracer`).
- **N** — toggle DLSS Ray Reconstruction (with the `dlss` feature; realtime path only).
- **B** — run a short benchmark over the flythrough (printed to the console).

A top-right overlay shows Solari's per-pass GPU timings (light tiles, world cache,
lighting, DLSS-RR) and world-cache occupancy.

## Hot-reloading shaders

To iterate on Solari's WGSL shaders (in `crates/bevy_solari/src/`) against this scene, run
with the `embedded_watcher` feature and edit the `.wgsl` files — they reload live:

```console
cargo run -p zero_day --release --features embedded_watcher
```

The watcher's asset root defaults to this crate, but the engine shaders live up in the
workspace, so `main` repoints the asset root to the workspace root when this feature is on
(the scene `.glb` still loads from this crate's `assets/`).

## Profiling

Bevy has built-in [Tracy](https://github.com/wolfpld/tracy) support (see
[`docs/profiling.md`](../../../docs/profiling.md)). This crate re-exposes it as a
passthrough feature, since `-p zero_day` resolves `--features` against this crate rather
than `bevy`. Start the Tracy UI, then:

```console
cargo run -p zero_day --release --features trace_tracy
# with memory-allocation tracking (higher overhead):
cargo run -p zero_day --release --features trace_tracy_memory
```

```console
Options:
  --scene      which ORCA measure to load: measure_one (default), measure_seven, or
               measure_seven_colored_lights. Each is a separate .glb built by convert.py.
  --emissive   emissive multiplier for the accent panels (they are the scene's only
               lights, so they must be bright). Defaults per measure (measure_one 150000,
               measure_seven 600000 -- a much larger space); override if it's too dark or
               blown out.
  --no-pulse   disable the synthetic emissive pulse (a wave of light sweeps the panels by
               default to evoke the film's animated lights, which weren't in the asset).
  --pathtracer start in Solari's reference path tracer instead of realtime ReSTIR lighting
               (P swaps at runtime either way). It converges to ground truth but is slow and
               accumulates across frames, so pause with Space to let a still frame resolve.
               DLSS is ignored in this mode.
  --resolution render resolution as WxH (default 1920x1080). Lower it (e.g. 1280x720) for
               more framerate -- Solari cost scales with pixel count.
  --dlss-quality  DLSS mode: auto (default), dlaa, quality, balanced, performance,
               ultra_performance. Lower renders internally smaller for more framerate.
  --help       display usage information
```
