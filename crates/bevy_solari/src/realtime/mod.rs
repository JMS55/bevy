mod extract;
mod node;
mod prepare;
mod variance_post;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{
        DeferredPrepass, DeferredPrepassDoubleBuffer, DepthPrepass, DepthPrepassDoubleBuffer,
        MotionVectorPrepass,
    },
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{component::Component, reflect::ReflectComponent, schedule::IntoScheduleConfigs};
use bevy_pbr::DefaultOpaqueRendererMethod;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use extract::extract_solari_lighting;
use node::{init_solari_lighting_pipelines, solari_lighting};
use prepare::prepare_solari_lighting_resources;
use tracing::warn;
use variance_post::{
    init_solari_variance_post_pipeline, prepare_solari_variance_post_resources,
    solari_variance_post,
};

/// Raytraced direct and indirect lighting.
///
/// When using this plugin, it's highly recommended to set `shadow_maps_enabled: false` on all lights, as Solari replaces
/// traditional shadow mapping.
pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wgsl");
        load_shader_library!(app, "realtime_bindings.wgsl");
        load_shader_library!(app, "presample_light_tiles.wgsl");
        load_shader_library!(app, "initial_path.wgsl");
        embedded_asset!(app, "restir.wgsl");
        load_shader_library!(app, "world_cache_query.wgsl");
        embedded_asset!(app, "world_cache_compact.wgsl");
        embedded_asset!(app, "world_cache_update.wgsl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wgsl");

        // Variance debug tooling (see `variance.wgsl` and `SolariVarianceDebug`).
        load_shader_library!(app, "variance.wgsl");
        embedded_asset!(app, "variance_accumulate.wgsl");
        embedded_asset!(app, "variance_present.wgsl");

        app.register_type::<SolariVarianceDebug>()
            .insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "SolariLightingPlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        render_app
            .add_systems(
                RenderStartup,
                (
                    init_solari_lighting_pipelines,
                    init_solari_variance_post_pipeline,
                ),
            )
            .add_systems(ExtractSchedule, extract_solari_lighting)
            .add_systems(
                Render,
                (
                    prepare_solari_lighting_resources,
                    prepare_solari_variance_post_resources,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                solari_lighting
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            )
            // Post-denoise variance tap + heatmap present: after all EarlyPostProcess
            // (so after DLSS Ray Reconstruction has denoised + upscaled into the view
            // target) and before all PostProcess (so it reads the denoised HDR before
            // tonemapping). Using the set labels avoids a cross-crate system reference.
            .add_systems(
                Core3d,
                solari_variance_post
                    .after(Core3dSystems::EarlyPostProcess)
                    .before(Core3dSystems::PostProcess),
            );
    }
}

/// A component for a 3d camera entity to enable the Solari raytraced lighting system.
///
/// Must be used with `CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING)`, and
/// `Msaa::Off`.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(
    Hdr,
    DeferredPrepass,
    DepthPrepass,
    MotionVectorPrepass,
    DeferredPrepassDoubleBuffer,
    DepthPrepassDoubleBuffer
)]
pub struct SolariLighting {
    /// Maximum confidence weight (effective temporal history length) a pixel
    /// can accumulate during temporal resampling.
    ///
    /// Higher values are more stable but slower to react to lighting changes
    /// and will lead to increased artifacts.
    pub confidence_weight_cap: f32,

    /// Number of direct light samples taken for the camera's primary hit during
    /// initial sampling.
    ///
    /// Higher values reduce noise in directly-lit areas at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub primary_di_samples: u32,

    /// Number of direct light samples taken at each indirect bounce during
    /// initial sampling.
    ///
    /// Higher values reduce noise in indirect lighting at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub secondary_di_samples: u32,

    /// Maximum number of bounces traced when generating an initial path.
    ///
    /// Higher values capture more indirect light for greater accuracy at the cost
    /// of more rays traced per frame. Lower values are faster but lose
    /// multi-bounce lighting for specular paths.
    pub max_bounces: u32,

    /// How responsive the world cache is to changes in lighting.
    ///
    /// Higher values accumulate more temporal history, giving more stable but
    /// less responsive (slower to update) lighting. Lower values react faster
    /// but are noisier and less stable.
    pub world_cache_max_temporal_samples: f32,

    /// How many direct light samples each world cache cell takes when updating
    /// each frame.
    ///
    /// Higher values reduce noise in cached lighting at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub world_cache_direct_light_sample_count: u32,

    /// Maximum distance to trace GI rays between two world cache cells.
    ///
    /// Higher values capture indirect light from farther away for more accurate
    /// GI at the cost of longer (more expensive) ray traversal and increased noise.
    /// Lower values are faster and less noisy but may miss distant lighting.
    pub world_cache_max_gi_ray_distance: f32,

    /// Soft upper limit on the number of world cache cells to update each frame.
    ///
    /// Higher values let the cache converge faster after lighting changes at the
    /// cost of more work per frame. Lower values are cheaper but make the cache
    /// slower to update.
    ///
    /// This is a stochastic target that only takes effect when the number of
    /// active cells exceeds it: each active cell is then updated with
    /// probability `target / active_cells`, so on average this many cells
    /// update, though individual frames may update more or fewer. When there
    /// are fewer active cells than the target, all of them update every frame.
    pub world_cache_cell_updates_soft_target: u32,

    /// Size of a world cache cell at the lowest LOD, in meters.
    ///
    /// Smaller values give finer spatial resolution and more detailed indirect
    /// lighting at the cost of more cells to fill and update. Larger values are
    /// cheaper but coarser, which can cause light leaking.
    pub world_cache_position_base_cell_size: f32,

    /// How fast the world cache transitions between LODs as a function of
    /// distance to the camera.
    ///
    /// Higher values keep cells small (high detail) out to greater distances for
    /// better quality at the cost of more cells to fill. Lower values transition
    /// to larger cells sooner, which is cheaper but coarser farther from the
    /// camera.
    pub world_cache_position_lod_scale: f32,

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self {
            confidence_weight_cap: 8.0,
            primary_di_samples: 8,
            secondary_di_samples: 4,
            max_bounces: 3,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            world_cache_cell_updates_soft_target: 40000,
            world_cache_position_base_cell_size: 0.15,
            world_cache_position_lod_scale: 15.0,
            reset: true, // No temporal history on the first frame
        }
    }
}

/// Which stage of the [`SolariLighting`] signal the variance debug view displays.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Default)]
pub enum VarianceDebugMode {
    /// No heatmap; the normal image is shown. Stats are still accumulated and
    /// reported for both stages while [`SolariVarianceDebug`] is present, so the
    /// numeric readout stays live for before/after comparison.
    #[default]
    Off,
    /// Heatmap of the raw ReSTIR output's temporal variance (the denoiser's input).
    PreDenoise,
    /// Heatmap of the denoised output's residual temporal variance (flicker the
    /// denoiser failed to remove). Identical to [`Self::PreDenoise`] when no
    /// denoiser (DLSS Ray Reconstruction) is running.
    PostDenoise,
}

/// Restricts the ReSTIR output to a single class of light path, so a given
/// artifact (noise, leaking, flicker) can be attributed to the path type that
/// produces it. Gated at candidate generation in `initial_path.wgsl`; pairs well
/// with the variance heatmap ([`SolariVarianceDebug::mode`]).
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Default)]
pub enum PathIsolation {
    /// No isolation: the normal, full lighting result.
    #[default]
    Normal,
    /// Bounce-0 direct lighting only (NEE + directly BRDF-sampled emitters); no
    /// indirect bounces and no world cache.
    Direct,
    /// Indirect only: contributions from bounce >= 1 plus the world cache, with
    /// bounce-0 direct lighting and directly-visible emission removed.
    Indirect,
    /// World-cache GI terminations only -- the diffuse indirect the cache serves.
    WorldCache,
    /// Everything except the world cache (the normal result minus cache GI).
    NoWorldCache,
}

/// Debug tooling for [`SolariLighting`]: estimates and visualizes per-pixel
/// *temporal* variance of the lit signal, to find the high-variance pixels that
/// visually stand out (fireflies, unstable specular, undersampled indirect)
/// before and after denoising. Also carries a [`PathIsolation`] selector for
/// attributing an artifact to a specific class of light path.
///
/// Add this component to a camera that already has [`SolariLighting`] to enable
/// the tool. Its mere presence turns on variance accumulation every frame and
/// publishes numeric stats as render diagnostics (under
/// `render/solari_lighting/variance_pre/*` and `.../variance_post/*`); [`mode`]
/// selects which stage, if any, is drawn as a full-screen heatmap.
///
/// The metric is *relative* variance (`Var / mean^2`), which is scale-invariant,
/// so noise reads as "hot" regardless of how bright the surface is.
///
/// [`mode`]: SolariVarianceDebug::mode
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Default, Clone)]
pub struct SolariVarianceDebug {
    /// Which stage's heatmap to display (or [`VarianceDebugMode::Off`]).
    pub mode: VarianceDebugMode,

    /// Relative-variance value mapped to the top (red) of the heatmap, and the
    /// cutoff for the "pixels over threshold" stat. Lower values make more of the
    /// image read as hot. A good starting point is `0.5`.
    pub threshold: f32,

    /// Capped sample count of the running moments estimate. Larger values give a
    /// smoother, more converged variance estimate that reacts more slowly to
    /// changes; smaller values react faster but are noisier. `64` is a reasonable
    /// default.
    pub history_length: f32,

    /// Restricts the lit output to one class of light path (see [`PathIsolation`]).
    /// [`PathIsolation::Normal`] disables isolation.
    pub path_isolation: PathIsolation,

    /// Skip the ReSTIR temporal resampling stage, to see how much of the noise the
    /// temporal reuse is (or isn't) removing.
    pub disable_temporal_reuse: bool,

    /// Skip the ReSTIR spatial resampling stage. Spatial reuse pulls a fresh random
    /// neighbor every frame; disabling it isolates the per-frame flicker that reuse
    /// injects versus the underlying per-pixel estimate.
    pub disable_spatial_reuse: bool,

    /// Force direct-lighting visibility to 1 (skip the shadow rays). If shimmer
    /// disappears, it was shadow-ray (penumbra/contact) noise.
    pub force_full_visibility: bool,

    /// Strip the specular lobe from the primary surface (treat it as pure diffuse).
    /// If shimmer collapses under this, the specular BRDF is the source; if it
    /// persists, specular is exonerated.
    pub force_diffuse: bool,

    /// Disable the temporal-reuse pixel permutation, reading a pixel's exact
    /// reprojected history instead of a frame-varying ±3px neighbor. If shimmer
    /// drops, the permutation is reading as temporal instability on static surfaces.
    pub disable_temporal_permutation: bool,

    /// Use two-level light RIS for direct lighting: RIS over light triangles with a
    /// cheap target to pick a promising light, then RIS over points on it with the
    /// full BRDF target. Concentrates the sample budget when uniform light selection
    /// over many lights wastes candidates. Experimental -- validate against the
    /// reference path tracer for bias.
    pub two_level_light_ris: bool,

    /// Disable vector-valued (chroma-marginalized) shading, reverting to shading
    /// from the single selected reservoir/light sample. Enable to A/B the color-
    /// noise reduction in the spatial merge and the light RIS.
    pub disable_color_noise_reduction: bool,

    /// Firefly clamp for the direct-lighting contribution, in post-exposure
    /// luminance (so ~1.0 is a midtone). Bounds how bright a single resampled
    /// sample can shade a pixel, killing the sparse `weight/target_function`
    /// spikes that survive denoising. `0.0` disables it (no clamp). Slightly
    /// biased -- it also clips genuine rare-bright highlights -- so tune it just
    /// above the real highlight range (watch the heatmap's `max` readout).
    ///
    /// This is a real rendering fix, parked here on the debug component while it's
    /// being tuned; promote it to [`SolariLighting`] once a good value is found.
    pub firefly_clamp: f32,
}

impl SolariVarianceDebug {
    /// Packs the reuse/visibility toggles into the `debug_flags` bitmask expected by
    /// the shaders (`DEBUG_FLAG_*` in `realtime_bindings.wgsl`).
    pub(crate) fn debug_flags(&self) -> u32 {
        (self.disable_temporal_reuse as u32)
            | ((self.disable_spatial_reuse as u32) << 1)
            | ((self.force_full_visibility as u32) << 2)
            | ((self.disable_color_noise_reduction as u32) << 3)
            | ((self.force_diffuse as u32) << 4)
            | ((self.disable_temporal_permutation as u32) << 5)
            | ((self.two_level_light_ris as u32) << 6)
    }
}

impl Default for SolariVarianceDebug {
    fn default() -> Self {
        Self {
            mode: VarianceDebugMode::Off,
            threshold: 0.5,
            history_length: 64.0,
            path_isolation: PathIsolation::Normal,
            disable_temporal_reuse: false,
            disable_spatial_reuse: false,
            force_full_visibility: false,
            force_diffuse: false,
            disable_temporal_permutation: false,
            two_level_light_ris: false,
            disable_color_noise_reduction: false,
            firefly_clamp: 0.0,
        }
    }
}
