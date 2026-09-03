mod extract;
mod node;
mod prepare;

use crate::{scene::RaytracingSceneBindings, SolariPlugins};
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
    init_gpu_resource, renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup,
    RenderSystems,
};
use bevy_shader::load_shader_library;
use extract::extract_solari_lighting;
use node::{init_solari_lighting_pipelines, solari_lighting};
use prepare::prepare_solari_lighting_resources;
use tracing::warn;

/// Raytraced direct and indirect lighting.
///
/// When using this plugin, it's highly recommended to set `shadow_maps_enabled: false` on all lights, as Solari replaces
/// traditional shadow mapping.
pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wesl");
        load_shader_library!(app, "bindings.wesl");
        load_shader_library!(app, "presample_light_tiles.wesl");
        load_shader_library!(app, "initial_path.wesl");
        embedded_asset!(app, "restir.wesl");
        load_shader_library!(app, "world_cache_query.wesl");
        embedded_asset!(app, "world_cache_compact.wesl");
        embedded_asset!(app, "world_cache_update.wesl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wesl");

        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
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
                init_solari_lighting_pipelines.after(init_gpu_resource::<RaytracingSceneBindings>),
            )
            .add_systems(ExtractSchedule, extract_solari_lighting)
            .add_systems(
                Render,
                prepare_solari_lighting_resources.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                solari_lighting
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
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

    /// How pixels are chosen to be shaded from an estimate that skipped temporal
    /// and spatial reuse, in order to decorrelate the noise between frames.
    ///
    /// Reusing samples is what keeps the lighting quiet, but it also means
    /// consecutive frames largely repeat the same estimate instead of being
    /// independent estimates of the same lighting. ML denoisers, DLSS Ray
    /// Reconstruction in particular, expect temporally independent samples: they
    /// under-denoise correlated noise and smear lighting changes. Shading some
    /// pixels from an unresampled estimate trades variance on those pixels for
    /// that independence.
    ///
    /// Intended for use with DLSS Ray Reconstruction. Leave this off when
    /// rendering without an ML denoiser, where it only adds noise.
    pub decorrelation_mode: DecorrelationMode,

    /// How large a fraction of pixels fall back to the unresampled estimate.
    ///
    /// Higher values feed the denoiser more temporally independent samples at the
    /// cost of a noisier input. Lower values keep more of the variance reduction
    /// that reuse buys, but leave the noise more correlated over time.
    ///
    /// Under [`DecorrelationMode::Stagnancy`] this is the fallback probability of a
    /// maximally stale pixel, with fresher pixels scaled down from it. Under
    /// [`DecorrelationMode::Uniform`] every pixel uses it directly.
    pub decorrelation_strength: f32,

    /// Exponent shaping how fast the fallback probability grows with staleness.
    ///
    /// Higher values confine the fallback to the stalest pixels, keeping variance
    /// reduction elsewhere but leaving mildly correlated pixels as they are. Lower
    /// values spread the fallback more evenly across the screen. The default is
    /// below 1, so the probability ramps up early and most of the screen is already
    /// contributing independent samples before any pixel saturates.
    ///
    /// Only used by [`DecorrelationMode::Stagnancy`].
    pub decorrelation_stagnancy_curve: f32,

    /// How much of the previous frame's staleness estimate is blended into the
    /// current frame's.
    ///
    /// Higher values give a smoother, more stable fallback probability, but make it
    /// slower to react as pixels start or stop reusing samples. Lower values react
    /// immediately, at the cost of a noisy probability that shows up as visible
    /// structure in the noise the denoiser is given.
    ///
    /// Only used by [`DecorrelationMode::Stagnancy`].
    pub decorrelation_stagnancy_smoothing: f32,

    /// How aggressively pixels that resampling made far brighter than their
    /// neighborhood are replaced with an unresampled estimate.
    ///
    /// Resampling occasionally latches onto a very high weight sample, which a
    /// denoiser will smear into a persistent blob. Higher values catch more of these
    /// fireflies at the cost of also replacing legitimately bright pixels, such as
    /// small highlights, with a noisier estimate. Set to 0 to disable the check.
    ///
    /// Independent of [`SolariLighting::decorrelation_mode`], and likewise intended
    /// for use with DLSS Ray Reconstruction.
    pub firefly_replacement_strength: f32,

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl SolariLighting {
    /// Whether any feature that needs the per-pixel decorrelation buffers is enabled.
    pub fn decorrelation_enabled(&self) -> bool {
        self.decorrelation_mode != DecorrelationMode::Off || self.firefly_replacement_strength > 0.0
    }
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
            decorrelation_mode: DecorrelationMode::Off,
            decorrelation_strength: 1.0,
            decorrelation_stagnancy_curve: 0.5,
            decorrelation_stagnancy_smoothing: 0.9,
            firefly_replacement_strength: 0.0,
            reset: true, // No temporal history on the first frame
        }
    }
}

/// How [`SolariLighting`] picks the pixels that get shaded from an estimate which
/// skipped temporal and spatial reuse.
#[derive(Clone, Copy, Reflect, Default, PartialEq, Eq, Hash, Debug)]
#[reflect(Default, Clone, PartialEq, Hash)]
pub enum DecorrelationMode {
    /// Never fall back. Every pixel is shaded from its resampled reservoir.
    #[default]
    Off,
    /// Fall back with the same probability on every pixel.
    ///
    /// Useful as a baseline when tuning, but spends as much variance on pixels that
    /// just generated a fresh sample as on pixels that have been reusing the same
    /// one for dozens of frames.
    Uniform,
    /// Fall back with a probability driven by how long the pixel has been reusing
    /// its current sample.
    ///
    /// Puts the added noise where the correlation actually is, and leaves pixels
    /// whose samples are already fresh alone.
    Stagnancy,
}
