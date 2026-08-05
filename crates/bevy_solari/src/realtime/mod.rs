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
pub use prepare::SOLARI_DEBUG_COUNTERS;

pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wgsl");
        load_shader_library!(app, "realtime_bindings.wgsl");
        load_shader_library!(app, "debug.wgsl");
        load_shader_library!(app, "presample_light_tiles.wgsl");
        load_shader_library!(app, "initial_path.wgsl");
        embedded_asset!(app, "restir.wgsl");
        load_shader_library!(app, "world_cache_query.wgsl");
        embedded_asset!(app, "world_cache_compact.wgsl");
        embedded_asset!(app, "world_cache_update.wgsl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wgsl");

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

    /// Maximum confidence weight for smooth surfaces, blending up to
    /// [`Self::confidence_weight_cap`] as perceptual roughness rises to 0.3.
    ///
    /// Smooth surfaces have a narrow BRDF lobe, so their target function varies
    /// enormously between candidate samples and resampling needs far more
    /// effective samples to converge than a diffuse surface does. Measured on the
    /// `pica_pica` scene, raising this from 8 to 32 cut specular noise by about a
    /// quarter while leaving diffuse untouched, and lowering it to 2 made specular
    /// noise substantially worse.
    ///
    /// Raising it lengthens temporal history, which also increases lag and
    /// ghosting under motion. That tradeoff is not visible in a
    /// fluctuation-based noise metric, so check it against the reference
    /// pathtracer before raising it far.
    ///
    /// Defaults to [`Self::confidence_weight_cap`], i.e. no special handling.
    pub specular_confidence_weight_cap: f32,

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

    /// Replaces the lit image with a false-colour visualization of one of
    /// Solari's internal signals, for diagnosing where noise comes from.
    ///
    /// See [`SolariDebugView`]. Costs an extra per-pixel buffer while set to
    /// anything other than [`SolariDebugView::None`].
    pub debug_view: SolariDebugView,

    /// Set to true to tally scene-wide rates (temporal rejections, reuse
    /// failures, world cache misses, etc.) and report them as render
    /// diagnostics under `solari_lighting/debug/*`.
    ///
    /// Every tally is a global atomic, so this is slow and should not be
    /// enabled while measuring performance.
    pub debug_counters: bool,
}

/// A false-colour visualization of one of Solari's internal signals, selected by
/// [`SolariLighting::debug_view`].
///
/// Noise in a ReSTIR path tracer can come from many places, and they need
/// different fixes. These views exist to tell them apart. Roughly, work down the
/// list: find where the noise is ([`Self::NoiseRelativeStdDev`]), find which
/// estimator owns it ([`Self::NonResampledShare`], [`Self::SampleProvenance`]),
/// then find why that estimator is starved ([`Self::ConfidenceWeight`],
/// [`Self::TemporalRejectReason`], [`Self::SpatialReuseFailure`],
/// [`Self::JacobianRejection`], [`Self::WorldCacheSampleCount`]).
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Default, Clone, PartialEq)]
#[repr(u32)]
pub enum SolariDebugView {
    /// Show the lit image (no debug visualization).
    #[default]
    None = 0,

    /// Frame-to-frame relative standard deviation of the shaded output, as a
    /// blue -> green -> red heatmap over 0% to 100%.
    ///
    /// This is the noise metric: it is what a denoiser has to remove, and it is
    /// the map to consult before changing any sampling heuristic. Note that it
    /// measures temporal *fluctuation*, which understates true estimator
    /// variance because ReSTIR deliberately correlates samples across frames.
    NoiseRelativeStdDev = 1,

    /// Fraction of this pixel's temporal fluctuation contributed by the
    /// non-resampled (ReSTIR-bypassing) term, as a heatmap over 0% to 100%.
    ///
    /// Red means the noise here is from paths that failed the reconnection test
    /// and got shaded with no reuse at all. Those pixels cannot be improved by
    /// tuning temporal or spatial reuse.
    ///
    /// Measured as the share of total variance *not* explained by the resampled
    /// term, so the residual also absorbs the covariance between the two terms.
    /// Read it as "how much of the flicker ReSTIR does not account for".
    NoiseNonResampledShare = 2,

    /// Like [`Self::NoiseRelativeStdDev`], but for the resampled (ReSTIR) term
    /// alone, ignoring the bypassing term and emissive surfaces.
    ///
    /// This is the noise that temporal and spatial reuse are responsible for, so
    /// it is the map to watch when tuning the confidence weight cap, the reuse
    /// radius, or the jacobian clamp. Compare against
    /// [`Self::NoiseRelativeStdDev`]: where the two agree, ReSTIR owns the noise;
    /// where the total is much worse, the bypassing term does.
    NoiseResampledStdDev = 15,

    /// Fraction of this pixel's *energy* that bypassed ReSTIR entirely, as a
    /// heatmap over 0% to 100%.
    NonResampledShare = 3,

    /// Show only the non-resampled term, i.e. raw 1-spp path tracing.
    NonResampledOnly = 4,

    /// Show only the resampled (ReSTIR) term.
    ResampledOnly = 5,

    /// Which estimator produced the winning sample, as flat colours:
    /// grey none, yellow direct light (NEE), pink directly-visible emissive,
    /// green reconnected NEE, orange reconnected emissive, blue world cache.
    SampleProvenance = 6,

    /// Effective temporal history length, as a heatmap over zero to
    /// [`SolariLighting::confidence_weight_cap`].
    ///
    /// Dark means history-starved: disocclusions, screen edges, and fast motion.
    ConfidenceWeight = 7,

    /// Why temporal reuse was rejected, as flat colours: black accepted,
    /// blue reprojected off-screen, red surface mismatch, magenta light
    /// despawned, grey no history yet.
    TemporalRejectReason = 8,

    /// How many of the five spatial neighbour candidates were rejected, as a
    /// heatmap over zero to five. Full red means no neighbour was found at all
    /// and the pixel got no spatial reuse.
    SpatialReuseFailure = 9,

    /// Where a reuse jacobian fell outside the merge's `0.125..8` clamp and the
    /// sample was discarded, as flat colours: green temporal, blue spatial,
    /// red both.
    JacobianRejection = 10,

    /// `log10` of the unbiased contribution weight, as a heatmap over 1 to 1e4.
    /// Finds the outlier weights that show up as fireflies.
    ContributionWeight = 11,

    /// Temporal sample count of the world cache cells this pixel queried, as a
    /// heatmap over zero to [`SolariLighting::world_cache_max_temporal_samples`].
    /// Dark means the pixel is reading cells that have not converged.
    WorldCacheSampleCount = 12,

    /// Red where a world cache query ran out of hash probe steps and silently
    /// returned black, losing energy.
    WorldCacheProbeFailure = 13,

    /// The world cache radiance at the primary hit, ignoring direct lighting.
    WorldCache = 14,

    /// How many frames the pixel has carried its current sample, as a heatmap over
    /// zero to 32 frames.
    ///
    /// This is the length of the sample's temporal correlation. Reuse lowers
    /// variance by sharing samples, but consecutive frames then stop being
    /// independent estimates, and a denoiser can only remove noise that is roughly
    /// independent. Red means this pixel has not been independently resampled in a
    /// long time, so its remaining error reads as structure and survives filtering.
    SampleAge = 16,

    /// Fraction of the pixels within a 9-pixel radius carrying the very same
    /// sample, as a heatmap saturating at 25%.
    ///
    /// The spatial counterpart to [`Self::SampleAge`]. Red means reuse has
    /// concentrated one sample across a neighbourhood, which is indistinguishable
    /// from real shading detail and so denoises into a blotch rather than averaging
    /// away. Expect the worst values exactly where valid samples are scarce: sharp
    /// specular and geometric edges.
    ///
    /// Measured over a radius rather than the immediate neighbours because
    /// permutation sampling reshuffles history within 4x4 tiles, which hides
    /// duplication at that scale without reducing it.
    SampleDuplication = 17,
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self {
            confidence_weight_cap: 8.0,
            specular_confidence_weight_cap: 8.0,
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
            debug_view: SolariDebugView::None,
            debug_counters: false,
        }
    }
}

impl SolariDebugView {
    /// Every view, for building a menu or key mapping over them.
    pub const ALL: [Self; 18] = [
        Self::None,
        Self::NoiseRelativeStdDev,
        Self::NoiseResampledStdDev,
        Self::NoiseNonResampledShare,
        Self::NonResampledShare,
        Self::NonResampledOnly,
        Self::ResampledOnly,
        Self::SampleProvenance,
        Self::SampleAge,
        Self::SampleDuplication,
        Self::ConfidenceWeight,
        Self::TemporalRejectReason,
        Self::SpatialReuseFailure,
        Self::JacobianRejection,
        Self::ContributionWeight,
        Self::WorldCacheSampleCount,
        Self::WorldCacheProbeFailure,
        Self::WorldCache,
    ];

    /// Whether the per-pixel debug buffers need to be allocated for this view.
    pub fn needs_debug_resources(self) -> bool {
        self != Self::None
    }

    /// A short human-readable name, for on-screen overlays.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::NoiseRelativeStdDev => "Noise: total",
            Self::NoiseResampledStdDev => "Noise: resampled only",
            Self::NoiseNonResampledShare => "Noise share: non-resampled",
            Self::NonResampledShare => "Energy share: non-resampled",
            Self::NonResampledOnly => "Non-resampled term only",
            Self::ResampledOnly => "Resampled (ReSTIR) term only",
            Self::SampleProvenance => "Sample provenance",
            Self::SampleAge => "Sample age (temporal correlation)",
            Self::SampleDuplication => "Sample duplication (spatial correlation)",
            Self::ConfidenceWeight => "Confidence weight",
            Self::TemporalRejectReason => "Temporal reject reason",
            Self::SpatialReuseFailure => "Spatial reuse failure",
            Self::JacobianRejection => "Jacobian rejection",
            Self::ContributionWeight => "Contribution weight (fireflies)",
            Self::WorldCacheSampleCount => "World cache sample count",
            Self::WorldCacheProbeFailure => "World cache probe failure",
            Self::WorldCache => "World cache radiance",
        }
    }
}
