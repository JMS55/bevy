mod extract;
mod node;
mod prepare;

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
            .add_systems(RenderStartup, init_solari_lighting_pipelines)
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

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,

    /// Give the denoiser the depth and motion of the surface seen *through* a mirror, rather than
    /// of the mirror itself.
    ///
    /// This is what every NVIDIA reference integration does under primary surface replacement, and
    /// it lets a reflection reproject at its true optical distance. Turning it off feeds the real
    /// mirror surface's depth and motion instead, which is the older Bevy behaviour.
    ///
    /// Only affects DLSS Ray Reconstruction. Exposed for A/B comparison; there is no known reason to
    /// prefer the old behaviour.
    pub psr_virtual_depth: bool,

    /// Place a mirror's virtual image by unfolding the reflection chain along the camera ray, rather
    /// than by reflecting the hit position about the first mirror's plane.
    ///
    /// The two are provably identical for a single planar mirror. They diverge once light bounces
    /// between two mirrors, because later mirrors must reflect about their own plane -- the older
    /// construction folds everything about the first one.
    ///
    /// Only affects DLSS Ray Reconstruction. Exposed for A/B comparison.
    pub psr_unfold_along_camera_ray: bool,

    /// Decline to replace the primary surface where it is visibly curving.
    ///
    /// Replacement assumes a flat reflector. On a curved one the virtual image is placed too far
    /// away, and near a silhouette it swings hard between neighbouring pixels — the denoiser reads
    /// that as a shattered depth field and blurs rather than reprojecting. Skipping those pixels
    /// falls back to the mirror's own surface, which is merely unhelpful instead of harmful.
    ///
    /// This is containment, not a fix: the real answer is to correct the virtual distance for
    /// curvature. Only affects DLSS Ray Reconstruction.
    pub psr_skip_curved_reflectors: bool,

    /// Decide what counts as a mirror by where the surface's energy goes, rather than by whether it
    /// is a metal.
    ///
    /// A smooth surface acts as a mirror because its specular lobe carries the reflectance, which a
    /// dielectric can do too — a black polished one almost entirely, a coloured one only a few
    /// percent. Turning this off restores the old all-or-nothing `metallic` test, under which no
    /// dielectric was ever replaced.
    ///
    /// Only affects DLSS Ray Reconstruction.
    pub psr_dielectric: bool,

    /// Multiply the albedo guides by the reflectance accumulated along the mirror chain.
    ///
    /// The guide buffers are meant to hold the pixel's reflectance, and a reflection seen in a gold
    /// mirror really is gold. Without this the denoiser is told the reflected surface's own colour,
    /// untinted, which biases it wherever a mirror is not neutral.
    ///
    /// Only affects DLSS Ray Reconstruction.
    pub psr_tint_albedo: bool,

    /// Describe how the reflection moves on surfaces that are polished but not mirrors.
    ///
    /// A glossy surface is never replaced — its own normal, roughness, albedo, depth and motion stay
    /// exactly as they are. The only thing it gains is a specular motion vector saying that what it
    /// reflects moves differently from the surface itself, which is otherwise the one thing the
    /// denoiser is never told and the reason reflections smear across polished floors.
    ///
    /// Turning this on also stops such a surface from walking a mirror chain (one ray along the lobe
    /// centre is all a widened lobe can support), exempts it from the curvature refusal (it writes no
    /// virtual depth, so there is no depth field to shatter), and gives it a motion vector for a
    /// reflection that hits nothing rather than dropping it.
    ///
    /// Only affects DLSS Ray Reconstruction.
    pub psr_glossy: bool,

    /// Replace the image with a false-colour map of how each pixel was classified for primary surface
    /// replacement.
    ///
    /// - dark grey — no delta lobe, so replacement was never considered. A glossy or rough surface.
    /// - green — replaced outright.
    /// - blue — blended: it has a delta lobe but meaningful non-delta energy too.
    /// - yellow — has a delta lobe, refused for being too curved.
    /// - red — has a delta lobe, but the reflection chain never landed on a non-mirror surface.
    /// - purple — glossy, and its reflection ray found nothing, so it was placed at infinity.
    ///
    /// Telling "not eligible" apart from "eligible but refused" is the point. Those look identical in
    /// the final image and have completely different causes.
    pub psr_debug_overlay: bool,



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
            psr_virtual_depth: true,
            psr_unfold_along_camera_ray: true,
            psr_skip_curved_reflectors: true,
            psr_dielectric: true,
            psr_tint_albedo: true,
            psr_glossy: true,
            psr_debug_overlay: false,
        }
    }
}
