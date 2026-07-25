#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]

//! Provides raytraced lighting.
//!
//! See [`SolariPlugins`] for more info.
//!
//! ![`bevy_solari` logo](https://raw.githubusercontent.com/bevyengine/bevy/refs/heads/main/assets/branding/bevy_solari.svg)

extern crate alloc;

pub mod pathtracer;
pub mod realtime;
pub mod scene;

/// The solari prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use super::SolariPlugins;
    pub use crate::realtime::SolariLighting;
    pub use crate::scene::RaytracingMesh3d;
}

use crate::realtime::SolariLightingPlugin;
use crate::scene::{tlas_build, RaytracingScenePlugin};
use bevy_app::{PluginGroup, PluginGroupBuilder};
use bevy_render::{renderer::RenderDevice, settings::WgpuFeatures};
use tracing::warn;

/// An experimental set of plugins for raytraced lighting.
///
/// This plugin group provides:
/// * [`SolariLightingPlugin`] - Raytraced direct and indirect lighting.
/// * [`RaytracingScenePlugin`] - BLAS building, resource and lighting binding.
///
/// There's also:
/// * [`pathtracer::PathtracingPlugin`] - A non-realtime pathtracer for validation purposes (not added by default).
///
/// To get started, add this plugin to your app, and then add `RaytracingMesh3d` and `MeshMaterial3d::<StandardMaterial>` to your entities.
pub struct SolariPlugins;

impl PluginGroup for SolariPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(RaytracingScenePlugin)
            .add(SolariLightingPlugin)
    }
}

impl SolariPlugins {
    /// [`WgpuFeatures`] required for these plugins to function.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
    }

    /// Whether this device can run Solari at all, warning about the first thing it lacks if not.
    ///
    /// Every Solari plugin gates on all of this rather than each on the part it uses directly, so
    /// that they all load or all decline together. They aren't independent: [`SolariLightingPlugin`]
    /// and [`PathtracingPlugin`] both read [`RaytracingSceneBindings`], which exists only if
    /// [`RaytracingScenePlugin`] loaded. One loading while another declined isn't a degraded mode,
    /// it's a missing resource panic at [`RenderStartup`].
    ///
    /// That case is reachable: Metal exposes every feature in [`Self::required_wgpu_features`] on
    /// raytracing-capable hardware, and has no TLAS build path.
    ///
    /// [`PathtracingPlugin`]: crate::pathtracer::PathtracingPlugin
    /// [`RaytracingSceneBindings`]: crate::scene::RaytracingSceneBindings
    /// [`RenderStartup`]: bevy_render::RenderStartup
    pub(crate) fn supported(render_device: &RenderDevice, plugin: &str) -> bool {
        let features = render_device.features();
        if !features.contains(Self::required_wgpu_features()) {
            warn!(
                "{plugin} not loaded. GPU lacks support for required features: {:?}.",
                Self::required_wgpu_features().difference(features)
            );
            return false;
        }

        // The TLAS is built through `wgpu_hal`, which means knowing the backend's instance
        // descriptor layout. There is no portable fallback: `wgpu-core`'s own build costs more CPU
        // per frame than everything else Solari does put together at scene scale.
        if !tlas_build::supported(render_device) {
            warn!(
                "{plugin} not loaded. No TLAS build path for this backend; Solari supports Vulkan \
                 and DX12."
            );
            return false;
        }

        true
    }
}
