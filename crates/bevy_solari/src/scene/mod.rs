mod binder;
mod blas;
mod extract;
mod tlas_build;
mod types;

use bevy_asset::embedded_asset;
use bevy_shader::load_shader_library;
pub use binder::RaytracingSceneBindings;
pub use types::RaytracingMesh3d;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    mesh::{
        allocator::{allocate_and_free_meshes, MeshAllocatorSettings},
        RenderMesh,
    },
    render_asset::prepare_assets,
    render_resource::{update_sparse_buffers, BufferUsages},
    renderer::{RenderDevice, RenderGraph, RenderGraphSystems},
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};
use binder::{
    build_raytracing_tlas, pack_raytracing_tlas_instances, prepare_raytracing_scene_bind_group,
    prepare_raytracing_scene_resources, TlasInstancePackPipeline,
};
use blas::{compact_raytracing_blas, prepare_dummy_blas, prepare_raytracing_blas, BlasManager};
use extract::{
    extract_raytracing_material_assets, extract_raytracing_scene_meshes_and_materials,
    extract_raytracing_scene_structural, extract_raytracing_scene_transforms,
    StandardMaterialAssets,
};
use tracing::warn;

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "brdf.wgsl");
        load_shader_library!(app, "raytracing_scene_bindings.wgsl");
        load_shader_library!(app, "sampling.wgsl");
        embedded_asset!(app, "tlas_instances.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "RaytracingScenePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        // The TLAS is built through `wgpu_hal`, which means knowing the backend's instance
        // descriptor layout. There is no portable fallback: `wgpu-core`'s own build costs more CPU
        // per frame than everything else Solari does put together at scene scale.
        if tlas_build::instance_layout(render_device).is_none() {
            warn!(
                "RaytracingScenePlugin not loaded. No TLAS build path for this backend; Solari \
                 supports Vulkan, DX12 and Metal."
            );
            return;
        }

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;

        render_app
            .init_gpu_resource::<BlasManager>()
            .init_gpu_resource::<StandardMaterialAssets>()
            .init_gpu_resource::<TlasInstancePackPipeline>()
            .init_gpu_resource::<RaytracingSceneBindings>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_raytracing_scene_structural,
                    extract_raytracing_scene_transforms,
                    extract_raytracing_scene_meshes_and_materials,
                    extract_raytracing_material_assets,
                ),
            )
            .add_systems(
                Render,
                (
                    // Dead instance slots point at this on backends where a null reference isn't
                    // legal, so it has to exist before any instance resolves.
                    prepare_dummy_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_raytracing_blas),
                    prepare_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    compact_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_raytracing_blas),
                    prepare_raytracing_scene_resources.in_set(RenderSystems::PrepareResources),
                    prepare_raytracing_scene_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                // The TLAS has to be built before any pass traces against it, and its instance
                // descriptors packed before that — which in turn needs this frame's transforms and
                // acceleration structure addresses to have reached the GPU.
                (pack_raytracing_tlas_instances, build_raytracing_tlas)
                    .chain()
                    .after(update_sparse_buffers)
                    .in_set(RenderGraphSystems::Begin),
            );
    }
}
