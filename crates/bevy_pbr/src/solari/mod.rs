mod asset_binder;
mod blas;
mod path_tracer;
mod scene_binder;
mod util;

use self::asset_binder::{copy_extracted_image_ids, prepare_asset_binding_arrays, AssetBindings};
use self::blas::{update_blas, BlasManager};
use self::path_tracer::{prepare_path_tracer_accumulation_texture, PathTracerNode};
use self::scene_binder::{extract_scene, prepare_scene_bindings, SceneBindings};
use crate::DefaultOpaqueRendererMethod;
use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{
    component::Component, prelude::resource_exists, schedule::IntoSystemConfigs, system::Resource,
};
use bevy_render::render_graph::{RenderGraphApp, ViewNodeRunner};
use bevy_render::render_resource::Shader;
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    mesh::{allocator::allocate_and_free_meshes, RenderMesh},
    render_asset::prepare_assets,
    renderer::RenderDevice,
    settings::WgpuFeatures,
    texture::GpuImage,
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use graph::NodeSolari;

pub mod graph {
    use bevy_render::render_graph::RenderLabel;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub enum NodeSolari {
        PathTracer,
    }
}

const SOLARI_BINDINGS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0717171717171717);
const SOLARI_PATH_TRACER_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1717171717171717);

pub struct SolariPlugin;

impl Plugin for SolariPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<SolariEnabled>::default(),
            ExtractComponentPlugin::<Solari>::default(),
        ))
        .insert_resource(DefaultOpaqueRendererMethod::deferred());

        load_internal_asset!(
            app,
            SOLARI_BINDINGS_SHADER_HANDLE,
            "solari_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SOLARI_PATH_TRACER_SHADER_HANDLE,
            "path_tracer.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        match render_app.world().get_resource::<RenderDevice>() {
            Some(render_device) if render_device.features().contains(Self::required_features()) => {
            }
            _ => return,
        }

        render_app
            .init_resource::<AssetBindings>()
            .init_resource::<SceneBindings>()
            .init_resource::<BlasManager>()
            .add_systems(
                ExtractSchedule,
                extract_scene.run_if(resource_exists::<SolariEnabled>),
            )
            .add_systems(
                Render,
                (
                    copy_extracted_image_ids
                        .in_set(RenderSet::PrepareAssets)
                        .before(prepare_assets::<GpuImage>),
                    prepare_asset_binding_arrays
                        .in_set(RenderSet::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(prepare_assets::<GpuImage>)
                        .after(allocate_and_free_meshes)
                        .after(copy_extracted_image_ids),
                    update_blas
                        .in_set(RenderSet::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    prepare_path_tracer_accumulation_texture.in_set(RenderSet::PrepareResources),
                    prepare_scene_bindings.in_set(RenderSet::PrepareBindGroups),
                )
                    .run_if(resource_exists::<SolariEnabled>),
            )
            .add_render_graph_node::<ViewNodeRunner<PathTracerNode>>(Core3d, NodeSolari::PathTracer)
            .add_render_graph_edges(Core3d, (NodeSolari::PathTracer, Node3d::EndMainPass));

        app.insert_resource(SolariSupported);
    }
}

impl SolariPlugin {
    pub fn required_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::STORAGE_RESOURCE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | WgpuFeatures::PUSH_CONSTANTS
    }
}

#[derive(Resource)]
pub struct SolariSupported;

#[derive(Resource, ExtractResource, Clone)]
pub struct SolariEnabled;

#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct Solari {
    pub debug_path_tracer: bool,
}
