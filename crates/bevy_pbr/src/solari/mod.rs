mod asset_binder;
mod blas_manager;
mod extract_asset_events;
mod gpu_types;
mod path_tracer;
mod scene_binder;

use self::{
    asset_binder::{prepare_asset_binding_arrays, AssetBindings},
    blas_manager::{prepare_new_blas, BlasManager},
    extract_asset_events::{
        extract_asset_events, ExtractAssetEventsSystemState, ExtractedAssetEvents,
    },
    graph::LabelsSolari,
    path_tracer::{prepare_path_tracer_accumulation_texture, PathTracerNode},
    scene_binder::{extract_scene, prepare_scene_bindings, ExtractedScene, SceneBindings},
};
use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::{Labels3d, SubGraph3d};
use bevy_ecs::{
    component::Component,
    schedule::{common_conditions::any_with_component, IntoSystemConfigs},
    system::Resource,
};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    mesh::Mesh,
    render_asset::prepare_assets,
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::Shader,
    renderer::RenderDevice,
    settings::WgpuFeatures,
    texture::Image,
    view::Msaa,
    ExtractSchedule, Render, RenderApp, RenderSet,
};

pub mod graph {
    use bevy_render::render_graph::RenderLabel;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub enum LabelsSolari {
        PathTracer,
    }
}

const SOLARI_BINDINGS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1717171717171717);
const SOLARI_PATH_TRACER_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(2717171717171717);

/// TODO: Docs
pub struct SolariPlugin;

impl Plugin for SolariPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Msaa::Off);

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
        match app.world.get_resource::<RenderDevice>() {
            Some(render_device) if render_device.features().contains(Self::required_features()) => {
            }
            _ => return,
        }

        app.insert_resource(SolariSupported)
            .init_resource::<ExtractAssetEventsSystemState>()
            .add_plugins(ExtractComponentPlugin::<SolariSettings>::default());

        let render_app = app.get_sub_app_mut(RenderApp).unwrap();
        render_app
            .init_resource::<ExtractedAssetEvents>()
            .init_resource::<ExtractedScene>()
            .init_resource::<BlasManager>()
            .init_resource::<AssetBindings>()
            .init_resource::<SceneBindings>()
            .add_systems(
                ExtractSchedule,
                (extract_asset_events, extract_scene).run_if(any_with_component::<SolariSettings>),
            )
            .add_systems(
                Render,
                (
                    prepare_new_blas
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Mesh>),
                    prepare_asset_binding_arrays
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Mesh>)
                        .after(prepare_assets::<Image>),
                    prepare_path_tracer_accumulation_texture.in_set(RenderSet::PrepareResources),
                    prepare_scene_bindings.in_set(RenderSet::PrepareBindGroups),
                )
                    .run_if(any_with_component::<SolariSettings>),
            )
            .add_render_graph_node::<ViewNodeRunner<PathTracerNode>>(
                SubGraph3d,
                LabelsSolari::PathTracer,
            )
            .add_render_graph_edges(
                SubGraph3d,
                (LabelsSolari::PathTracer, Labels3d::EndMainPass),
            );
    }
}

impl SolariPlugin {
    /// TODO: Docs
    pub fn required_features() -> WgpuFeatures {
        WgpuFeatures::RAY_TRACING_ACCELERATION_STRUCTURE
            | WgpuFeatures::RAY_QUERY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::STORAGE_RESOURCE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | WgpuFeatures::PUSH_CONSTANTS
    }
}

/// TODO: Docs
#[derive(Resource)]
pub struct SolariSupported;

/// TODO: Docs
// Requires MSAA off, HDR, CameraMainTextureUsages::with_storage_binding(), and should disable shadows for all lights
#[derive(Component, ExtractComponent, Clone)]
pub struct SolariSettings {
    pub debug_path_tracer: bool,
}

impl Default for SolariSettings {
    fn default() -> Self {
        Self {
            debug_path_tracer: false,
        }
    }
}
