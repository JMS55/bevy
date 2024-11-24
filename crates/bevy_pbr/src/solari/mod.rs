mod blas;

use self::blas::{update_blas, BlasManager};
use bevy_app::{App, Plugin};
use bevy_ecs::{
    component::Component, prelude::resource_exists, schedule::IntoSystemConfigs, system::Resource,
};
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    mesh::{allocator::allocate_and_free_meshes, RenderMesh},
    render_asset::prepare_assets,
    renderer::RenderDevice,
    settings::WgpuFeatures,
    Render, RenderApp, RenderSet,
};

pub struct SolariPlugin;

impl Plugin for SolariPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SolariEnabled>::default());
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

        render_app.init_resource::<BlasManager>().add_systems(
            Render,
            update_blas
                .in_set(RenderSet::PrepareAssets)
                .after(allocate_and_free_meshes)
                .before(prepare_assets::<RenderMesh>)
                .run_if(resource_exists::<SolariEnabled>),
        );

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

#[derive(Component)]
pub struct Solari {}
