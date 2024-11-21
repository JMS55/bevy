use bevy_app::{App, Plugin};
use bevy_ecs::{component::Component, system::Resource};
use bevy_render::{renderer::RenderDevice, settings::WgpuFeatures};

pub struct SolariPlugin;

impl Plugin for SolariPlugin {
    fn build(&self, app: &mut App) {}

    fn finish(&self, app: &mut App) {
        match app.world().get_resource::<RenderDevice>() {
            Some(render_device) if render_device.features().contains(Self::required_features()) => {
            }
            _ => return,
        }

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

#[derive(Component)]
pub struct Solari {}
