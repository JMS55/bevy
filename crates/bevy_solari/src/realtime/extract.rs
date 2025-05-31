use super::{prepare::SolariLightingResources, SolariLighting};
use bevy_ecs::system::{Commands, Query};
use bevy_render::{camera::Camera, sync_world::RenderEntity, Extract};

pub fn extract_solari_lighting(
    cameras_3d: Extract<Query<(RenderEntity, &Camera, Option<&SolariLighting>)>>,
    mut commands: Commands,
) {
    for (entity, camera, solari_lighting) in &cameras_3d {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if solari_lighting.is_some() && camera.is_active {
            entity_commands.insert(solari_lighting.unwrap().clone());
        } else {
            entity_commands.remove::<(SolariLighting, SolariLightingResources)>();
        }
    }
}
