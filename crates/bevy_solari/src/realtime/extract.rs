use super::{prepare::SolariLightingResources, SolariLighting};
use bevy_camera::Camera;
use bevy_ecs::system::{Commands, ResMut};
use bevy_pbr::deferred::SkipDeferredLighting;
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_solari_lighting(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut cameras_3d = main_world.query::<(RenderEntity, &Camera, Option<&mut SolariLighting>)>();

    for (entity, camera, solari_lighting) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(mut solari_lighting) = solari_lighting
            && camera.is_active
        {
            entity_commands.insert((solari_lighting.clone(), SkipDeferredLighting));
            // `reset` only lasts one frame. Keep it while the camera doesn't render, e.g. while its window is
            // minimized, since the render world skips the view then.
            if camera
                .physical_target_size()
                .is_some_and(|size| size.x > 0 && size.y > 0)
            {
                solari_lighting.reset = false;
            }
        } else {
            entity_commands.remove::<(
                SolariLighting,
                SolariLightingResources,
                SkipDeferredLighting,
            )>();
        }
    }
}
