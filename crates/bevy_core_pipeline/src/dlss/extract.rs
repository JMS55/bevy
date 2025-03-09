use super::Dlss;
use crate::{
    core_3d::{Camera3d, MainPassViewportOverride},
    prepass::{DepthPrepass, MotionVectorPrepass},
};
use bevy_ecs::{
    query::With,
    system::{Commands, ResMut},
};
use bevy_render::{
    camera::{Camera, Projection, TemporalJitter},
    sync_world::RenderEntity,
    MainWorld,
};

pub fn extract_dlss(mut commands: Commands, mut main_world: ResMut<MainWorld>) {
    let mut cameras_3d = main_world
        .query_filtered::<(RenderEntity, &Camera, &Projection, &mut Dlss), (
            With<Camera3d>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        )>();

    for (entity, camera, camera_projection, mut dlss) in cameras_3d.iter_mut(&mut main_world) {
        let has_perspective_projection = matches!(camera_projection, Projection::Perspective(_));
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if camera.is_active && has_perspective_projection {
            entity_commands.insert(dlss.clone());
            dlss.reset = false;
        } else {
            entity_commands.remove::<(Dlss, ViewDlssContext, MainPassViewportOverride)>();
        }
    }
}
