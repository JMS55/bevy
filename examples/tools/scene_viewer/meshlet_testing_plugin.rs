//! Allows converting a loaded scene to meshlet meshes for developers working on testing meshlets.
//!
//! Not meant for general usage.

use crate::scene_viewer_plugin::SceneHandle;
use bevy::{
    app::{App, Plugin, Update},
    asset::{Asset, Assets},
    input::ButtonInput,
    log::info,
    pbr::{
        experimental::meshlet::{
            MeshletMesh, MeshletMesh3d, MeshletPlugin, DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR,
        },
        Material, MaterialPlugin, MeshMaterial3d, StandardMaterial,
    },
    prelude::{Commands, Entity, KeyCode, Local, Mesh, Mesh3d, Msaa, Query, Res, ResMut},
    reflect::TypePath,
    render::render_resource::AsBindGroup,
};

pub struct MeshletTestingPlugin;

impl Plugin for MeshletTestingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MeshletPlugin {
                cluster_buffer_slots: 8192, // Adjust for your scene as needed
            },
            MaterialPlugin::<MeshletDebugMaterial>::default(),
        ))
        .add_systems(Update, (swap_meshes_to_meshlet_meshes, remove_camera_msaa));
    }
}

fn swap_meshes_to_meshlet_meshes(
    key_input: Res<ButtonInput<KeyCode>>,
    scene_handle: Res<SceneHandle>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut meshlet_meshes: ResMut<Assets<MeshletMesh>>,
    mut commands: Commands,
    entities: Query<(Entity, &Mesh3d)>,
    mut meshlet_debug_materials: ResMut<Assets<MeshletDebugMaterial>>,
    mut has_ran: Local<bool>,
) {
    if !(key_input.pressed(KeyCode::BracketRight) && scene_handle.is_loaded && !*has_ran) {
        return;
    }
    *has_ran = true;

    info!("Starting mesh to meshlet mesh conversion");

    let meshlet_debug_material =
        MeshMaterial3d(meshlet_debug_materials.add(MeshletDebugMaterial::default()));

    for (mesh_id, mesh) in meshes.iter_mut() {
        mesh.remove_attribute(Mesh::ATTRIBUTE_TANGENT.id);

        let Ok(meshlet_mesh) =
            MeshletMesh::from_mesh(mesh, DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR)
        else {
            info!("glTF mesh was not valid to convert to MeshletMesh, skipping");
            continue;
        };

        let meshlet_mesh = MeshletMesh3d(meshlet_meshes.add(meshlet_mesh));

        for (entity, mesh) in &entities {
            if mesh.id() == mesh_id {
                commands
                    .entity(entity)
                    .insert((meshlet_mesh.clone(), meshlet_debug_material.clone()))
                    .remove::<(Mesh3d, MeshMaterial3d<StandardMaterial>)>();
            }
        }
    }

    info!("Finished converting meshes to meshlet meshes");
}

fn remove_camera_msaa(mut commands: Commands, cameras: Query<(Entity, &Msaa)>) {
    for (camera, msaa) in &cameras {
        if *msaa != Msaa::Off {
            commands.entity(camera).insert(Msaa::Off);
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct MeshletDebugMaterial {
    _dummy: (),
}
impl Material for MeshletDebugMaterial {}
