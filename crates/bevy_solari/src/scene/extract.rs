use super::RaytracingMesh3d;
use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    lifecycle::RemovedComponents,
    message::MessageReader,
    query::{Changed, Or},
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;
use bevy_utils::Parallel;
use core::ops::Deref;

pub struct ExtractedRaytracingInstance {
    pub mesh: RaytracingMesh3d,
    pub material: MeshMaterial3d<StandardMaterial>,
    pub transform: GlobalTransform,
    pub previous_transform: Option<PreviousGlobalTransform>,
}

#[derive(Resource, Default)]
pub struct ExtractedRaytracingScene {
    pub instances: EntityHashMap<ExtractedRaytracingInstance>,
    pub changed_instances: Vec<Entity>,
    pub topology_revision: u64,
    main_to_render_entity: EntityHashMap<Entity>,
}

#[derive(Resource, Default)]
pub struct RaytracingSceneExtractionQueues(
    Parallel<Vec<(Entity, Entity, ExtractedRaytracingInstance)>>,
);

pub fn extract_raytracing_scene(
    mut extracted_scene: ResMut<ExtractedRaytracingScene>,
    mut extraction_queues: ResMut<RaytracingSceneExtractionQueues>,
    instances: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &RaytracingMesh3d,
                &MeshMaterial3d<StandardMaterial>,
                &GlobalTransform,
                Option<&PreviousGlobalTransform>,
            ),
            Or<(
                Changed<RaytracingMesh3d>,
                Changed<MeshMaterial3d<StandardMaterial>>,
                Changed<GlobalTransform>,
                Changed<PreviousGlobalTransform>,
            )>,
        >,
    >,
    all_instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
        )>,
    >,
    (mut removed_raytracing_meshes, mut removed_materials, mut removed_transforms): (
        Extract<RemovedComponents<RaytracingMesh3d>>,
        Extract<RemovedComponents<MeshMaterial3d<StandardMaterial>>>,
        Extract<RemovedComponents<GlobalTransform>>,
    ),
    mut removed_previous_transforms: Extract<RemovedComponents<PreviousGlobalTransform>>,
    render_entities: Extract<Query<RenderEntity>>,
) {
    extracted_scene.changed_instances.clear();

    instances.par_iter().for_each_init(
        || extraction_queues.0.borrow_local_mut(),
        |queue, (main_entity, render_entity, mesh, material, transform, previous_transform)| {
            queue.push((
                main_entity,
                render_entity,
                ExtractedRaytracingInstance {
                    mesh: mesh.clone(),
                    material: material.clone(),
                    transform: *transform,
                    previous_transform: previous_transform.cloned(),
                },
            ));
        },
    );

    for (main_entity, render_entity, instance) in extraction_queues.0.drain() {
        update_extracted_instance(&mut extracted_scene, main_entity, render_entity, instance);
    }

    for main_entity in removed_raytracing_meshes
        .read()
        .chain(removed_materials.read())
        .chain(removed_transforms.read())
    {
        // If all required components are present, this component was removed
        // and re-added in the same frame and the changed query above wins.
        if all_instances.contains(main_entity) {
            continue;
        }
        let render_entity = extracted_scene
            .main_to_render_entity
            .remove(&main_entity)
            .or_else(|| render_entities.get(main_entity).ok());
        if render_entity
            .is_some_and(|render_entity| extracted_scene.instances.remove(&render_entity).is_some())
        {
            extracted_scene.topology_revision = extracted_scene.topology_revision.wrapping_add(1);
        }
    }

    // Component removal is not observed by `Changed<PreviousGlobalTransform>`.
    for main_entity in removed_previous_transforms.read() {
        let Ok((render_entity, mesh, material, transform, previous_transform)) =
            all_instances.get(main_entity)
        else {
            continue;
        };
        update_extracted_instance(
            &mut extracted_scene,
            main_entity,
            render_entity,
            ExtractedRaytracingInstance {
                mesh: mesh.clone(),
                material: material.clone(),
                transform: *transform,
                previous_transform: previous_transform.cloned(),
            },
        );
    }
}

fn update_extracted_instance(
    scene: &mut ExtractedRaytracingScene,
    main_entity: Entity,
    render_entity: Entity,
    instance: ExtractedRaytracingInstance,
) {
    let mut topology_changed = scene.instances.get(&render_entity).is_none_or(|previous| {
        previous.mesh.id() != instance.mesh.id()
            || previous.material.id() != instance.material.id()
            || previous.previous_transform.is_some() != instance.previous_transform.is_some()
    });
    let previous_render_entity = scene
        .main_to_render_entity
        .insert(main_entity, render_entity);
    if previous_render_entity.is_some_and(|previous_render_entity| {
        previous_render_entity != render_entity
            && scene.instances.remove(&previous_render_entity).is_some()
    }) {
        topology_changed = true;
    }
    scene.instances.insert(render_entity, instance);
    scene.changed_instances.push(render_entity);
    if topology_changed {
        scene.topology_revision = scene.topology_revision.wrapping_add(1);
    }
}

#[derive(Resource, Default)]
pub struct StandardMaterialAssets {
    assets: HashMap<AssetId<StandardMaterial>, StandardMaterial>,
    initialized: bool,
}

impl Deref for StandardMaterialAssets {
    type Target = HashMap<AssetId<StandardMaterial>, StandardMaterial>;

    fn deref(&self) -> &Self::Target {
        &self.assets
    }
}

pub fn extract_standard_material_assets(
    source: Extract<Res<Assets<StandardMaterial>>>,
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
    mut extracted: ResMut<StandardMaterialAssets>,
) {
    if !extracted.initialized {
        extracted.assets.extend(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone())),
        );
        extracted.initialized = true;
    }

    for event in events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                if let Some(material) = source.get(*id) {
                    extracted.assets.insert(*id, material.clone());
                }
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                extracted.assets.remove(id);
            }
        }
    }
}
