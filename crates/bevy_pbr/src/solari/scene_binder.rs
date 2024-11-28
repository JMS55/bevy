use super::{asset_binder::AssetBindings, blas::BlasManager};
use crate::{MeshMaterial3d, StandardMaterial};
use bevy_asset::{AssetId, Assets};
use bevy_color::ColorToComponents;
use bevy_ecs::{
    system::{Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::{Mat4, Vec4};
use bevy_render::{
    mesh::{Mesh, Mesh3d},
    render_resource::{binding_types::storage_buffer_read_only, *},
    renderer::{RenderDevice, RenderQueue},
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::HashMap;
use std::ops::DerefMut;

#[derive(Resource)]
pub struct SceneBindings {
    extracted_instances: Vec<(AssetId<Mesh>, AssetId<StandardMaterial>, GlobalTransform)>,
    extracted_materials: Vec<(AssetId<StandardMaterial>, StandardMaterial)>,
    instance_data: StorageBuffer<Vec<InstanceData>>,
    materal_data: StorageBuffer<Vec<MaterialData>>,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: Option<BindGroup>,
}

impl FromWorld for SceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self {
            extracted_instances: Vec::new(),
            extracted_materials: Vec::new(),
            instance_data: StorageBuffer::default(),
            materal_data: StorageBuffer::default(),
            bind_group_layout: render_device.create_bind_group_layout(
                "solari_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        BindingType::AccelerationStructure,
                        storage_buffer_read_only::<InstanceData>(false),
                        storage_buffer_read_only::<MaterialData>(false),
                    ),
                ),
            ),
            bind_group: None,
        }
    }
}

pub fn extract_scene(
    instances: Extract<Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>, &GlobalTransform)>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    mut scene_bindings: ResMut<SceneBindings>,
) {
    scene_bindings.extracted_instances.clear();
    scene_bindings.extracted_materials.clear();

    for (mesh_3d, mesh_material_3d, global_transform) in &instances {
        scene_bindings.extracted_instances.push((
            mesh_3d.id(),
            mesh_material_3d.id(),
            global_transform.clone(),
        ));
    }

    for (material_id, material) in materials.iter() {
        scene_bindings
            .extracted_materials
            .push((material_id, material.clone()));
    }
}

pub fn prepare_scene_bindings(
    mut scene_bindings: ResMut<SceneBindings>,
    asset_bindings: Res<AssetBindings>,
    blas_manager: Res<BlasManager>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let scene_bindings = scene_bindings.deref_mut();
    if scene_bindings.extracted_instances.is_empty() {
        return;
    }

    scene_bindings.instance_data.get_mut().clear();
    scene_bindings.materal_data.get_mut().clear();

    // Create TLAS
    let mut tlas = TlasPackage::new(render_device.wgpu_device().create_tlas(
        &CreateTlasDescriptor {
            label: Some("tlas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: scene_bindings.extracted_instances.len() as u32,
        },
    ));

    // Write material data
    let mut material_indices = HashMap::new();
    for (material_id, material) in scene_bindings.extracted_materials.drain(..) {
        material_indices.insert(material_id, scene_bindings.materal_data.get().len() as u32);

        scene_bindings.materal_data.get_mut().push(MaterialData {
            base_color: material.base_color.to_linear().to_vec4(),
            emissive: material.emissive.to_vec4(),
            base_color_texture: asset_bindings.get_image_index(material.base_color_texture),
            emissive_texture: asset_bindings.get_image_index(material.emissive_texture),
            normal_map_texture: asset_bindings.get_image_index(material.normal_map_texture),
            _padding: 0,
        });
    }

    // Write instance data and TLAS entry for each instance
    let mut instance_id = 0;
    for (mesh_id, material_id, transform) in &scene_bindings.extracted_instances {
        if let (
            Some([vertex_buffer, vertex_offset, index_buffer, index_offset]),
            Some(blas),
            Some(material),
        ) = (
            asset_bindings.mesh_indices.get(mesh_id).copied(),
            blas_manager.get(mesh_id),
            material_indices.get(material_id).copied(),
        ) {
            let transform = transform.compute_matrix();

            scene_bindings.instance_data.get_mut().push(InstanceData {
                vertex_buffer,
                vertex_offset,
                index_buffer,
                index_offset,
                material,
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
                transform,
            });

            *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
                blas,
                tlas_transform(&transform),
                instance_id as u32,
                0xFF,
            ));
            instance_id += 1;
        }
    }

    // Upload GPU buffers
    scene_bindings
        .instance_data
        .write_buffer(&render_device, &render_queue);
    scene_bindings
        .materal_data
        .write_buffer(&render_device, &render_queue);

    // Build the TLAS
    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_tlas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&[], [&tlas]);
    render_queue.submit([command_encoder.finish()]);

    // Create the bind group
    scene_bindings.bind_group = Some(render_device.create_bind_group(
        "solari_scene_bind_group",
        &scene_bindings.bind_group_layout,
        &BindGroupEntries::sequential((
            tlas.as_binding(),
            scene_bindings.instance_data.binding().unwrap(),
            scene_bindings.materal_data.binding().unwrap(),
        )),
    ));
}

#[derive(ShaderType)]
#[repr(C)]
struct InstanceData {
    vertex_buffer: u32,
    vertex_offset: u32,
    index_buffer: u32,
    index_offset: u32,
    material: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    transform: Mat4,
}

#[derive(ShaderType)]
#[repr(C)]
struct MaterialData {
    base_color: Vec4,
    emissive: Vec4,
    base_color_texture: u32,
    emissive_texture: u32,
    normal_map_texture: u32,
    _padding: u32,
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
