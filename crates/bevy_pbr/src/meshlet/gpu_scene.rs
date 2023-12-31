use super::{persistent_buffer::PersistentGpuBuffer, Meshlet, MeshletBoundingSphere, MeshletMesh};
use crate::{
    Material, MeshFlags, MeshTransforms, MeshUniform, NotShadowCaster, NotShadowReceiver,
    PreviousGlobalTransform, RenderMaterialInstances, ShadowView,
};
use bevy_asset::{AssetId, Assets, Handle, UntypedAssetId};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{AnyOf, Has},
    system::{Commands, Query, Res, ResMut, Resource, SystemState},
    world::{FromWorld, World},
};
use bevy_render::{
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, ViewDepthTexture, ViewUniform, ViewUniforms},
    MainWorld,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{default, EntityHashMap, HashMap, HashSet};
use std::{
    ops::{DerefMut, Range},
    sync::Arc,
};

pub fn extract_meshlet_meshes(
    // TODO: Replace main_world when Extract<ResMut<Assets<MeshletMesh>>> is possible
    mut main_world: ResMut<MainWorld>,
    mut gpu_scene: ResMut<MeshletGpuScene>,
) {
    let mut system_state: SystemState<(
        Query<(
            Entity,
            &Handle<MeshletMesh>,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
            Has<NotShadowReceiver>,
            Has<NotShadowCaster>,
        )>,
        ResMut<Assets<MeshletMesh>>,
    )> = SystemState::new(&mut main_world);
    let (query, mut assets) = system_state.get_mut(&mut main_world);

    gpu_scene.reset();

    // TODO: Handle not_shadow_caster
    for (
        instance_index,
        (instance, handle, transform, previous_transform, not_shadow_receiver, _not_shadow_caster),
    ) in query.iter().enumerate()
    {
        gpu_scene.queue_meshlet_mesh_upload(instance, handle, &mut assets, instance_index as u32);

        let transform = transform.affine();
        let previous_transform = previous_transform.map(|t| t.0).unwrap_or(transform);
        let mut flags = if not_shadow_receiver {
            MeshFlags::empty()
        } else {
            MeshFlags::SHADOW_RECEIVER
        };
        if transform.matrix3.determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }
        let transforms = MeshTransforms {
            transform: (&transform).into(),
            previous_transform: (&previous_transform).into(),
            flags: flags.bits(),
        };
        gpu_scene
            .instance_uniforms
            .get_mut()
            .push(MeshUniform::from(&transforms));
    }
}

pub fn perform_pending_meshlet_mesh_writes(
    mut gpu_scene: ResMut<MeshletGpuScene>,
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
) {
    gpu_scene
        .vertex_data
        .perform_writes(&render_queue, &render_device);
    gpu_scene
        .vertex_ids
        .perform_writes(&render_queue, &render_device);
    gpu_scene
        .indices
        .perform_writes(&render_queue, &render_device);
    gpu_scene
        .meshlets
        .perform_writes(&render_queue, &render_device);
    gpu_scene
        .meshlet_bounding_spheres
        .perform_writes(&render_queue, &render_device);
}

pub fn queue_material_meshlet_meshes<M: Material>(
    mut gpu_scene: ResMut<MeshletGpuScene>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
) {
    let gpu_scene = gpu_scene.deref_mut();
    for instance in &gpu_scene.instances {
        let material_asset_id = render_material_instances
            .get(instance)
            .expect("TODO")
            .untyped();

        let material_id = *gpu_scene
            .material_id_lookup
            .get(&material_asset_id)
            .expect("TODO: Will this error ever occur?");

        gpu_scene.material_ids_present_in_scene.insert(material_id);
        gpu_scene.instance_material_ids.get_mut().push(material_id);
    }
}

pub fn prepare_meshlet_per_frame_resources(
    mut gpu_scene: ResMut<MeshletGpuScene>,
    views: Query<(Entity, &ExtractedView, Has<ShadowView>)>,
    mut texture_cache: ResMut<TextureCache>,
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    gpu_scene
        .previous_thread_id_starts
        .retain(|_, (_, active)| *active);

    if gpu_scene.scene_meshlet_count == 0 {
        return;
    }

    gpu_scene
        .instance_uniforms
        .write_buffer(&render_device, &render_queue);
    gpu_scene
        .instance_material_ids
        .write_buffer(&render_device, &render_queue);
    gpu_scene
        .thread_instance_ids
        .write_buffer(&render_device, &render_queue);
    gpu_scene
        .thread_meshlet_ids
        .write_buffer(&render_device, &render_queue);
    gpu_scene
        .previous_thread_ids
        .write_buffer(&render_device, &render_queue);

    // TODO: Should draw_index_buffer be per-view, or a single resource shared between all views?
    let visibility_buffer_draw_index_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("meshlet_visibility_buffer_draw_index_buffer"),
        size: 4 * gpu_scene.scene_index_count,
        usage: BufferUsages::STORAGE | BufferUsages::INDEX,
        mapped_at_creation: false,
    });

    for (view_entity, view, is_shadow_view) in &views {
        let previous_occlusion_buffer = gpu_scene
            .previous_occlusion_buffers
            .get(&view_entity)
            .map(Buffer::clone)
            .unwrap_or_else(|| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some("meshlet_occlusion_buffer"),
                    size: 4,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            });

        let occlusion_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("meshlet_occlusion_buffer"),
            size: gpu_scene.scene_meshlet_count as u64 * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        gpu_scene
            .previous_occlusion_buffers
            .insert(view_entity, occlusion_buffer.clone());

        let visibility_buffer = TextureDescriptor {
            label: Some("meshlet_visibility_buffer"),
            size: Extent3d {
                width: view.viewport.z,
                height: view.viewport.w,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let visibility_buffer_draw_command_buffer_first =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("meshlet_visibility_buffer_draw_command_buffer_first"),
                contents: DrawIndexedIndirect {
                    vertex_count: 0,
                    instance_count: 1,
                    base_index: 0,
                    vertex_offset: 0,
                    base_instance: 0,
                }
                .as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });
        let visibility_buffer_draw_command_buffer_second =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("meshlet_visibility_buffer_draw_command_buffer_second"),
                contents: DrawIndexedIndirect {
                    vertex_count: 0,
                    instance_count: 1,
                    base_index: 0,
                    vertex_offset: 0,
                    base_instance: 0,
                }
                .as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });

        let depth_pyramid = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("meshlet_depth_pyramid"),
                size: Extent3d {
                    // Round down to nearest power of 2 to ensure depth is conservative
                    width: (1 << (31 - view.viewport.z.leading_zeros())) / 2,
                    height: (1 << (31 - view.viewport.w.leading_zeros())) / 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 6,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let depth_pyramid_mips = [0, 1, 2, 3, 4, 5].map(|i| {
            depth_pyramid.texture.create_view(&TextureViewDescriptor {
                label: Some("meshlet_depth_pyramid_texture_view"),
                format: Some(TextureFormat::R32Float),
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: i,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            })
        });

        let material_depth_color = TextureDescriptor {
            label: Some("meshlet_material_depth_color"),
            size: Extent3d {
                width: view.viewport.z,
                height: view.viewport.w,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Uint,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let material_depth = TextureDescriptor {
            label: Some("meshlet_material_depth"),
            size: Extent3d {
                width: view.viewport.z,
                height: view.viewport.w,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth16Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        commands.entity(view_entity).insert(MeshletViewResources {
            scene_meshlet_count: gpu_scene.scene_meshlet_count,
            previous_occlusion_buffer,
            occlusion_buffer,
            visibility_buffer: (!is_shadow_view)
                .then(|| texture_cache.get(&render_device, visibility_buffer)),
            visibility_buffer_draw_command_buffer_first,
            visibility_buffer_draw_command_buffer_second,
            visibility_buffer_draw_index_buffer: visibility_buffer_draw_index_buffer.clone(),
            depth_pyramid,
            depth_pyramid_mips,
            material_depth_color: (!is_shadow_view)
                .then(|| texture_cache.get(&render_device, material_depth_color)),
            material_depth: (!is_shadow_view)
                .then(|| texture_cache.get(&render_device, material_depth)),
        });
    }
}

pub fn prepare_meshlet_view_bind_groups(
    gpu_scene: Res<MeshletGpuScene>,
    views: Query<(
        Entity,
        &MeshletViewResources,
        AnyOf<(&ViewDepthTexture, &ShadowView)>,
    )>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    for (view_entity, view_resources, view_depth) in &views {
        let entries = BindGroupEntries::sequential((
            gpu_scene.meshlets.binding(),
            gpu_scene.instance_uniforms.binding().unwrap(),
            gpu_scene.thread_instance_ids.binding().unwrap(),
            gpu_scene.thread_meshlet_ids.binding().unwrap(),
            gpu_scene.previous_thread_ids.binding().unwrap(),
            view_resources.previous_occlusion_buffer.as_entire_binding(),
            view_resources.occlusion_buffer.as_entire_binding(),
            gpu_scene.meshlet_bounding_spheres.binding(),
            view_resources
                .visibility_buffer_draw_command_buffer_first
                .as_entire_binding(),
            view_resources
                .visibility_buffer_draw_index_buffer
                .as_entire_binding(),
            view_uniforms.clone(),
        ));
        let culling_first = render_device.create_bind_group(
            "meshlet_culling_first_bind_group",
            &gpu_scene.culling_first_bind_group_layout,
            &entries,
        );

        let entries = BindGroupEntries::sequential((
            gpu_scene.meshlets.binding(),
            gpu_scene.instance_uniforms.binding().unwrap(),
            gpu_scene.thread_instance_ids.binding().unwrap(),
            gpu_scene.thread_meshlet_ids.binding().unwrap(),
            gpu_scene.previous_thread_ids.binding().unwrap(),
            view_resources.previous_occlusion_buffer.as_entire_binding(),
            view_resources.occlusion_buffer.as_entire_binding(),
            gpu_scene.meshlet_bounding_spheres.binding(),
            view_resources
                .visibility_buffer_draw_command_buffer_second
                .as_entire_binding(),
            view_resources
                .visibility_buffer_draw_index_buffer
                .as_entire_binding(),
            view_uniforms.clone(),
            &view_resources.depth_pyramid.default_view,
            &gpu_scene.depth_pyramid_sampler,
        ));
        let culling_second = render_device.create_bind_group(
            "meshlet_culling_second_bind_group",
            &gpu_scene.culling_second_bind_group_layout,
            &entries,
        );

        let view_depth_texture = match view_depth {
            (Some(view_depth), None) => view_depth.view(),
            (None, Some(shadow_view)) => &shadow_view.depth_attachment.view,
            _ => unreachable!(),
        };
        let downsample_depth = [0, 1, 2, 3, 4, 5].map(|i| {
            render_device.create_bind_group(
                "meshlet_downsample_depth_bind_group",
                &gpu_scene.downsample_depth_bind_group_layout,
                &BindGroupEntries::sequential((
                    if i == 0 {
                        view_depth_texture
                    } else {
                        &view_resources.depth_pyramid_mips[i - 1]
                    },
                    &gpu_scene.depth_pyramid_sampler,
                )),
            )
        });

        let entries = BindGroupEntries::sequential((
            gpu_scene.meshlets.binding(),
            gpu_scene.instance_uniforms.binding().unwrap(),
            gpu_scene.thread_instance_ids.binding().unwrap(),
            gpu_scene.thread_meshlet_ids.binding().unwrap(),
            gpu_scene.vertex_data.binding(),
            gpu_scene.vertex_ids.binding(),
            gpu_scene.indices.binding(),
            gpu_scene.instance_material_ids.binding().unwrap(),
            view_uniforms.clone(),
        ));
        let visibility_buffer = render_device.create_bind_group(
            "meshlet_visibility_buffer_bind_group",
            &gpu_scene.visibility_buffer_bind_group_layout,
            &entries,
        );

        let copy_material_depth =
            view_resources
                .material_depth_color
                .as_ref()
                .map(|material_depth_color| {
                    render_device.create_bind_group(
                        "meshlet_copy_material_depth_bind_group",
                        &gpu_scene.copy_material_depth_bind_group_layout,
                        &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(
                                &material_depth_color.default_view,
                            ),
                        }],
                    )
                });

        let material_draw = view_resources
            .visibility_buffer
            .as_ref()
            .map(|visibility_buffer| {
                let entries = BindGroupEntries::sequential((
                    gpu_scene.meshlets.binding(),
                    gpu_scene.instance_uniforms.binding().unwrap(),
                    gpu_scene.thread_instance_ids.binding().unwrap(),
                    gpu_scene.thread_meshlet_ids.binding().unwrap(),
                    gpu_scene.vertex_data.binding(),
                    gpu_scene.vertex_ids.binding(),
                    gpu_scene.indices.binding(),
                    &visibility_buffer.default_view,
                ));
                render_device.create_bind_group(
                    "meshlet_mesh_material_draw_bind_group",
                    &gpu_scene.material_draw_bind_group_layout,
                    &entries,
                )
            });

        commands.entity(view_entity).insert(MeshletViewBindGroups {
            culling_first,
            culling_second,
            downsample_depth,
            visibility_buffer,
            copy_material_depth,
            material_draw,
        });
    }
}

#[derive(Resource)]
pub struct MeshletGpuScene {
    vertex_data: PersistentGpuBuffer<Arc<[u8]>>,
    vertex_ids: PersistentGpuBuffer<Arc<[u32]>>,
    indices: PersistentGpuBuffer<Arc<[u8]>>,
    meshlets: PersistentGpuBuffer<Arc<[Meshlet]>>,
    meshlet_bounding_spheres: PersistentGpuBuffer<Arc<[MeshletBoundingSphere]>>,
    meshlet_mesh_slices: HashMap<AssetId<MeshletMesh>, (Range<u32>, u64)>,

    scene_meshlet_count: u32,
    scene_index_count: u64,
    next_material_id: u32,
    material_id_lookup: HashMap<UntypedAssetId, u32>,
    material_ids_present_in_scene: HashSet<u32>,
    instances: Vec<Entity>,
    instance_uniforms: StorageBuffer<Vec<MeshUniform>>,
    instance_material_ids: StorageBuffer<Vec<u32>>,
    thread_instance_ids: StorageBuffer<Vec<u32>>,
    thread_meshlet_ids: StorageBuffer<Vec<u32>>,
    previous_thread_ids: StorageBuffer<Vec<u32>>,
    previous_thread_id_starts: HashMap<(Entity, AssetId<MeshletMesh>), (u32, bool)>,
    previous_occlusion_buffers: EntityHashMap<Entity, Buffer>,

    culling_first_bind_group_layout: BindGroupLayout,
    culling_second_bind_group_layout: BindGroupLayout,
    visibility_buffer_bind_group_layout: BindGroupLayout,
    downsample_depth_bind_group_layout: BindGroupLayout,
    copy_material_depth_bind_group_layout: BindGroupLayout,
    material_draw_bind_group_layout: BindGroupLayout,
    depth_pyramid_sampler: Sampler,
}

impl FromWorld for MeshletGpuScene {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            vertex_data: PersistentGpuBuffer::new("meshlet_vertex_data", render_device),
            vertex_ids: PersistentGpuBuffer::new("meshlet_vertex_ids", render_device),
            indices: PersistentGpuBuffer::new("meshlet_indices", render_device),
            meshlets: PersistentGpuBuffer::new("meshlets", render_device),
            meshlet_bounding_spheres: PersistentGpuBuffer::new(
                "meshlet_bounding_spheres",
                render_device,
            ),
            meshlet_mesh_slices: HashMap::new(),

            scene_meshlet_count: 0,
            scene_index_count: 0,
            next_material_id: 0,
            material_id_lookup: HashMap::new(),
            material_ids_present_in_scene: HashSet::new(),
            instances: Vec::new(),
            instance_uniforms: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_uniforms"));
                buffer
            },
            instance_material_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_material_ids"));
                buffer
            },
            thread_instance_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_thread_instance_ids"));
                buffer
            },
            thread_meshlet_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_thread_meshlet_ids"));
                buffer
            },
            previous_thread_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_previous_thread_ids"));
                buffer
            },
            previous_thread_id_starts: HashMap::new(),
            previous_occlusion_buffers: EntityHashMap::default(),

            // TODO: Buffer min sizes
            culling_first_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_culling_first_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        uniform_buffer::<ViewUniform>(true),
                    ),
                ),
            ),
            culling_second_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_culling_second_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        uniform_buffer::<ViewUniform>(true),
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        sampler(SamplerBindingType::NonFiltering),
                    ),
                ),
            ),
            downsample_depth_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_downsample_depth_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        sampler(SamplerBindingType::NonFiltering),
                    ),
                ),
            ),
            visibility_buffer_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_visibility_buffer_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::VERTEX,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        uniform_buffer::<ViewUniform>(true),
                    ),
                ),
            ),
            copy_material_depth_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_copy_material_depth_bind_group_layout",
                &BindGroupLayoutEntries::single(
                    ShaderStages::FRAGMENT,
                    texture_2d(TextureSampleType::Uint),
                ),
            ),
            material_draw_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_mesh_material_draw_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_2d(TextureSampleType::Uint),
                    ),
                ),
            ),
            depth_pyramid_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("meshlet_depth_pyramid_sampler"),
                ..default()
            }),
        }
    }
}

impl MeshletGpuScene {
    fn reset(&mut self) {
        // TODO: Shrink capacity if saturation is low
        self.scene_meshlet_count = 0;
        self.scene_index_count = 0;
        self.next_material_id = 0;
        self.material_id_lookup.clear();
        self.material_ids_present_in_scene.clear();
        self.instances.clear();
        self.instance_uniforms.get_mut().clear();
        self.instance_material_ids.get_mut().clear();
        self.thread_instance_ids.get_mut().clear();
        self.thread_meshlet_ids.get_mut().clear();
        self.previous_thread_ids.get_mut().clear();
        self.previous_thread_id_starts
            .values_mut()
            .for_each(|(_, active)| *active = false);
    }

    fn queue_meshlet_mesh_upload(
        &mut self,
        instance: Entity,
        handle: &Handle<MeshletMesh>,
        assets: &mut Assets<MeshletMesh>,
        instance_index: u32,
    ) {
        let queue_meshlet_mesh = |asset_id: &AssetId<MeshletMesh>| {
            let meshlet_mesh = assets.remove_untracked(*asset_id).expect(
                "MeshletMesh asset was already unloaded but is not registered with MeshletGpuScene",
            );

            let vertex_data_slice = self
                .vertex_data
                .queue_write(Arc::clone(&meshlet_mesh.vertex_data), ());
            let vertex_ids_slice = self.vertex_ids.queue_write(
                Arc::clone(&meshlet_mesh.vertex_ids),
                vertex_data_slice.start,
            );
            let indices_slice = self
                .indices
                .queue_write(Arc::clone(&meshlet_mesh.indices), ());
            let meshlets_slice = self.meshlets.queue_write(
                Arc::clone(&meshlet_mesh.meshlets),
                (vertex_ids_slice.start, indices_slice.start),
            );
            self.meshlet_bounding_spheres
                .queue_write(Arc::clone(&meshlet_mesh.meshlet_bounding_spheres), ());

            (
                (meshlets_slice.start as u32 / 12)..(meshlets_slice.end as u32 / 12),
                meshlet_mesh.total_meshlet_indices,
            )
        };

        let (meshlets_slice, index_count) = self
            .meshlet_mesh_slices
            .entry(handle.id())
            .or_insert_with_key(queue_meshlet_mesh)
            .clone();

        let current_thread_id_start = self.scene_meshlet_count;

        self.scene_meshlet_count += meshlets_slice.end - meshlets_slice.start;
        self.scene_index_count += index_count;
        self.instances.push(instance);

        let previous_thread_id_start = self
            .previous_thread_id_starts
            .entry((instance, handle.id()))
            .or_insert((0, true));

        for (i, meshlet_index) in meshlets_slice.into_iter().enumerate() {
            self.thread_instance_ids.get_mut().push(instance_index);
            self.thread_meshlet_ids.get_mut().push(meshlet_index);
            self.previous_thread_ids
                .get_mut()
                .push(if previous_thread_id_start.1 {
                    0
                } else {
                    previous_thread_id_start.0 + i as u32
                });
        }

        *previous_thread_id_start = (current_thread_id_start, true);
    }

    pub fn get_material_id(&mut self, material_id: UntypedAssetId) -> u32 {
        *self
            .material_id_lookup
            .entry(material_id)
            .or_insert_with(|| {
                self.next_material_id += 1;
                self.next_material_id
            })
    }

    pub fn material_present_in_scene(&self, material_id: &u32) -> bool {
        self.material_ids_present_in_scene.contains(material_id)
    }

    pub fn culling_first_bind_group_layout(&self) -> BindGroupLayout {
        self.culling_first_bind_group_layout.clone()
    }

    pub fn culling_second_bind_group_layout(&self) -> BindGroupLayout {
        self.culling_second_bind_group_layout.clone()
    }

    pub fn downsample_depth_bind_group_layout(&self) -> BindGroupLayout {
        self.downsample_depth_bind_group_layout.clone()
    }

    pub fn visibility_buffer_bind_group_layout(&self) -> BindGroupLayout {
        self.visibility_buffer_bind_group_layout.clone()
    }

    pub fn copy_material_depth_bind_group_layout(&self) -> BindGroupLayout {
        self.copy_material_depth_bind_group_layout.clone()
    }

    pub fn material_draw_bind_group_layout(&self) -> BindGroupLayout {
        self.material_draw_bind_group_layout.clone()
    }
}

#[derive(Component)]
pub struct MeshletViewResources {
    pub scene_meshlet_count: u32,
    previous_occlusion_buffer: Buffer,
    occlusion_buffer: Buffer,
    pub visibility_buffer: Option<CachedTexture>,
    pub visibility_buffer_draw_command_buffer_first: Buffer,
    pub visibility_buffer_draw_command_buffer_second: Buffer,
    pub visibility_buffer_draw_index_buffer: Buffer,
    pub depth_pyramid: CachedTexture,
    // TODO: Dynamic number of mips based on view resolution
    pub depth_pyramid_mips: [TextureView; 6],
    pub material_depth_color: Option<CachedTexture>,
    pub material_depth: Option<CachedTexture>,
}

#[derive(Component)]
pub struct MeshletViewBindGroups {
    pub culling_first: BindGroup,
    pub culling_second: BindGroup,
    pub downsample_depth: [BindGroup; 6],
    pub visibility_buffer: BindGroup,
    pub copy_material_depth: Option<BindGroup>,
    pub material_draw: Option<BindGroup>,
}
