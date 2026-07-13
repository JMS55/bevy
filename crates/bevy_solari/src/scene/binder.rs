use super::{
    blas::BlasManager,
    extract::{ExtractedRaytracingScene, StandardMaterialAssets},
};
use alloc::sync::Arc;
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::{Entity, EntityHashMap},
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_math::{ops::cos, Affine3, Affine3Ext, Mat4, Vec3, Vec4};
use bevy_pbr::{DfgLut, ExtractedDirectionalLight, StandardMaterial};
use bevy_platform::{collections::HashMap, hash::FixedHasher};
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::allocator::MeshAllocator,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use bytemuck::{Pod, Zeroable};
use core::{
    f32::consts::TAU,
    hash::{Hash, Hasher},
    num::NonZeroU32,
    ops::Deref,
};
use tracing::info_span;

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    previous_frame_light_entities: Vec<Entity>,
    topology_fingerprint: Option<u64>,
    dynamic_fingerprint: Option<u64>,
    settle_previous_frame_data: bool,
    material_order: Vec<AssetId<StandardMaterial>>,
    instance_ids: EntityHashMap<u32>,
    materials: StorageBufferList<GpuMaterial>,
    tlas: Option<Tlas>,
    transforms: AtomicSparseBufferVec<GpuTransform>,
    previous_frame_transforms: AtomicSparseBufferVec<GpuTransform>,
    geometry_ids: StorageBufferList<GpuInstanceGeometryIds>,
    material_ids: StorageBufferList<u32>,
    light_sources: StorageBufferList<GpuLightSource>,
    directional_lights: StorageBufferList<GpuDirectionalLight>,
    previous_frame_light_id_translations: StorageBufferList<u32>,
}

pub fn prepare_raytracing_scene_bindings(
    extracted_scene: Res<ExtractedRaytracingScene>,
    directional_lights_query: Query<(Entity, &ExtractedDirectionalLight)>,
    mesh_allocator: Res<MeshAllocator>,
    blas_manager: Res<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut sparse_buffer_update_jobs: ResMut<SparseBufferUpdateJobs>,
    mut sparse_buffer_update_bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    sparse_buffer_update_pipelines: Res<SparseBufferUpdatePipelines>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
    mut raytracing_scene_bindings: ResMut<RaytracingSceneBindings>,
) {
    let (topology_fingerprint, dynamic_fingerprint) = {
        let _span = info_span!("raytracing_scene/fingerprints").entered();
        scene_fingerprints(
            &directional_lights_query,
            &material_assets,
            extracted_scene.topology_revision,
            blas_manager.revision(),
        )
    };
    let topology_changed = raytracing_scene_bindings.topology_fingerprint
        != Some(topology_fingerprint)
        || texture_assets.is_changed()
        || fallback_texture.is_changed()
        || dfg_lut.is_changed();

    if !topology_changed {
        let dynamic_changed = raytracing_scene_bindings.dynamic_fingerprint
            != Some(dynamic_fingerprint)
            || !extracted_scene.changed_instances.is_empty();
        if !dynamic_changed && !raytracing_scene_bindings.settle_previous_frame_data {
            return;
        }
        update_dynamic_scene_bindings(
            &extracted_scene,
            &directional_lights_query,
            &blas_manager,
            &material_assets,
            &render_device,
            &render_queue,
            &pipeline_cache,
            &mut sparse_buffer_update_jobs,
            &mut sparse_buffer_update_bind_groups,
            &sparse_buffer_update_pipelines,
            diagnostics.as_deref_mut(),
            &mut raytracing_scene_bindings,
        );
        raytracing_scene_bindings.dynamic_fingerprint = Some(dynamic_fingerprint);
        raytracing_scene_bindings.settle_previous_frame_data = false;
        return;
    }
    raytracing_scene_bindings.topology_fingerprint = Some(topology_fingerprint);
    raytracing_scene_bindings.dynamic_fingerprint = Some(dynamic_fingerprint);
    raytracing_scene_bindings.settle_previous_frame_data = true;
    raytracing_scene_bindings.bind_group = None;

    let mut this_frame_entity_to_light_id = EntityHashMap::<u32>::default();
    let previous_frame_light_entities: Vec<_> = raytracing_scene_bindings
        .previous_frame_light_entities
        .drain(..)
        .collect();

    if extracted_scene.instances.is_empty() {
        return;
    }

    let mut vertex_buffers = CachedBindingArray::new();
    let mut index_buffers = CachedBindingArray::new();
    let mut textures = CachedBindingArray::new();
    let mut samplers = Vec::new();
    let mut materials = StorageBufferList::<GpuMaterial>::default();
    let mut material_order = Vec::new();
    let mut instance_ids = EntityHashMap::default();
    let mut tlas = render_device
        .wgpu_device()
        .create_tlas(&CreateTlasDescriptor {
            label: Some("tlas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_UPDATE,
            update_mode: AccelerationStructureUpdateMode::PreferUpdate,
            max_instances: extracted_scene.instances.len() as u32,
        });
    let mut transforms = AtomicSparseBufferVec::<GpuTransform>::new(
        BufferUsages::STORAGE,
        Arc::from("raytracing scene transforms"),
    );
    let mut previous_frame_transforms = AtomicSparseBufferVec::<GpuTransform>::new(
        BufferUsages::STORAGE,
        Arc::from("raytracing scene previous frame transforms"),
    );
    let mut geometry_ids = StorageBufferList::<GpuInstanceGeometryIds>::default();
    let mut material_ids = StorageBufferList::<u32>::default();
    let mut light_sources = StorageBufferList::<GpuLightSource>::default();
    let mut directional_lights = StorageBufferList::<GpuDirectionalLight>::default();
    let mut previous_frame_light_id_translations = StorageBufferList::<u32>::default();

    let material_span = info_span!("raytracing_scene/collect_materials").entered();
    let mut material_id_map: HashMap<AssetId<StandardMaterial>, u32, FixedHasher> =
        HashMap::default();
    let mut material_id = 0;
    let mut process_texture = |texture_handle: &Option<Handle<_>>| -> Option<u32> {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) => {
                    let (texture_id, is_new) =
                        textures.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    if is_new {
                        samplers.push(texture.sampler.deref());
                    }
                    Some(texture_id)
                }
                None => None,
            },
            None => Some(TEXTURE_MAP_NONE),
        }
    };
    for (asset_id, material) in material_assets.iter() {
        let Some(base_color_texture_id) = process_texture(&material.base_color_texture) else {
            continue;
        };
        let Some(normal_map_texture_id) = process_texture(&material.normal_map_texture) else {
            continue;
        };
        let Some(emissive_texture_id) = process_texture(&material.emissive_texture) else {
            continue;
        };
        let Some(metallic_roughness_texture_id) =
            process_texture(&material.metallic_roughness_texture)
        else {
            continue;
        };

        materials.get_mut().push(GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: material.emissive.to_vec3(),
            metallic: material.metallic,
            reflectance: material.reflectance,
            _padding: Default::default(),
        });

        material_id_map.insert(*asset_id, material_id);
        material_order.push(*asset_id);
        material_id += 1;
    }
    drop(material_span);

    if material_id == 0 {
        return;
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    let instance_span = info_span!("raytracing_scene/collect_instances").entered();
    let mut instance_id = 0;
    for (&entity, instance) in &extracted_scene.instances {
        let Some(blas) = blas_manager.get(&instance.mesh.id()) else {
            continue;
        };
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&instance.mesh.id()) else {
            continue;
        };
        let Some(index_slice) = mesh_allocator.mesh_index_slice(&instance.mesh.id()) else {
            continue;
        };
        let Some(material_id) = material_id_map.get(&instance.material.id()).copied() else {
            continue;
        };
        let Some(material) = materials.get().get(material_id as usize) else {
            continue;
        };

        let transform = instance.transform.to_matrix();
        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
            tlas_transform(&transform),
            Default::default(),
            0xFF,
        ));

        transforms.push(GpuTransform::new(
            Affine3::from(instance.transform.affine()).to_transpose(),
        ));
        previous_frame_transforms.push(
            instance
                .previous_transform
                .as_ref()
                .map(|transform| GpuTransform::new(Affine3::from(transform.0).to_transpose()))
                .unwrap_or_else(|| {
                    GpuTransform::new(Affine3::from(instance.transform.affine()).to_transpose())
                }),
        );

        let (vertex_buffer_id, _) = vertex_buffers.push_if_absent(
            vertex_slice.buffer.as_entire_buffer_binding(),
            vertex_slice.buffer.id(),
        );
        let (index_buffer_id, _) = index_buffers.push_if_absent(
            index_slice.buffer.as_entire_buffer_binding(),
            index_slice.buffer.id(),
        );

        geometry_ids.get_mut().push(GpuInstanceGeometryIds {
            vertex_buffer_id,
            vertex_buffer_offset: vertex_slice.range.start,
            index_buffer_id,
            index_buffer_offset: index_slice.range.start,
            triangle_count: (index_slice.range.len() / 3) as u32,
        });

        material_ids.get_mut().push(material_id);
        instance_ids.insert(entity, instance_id as u32);

        if material.emissive != Vec3::ZERO {
            light_sources
                .get_mut()
                .push(GpuLightSource::new_emissive_mesh_light(
                    instance_id as u32,
                    (index_slice.range.len() / 3) as u32,
                ));

            this_frame_entity_to_light_id.insert(entity, light_sources.get().len() as u32 - 1);
            raytracing_scene_bindings
                .previous_frame_light_entities
                .push(entity);
        }

        instance_id += 1;
    }
    drop(instance_span);

    if instance_id == 0 {
        return;
    }

    let light_span = info_span!("raytracing_scene/collect_lights").entered();
    for (entity, directional_light) in &directional_lights_query {
        let directional_lights = directional_lights.get_mut();
        let directional_light_id = directional_lights.len() as u32;

        directional_lights.push(GpuDirectionalLight::new(directional_light));

        light_sources
            .get_mut()
            .push(GpuLightSource::new_directional_light(directional_light_id));

        this_frame_entity_to_light_id.insert(entity, light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_entities
            .push(entity);
    }

    for previous_frame_light_entity in previous_frame_light_entities {
        let current_frame_index = this_frame_entity_to_light_id
            .get(&previous_frame_light_entity)
            .copied()
            .unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);
        previous_frame_light_id_translations
            .get_mut()
            .push(current_frame_index);
    }
    drop(light_span);

    if light_sources.get().len() > u16::MAX as usize {
        panic!("Too many light sources in the scene, maximum is 65535.");
    }

    {
        let _span = info_span!("raytracing_scene/write_buffers").entered();
        materials.write_buffer(&render_device, &render_queue);
        write_transform_buffers(
            &mut transforms,
            &mut previous_frame_transforms,
            &render_device,
            &render_queue,
            &pipeline_cache,
            &mut sparse_buffer_update_jobs,
            &mut sparse_buffer_update_bind_groups,
            &sparse_buffer_update_pipelines,
        );
        geometry_ids.write_buffer(&render_device, &render_queue);
        material_ids.write_buffer(&render_device, &render_queue);
        light_sources.write_buffer(&render_device, &render_queue);
        directional_lights.write_buffer(&render_device, &render_queue);
        previous_frame_light_id_translations.write_buffer(&render_device, &render_queue);
    }

    let tlas_span = info_span!("raytracing_scene/build_tlas").entered();
    let mut command_encoder = {
        let _span = info_span!("raytracing_scene/build_tlas/create_encoder").entered();
        render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build_tlas_command_encoder"),
        })
    };
    let time_span = diagnostics.as_mut().map(|diagnostics| {
        diagnostics.time_span(&mut command_encoder, "raytracing_scene/tlas_build")
    });
    {
        let _span = info_span!("raytracing_scene/build_tlas/encode").entered();
        command_encoder.build_acceleration_structures(&[], [&tlas]);
    }
    if let Some(time_span) = time_span {
        time_span.end(&mut command_encoder);
    }
    let command_buffer = {
        let _span = info_span!("raytracing_scene/build_tlas/finish_encoder").entered();
        command_encoder.finish()
    };
    {
        let _span = info_span!("raytracing_scene/build_tlas/submit").entered();
        render_queue.submit([command_buffer]);
    }
    drop(tlas_span);

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|img| (&img.texture_view, &img.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    raytracing_scene_bindings.material_order = material_order;
    raytracing_scene_bindings.instance_ids = instance_ids;
    raytracing_scene_bindings.materials = materials;
    raytracing_scene_bindings.tlas = Some(tlas);
    raytracing_scene_bindings.transforms = transforms;
    raytracing_scene_bindings.previous_frame_transforms = previous_frame_transforms;
    raytracing_scene_bindings.geometry_ids = geometry_ids;
    raytracing_scene_bindings.material_ids = material_ids;
    raytracing_scene_bindings.light_sources = light_sources;
    raytracing_scene_bindings.directional_lights = directional_lights;
    raytracing_scene_bindings.previous_frame_light_id_translations =
        previous_frame_light_id_translations;

    let bind_group_span = info_span!("raytracing_scene/create_bind_group").entered();
    raytracing_scene_bindings.bind_group = Some(
        render_device.create_bind_group(
            "raytracing_scene_bind_group",
            &pipeline_cache.get_bind_group_layout(&raytracing_scene_bindings.bind_group_layout),
            &BindGroupEntries::sequential((
                vertex_buffers.as_slice(),
                index_buffers.as_slice(),
                textures.as_slice(),
                samplers.as_slice(),
                raytracing_scene_bindings.materials.binding().unwrap(),
                raytracing_scene_bindings
                    .tlas
                    .as_ref()
                    .unwrap()
                    .as_binding(),
                raytracing_scene_bindings
                    .transforms
                    .buffer()
                    .unwrap()
                    .as_entire_binding(),
                raytracing_scene_bindings
                    .previous_frame_transforms
                    .buffer()
                    .unwrap()
                    .as_entire_binding(),
                raytracing_scene_bindings.geometry_ids.binding().unwrap(),
                raytracing_scene_bindings.material_ids.binding().unwrap(),
                raytracing_scene_bindings.light_sources.binding().unwrap(),
                raytracing_scene_bindings
                    .directional_lights
                    .binding()
                    .unwrap(),
                raytracing_scene_bindings
                    .previous_frame_light_id_translations
                    .binding()
                    .unwrap(),
                dfg_view,
                dfg_sampler,
            )),
        ),
    );
    drop(bind_group_span);
}

fn update_dynamic_scene_bindings(
    extracted_scene: &ExtractedRaytracingScene,
    directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>,
    blas_manager: &BlasManager,
    material_assets: &StandardMaterialAssets,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    sparse_buffer_update_jobs: &mut SparseBufferUpdateJobs,
    sparse_buffer_update_bind_groups: &mut SparseBufferUpdateBindGroups,
    sparse_buffer_update_pipelines: &SparseBufferUpdatePipelines,
    diagnostics: Option<&mut DiagnosticsRecorder>,
    bindings: &mut RaytracingSceneBindings,
) {
    let Some(tlas) = bindings.tlas.as_mut() else {
        return;
    };

    {
        let _span = info_span!("raytracing_scene/update_materials").entered();
        for (gpu_material, asset_id) in bindings
            .materials
            .get_mut()
            .iter_mut()
            .zip(&bindings.material_order)
        {
            let Some(material) = material_assets.get(asset_id) else {
                continue;
            };
            gpu_material.base_color = LinearRgba::from(material.base_color).to_vec3();
            gpu_material.perceptual_roughness = material.perceptual_roughness;
            gpu_material.emissive = material.emissive.to_vec3();
            gpu_material.metallic = material.metallic;
            gpu_material.reflectance = material.reflectance;
        }
    }

    let mut tlas_changed = false;
    {
        let _span = info_span!("raytracing_scene/update_instances").entered();
        for entity in &extracted_scene.changed_instances {
            let Some(instance) = extracted_scene.instances.get(entity) else {
                continue;
            };
            let Some(instance_id) = bindings.instance_ids.get(entity).copied() else {
                continue;
            };
            let Some(blas) = blas_manager.get(&instance.mesh.id()) else {
                continue;
            };
            let transform = instance.transform.to_matrix();
            *tlas.get_mut_single(instance_id as usize).unwrap() = Some(TlasInstance::new(
                blas,
                tlas_transform(&transform),
                Default::default(),
                0xFF,
            ));
            bindings.transforms.set(
                instance_id,
                GpuTransform::new(Affine3::from(instance.transform.affine()).to_transpose()),
            );
            bindings.previous_frame_transforms.set(
                instance_id,
                instance
                    .previous_transform
                    .as_ref()
                    .map(|transform| GpuTransform::new(Affine3::from(transform.0).to_transpose()))
                    .unwrap_or_else(|| {
                        GpuTransform::new(Affine3::from(instance.transform.affine()).to_transpose())
                    }),
            );
            tlas_changed = true;
        }
    }

    {
        let _span = info_span!("raytracing_scene/update_lights").entered();
        bindings.directional_lights.get_mut().clear();
        for (_, directional_light) in directional_lights {
            bindings
                .directional_lights
                .get_mut()
                .push(GpuDirectionalLight::new(directional_light));
        }

        // The topology is unchanged, so light ordering is unchanged and the
        // previous-to-current translation is the identity mapping.
        let light_count = bindings.previous_frame_light_entities.len() as u32;
        bindings
            .previous_frame_light_id_translations
            .set((0..light_count).collect());
    }

    {
        let _span = info_span!("raytracing_scene/write_dynamic_buffers").entered();
        bindings.materials.write_buffer(render_device, render_queue);
        if tlas_changed {
            write_transform_buffers(
                &mut bindings.transforms,
                &mut bindings.previous_frame_transforms,
                render_device,
                render_queue,
                pipeline_cache,
                sparse_buffer_update_jobs,
                sparse_buffer_update_bind_groups,
                sparse_buffer_update_pipelines,
            );
        }
        bindings
            .directional_lights
            .write_buffer(render_device, render_queue);
        bindings
            .previous_frame_light_id_translations
            .write_buffer(render_device, render_queue);
    }

    if tlas_changed {
        let _span = info_span!("raytracing_scene/build_tlas").entered();
        let mut command_encoder = {
            let _span = info_span!("raytracing_scene/build_tlas/create_encoder").entered();
            render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("update_tlas_command_encoder"),
            })
        };
        let time_span = diagnostics.map(|diagnostics| {
            diagnostics.time_span(&mut command_encoder, "raytracing_scene/tlas_build")
        });
        {
            let _span = info_span!("raytracing_scene/build_tlas/encode").entered();
            command_encoder.build_acceleration_structures(&[], [&*tlas]);
        }
        if let Some(time_span) = time_span {
            time_span.end(&mut command_encoder);
        }
        let command_buffer = {
            let _span = info_span!("raytracing_scene/build_tlas/finish_encoder").entered();
            command_encoder.finish()
        };
        {
            let _span = info_span!("raytracing_scene/build_tlas/submit").entered();
            render_queue.submit([command_buffer]);
        }
    }
}

fn write_transform_buffers(
    transforms: &mut AtomicSparseBufferVec<GpuTransform>,
    previous_frame_transforms: &mut AtomicSparseBufferVec<GpuTransform>,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    sparse_buffer_update_jobs: &mut SparseBufferUpdateJobs,
    sparse_buffer_update_bind_groups: &mut SparseBufferUpdateBindGroups,
    sparse_buffer_update_pipelines: &SparseBufferUpdatePipelines,
) {
    transforms.write_buffers(render_device, render_queue);
    previous_frame_transforms.write_buffers(render_device, render_queue);

    transforms.prepare_to_populate_buffers(
        render_device,
        pipeline_cache,
        sparse_buffer_update_jobs,
        sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    );
    previous_frame_transforms.prepare_to_populate_buffers(
        render_device,
        pipeline_cache,
        sparse_buffer_update_jobs,
        sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    );
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT),
                        storage_buffer_read_only_sized(false, None),
                        acceleration_structure(),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                    ),
                ),
            ),
            previous_frame_light_entities: Vec::new(),
            topology_fingerprint: None,
            dynamic_fingerprint: None,
            settle_previous_frame_data: false,
            material_order: Vec::new(),
            instance_ids: Default::default(),
            materials: Default::default(),
            tlas: None,
            transforms: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("raytracing scene transforms"),
            ),
            previous_frame_transforms: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("raytracing scene previous frame transforms"),
            ),
            geometry_ids: Default::default(),
            material_ids: Default::default(),
            light_sources: Default::default(),
            directional_lights: Default::default(),
            previous_frame_light_id_translations: Default::default(),
        }
    }
}

#[derive(Default)]
struct SceneFingerprintHasher(u64);

impl Hasher for SceneFingerprintHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // FNV-1a is sufficient here: this is a change detector, not a key used
        // for correctness or security.
        if self.0 == 0 {
            self.0 = 0xcbf29ce484222325;
        }
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn scene_fingerprints(
    directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>,
    materials: &StandardMaterialAssets,
    scene_topology_revision: u64,
    blas_revision: u64,
) -> (u64, u64) {
    let mut topology_hasher = SceneFingerprintHasher::default();
    let mut dynamic_hasher = SceneFingerprintHasher::default();
    scene_topology_revision.hash(&mut topology_hasher);
    blas_revision.hash(&mut topology_hasher);

    // Texture bindings and whether a material contributes an emissive mesh
    // light affect binding topology. Scalar material values do not.
    let mut material_fingerprint = 0_u64;
    let mut dynamic_material_fingerprint = 0_u64;
    for (asset_id, material) in materials.iter() {
        let mut material_hasher = SceneFingerprintHasher::default();
        let mut dynamic_material_hasher = SceneFingerprintHasher::default();
        asset_id.hash(&mut material_hasher);
        asset_id.hash(&mut dynamic_material_hasher);
        material.base_color_texture.hash(&mut material_hasher);
        material.normal_map_texture.hash(&mut material_hasher);
        material.emissive_texture.hash(&mut material_hasher);
        material
            .metallic_roughness_texture
            .hash(&mut material_hasher);
        (material.emissive.to_vec3() != Vec3::ZERO).hash(&mut material_hasher);
        hash_vec3(
            &mut dynamic_material_hasher,
            LinearRgba::from(material.base_color).to_vec3(),
        );
        hash_f32(&mut dynamic_material_hasher, material.perceptual_roughness);
        hash_vec3(&mut dynamic_material_hasher, material.emissive.to_vec3());
        hash_f32(&mut dynamic_material_hasher, material.metallic);
        hash_f32(&mut dynamic_material_hasher, material.reflectance);
        material_fingerprint ^= material_hasher.finish();
        dynamic_material_fingerprint ^= dynamic_material_hasher.finish();
    }
    material_fingerprint.hash(&mut topology_hasher);
    dynamic_material_fingerprint.hash(&mut dynamic_hasher);

    for (entity, light) in directional_lights {
        entity.hash(&mut topology_hasher);
        let light = GpuDirectionalLight::new(light);
        hash_vec3(&mut dynamic_hasher, light.direction_to_light);
        hash_f32(&mut dynamic_hasher, light.cos_theta_max);
        hash_vec3(&mut dynamic_hasher, light.luminance);
        hash_f32(&mut dynamic_hasher, light.inverse_pdf);
    }

    (topology_hasher.finish(), dynamic_hasher.finish())
}

fn hash_vec3(hasher: &mut impl Hasher, vector: Vec3) {
    for value in vector.to_array() {
        hash_f32(hasher, value);
    }
}

fn hash_f32(hasher: &mut impl Hasher, value: f32) {
    value.to_bits().hash(hasher);
}

impl Default for RaytracingSceneBindings {
    fn default() -> Self {
        Self::new()
    }
}

struct CachedBindingArray<T, I: Eq + Hash> {
    map: HashMap<I, u32>,
    vec: Vec<T>,
}

impl<T, I: Eq + Hash> CachedBindingArray<T, I> {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            vec: Vec::default(),
        }
    }

    fn push_if_absent(&mut self, item: T, item_id: I) -> (u32, bool) {
        let mut is_new = false;
        let i = *self.map.entry(item_id).or_insert_with(|| {
            is_new = true;
            let i = self.vec.len() as u32;
            self.vec.push(item);
            i
        });
        (i, is_new)
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
}

type StorageBufferList<T> = StorageBuffer<Vec<T>>;

/// An affine transform stored transposed so it maps directly to WGSL's
/// `mat3x4<f32>` layout without carrying the redundant final matrix row.
#[derive(Clone, Copy, Default, Pod, ShaderType, Zeroable)]
#[repr(C)]
struct GpuTransform {
    affine_transpose: [Vec4; 3],
}

bevy_render::impl_atomic_pod!(GpuTransform, GpuTransformBlob);

impl GpuTransform {
    fn new(affine_transpose: [Vec4; 3]) -> Self {
        Self { affine_transpose }
    }
}

#[derive(ShaderType)]
struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
}

#[derive(ShaderType)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    _padding: Vec3,
    reflectance: f32,
}

#[derive(ShaderType)]
struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        if triangle_count > u16::MAX as u32 {
            panic!("Too many triangles ({triangle_count}) in an emissive mesh, maximum is 65535.");
        }

        Self {
            kind: triangle_count << 1,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }
}

#[derive(ShaderType, Default)]
struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
