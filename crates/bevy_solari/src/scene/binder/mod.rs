mod assets;
mod bind_group;
mod buffers;
mod instances;
mod lights;
mod slots;
mod tlas;
mod tlas_build;

use self::assets::{AssetState, MAX_TEXTURE_COUNT};
pub use self::bind_group::prepare_raytracing_scene_bind_group;
use self::bind_group::BindGroupCacheState;
use self::instances::{
    ChangedInstanceFilter, InstanceQueryData, InstanceState, MAX_MESH_SLAB_COUNT,
};
use self::lights::LightState;
use self::tlas::TlasState;
pub use self::tlas::{
    build_raytracing_tlas, pack_raytracing_tlas_instances, retire_raytracing_resources,
    TlasInstancePackPipeline,
};
use super::{blas::BlasManager, extract::StandardMaterialAssets, RaytracingMesh3d};
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    resource::Resource,
    system::{Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_render::{
    mesh::allocator::MeshAllocator,
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};

// ---------------------------------------------------------------------------
// The scene bindings
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,

    assets: AssetState,
    instances: InstanceState,
    lights: LightState,
    tlas: TlasState,

    bind_groups: BindGroupCacheState,
}

impl FromWorld for RaytracingSceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Binding arrays are dense slices, so freed slots still need something valid bound into
        // them. A few elements' worth of zeroes covers the shader's runtime-sized arrays.
        let dummy_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_dummy_binding_array_buffer"),
            size: 256,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

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

            assets: AssetState::new(buffers::new_storage_buffer("solari_materials")),
            instances: InstanceState::new(),
            lights: LightState::new(
                buffers::new_storage_buffer("solari_light_sources"),
                buffers::new_storage_buffer("solari_directional_lights"),
                buffers::new_storage_buffer("solari_previous_frame_light_id_translations"),
            ),
            tlas: TlasState::new(),

            bind_groups: BindGroupCacheState::new(dummy_buffer),
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

impl RaytracingSceneBindings {
    /// Phase 1: resolve changed assets and poll readiness retries.
    fn prepare_asset_updates(
        &mut self,
        material_assets: &StandardMaterialAssets,
        texture_assets: &RenderAssets<GpuImage>,
        extracted_images: &ExtractedAssets<GpuImage>,
    ) {
        self.assets
            .update_materials(&mut self.instances, material_assets, texture_assets);
        self.assets.update_textures(
            &mut self.instances,
            extracted_images,
            texture_assets,
            material_assets,
        );
    }

    /// Phase 2: apply structural changes after asset slots are current.
    fn prepare_instance_updates(
        &mut self,
        removed: impl IntoIterator<Item = Entity>,
        instances: &Query<InstanceQueryData>,
        changed_instances: &Query<Entity, ChangedInstanceFilter>,
        blas_manager: &BlasManager,
        mesh_allocator: &MeshAllocator,
    ) {
        self.instances.remove_instances(&mut self.lights, removed);
        self.instances.refresh_instances(
            &self.assets,
            &mut self.lights,
            instances,
            changed_instances,
            blas_manager,
            mesh_allocator,
        );
    }

    /// Phase 3: finalize the dense light set after emissive instances are resolved.
    fn prepare_light_updates(
        &mut self,
        directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>,
    ) {
        self.lights.update(directional_lights);
    }

    /// Phase 4: make every sparse write from the earlier phases available for GPU upload.
    fn write_scene_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        buffers::write_sparse_buffers(self, render_device, render_queue);
    }

    /// Phase 5: advance parity only when the later pack/build systems can finish the new TLAS.
    fn prepare_tlas_update(
        &mut self,
        render_device: &RenderDevice,
        blas_manager: &BlasManager,
        build_ready: bool,
    ) {
        self.tlas.advance(
            &self.instances,
            &mut self.bind_groups,
            render_device,
            blas_manager,
            build_ready,
        );
    }
}

/// Applies this frame's scene changes to the retained buffers, binding arrays and TLAS.
///
/// These dependency phases deliberately remain in one render system. Splitting them into several
/// systems would not expose parallelism because every phase mutates the same façade resource, and
/// it would make the required asset -> instance -> light -> upload -> TLAS order less local. The
/// independent transform hot path remains a shared, parallel extraction system.
pub fn prepare_raytracing_scene_resources(
    instances: Query<InstanceQueryData>,
    changed_instances: Query<Entity, ChangedInstanceFilter>,
    mut removed_instances: RemovedComponents<RaytracingMesh3d>,
    directional_lights: Query<(Entity, &ExtractedDirectionalLight)>,
    mesh_allocator: Res<MeshAllocator>,
    blas_manager: Res<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    extracted_images: Res<ExtractedAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    instance_pack_pipeline: Res<TlasInstancePackPipeline>,
    mut bindings: ResMut<RaytracingSceneBindings>,
) {
    let bindings = &mut *bindings;

    // Reset before any removal or compaction writes this frame's translations.
    bindings.lights.reset_id_translations();
    bindings.prepare_asset_updates(&material_assets, &texture_assets, &extracted_images);
    bindings.prepare_instance_updates(
        removed_instances.read(),
        &instances,
        &changed_instances,
        &blas_manager,
        &mesh_allocator,
    );
    bindings.prepare_light_updates(&directional_lights);
    bindings.write_scene_buffers(&render_device, &render_queue);

    // The raw path can't build until the pack shader exists to fill the descriptors, and the
    // pipeline cache takes a few frames to get there from a cold start.
    let build_ready = instance_pack_pipeline
        .id
        .and_then(|id| pipeline_cache.get_compute_pipeline(id))
        .is_some();
    bindings.prepare_tlas_update(&render_device, &blas_manager, build_ready);
}
