mod allocator;
mod assets;
mod bind_group;
mod buffers;
mod instances;
mod lights;
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
    build_raytracing_tlas, pack_raytracing_tlas_instances, TlasInstancePackPipeline,
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

        let bind_group_layout = BindGroupLayoutDescriptor::new(
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
        );

        Self {
            bind_group: None,
            bind_group_layout,
            assets: AssetState::new(),
            instances: InstanceState::new(),
            lights: LightState::new(),
            tlas: TlasState::new(render_device),
            bind_groups: BindGroupCacheState::new(render_device),
        }
    }
}

/// Applies this frame's scene changes to the retained buffers, binding arrays and TLAS.
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

    // Reset lights before any removal or compaction writes this frame's light id translations
    bindings.lights.reset_id_translations();

    // Update material and texture assets
    bindings
        .assets
        .update_materials(&mut bindings.instances, &material_assets, &texture_assets);
    bindings.assets.update_textures(
        &mut bindings.instances,
        &extracted_images,
        &texture_assets,
        &material_assets,
    );

    // Apply structural instance changes, now that asset slots are current
    bindings
        .instances
        .remove_instances(&mut bindings.lights, removed_instances.read());
    bindings.instances.refresh_instances(
        &bindings.assets,
        &mut bindings.lights,
        &instances,
        &changed_instances,
        &blas_manager,
        &mesh_allocator,
    );

    // Update the light set, now that emissive instances are resolved
    bindings.lights.update(&directional_lights);

    // Make every sparse write above available for GPU upload
    buffers::write_sparse_buffers(bindings, &render_device, &render_queue);

    // Prepare the next TLAS
    let build_ready = !bindings.tlas.uses_raw_build()
        || instance_pack_pipeline
            .id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id))
            .is_some();
    bindings.tlas.advance(
        &bindings.instances,
        &mut bindings.bind_groups,
        &render_device,
        build_ready,
    );
}
