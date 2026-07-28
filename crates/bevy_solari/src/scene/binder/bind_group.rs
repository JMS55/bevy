use super::{
    buffers::SPARSE_BUFFER_COUNT, lights::GpuLightSource, slots::RetainedBindingArray,
    RaytracingSceneBindings, TlasInstancePackPipeline,
};
use bevy_ecs::system::{Res, ResMut};
use bevy_pbr::DfgLut;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, Buffer, BufferBinding, BufferId, BufferSize,
        PipelineCache, Sampler, SamplerId, SparseBufferUpdateBindGroups, SparseBufferUpdateJobs,
        SparseBufferUpdatePipelines, TextureView, TextureViewId,
    },
    renderer::RenderDevice,
    texture::{FallbackImage, GpuImage},
};
use core::{mem::size_of, ops::Deref};
use tracing::info_span;

pub struct BindGroupCacheState {
    cached: [Option<BindGroup>; 2],
    pub invalid: bool,
    last_buffer_ids: [Option<BufferId>; SPARSE_BUFFER_COUNT],
    last_light_count: u32,
    last_dfg_ids: Option<(TextureViewId, SamplerId)>,
    pub dummy_buffer: Buffer,
}

impl BindGroupCacheState {
    pub fn new(dummy_buffer: Buffer) -> Self {
        Self {
            cached: [None, None],
            invalid: true,
            last_buffer_ids: [None; SPARSE_BUFFER_COUNT],
            last_light_count: 0,
            last_dfg_ids: None,
            dummy_buffer,
        }
    }
}

fn buffer_bindings<'a>(
    buffers: &'a RetainedBindingArray<BufferId, Buffer>,
    dummy: &'a Buffer,
) -> Vec<BufferBinding<'a>> {
    buffers
        .iter()
        .map(|buffer| buffer.unwrap_or(dummy).as_entire_buffer_binding())
        .collect()
}

impl RaytracingSceneBindings {
    fn buffer_ids(&self) -> [Option<BufferId>; SPARSE_BUFFER_COUNT] {
        let buffers = super::buffers::sparse_buffers(self);
        core::array::from_fn(|index| buffers[index].buffer_id())
    }

    fn take_bind_group_invalidation(
        &mut self,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> bool {
        let mut invalid = self.bind_groups.invalid;
        self.bind_groups.invalid = false;

        for dirty in [
            &mut self.instances.vertex_buffers.dirty,
            &mut self.instances.index_buffers.dirty,
            &mut self.assets.textures.dirty,
        ] {
            invalid |= core::mem::replace(dirty, false);
        }

        let buffer_ids = self.buffer_ids();
        if self.bind_groups.last_buffer_ids != buffer_ids {
            self.bind_groups.last_buffer_ids = buffer_ids;
            invalid = true;
        }

        let light_count = self.lights.index.len() as u32;
        if self.bind_groups.last_light_count != light_count {
            self.bind_groups.last_light_count = light_count;
            invalid = true;
        }

        let dfg_ids = Some((dfg_view.id(), dfg_sampler.id()));
        if self.bind_groups.last_dfg_ids != dfg_ids {
            self.bind_groups.last_dfg_ids = dfg_ids;
            invalid = true;
        }

        invalid
    }

    fn create_bind_group(
        &self,
        parity: usize,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        fallback_texture: &FallbackImage,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> BindGroup {
        let _span = info_span!("create_bind_group").entered();
        let dummy = &self.bind_groups.dummy_buffer;
        let vertex_buffers = buffer_bindings(&self.instances.vertex_buffers, dummy);
        let index_buffers = buffer_bindings(&self.instances.index_buffers, dummy);

        let (mut textures, mut samplers): (Vec<_>, Vec<_>) = self
            .assets
            .textures
            .iter()
            .map(|texture| match texture {
                Some((view, sampler)) => (view.deref(), sampler.deref()),
                None => (
                    fallback_texture.d2.texture_view.deref(),
                    fallback_texture.d2.sampler.deref(),
                ),
            })
            .unzip();
        if textures.is_empty() {
            textures.push(fallback_texture.d2.texture_view.deref());
            samplers.push(fallback_texture.d2.sampler.deref());
        }

        let light_sources = BufferBinding {
            buffer: self.lights.sources.buffer().unwrap(),
            offset: 0,
            size: BufferSize::new(
                self.lights.index.len() as u64 * size_of::<GpuLightSource>() as u64,
            ),
        };

        let materials = self
            .assets
            .materials
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let transforms = self
            .instances
            .transforms
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let previous_frame_transforms = self
            .instances
            .previous_frame_transforms
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let geometry_ids = self
            .instances
            .geometry_ids
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let material_ids = self
            .instances
            .material_ids
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let directional_lights = self
            .lights
            .directional_lights
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let translations = self
            .lights
            .previous_frame_id_translations
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();

        let current = self.tlas.structures[parity].as_ref().unwrap();
        let previous = self.tlas.structures[parity ^ 1]
            .as_ref()
            .filter(|_| self.tlas.built[parity ^ 1])
            .unwrap_or(current);

        render_device.create_bind_group(
            "raytracing_scene_bind_group",
            layout,
            &BindGroupEntries::sequential((
                vertex_buffers.as_slice(),
                index_buffers.as_slice(),
                textures.as_slice(),
                samplers.as_slice(),
                materials,
                current.as_binding(),
                previous.as_binding(),
                transforms,
                previous_frame_transforms,
                geometry_ids,
                material_ids,
                light_sources,
                directional_lights,
                translations,
                dfg_view,
                dfg_sampler,
            )),
        )
    }

    fn cache_bind_group(
        &mut self,
        parity: usize,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        fallback_texture: &FallbackImage,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> BindGroup {
        if let Some(bind_group) = &self.bind_groups.cached[parity] {
            return bind_group.clone();
        }

        let bind_group = self.create_bind_group(
            parity,
            render_device,
            layout,
            fallback_texture,
            dfg_view,
            dfg_sampler,
        );
        if self.tlas.built[parity ^ 1] {
            self.bind_groups.cached[parity] = Some(bind_group.clone());
        }
        bind_group
    }
}

/// Finalizes sparse uploads and selects the cached bind group for this TLAS parity.
pub fn prepare_raytracing_scene_bind_group(
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    sparse_buffer_update_pipelines: Res<SparseBufferUpdatePipelines>,
    instance_pack_pipeline: Res<TlasInstancePackPipeline>,
    mut sparse_buffer_update_jobs: ResMut<SparseBufferUpdateJobs>,
    mut sparse_buffer_update_bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    mut bindings: ResMut<RaytracingSceneBindings>,
) {
    let bindings = &mut *bindings;

    {
        let _span = info_span!("prepare_sparse_uploads").entered();
        for buffer in super::buffers::sparse_buffers_mut(bindings) {
            buffer.prepare_upload(
                &render_device,
                &pipeline_cache,
                &mut sparse_buffer_update_jobs,
                &mut sparse_buffer_update_bind_groups,
                &sparse_buffer_update_pipelines,
            );
        }
    }

    bindings.tlas.update_instance_pack_bind_group(
        &bindings.instances,
        &render_device,
        &pipeline_cache,
        &instance_pack_pipeline,
    );
    bindings.bind_group = None;

    if bindings.instances.live_count == 0 || bindings.lights.index.is_empty() {
        return;
    }
    if bindings.tlas.structures[bindings.tlas.frame_parity].is_none() {
        return;
    }

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|image| (&image.texture_view, &image.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    if bindings.take_bind_group_invalidation(dfg_view, dfg_sampler) {
        bindings.bind_groups.cached = [None, None];
    }

    let parity = bindings.tlas.frame_parity;
    let layout = pipeline_cache.get_bind_group_layout(&bindings.bind_group_layout);
    bindings.bind_group = Some(bindings.cache_bind_group(
        parity,
        &render_device,
        &layout,
        &fallback_texture,
        dfg_view,
        dfg_sampler,
    ));
}
