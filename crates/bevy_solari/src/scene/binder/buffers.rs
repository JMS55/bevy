use super::RaytracingSceneBindings;
use alloc::sync::Arc;
use bevy_render::{
    impl_atomic_pod,
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, BufferId, BufferUsages, PipelineCache,
        SparseBufferUpdateBindGroups, SparseBufferUpdateJobs, SparseBufferUpdatePipelines,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};
use tracing::info_span;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
pub struct GpuU32(pub u32);

impl_atomic_pod!(GpuU32, GpuU32Blob);

pub fn set_at<T: AtomicPod + PartialEq>(
    buffer: &mut AtomicSparseBufferVec<T>,
    index: u32,
    value: T,
) {
    if buffer.len() > index {
        if buffer.get(index) == value {
            return;
        }
    } else {
        buffer.grow(index + 1);
    }
    buffer.set(index, value);
}

/// Allocation-free existing-slot write used by the parallel transform path.
pub fn set_existing<T: AtomicPod + PartialEq>(
    buffer: &AtomicSparseBufferVec<T>,
    index: u32,
    value: T,
) {
    debug_assert!(
        index < buffer.len(),
        "buffer was not grown past index {index}"
    );
    if buffer.get(index) != value {
        buffer.set(index, value);
    }
}

pub fn new_storage_buffer<T: AtomicPod>(label: &'static str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, Arc::from(label))
}

pub const SPARSE_BUFFER_COUNT: usize = 9;

pub trait SceneBuffer {
    fn grow_to(&mut self, len: u32);
    fn buffer_id(&self) -> Option<BufferId>;
    fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue);
    fn prepare_upload(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        jobs: &mut SparseBufferUpdateJobs,
        bind_groups: &mut SparseBufferUpdateBindGroups,
        pipelines: &SparseBufferUpdatePipelines,
    );
}

impl<T: AtomicPod> SceneBuffer for AtomicSparseBufferVec<T> {
    fn grow_to(&mut self, len: u32) {
        self.grow(len);
    }

    fn buffer_id(&self) -> Option<BufferId> {
        Some(self.buffer()?.id())
    }

    fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.write_buffers(render_device, render_queue);
    }

    fn prepare_upload(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        jobs: &mut SparseBufferUpdateJobs,
        bind_groups: &mut SparseBufferUpdateBindGroups,
        pipelines: &SparseBufferUpdatePipelines,
    ) {
        self.prepare_to_populate_buffers(
            render_device,
            pipeline_cache,
            jobs,
            bind_groups,
            pipelines,
        );
    }
}

pub fn sparse_buffers(
    bindings: &RaytracingSceneBindings,
) -> [&dyn SceneBuffer; SPARSE_BUFFER_COUNT] {
    [
        &bindings.assets.materials,
        &bindings.instances.transforms,
        &bindings.instances.previous_frame_transforms,
        &bindings.instances.geometry_ids,
        &bindings.instances.material_ids,
        &bindings.instances.blas_refs,
        &bindings.lights.sources,
        &bindings.lights.directional_lights,
        &bindings.lights.previous_frame_id_translations,
    ]
}

pub fn sparse_buffers_mut(
    bindings: &mut RaytracingSceneBindings,
) -> [&mut dyn SceneBuffer; SPARSE_BUFFER_COUNT] {
    [
        &mut bindings.assets.materials,
        &mut bindings.instances.transforms,
        &mut bindings.instances.previous_frame_transforms,
        &mut bindings.instances.geometry_ids,
        &mut bindings.instances.material_ids,
        &mut bindings.instances.blas_refs,
        &mut bindings.lights.sources,
        &mut bindings.lights.directional_lights,
        &mut bindings.lights.previous_frame_id_translations,
    ]
}

pub fn write_sparse_buffers(
    bindings: &mut RaytracingSceneBindings,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) {
    let _span = info_span!("write_buffers").entered();
    for buffer in sparse_buffers_mut(bindings) {
        buffer.grow_to(1);
        buffer.write(render_device, render_queue);
    }
}
