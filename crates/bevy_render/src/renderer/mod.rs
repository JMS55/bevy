mod graph_runner;
mod render_device;

pub use graph_runner::*;
pub use render_device::*;

use crate::{
    render_graph::RenderGraph,
    render_phase::TrackedRenderPass,
    render_resource::RenderPassDescriptor,
    settings::{WgpuSettings, WgpuSettingsPriority},
    view::{ExtractedWindows, ViewTarget},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::prelude::*;
use bevy_time::TimeSender;
use bevy_utils::tracing::{error, info, info_span};
use bevy_utils::Instant;
use std::{
    mem,
    sync::{Arc, Mutex},
};
use wgpu::{
    Adapter, AdapterInfo, CommandBuffer, CommandEncoder, Features, Instance, Maintain, Queue,
    RequestAdapterOptions,
};
use wgpu_profiler::{GpuProfiler, GpuTimerScopeResult};

/// Updates the [`RenderGraph`] with all of its nodes and then runs it to render the entire frame.
pub fn render_system(world: &mut World) {
    world.resource_scope(|world, mut graph: Mut<RenderGraph>| {
        graph.update(world);
    });
    let graph = world.resource::<RenderGraph>();
    let render_device = world.resource::<RenderDevice>();
    let render_queue = world.resource::<RenderQueue>();

    match RenderGraphRunner::run(
        graph,
        render_device.clone(), // TODO: is this clone really necessary?
        &render_queue.0,
        world,
    ) {
        Ok(Some(gpu_timers)) => world.resource_mut::<GpuTimerScopes>().set(gpu_timers),
        Ok(None) => {}
        Err(e) => {
            error!("Error running render graph:");
            {
                let mut src: &dyn std::error::Error = &e;
                loop {
                    error!("> {}", src);
                    match src.source() {
                        Some(s) => src = s,
                        None => break,
                    }
                }
            }

            panic!("Error running render graph: {e}");
        }
    }

    {
        let _span = info_span!("present_frames").entered();

        // Remove ViewTarget components to ensure swap chain TextureViews are dropped.
        // If all TextureViews aren't dropped before present, acquiring the next swap chain texture will fail.
        let view_entities = world
            .query_filtered::<Entity, With<ViewTarget>>()
            .iter(world)
            .collect::<Vec<_>>();
        for view_entity in view_entities {
            world.entity_mut(view_entity).remove::<ViewTarget>();
        }

        let mut windows = world.resource_mut::<ExtractedWindows>();
        for window in windows.values_mut() {
            if let Some(texture_view) = window.swap_chain_texture.take() {
                if let Some(surface_texture) = texture_view.take_surface_texture() {
                    surface_texture.present();
                }
            }
        }

        #[cfg(feature = "tracing-tracy")]
        bevy_utils::tracing::event!(
            bevy_utils::tracing::Level::INFO,
            message = "finished frame",
            tracy.frame_mark = true
        );
    }

    // update the time and send it to the app world
    let time_sender = world.resource::<TimeSender>();
    time_sender.0.try_send(Instant::now()).expect(
        "The TimeSender channel should always be empty during render. You might need to add the bevy::core::time_system to your app.",
    );
}

/// This queue is used to enqueue tasks for the GPU to execute asynchronously.
#[derive(Resource, Clone, Deref, DerefMut)]
pub struct RenderQueue(pub Arc<Queue>);

/// The handle to the physical device being used for rendering.
/// See [`wgpu::Adapter`] for more info.
#[derive(Resource, Clone, Debug, Deref, DerefMut)]
pub struct RenderAdapter(pub Arc<Adapter>);

/// The GPU instance is used to initialize the [`RenderQueue`] and [`RenderDevice`],
/// as well as to create [`WindowSurfaces`](crate::view::window::WindowSurfaces).
#[derive(Resource, Deref, DerefMut)]
pub struct RenderInstance(pub Instance);

/// The `AdapterInfo` of the adapter in use by the renderer.
#[derive(Resource, Clone, Deref, DerefMut)]
pub struct RenderAdapterInfo(pub AdapterInfo);

const GPU_NOT_FOUND_ERROR_MESSAGE: &str = if cfg!(target_os = "linux") {
    "Unable to find a GPU! Make sure you have installed required drivers! For extra information, see: https://github.com/bevyengine/bevy/blob/latest/docs/linux_dependencies.md"
} else {
    "Unable to find a GPU! Make sure you have installed required drivers!"
};

/// Initializes the renderer by retrieving and preparing the GPU instance, device and queue
/// for the specified backend.
pub async fn initialize_renderer(
    instance: &Instance,
    options: &WgpuSettings,
    request_adapter_options: &RequestAdapterOptions<'_>,
) -> (RenderDevice, RenderQueue, RenderAdapterInfo, RenderAdapter) {
    let adapter = instance
        .request_adapter(request_adapter_options)
        .await
        .expect(GPU_NOT_FOUND_ERROR_MESSAGE);

    let adapter_info = adapter.get_info();
    info!("{:?}", adapter_info);

    #[cfg(feature = "wgpu_trace")]
    let trace_path = {
        let path = std::path::Path::new("wgpu_trace");
        // ignore potential error, wgpu will log it
        let _ = std::fs::create_dir(path);
        Some(path)
    };
    #[cfg(not(feature = "wgpu_trace"))]
    let trace_path = None;

    // Maybe get features and limits based on what is supported by the adapter/backend
    let mut features = wgpu::Features::empty();
    let mut limits = options.limits.clone();
    if matches!(options.priority, WgpuSettingsPriority::Functionality) {
        features = adapter.features();
        if adapter_info.device_type == wgpu::DeviceType::DiscreteGpu {
            // `MAPPABLE_PRIMARY_BUFFERS` can have a significant, negative performance impact for
            // discrete GPUs due to having to transfer data across the PCI-E bus and so it
            // should not be automatically enabled in this case. It is however beneficial for
            // integrated GPUs.
            features -= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        // TIMESTAMP_QUERY indicates to time GPU passes, which affects performance. Don't enable by default.
        features -= wgpu::Features::TIMESTAMP_QUERY;
        limits = adapter.limits();
    }

    // Enforce the disabled features
    if let Some(disabled_features) = options.disabled_features {
        features -= disabled_features;
    }
    // NOTE: |= is used here to ensure that any explicitly-enabled features are respected.
    features |= options.features;

    // Enforce the limit constraints
    if let Some(constrained_limits) = options.constrained_limits.as_ref() {
        // NOTE: Respect the configured limits as an 'upper bound'. This means for 'max' limits, we
        // take the minimum of the calculated limits according to the adapter/backend and the
        // specified max_limits. For 'min' limits, take the maximum instead. This is intended to
        // err on the side of being conservative. We can't claim 'higher' limits that are supported
        // but we can constrain to 'lower' limits.
        limits = wgpu::Limits {
            max_texture_dimension_1d: limits
                .max_texture_dimension_1d
                .min(constrained_limits.max_texture_dimension_1d),
            max_texture_dimension_2d: limits
                .max_texture_dimension_2d
                .min(constrained_limits.max_texture_dimension_2d),
            max_texture_dimension_3d: limits
                .max_texture_dimension_3d
                .min(constrained_limits.max_texture_dimension_3d),
            max_texture_array_layers: limits
                .max_texture_array_layers
                .min(constrained_limits.max_texture_array_layers),
            max_bind_groups: limits
                .max_bind_groups
                .min(constrained_limits.max_bind_groups),
            max_dynamic_uniform_buffers_per_pipeline_layout: limits
                .max_dynamic_uniform_buffers_per_pipeline_layout
                .min(constrained_limits.max_dynamic_uniform_buffers_per_pipeline_layout),
            max_dynamic_storage_buffers_per_pipeline_layout: limits
                .max_dynamic_storage_buffers_per_pipeline_layout
                .min(constrained_limits.max_dynamic_storage_buffers_per_pipeline_layout),
            max_sampled_textures_per_shader_stage: limits
                .max_sampled_textures_per_shader_stage
                .min(constrained_limits.max_sampled_textures_per_shader_stage),
            max_samplers_per_shader_stage: limits
                .max_samplers_per_shader_stage
                .min(constrained_limits.max_samplers_per_shader_stage),
            max_storage_buffers_per_shader_stage: limits
                .max_storage_buffers_per_shader_stage
                .min(constrained_limits.max_storage_buffers_per_shader_stage),
            max_storage_textures_per_shader_stage: limits
                .max_storage_textures_per_shader_stage
                .min(constrained_limits.max_storage_textures_per_shader_stage),
            max_uniform_buffers_per_shader_stage: limits
                .max_uniform_buffers_per_shader_stage
                .min(constrained_limits.max_uniform_buffers_per_shader_stage),
            max_uniform_buffer_binding_size: limits
                .max_uniform_buffer_binding_size
                .min(constrained_limits.max_uniform_buffer_binding_size),
            max_storage_buffer_binding_size: limits
                .max_storage_buffer_binding_size
                .min(constrained_limits.max_storage_buffer_binding_size),
            max_vertex_buffers: limits
                .max_vertex_buffers
                .min(constrained_limits.max_vertex_buffers),
            max_vertex_attributes: limits
                .max_vertex_attributes
                .min(constrained_limits.max_vertex_attributes),
            max_vertex_buffer_array_stride: limits
                .max_vertex_buffer_array_stride
                .min(constrained_limits.max_vertex_buffer_array_stride),
            max_push_constant_size: limits
                .max_push_constant_size
                .min(constrained_limits.max_push_constant_size),
            min_uniform_buffer_offset_alignment: limits
                .min_uniform_buffer_offset_alignment
                .max(constrained_limits.min_uniform_buffer_offset_alignment),
            min_storage_buffer_offset_alignment: limits
                .min_storage_buffer_offset_alignment
                .max(constrained_limits.min_storage_buffer_offset_alignment),
            max_inter_stage_shader_components: limits
                .max_inter_stage_shader_components
                .min(constrained_limits.max_inter_stage_shader_components),
            max_compute_workgroup_storage_size: limits
                .max_compute_workgroup_storage_size
                .min(constrained_limits.max_compute_workgroup_storage_size),
            max_compute_invocations_per_workgroup: limits
                .max_compute_invocations_per_workgroup
                .min(constrained_limits.max_compute_invocations_per_workgroup),
            max_compute_workgroup_size_x: limits
                .max_compute_workgroup_size_x
                .min(constrained_limits.max_compute_workgroup_size_x),
            max_compute_workgroup_size_y: limits
                .max_compute_workgroup_size_y
                .min(constrained_limits.max_compute_workgroup_size_y),
            max_compute_workgroup_size_z: limits
                .max_compute_workgroup_size_z
                .min(constrained_limits.max_compute_workgroup_size_z),
            max_compute_workgroups_per_dimension: limits
                .max_compute_workgroups_per_dimension
                .min(constrained_limits.max_compute_workgroups_per_dimension),
            max_buffer_size: limits
                .max_buffer_size
                .min(constrained_limits.max_buffer_size),
            max_bindings_per_bind_group: limits
                .max_bindings_per_bind_group
                .min(constrained_limits.max_bindings_per_bind_group),
        };
    }

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: options.device_label.as_ref().map(|a| a.as_ref()),
                features,
                limits,
            },
            trace_path,
        )
        .await
        .unwrap();
    let queue = Arc::new(queue);
    let adapter = Arc::new(adapter);
    (
        RenderDevice::from(device),
        RenderQueue(queue),
        RenderAdapterInfo(adapter_info),
        RenderAdapter(adapter),
    )
}

/// The context with all information required to interact with the GPU.
///
/// The [`RenderDevice`] is used to create render resources and the
/// the [`CommandEncoder`] is used to record a series of GPU operations.
pub struct RenderContext {
    render_device: RenderDevice,
    command_encoder: Option<CommandEncoder>,
    command_buffers: Vec<CommandBuffer>,
    gpu_profiler: Option<GpuProfiler>,
}

impl RenderContext {
    /// Creates a new [`RenderContext`] from a [`RenderDevice`].
    pub fn new(render_device: RenderDevice, render_queue: &Queue) -> Self {
        let mut gpu_profiler = None;
        let usable_features = render_device.features();
        if usable_features.contains(Features::TIMESTAMP_QUERY) {
            gpu_profiler = Some(GpuProfiler::new(
                4,
                render_queue.get_timestamp_period(),
                usable_features,
            ))
        }

        Self {
            render_device,
            command_encoder: None,
            command_buffers: Vec::new(),
            gpu_profiler,
        }
    }

    /// Gets the underlying [`RenderDevice`].
    pub fn render_device(&self) -> &RenderDevice {
        &self.render_device
    }

    /// Gets the current [`CommandEncoder`].
    pub fn command_encoder(&mut self) -> &mut CommandEncoder {
        self.command_encoder.get_or_insert_with(|| {
            self.render_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        })
    }

    /// Creates a new [`TrackedRenderPass`] for the context,
    /// configured using the provided `descriptor`.
    pub fn begin_tracked_render_pass<'a>(
        &'a mut self,
        descriptor: RenderPassDescriptor<'a, '_>,
    ) -> TrackedRenderPass<'a> {
        // Cannot use command_encoder() as we need to split the borrow on self
        let command_encoder = self.command_encoder.get_or_insert_with(|| {
            self.render_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        });
        let render_pass = command_encoder.begin_render_pass(&descriptor);
        TrackedRenderPass::new(&self.render_device, render_pass)
    }

    /// Append a [`CommandBuffer`] to the queue.
    ///
    /// If present, this will flush the currently unflushed [`CommandEncoder`]
    /// into a [`CommandBuffer`] into the queue before append the provided
    /// buffer.
    pub fn add_command_buffer(&mut self, command_buffer: CommandBuffer) {
        self.flush_command_encoder();
        self.command_buffers.push(command_buffer);
    }

    pub fn begin_debug_scope(&mut self, label: &str) {
        if let Some(gpu_profiler) = self.gpu_profiler.as_mut() {
            let command_encoder = self.command_encoder.get_or_insert_with(|| {
                self.render_device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
            });

            gpu_profiler.begin_scope(label, command_encoder, self.render_device.wgpu_device());
        }
    }

    pub fn end_debug_scope(&mut self) {
        if let Some(gpu_profiler) = self.gpu_profiler.as_mut() {
            gpu_profiler.end_scope(self.command_encoder.as_mut().unwrap());
        }
    }

    pub fn finish(
        mut self,
        queue: &Queue,
        render_device: &RenderDevice,
    ) -> Option<Vec<GpuTimerScopeResult>> {
        self.flush_command_encoder();

        let submission_index = {
            #[cfg(feature = "trace")]
            let _span = info_span!("submit_graph_commands").entered();
            queue.submit(self.command_buffers)
        };

        if let Some(gpu_profiler) = self.gpu_profiler.as_mut() {
            gpu_profiler.end_frame().unwrap();
            // This poll is not optimal, but unsure how to ensure buffers are mapped otherwise
            render_device.poll(Maintain::WaitForSubmissionIndex(submission_index));
            return gpu_profiler.process_finished_frame();
        }

        None
    }

    fn flush_command_encoder(&mut self) {
        if let Some(mut command_encoder) = self.command_encoder.take() {
            if let Some(gpu_profiler) = self.gpu_profiler.as_mut() {
                gpu_profiler.resolve_queries(&mut command_encoder);
            }

            self.command_buffers.push(command_encoder.finish());
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct GpuTimerScopes(Arc<Mutex<Vec<GpuTimerScopeResult>>>);

impl GpuTimerScopes {
    pub fn set(&self, gpu_timers: Vec<GpuTimerScopeResult>) {
        *self.0.lock().unwrap() = gpu_timers;
    }

    pub fn take(&self) -> Vec<GpuTimerScopeResult> {
        mem::take(&mut self.0.lock().unwrap())
    }
}
