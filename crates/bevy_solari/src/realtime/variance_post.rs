//! Debug tooling: post-denoise temporal-variance tap and heatmap present pass.
//!
//! Companion to the pre-denoise variance accumulation done inside the main
//! Solari lighting node (see `variance_accumulate.wgsl`). This node runs after
//! DLSS Ray Reconstruction (or, with no denoiser, straight after the main pass)
//! and before tonemapping, at output resolution: it measures the temporal
//! variance of the *denoised* signal, publishes its stats as diagnostics, and
//! draws the selected heatmap (pre- or post-denoise) into the view target. See
//! [`SolariVarianceDebug`](super::SolariVarianceDebug) and `variance.wgsl`.

use super::{prepare::SolariLightingResources, SolariVarianceDebug};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, Query, Res},
};
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics as _,
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_depth_2d,
            texture_storage_2d, uniform_buffer_sized,
        },
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
        TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::ViewTarget,
};
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};

/// Size of a `VarianceMoments` (`vec4<f32>`) entry, in bytes (see `variance.wgsl`).
const VARIANCE_MOMENTS_STRUCT_SIZE: u64 = 16;
/// Size of the `VarianceStats` shader struct, in bytes.
const VARIANCE_STATS_STRUCT_SIZE: u64 = 16;

/// GPU uniform for the present pass. Field order/types must match
/// `VariancePostUniforms` in `variance_present.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VariancePostUniforms {
    render_size: [u32; 2],
    output_size: [u32; 2],
    mode: u32,
    threshold: f32,
    history_length: f32,
    reset: u32,
}

/// Render pipeline + bind group layout for the post-denoise variance present pass.
#[derive(Resource)]
pub struct SolariVariancePostPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

/// Per-view resources for the post-denoise variance present pass. Present only
/// while the [`SolariVarianceDebug`] component is on the camera.
#[derive(bevy_ecs::component::Component)]
pub struct SolariVariancePostResources {
    uniforms: Buffer,
    /// Ping-ponged output-resolution moments buffers (see the main node's docs).
    moments_a: Buffer,
    moments_b: Buffer,
    /// Global stats reduction, read back to the CPU via the diagnostics path.
    stats: Buffer,
    output_size: UVec2,
}

/// Initializes the post-denoise variance present pipeline at render startup.
pub fn init_solari_variance_post_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "solari_variance_post_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // view target (read denoised color, write heatmap)
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                texture_depth_2d(),                                        // depth
                texture_2d(TextureSampleType::Float { filterable: true }), // motion vectors
                texture_depth_2d(),                                        // previous depth
                storage_buffer_read_only_sized(false, None),               // post moments read
                storage_buffer_sized(false, None),                         // post moments write
                storage_buffer_sized(false, None),                         // post stats
                storage_buffer_read_only_sized(false, None),               // pre moments (display)
                uniform_buffer_sized(false, None),
            ),
        ),
    );

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_variance_post_present_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "variance_present.wgsl"),
        entry_point: Some("variance_present".into()),
        ..default()
    });

    commands.insert_resource(SolariVariancePostPipeline {
        bind_group_layout,
        pipeline,
    });
}

/// Creates/resizes the per-view post-denoise variance resources, mirroring the
/// (optional) [`SolariVarianceDebug`] settings into the uniform every frame.
pub fn prepare_solari_variance_post_resources(
    query: Query<(
        Entity,
        &ExtractedCamera,
        Option<&SolariVarianceDebug>,
        Option<&MainPassResolutionOverride>,
        Option<&SolariVariancePostResources>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    for (entity, camera, variance_debug, resolution_override, resources) in &query {
        let Some(variance_debug) = variance_debug else {
            // Tooling off: drop any resources so the node no-ops for this view.
            if resources.is_some() {
                commands
                    .entity(entity)
                    .remove::<SolariVariancePostResources>();
            }
            continue;
        };

        let Some(output_size) = camera.physical_viewport_size else {
            continue;
        };
        let render_size = resolution_override
            .map(|r| r.0)
            .unwrap_or(output_size);

        let uniforms = VariancePostUniforms {
            render_size: render_size.to_array(),
            output_size: output_size.to_array(),
            mode: variance_debug.mode as u32,
            threshold: variance_debug.threshold,
            history_length: variance_debug.history_length,
            // Reset the history the frame the buffers are (re)created; disocclusion
            // handles the rest.
            reset: resources.is_none_or(|r| r.output_size != output_size) as u32,
        };

        if let Some(resources) = resources
            && resources.output_size == output_size
        {
            render_queue.write_buffer(&resources.uniforms, 0, bytemuck::bytes_of(&uniforms));
            continue;
        }

        let uniforms_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_variance_post_uniforms"),
            size: size_of::<VariancePostUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        let moments_size =
            (output_size.x * output_size.y) as u64 * VARIANCE_MOMENTS_STRUCT_SIZE;
        let moments_buffer = |name| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: moments_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        let stats = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_variance_post_stats"),
            size: VARIANCE_STATS_STRUCT_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands
            .entity(entity)
            .insert(SolariVariancePostResources {
                uniforms: uniforms_buffer,
                moments_a: moments_buffer("solari_variance_post_moments_a"),
                moments_b: moments_buffer("solari_variance_post_moments_b"),
                stats,
                output_size,
            });
    }
}

type SolariVariancePostViewQuery = (
    &'static SolariVariancePostResources,
    &'static SolariLightingResources,
    &'static ViewTarget,
    &'static ViewPrepassTextures,
);

/// Post-denoise variance tap + heatmap present. No-ops for views without the
/// debug tooling (no [`SolariVariancePostResources`]).
pub fn solari_variance_post(
    view: ViewQuery<SolariVariancePostViewQuery>,
    pipeline: Option<Res<SolariVariancePostPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    mut ctx: RenderContext,
) {
    let (post, solari, view_target, prepass_textures) = view.into_inner();

    let Some(pipeline) = pipeline else {
        return;
    };
    let (
        Some(present_pipeline),
        Some(depth),
        Some(motion_vectors),
        Some(previous_depth),
    ) = (
        pipeline_cache.get_compute_pipeline(pipeline.pipeline),
        prepass_textures.depth_view(),
        prepass_textures.motion_vectors_view(),
        prepass_textures.previous_depth_view(),
    ) else {
        return;
    };

    // Ping-pong the output-res moments by frame parity, matching the pre-denoise
    // buffers. The pre-denoise moments to *display* are the buffer the main node
    // wrote this frame (its write target), selected by the same parity.
    let even = frame_count.0 & 1 == 0;
    let (moments_read, moments_write) = if even {
        (&post.moments_a, &post.moments_b)
    } else {
        (&post.moments_b, &post.moments_a)
    };
    let pre_moments = if even {
        &solari.variance_moments_b
    } else {
        &solari.variance_moments_a
    };

    let view_target_attachment = view_target.get_unsampled_color_attachment();
    let output_size = post.output_size;

    let bind_group = render_device.create_bind_group(
        "solari_variance_post_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
        &BindGroupEntries::sequential((
            view_target_attachment.view,
            depth,
            motion_vectors,
            previous_depth,
            moments_read.as_entire_binding(),
            moments_write.as_entire_binding(),
            post.stats.as_entire_binding(),
            pre_moments.as_entire_binding(),
            post.uniforms.as_entire_binding(),
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let command_encoder = ctx.command_encoder();
    // Zero the per-frame stats before the atomics accumulate this frame's pixels.
    command_encoder.clear_buffer(&post.stats, 0, None);

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("solari_variance_post"),
        timestamp_writes: None,
    });
    let d = diagnostics.time_span(&mut pass, "solari_lighting/variance_post");
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_pipeline(present_pipeline);
    pass.dispatch_workgroups(output_size.x.div_ceil(8), output_size.y.div_ceil(8), 1);
    d.end(&mut pass);
    drop(pass);

    // Publish the post-denoise stats as diagnostics for the HUD readback (see
    // `VarianceStats` in `variance.wgsl` for layout/units).
    let command_encoder = ctx.command_encoder();
    diagnostics.record_u32(
        command_encoder,
        &post.stats.slice(0..4),
        "solari_lighting/variance_post/sum_relative_fixed",
    );
    diagnostics.record_u32(
        command_encoder,
        &post.stats.slice(4..8),
        "solari_lighting/variance_post/max_relative_bits",
    );
    diagnostics.record_u32(
        command_encoder,
        &post.stats.slice(8..12),
        "solari_lighting/variance_post/count_over_threshold",
    );
    diagnostics.record_u32(
        command_encoder,
        &post.stats.slice(12..16),
        "solari_lighting/variance_post/valid_count",
    );
}
