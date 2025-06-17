use super::{prepare::SolariLightingResources, SolariLighting};
use crate::scene::RaytracingSceneBindings;
use bevy_asset::load_embedded_asset;
use bevy_core_pipeline::prepass::{
    PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms, ViewPrepassTextures,
};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    query::QueryItem,
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_depth_2d,
            texture_storage_2d, uniform_buffer,
        },
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache,
        PushConstantRange, ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};

pub mod graph {
    use bevy_render::render_graph::RenderLabel;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct SolariLightingNode;
}

pub struct SolariLightingNode {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl ViewNode for SolariLightingNode {
    type ViewQuery = (
        &'static SolariLighting,
        &'static SolariLightingResources,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            solari_lighting,
            solari_lighting_resources,
            camera,
            view_target,
            view_prepass_textures,
            view_uniform_offset,
            previous_view_uniform_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let scene_bindings = world.resource::<RaytracingSceneBindings>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let previous_view_uniforms = world.resource::<PreviousViewUniforms>();
        let frame_count = world.resource::<FrameCount>();
        let (
            Some(pipeline),
            Some(scene_bindings),
            Some(viewport),
            Some(gbuffer),
            Some(depth_buffer),
            Some(motion_vectors),
            Some(view_uniforms),
            Some(previous_view_uniforms),
        ) = (
            pipeline_cache.get_compute_pipeline(self.pipeline),
            &scene_bindings.bind_group,
            camera.physical_viewport_size,
            view_prepass_textures.deferred_view(),
            view_prepass_textures.depth_view(),
            view_prepass_textures.motion_vectors_view(),
            view_uniforms.uniforms.binding(),
            previous_view_uniforms.uniforms.binding(),
        )
        else {
            return Ok(());
        };

        let (reservoirs, previous_reservoirs) = if frame_count.0 % 2 == 0 {
            (
                &solari_lighting_resources.reservoirs_a,
                &solari_lighting_resources.reservoirs_b,
            )
        } else {
            (
                &solari_lighting_resources.reservoirs_b,
                &solari_lighting_resources.reservoirs_a,
            )
        };

        let bind_group = render_context.render_device().create_bind_group(
            "solari_lighting_bind_group",
            &self.bind_group_layout,
            &BindGroupEntries::sequential((
                view_target.get_unsampled_color_attachment().view,
                previous_reservoirs.as_entire_binding(),
                reservoirs.as_entire_binding(),
                gbuffer,
                depth_buffer,
                motion_vectors,
                &solari_lighting_resources.previous_gbuffer.1,
                &solari_lighting_resources.previous_depth.1,
                view_uniforms,
                previous_view_uniforms,
                &solari_lighting_resources.accumulation_texture,
            )),
        );

        let frame_index = frame_count.0.wrapping_mul(5782582);
        let command_encoder = render_context.command_encoder();

        {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("solari_lighting"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, scene_bindings, &[]);
            pass.set_bind_group(
                1,
                &bind_group,
                &[
                    view_uniform_offset.offset,
                    previous_view_uniform_offset.offset,
                ],
            );
            pass.set_push_constants(
                0,
                bytemuck::cast_slice(&[frame_index, solari_lighting.reset as u32]),
            );
            pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
        }

        // TODO: Remove these copies, and double buffer instead
        command_encoder.copy_texture_to_texture(
            view_prepass_textures
                .deferred
                .clone()
                .unwrap()
                .texture
                .texture
                .as_image_copy(),
            solari_lighting_resources.previous_gbuffer.0.as_image_copy(),
            Extent3d {
                width: viewport.x,
                height: viewport.y,
                depth_or_array_layers: 1,
            },
        );
        command_encoder.copy_texture_to_texture(
            view_prepass_textures
                .depth
                .clone()
                .unwrap()
                .texture
                .texture
                .as_image_copy(),
            solari_lighting_resources.previous_depth.0.as_image_copy(),
            Extent3d {
                width: viewport.x,
                height: viewport.y,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }
}

impl FromWorld for SolariLightingNode {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let scene_bindings = world.resource::<RaytracingSceneBindings>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "solari_lighting_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(
                        ViewTarget::TEXTURE_FORMAT_HDR,
                        StorageTextureAccess::WriteOnly,
                    ),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    texture_2d(TextureSampleType::Uint),
                    texture_depth_2d(),
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Uint),
                    texture_depth_2d(),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<PreviousViewData>(true),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                ),
            ),
        );

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("solari_lighting_restir_di_pipeline".into()),
            layout: vec![
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..8,
            }],
            shader: load_embedded_asset!(world, "restir_di.wgsl"),
            shader_defs: vec![],
            entry_point: "restir_di".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}
