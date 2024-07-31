use super::{
    gpu_scene::{MeshletViewBindGroups, MeshletViewResources},
    pipelines::MeshletPipelines,
};
use crate::{LightEntity, ShadowView, ViewLightEntities};
use bevy_color::LinearRgba;
use bevy_core_pipeline::prepass::PreviousViewUniformOffset;
use bevy_ecs::{
    query::QueryState,
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    render_resource::*,
    renderer::RenderContext,
    view::{ViewDepthTexture, ViewUniformOffset},
};
use std::sync::atomic::Ordering;

/// Rasterize meshlets into a depth buffer, and optional visibility buffer + material depth buffer for shading passes.
pub struct MeshletVisibilityBufferRasterPassNode {
    main_view_query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
        &'static ViewLightEntities,
    )>,
    view_light_query: QueryState<(
        &'static ShadowView,
        &'static LightEntity,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    )>,
}

impl FromWorld for MeshletVisibilityBufferRasterPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for MeshletVisibilityBufferRasterPassNode {
    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

    // TODO: Reuse compute/render passes between logical passes where possible, as they're expensive
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Ok((
            camera,
            view_depth,
            view_offset,
            previous_view_offset,
            meshlet_view_bind_groups,
            meshlet_view_resources,
            lights,
        )) = self.main_view_query.get_manual(world, graph.view_entity())
        else {
            return Ok(());
        };

        let Some((
            fill_cluster_buffers_pipeline,
            culling_first_pipeline,
            culling_second_pipeline,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
            downsample_depth_first_shadow_view_pipeline,
            downsample_depth_second_shadow_view_pipeline,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_software_raster_depth_only_pipeline,
            visibility_buffer_software_raster_depth_only_clamp_ortho,
            visibility_buffer_hardware_raster_pipeline,
            visibility_buffer_hardware_raster_depth_only_pipeline,
            visibility_buffer_hardware_raster_depth_only_clamp_ortho,
            resolve_depth_pipeline,
            resolve_depth_shadow_view_pipeline,
            resolve_material_depth_pipeline,
            remap_1d_to_2d_dispatch_pipeline,
        )) = MeshletPipelines::get(world)
        else {
            return Ok(());
        };

        let first_node = meshlet_view_bind_groups
            .first_node
            .fetch_and(false, Ordering::SeqCst);

        let thread_per_cluster_workgroups =
            (meshlet_view_resources.scene_cluster_count.div_ceil(128) as f32)
                .cbrt()
                .ceil() as u32;

        render_context
            .command_encoder()
            .push_debug_group("meshlet_visibility_buffer_raster");
        render_context.command_encoder().clear_buffer(
            &meshlet_view_resources.second_pass_candidates_buffer,
            0,
            None,
        );
        if first_node {
            fill_cluster_buffers_pass(
                render_context,
                &meshlet_view_bind_groups.fill_cluster_buffers,
                fill_cluster_buffers_pipeline,
                thread_per_cluster_workgroups,
                meshlet_view_resources.scene_cluster_count,
            );
        }
        cull_pass(
            "culling_first",
            render_context,
            &meshlet_view_bind_groups.culling_first,
            view_offset,
            previous_view_offset,
            culling_first_pipeline,
            thread_per_cluster_workgroups,
            meshlet_view_resources.raster_cluster_rightmost_slot,
            meshlet_view_bind_groups
                .remap_1d_to_2d_dispatch
                .as_ref()
                .map(|(bg1, _)| bg1),
            remap_1d_to_2d_dispatch_pipeline,
        );
        raster_pass(
            true,
            render_context,
            &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_first,
            &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_first,
            &meshlet_view_resources.dummy_render_target.default_view,
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_hardware_raster_pipeline,
            Some(camera),
            meshlet_view_resources.raster_cluster_rightmost_slot,
        );
        downsample_depth(
            render_context,
            meshlet_view_resources,
            meshlet_view_bind_groups,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
        );
        cull_pass(
            "culling_second",
            render_context,
            &meshlet_view_bind_groups.culling_second,
            view_offset,
            previous_view_offset,
            culling_second_pipeline,
            thread_per_cluster_workgroups,
            meshlet_view_resources.raster_cluster_rightmost_slot,
            meshlet_view_bind_groups
                .remap_1d_to_2d_dispatch
                .as_ref()
                .map(|(_, bg2)| bg2),
            remap_1d_to_2d_dispatch_pipeline,
        );
        raster_pass(
            false,
            render_context,
            &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_second,
            &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_second,
            &meshlet_view_resources.dummy_render_target.default_view,
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_hardware_raster_pipeline,
            Some(camera),
            meshlet_view_resources.raster_cluster_rightmost_slot,
        );
        resolve_depth(
            render_context,
            view_depth.get_attachment(StoreOp::Store),
            meshlet_view_resources,
            meshlet_view_bind_groups,
            resolve_depth_pipeline,
            camera,
        );
        resolve_material_depth(
            render_context,
            meshlet_view_resources,
            meshlet_view_bind_groups,
            resolve_material_depth_pipeline,
            camera,
        );
        downsample_depth(
            render_context,
            meshlet_view_resources,
            meshlet_view_bind_groups,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
        );
        render_context.command_encoder().pop_debug_group();

        for light_entity in &lights.lights {
            let Ok((
                shadow_view,
                light_type,
                view_offset,
                previous_view_offset,
                meshlet_view_bind_groups,
                meshlet_view_resources,
            )) = self.view_light_query.get_manual(world, *light_entity)
            else {
                continue;
            };

            let (
                shadow_visibility_buffer_software_raster_pipeline,
                shadow_visibility_buffer_hardware_raster_pipeline,
            ) = match light_type {
                LightEntity::Directional { .. } => (
                    visibility_buffer_software_raster_depth_only_clamp_ortho,
                    visibility_buffer_hardware_raster_depth_only_clamp_ortho,
                ),
                _ => (
                    visibility_buffer_software_raster_depth_only_pipeline,
                    visibility_buffer_hardware_raster_depth_only_pipeline,
                ),
            };

            render_context.command_encoder().push_debug_group(&format!(
                "meshlet_visibility_buffer_raster: {}",
                shadow_view.pass_name
            ));
            render_context.command_encoder().clear_buffer(
                &meshlet_view_resources.second_pass_candidates_buffer,
                0,
                None,
            );
            cull_pass(
                "culling_first",
                render_context,
                &meshlet_view_bind_groups.culling_first,
                view_offset,
                previous_view_offset,
                culling_first_pipeline,
                thread_per_cluster_workgroups,
                meshlet_view_resources.raster_cluster_rightmost_slot,
                meshlet_view_bind_groups
                    .remap_1d_to_2d_dispatch
                    .as_ref()
                    .map(|(bg1, _)| bg1),
                remap_1d_to_2d_dispatch_pipeline,
            );
            raster_pass(
                true,
                render_context,
                &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_first,
                &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_first,
                &meshlet_view_resources.dummy_render_target.default_view,
                meshlet_view_bind_groups,
                view_offset,
                shadow_visibility_buffer_software_raster_pipeline,
                shadow_visibility_buffer_hardware_raster_pipeline,
                None,
                meshlet_view_resources.raster_cluster_rightmost_slot,
            );
            downsample_depth(
                render_context,
                meshlet_view_resources,
                meshlet_view_bind_groups,
                downsample_depth_first_shadow_view_pipeline,
                downsample_depth_second_shadow_view_pipeline,
            );
            cull_pass(
                "culling_second",
                render_context,
                &meshlet_view_bind_groups.culling_second,
                view_offset,
                previous_view_offset,
                culling_second_pipeline,
                thread_per_cluster_workgroups,
                meshlet_view_resources.raster_cluster_rightmost_slot,
                meshlet_view_bind_groups
                    .remap_1d_to_2d_dispatch
                    .as_ref()
                    .map(|(_, bg2)| bg2),
                remap_1d_to_2d_dispatch_pipeline,
            );
            raster_pass(
                false,
                render_context,
                &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_second,
                &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_second,
                &meshlet_view_resources.dummy_render_target.default_view,
                meshlet_view_bind_groups,
                view_offset,
                shadow_visibility_buffer_software_raster_pipeline,
                shadow_visibility_buffer_hardware_raster_pipeline,
                None,
                meshlet_view_resources.raster_cluster_rightmost_slot,
            );
            resolve_depth(
                render_context,
                shadow_view.depth_attachment.get_attachment(StoreOp::Store),
                meshlet_view_resources,
                meshlet_view_bind_groups,
                resolve_depth_shadow_view_pipeline,
                camera,
            );
            downsample_depth(
                render_context,
                meshlet_view_resources,
                meshlet_view_bind_groups,
                downsample_depth_first_shadow_view_pipeline,
                downsample_depth_second_shadow_view_pipeline,
            );
            render_context.command_encoder().pop_debug_group();
        }

        Ok(())
    }
}

fn fill_cluster_buffers_pass(
    render_context: &mut RenderContext,
    fill_cluster_buffers_bind_group: &BindGroup,
    fill_cluster_buffers_pass_pipeline: &ComputePipeline,
    fill_cluster_buffers_pass_workgroups: u32,
    cluster_count: u32,
) {
    let command_encoder = render_context.command_encoder();
    let mut fill_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fill_cluster_buffers"),
        timestamp_writes: None,
    });
    fill_pass.set_pipeline(fill_cluster_buffers_pass_pipeline);
    fill_pass.set_push_constants(0, &cluster_count.to_le_bytes());
    fill_pass.set_bind_group(0, fill_cluster_buffers_bind_group, &[]);
    fill_pass.dispatch_workgroups(
        fill_cluster_buffers_pass_workgroups,
        fill_cluster_buffers_pass_workgroups,
        fill_cluster_buffers_pass_workgroups,
    );
}

fn cull_pass(
    label: &'static str,
    render_context: &mut RenderContext,
    culling_bind_group: &BindGroup,
    view_offset: &ViewUniformOffset,
    previous_view_offset: &PreviousViewUniformOffset,
    culling_pipeline: &ComputePipeline,
    culling_workgroups: u32,
    raster_cluster_rightmost_slot: u32,
    remap_1d_to_2d_dispatch_bind_group: Option<&BindGroup>,
    remap_1d_to_2d_dispatch_pipeline: Option<&ComputePipeline>,
) {
    let command_encoder = render_context.command_encoder();
    let mut cull_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cull_pass.set_pipeline(culling_pipeline);
    cull_pass.set_push_constants(0, &raster_cluster_rightmost_slot.to_le_bytes());
    cull_pass.set_bind_group(
        0,
        culling_bind_group,
        &[view_offset.offset, previous_view_offset.offset],
    );
    cull_pass.dispatch_workgroups(culling_workgroups, culling_workgroups, culling_workgroups);

    if let (Some(remap_1d_to_2d_dispatch_pipeline), Some(remap_1d_to_2d_dispatch_bind_group)) = (
        remap_1d_to_2d_dispatch_pipeline,
        remap_1d_to_2d_dispatch_bind_group,
    ) {
        cull_pass.set_pipeline(remap_1d_to_2d_dispatch_pipeline);
        cull_pass.set_bind_group(0, remap_1d_to_2d_dispatch_bind_group, &[]);
        cull_pass.dispatch_workgroups(1, 1, 1);
    }
}

#[allow(clippy::too_many_arguments)]
fn raster_pass(
    first_pass: bool,
    render_context: &mut RenderContext,
    visibility_buffer_hardware_software_indirect_args: &Buffer,
    visibility_buffer_hardware_raster_indirect_args: &Buffer,
    dummy_render_target: &TextureView,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    visibility_buffer_hardware_software_pipeline: &ComputePipeline,
    visibility_buffer_hardware_raster_pipeline: &RenderPipeline,
    camera: Option<&ExtractedCamera>,
    raster_cluster_rightmost_slot: u32,
) {
    let command_encoder = render_context.command_encoder();
    let mut software_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(if first_pass {
            "raster_software_first"
        } else {
            "raster_software_second"
        }),
        timestamp_writes: None,
    });
    software_pass.set_pipeline(visibility_buffer_hardware_software_pipeline);
    software_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer_raster,
        &[view_offset.offset],
    );
    software_pass
        .dispatch_workgroups_indirect(visibility_buffer_hardware_software_indirect_args, 0);
    drop(software_pass);

    let mut hardware_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(if first_pass {
            "raster_hardware_first"
        } else {
            "raster_hardware_second"
        }),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: dummy_render_target,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::BLACK.into()),
                store: StoreOp::Discard,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = camera.and_then(|camera| camera.viewport.as_ref()) {
        hardware_pass.set_camera_viewport(viewport);
    }
    hardware_pass.set_render_pipeline(visibility_buffer_hardware_raster_pipeline);
    hardware_pass.set_push_constants(
        ShaderStages::VERTEX,
        0,
        &raster_cluster_rightmost_slot.to_le_bytes(),
    );
    hardware_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer_raster,
        &[view_offset.offset],
    );
    hardware_pass.draw_indirect(visibility_buffer_hardware_raster_indirect_args, 0);
}

fn downsample_depth(
    render_context: &mut RenderContext,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    downsample_depth_first_pipeline: &ComputePipeline,
    downsample_depth_second_pipeline: &ComputePipeline,
) {
    let command_encoder = render_context.command_encoder();
    let mut downsample_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("downsample_depth"),
        timestamp_writes: None,
    });
    downsample_pass.set_pipeline(downsample_depth_first_pipeline);
    downsample_pass.set_push_constants(
        0,
        bytemuck::cast_slice(&[
            meshlet_view_resources.depth_pyramid_mip_count,
            meshlet_view_resources.view_size.x,
        ]),
    );
    downsample_pass.set_bind_group(0, &meshlet_view_bind_groups.downsample_depth, &[]);
    downsample_pass.dispatch_workgroups(
        meshlet_view_resources.view_size.x.div_ceil(64),
        meshlet_view_resources.view_size.y.div_ceil(64),
        1,
    );

    if meshlet_view_resources.depth_pyramid_mip_count >= 7 {
        downsample_pass.set_pipeline(downsample_depth_second_pipeline);
        downsample_pass.dispatch_workgroups(1, 1, 1);
    }
}

fn resolve_depth(
    render_context: &mut RenderContext,
    depth_stencil_attachment: RenderPassDepthStencilAttachment,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    resolve_depth_pipeline: &RenderPipeline,
    camera: &ExtractedCamera,
) {
    let mut resolve_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("resolve_depth"),
        color_attachments: &[],
        depth_stencil_attachment: Some(depth_stencil_attachment),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = &camera.viewport {
        resolve_pass.set_camera_viewport(viewport);
    }
    resolve_pass.set_render_pipeline(resolve_depth_pipeline);
    resolve_pass.set_push_constants(
        ShaderStages::FRAGMENT,
        0,
        &meshlet_view_resources.view_size.x.to_le_bytes(),
    );
    resolve_pass.set_bind_group(0, &meshlet_view_bind_groups.resolve_depth, &[]);
    resolve_pass.draw(0..3, 0..1);
}

fn resolve_material_depth(
    render_context: &mut RenderContext,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    resolve_material_depth_pipeline: &RenderPipeline,
    camera: &ExtractedCamera,
) {
    if let (Some(material_depth), Some(resolve_material_depth_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.resolve_material_depth.as_ref(),
    ) {
        let mut resolve_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("resolve_material_depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &material_depth.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = &camera.viewport {
            resolve_pass.set_camera_viewport(viewport);
        }
        resolve_pass.set_render_pipeline(resolve_material_depth_pipeline);
        resolve_pass.set_push_constants(
            ShaderStages::FRAGMENT,
            0,
            &meshlet_view_resources.view_size.x.to_le_bytes(),
        );
        resolve_pass.set_bind_group(0, resolve_material_depth_bind_group, &[]);
        resolve_pass.draw(0..3, 0..1);
    }
}
