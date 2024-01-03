use super::{
    gpu_scene::{MeshletViewBindGroups, MeshletViewResources},
    pipelines::MeshletPipelines,
};
use crate::{LightEntity, ShadowView, ViewLightEntities};
use bevy_ecs::{
    query::QueryState,
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    color::Color,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    render_resource::*,
    renderer::RenderContext,
    view::{ViewDepthTexture, ViewUniformOffset},
};

pub mod draw_3d_graph {
    pub mod node {
        pub const MESHLET_VISIBILITY_BUFFER_PASS: &str = "meshlet_visibility_buffer_pass";
    }
}

pub struct MeshletVisibilityBufferPassNode {
    main_view_query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
        &'static ViewLightEntities,
    )>,
    view_light_query: QueryState<(
        &'static ShadowView,
        &'static LightEntity,
        &'static ViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    )>,
}

impl FromWorld for MeshletVisibilityBufferPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for MeshletVisibilityBufferPassNode {
    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

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
            meshlet_view_bind_groups,
            meshlet_view_resources,
            lights,
        )) = self.main_view_query.get_manual(world, graph.view_entity())
        else {
            return Ok(());
        };

        let Some((
            culling_first_pipeline,
            culling_second_pipeline,
            downsample_depth_pipeline,
            visibility_buffer_pipeline,
            visibility_buffer_with_output_pipeline,
            visibility_buffer_depth_clamp_ortho,
            copy_material_depth_pipeline,
        )) = MeshletPipelines::get(world)
        else {
            return Ok(());
        };

        let culling_workgroups = (meshlet_view_resources.scene_meshlet_count + 127) / 128;

        render_context
            .command_encoder()
            .push_debug_group(draw_3d_graph::node::MESHLET_VISIBILITY_BUFFER_PASS);
        cull_first_pass(
            render_context,
            meshlet_view_bind_groups,
            view_offset,
            culling_first_pipeline,
            culling_workgroups,
        );
        draw_first_pass(
            render_context,
            view_depth.get_attachment(StoreOp::Store),
            Some(camera),
            meshlet_view_resources,
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_pipeline,
        );
        downsample_depth(
            render_context,
            meshlet_view_resources,
            meshlet_view_bind_groups,
            downsample_depth_pipeline,
        );
        cull_second_pass(
            render_context,
            meshlet_view_bind_groups,
            view_offset,
            culling_second_pipeline,
            culling_workgroups,
        );
        draw_second_pass(
            meshlet_view_resources,
            render_context,
            view_depth.get_attachment(StoreOp::Store),
            Some(camera),
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_with_output_pipeline,
        );
        copy_material_depth_pass(
            meshlet_view_resources,
            meshlet_view_bind_groups,
            render_context,
            camera,
            copy_material_depth_pipeline,
        );
        render_context.command_encoder().pop_debug_group();

        for light_entity in &lights.lights {
            let Ok((
                shadow_view,
                light_type,
                view_offset,
                meshlet_view_bind_groups,
                meshlet_view_resources,
            )) = self.view_light_query.get_manual(world, *light_entity)
            else {
                continue;
            };

            let shadow_visibility_buffer_pipeline = match light_type {
                LightEntity::Directional { .. } => visibility_buffer_depth_clamp_ortho,
                _ => visibility_buffer_pipeline,
            };

            render_context.command_encoder().push_debug_group(&format!(
                "{}: {}",
                draw_3d_graph::node::MESHLET_VISIBILITY_BUFFER_PASS,
                shadow_view.pass_name
            ));
            cull_first_pass(
                render_context,
                meshlet_view_bind_groups,
                view_offset,
                culling_first_pipeline,
                culling_workgroups,
            );
            draw_first_pass(
                render_context,
                shadow_view.depth_attachment.get_attachment(StoreOp::Store),
                None,
                meshlet_view_resources,
                meshlet_view_bind_groups,
                view_offset,
                shadow_visibility_buffer_pipeline,
            );
            downsample_depth(
                render_context,
                meshlet_view_resources,
                meshlet_view_bind_groups,
                downsample_depth_pipeline,
            );
            cull_second_pass(
                render_context,
                meshlet_view_bind_groups,
                view_offset,
                culling_second_pipeline,
                culling_workgroups,
            );
            draw_second_pass(
                meshlet_view_resources,
                render_context,
                shadow_view.depth_attachment.get_attachment(StoreOp::Store),
                None,
                meshlet_view_bind_groups,
                view_offset,
                shadow_visibility_buffer_pipeline,
            );
            render_context.command_encoder().pop_debug_group();
        }

        Ok(())
    }
}

fn cull_first_pass(
    render_context: &mut RenderContext,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    culling_first_pipeline: &ComputePipeline,
    culling_workgroups: u32,
) {
    let command_encoder = render_context.command_encoder();
    let mut cull_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("meshlet_culling_first_pass"),
        timestamp_writes: None,
    });
    cull_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.culling_first,
        &[view_offset.offset],
    );
    cull_pass.set_pipeline(culling_first_pipeline);
    cull_pass.dispatch_workgroups(culling_workgroups, 1, 1);
}

fn draw_first_pass(
    render_context: &mut RenderContext,
    depth_stencil_attachment: RenderPassDepthStencilAttachment,
    camera: Option<&ExtractedCamera>,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    first_pass_visibility_buffer_pipeline: &RenderPipeline,
) {
    let mut draw_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meshlet_visibility_buffer_first_pass"),
        color_attachments: &[],
        depth_stencil_attachment: Some(depth_stencil_attachment),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = camera.and_then(|camera| camera.viewport.as_ref()) {
        draw_pass.set_camera_viewport(viewport);
    }

    draw_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer,
        &[view_offset.offset],
    );
    draw_pass.set_render_pipeline(first_pass_visibility_buffer_pipeline);
    draw_pass.draw_indirect(
        &meshlet_view_resources.visibility_buffer_draw_command_buffer_first,
        0,
    );
}

fn downsample_depth(
    render_context: &mut RenderContext,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    downsample_depth_pipeline: &RenderPipeline,
) {
    render_context
        .command_encoder()
        .push_debug_group("meshlet_downsample_depth");

    for i in 0..meshlet_view_resources.depth_pyramid_mips.len() {
        let downsample_pass = RenderPassDescriptor {
            label: Some("meshlet_downsample_depth_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &meshlet_view_resources.depth_pyramid_mips[i],
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK.into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut downsample_pass = render_context.begin_tracked_render_pass(downsample_pass);
        downsample_pass.set_bind_group(0, &meshlet_view_bind_groups.downsample_depth[i], &[]);
        downsample_pass.set_render_pipeline(downsample_depth_pipeline);
        downsample_pass.draw(0..3, 0..1);
    }

    render_context.command_encoder().pop_debug_group();
}

fn cull_second_pass(
    render_context: &mut RenderContext,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    culling_second_pipeline: &ComputePipeline,
    culling_workgroups: u32,
) {
    let command_encoder = render_context.command_encoder();
    let mut cull_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("meshlet_culling_second_pass"),
        timestamp_writes: None,
    });
    cull_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.culling_second,
        &[view_offset.offset],
    );
    cull_pass.set_pipeline(culling_second_pipeline);
    cull_pass.dispatch_workgroups(culling_workgroups, 1, 1);
}

fn draw_second_pass(
    meshlet_view_resources: &MeshletViewResources,
    render_context: &mut RenderContext,
    depth_stencil_attachment: RenderPassDepthStencilAttachment,
    camera: Option<&ExtractedCamera>,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    second_pass_visibility_buffer_pipeline: &RenderPipeline,
) {
    let mut color_attachments_filled = [None, None];
    if let (Some(visibility_buffer), Some(material_depth_color)) = (
        meshlet_view_resources.visibility_buffer.as_ref(),
        meshlet_view_resources.material_depth_color.as_ref(),
    ) {
        color_attachments_filled = [
            Some(RenderPassColorAttachment {
                view: &visibility_buffer.default_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK.into()),
                    store: StoreOp::Store,
                },
            }),
            Some(RenderPassColorAttachment {
                view: &material_depth_color.default_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK.into()),
                    store: StoreOp::Store,
                },
            }),
        ];
    }

    let mut draw_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meshlet_visibility_buffer_second_pass"),
        color_attachments: if color_attachments_filled[0].is_none() {
            &[]
        } else {
            &color_attachments_filled
        },
        depth_stencil_attachment: Some(depth_stencil_attachment),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = camera.and_then(|camera| camera.viewport.as_ref()) {
        draw_pass.set_camera_viewport(viewport);
    }

    draw_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer,
        &[view_offset.offset],
    );
    draw_pass.set_render_pipeline(second_pass_visibility_buffer_pipeline);
    draw_pass.draw_indirect(
        &meshlet_view_resources.visibility_buffer_draw_command_buffer_second,
        0,
    );
}

fn copy_material_depth_pass(
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    render_context: &mut RenderContext,
    camera: &ExtractedCamera,
    copy_material_depth_pipeline: &RenderPipeline,
) {
    if let (Some(material_depth), Some(copy_material_depth_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.copy_material_depth.as_ref(),
    ) {
        let mut copy_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meshlet_copy_material_depth_pass"),
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
            copy_pass.set_camera_viewport(viewport);
        }

        copy_pass.set_bind_group(0, copy_material_depth_bind_group, &[]);
        copy_pass.set_render_pipeline(copy_material_depth_pipeline);
        copy_pass.draw(0..3, 0..1);
    }
}
