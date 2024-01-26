use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryItem;
use bevy_render::{
    camera::ExtractedCamera,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_phase::{RenderPhase, TrackedRenderPass},
    render_resource::{CommandEncoderDescriptor, RenderPassDescriptor, StoreOp},
    renderer::RenderContext,
    view::ViewDepthTexture,
};
#[cfg(feature = "trace")]
use bevy_utils::tracing::info_span;

use super::{AlphaMask3dPrepass, DeferredPrepass, Opaque3dPrepass, ViewPrepassTextures};

/// Render node used by the prepass.
///
/// By default, inserted before the main pass in the render graph.
#[derive(Default)]
pub struct PrepassNode;

impl ViewNode for PrepassNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static RenderPhase<Opaque3dPrepass>,
        &'static RenderPhase<AlphaMask3dPrepass>,
        &'static ViewDepthTexture,
        &'static ViewPrepassTextures,
        Option<&'static DeferredPrepass>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            camera,
            opaque_prepass_phase,
            alpha_mask_prepass_phase,
            view_depth_texture,
            view_prepass_textures,
            deferred_prepass,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let mut color_attachments = vec![
            view_prepass_textures
                .normal
                .as_ref()
                .map(|normals_texture| normals_texture.get_attachment()),
            view_prepass_textures
                .motion_vectors
                .as_ref()
                .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
            // Use None in place of Deferred attachments
            None,
            None,
        ];

        if color_attachments.iter().all(Option::is_none) {
            // all attachments are none: clear the attachment list so that no fragment shader is required
            color_attachments.clear();
        }

        render_context.add_command_buffer_generation_task(move |render_device| {
            let mut command_encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("prepass_command_encoder"),
                });

            // Set up the pass descriptor with the depth attachment and optional color attachments
            let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("prepass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(view_depth_texture.get_attachment(StoreOp::Store)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);

            if let Some(viewport) = camera.viewport.as_ref() {
                render_pass.set_camera_viewport(viewport);
            }

            // Always run opaque pass to ensure screen is cleared
            {
                // Run the prepass, sorted front-to-back
                #[cfg(feature = "trace")]
                let _opaque_prepass_span = info_span!("opaque_prepass").entered();
                opaque_prepass_phase.render(&mut render_pass, world, view_entity);
            }

            if !alpha_mask_prepass_phase.items.is_empty() {
                // Run the prepass, sorted front-to-back
                #[cfg(feature = "trace")]
                let _alpha_mask_prepass_span = info_span!("alpha_mask_prepass").entered();
                alpha_mask_prepass_phase.render(&mut render_pass, world, view_entity);
            }

            if deferred_prepass.is_none() {
                // Copy if deferred isn't going to
                if let Some(prepass_depth_texture) = &view_prepass_textures.depth {
                    // Copy depth buffer to texture
                    command_encoder.copy_texture_to_texture(
                        view_depth_texture.texture.as_image_copy(),
                        prepass_depth_texture.texture.texture.as_image_copy(),
                        view_prepass_textures.size,
                    );
                }
            }

            command_encoder.finish()
        });

        Ok(())
    }
}
