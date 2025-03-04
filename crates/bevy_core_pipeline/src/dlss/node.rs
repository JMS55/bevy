use super::{Dlss, DlssResource};
use crate::{core_3d::MainPassViewportOverride, prepass::ViewPrepassTextures};
use bevy_ecs::{query::QueryItem, world::World};
use bevy_math::Vec4Swizzles;
use bevy_render::{
    camera::TemporalJitter,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    renderer::{RenderAdapter, RenderContext},
    view::{ExtractedView, ViewTarget},
};
use dlss_wgpu::{DlssExposure, DlssFeatureFlags, DlssRenderParameters, DlssTexture};

#[derive(Default)]
pub struct DlssNode;

impl ViewNode for DlssNode {
    type ViewQuery = (
        &'static ExtractedView,
        &'static Dlss,
        &'static MainPassViewportOverride,
        &'static TemporalJitter,
        &'static ViewTarget,
        &'static ViewPrepassTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view, dlss, viewport_override, temporal_jitter, view_target, prepass_textures): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let adapter = world.resource::<RenderAdapter>();
        let dlss_resource = world.non_send_resource::<DlssResource>();
        let (Some(prepass_motion_vectors_texture), Some(prepass_depth_texture)) =
            (&prepass_textures.motion_vectors, &prepass_textures.depth)
        else {
            return Ok(());
        };

        let render_resolution = viewport_override.0.physical_size;
        let upscaled_resolution = view.viewport.zw();
        let mut dlss_feature_flags = DlssFeatureFlags::LowResolutionMotionVectors
            | DlssFeatureFlags::InvertedDepth
            | DlssFeatureFlags::AutoExposure; // TODO
        if view.hdr {
            dlss_feature_flags |= DlssFeatureFlags::HighDynamicRange;
        }

        let mut dlss_context = dlss_resource.context_cache[&(
            upscaled_resolution,
            dlss.perf_quality_mode,
            dlss_feature_flags,
        )]
            .0
            .lock()
            .unwrap();
        let view_target = view_target.post_process_write();

        dlss_context
            .render(
                DlssRenderParameters {
                    color: DlssTexture {
                        texture: &view_target.source_texture,
                        view: &view_target.source,
                    },
                    depth: DlssTexture {
                        texture: &prepass_depth_texture.texture.texture,
                        view: &prepass_depth_texture.texture.default_view,
                    },
                    motion_vectors: DlssTexture {
                        texture: &prepass_motion_vectors_texture.texture.texture,
                        view: &prepass_motion_vectors_texture.texture.default_view,
                    },
                    exposure: DlssExposure::Automatic, // TODO
                    transparency_mask: None,           // TODO
                    bias: None,                        // TODO
                    dlss_output: DlssTexture {
                        texture: &view_target.destination_texture,
                        view: &view_target.destination,
                    },
                    reset: dlss.reset,
                    jitter_offset: temporal_jitter.offset,
                    partial_texture_size: Some(render_resolution),
                    motion_vector_scale: Some(-render_resolution.as_vec2()),
                },
                render_context.command_encoder(),
                &adapter,
            )
            .expect("Failed to render DLSS");

        Ok(())
    }
}
