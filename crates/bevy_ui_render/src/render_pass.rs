use core::ops::Range;

use super::{ImageNodeBindGroups, UiBatch, UiMeta, UiViewTarget};

use crate::{
    srgb_target::{run_srgb_conversion, SrgbConversionPipeline},
    UiCameraView, UiRenderTarget,
};
use bevy_ecs::{
    entity::EntityHash,
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::FloatOrd;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::*,
    render_resource::{
        CachedRenderPipelineId, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
    },
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::*,
};
use indexmap::IndexMap;
use tracing::error;

pub fn ui_pass(
    world: &World,
    view: ViewQuery<&UiCameraView>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    ui_view_target_query: Query<(&ViewTarget, &ExtractedCamera, &UiRenderTarget)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    conversion_pipeline: Res<SrgbConversionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let ui_camera_view = view.into_inner();
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        return;
    };

    let Ok((target, camera, ui_render_target)) = ui_view_target_query.get(ui_view_target.0) else {
        return;
    };

    let Some(transparent_phase) =
        transparent_render_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };

    if transparent_phase.items.is_empty() {
        return;
    }

    // UI shaders output sRGB-encoded colors. Each arm renders them to a target
    // that stores them as they are, so the hardware blends them in sRGB space.
    match *ui_render_target {
        UiRenderTarget::Direct(_) => {
            let attachment = target.get_unsampled_color_attachment();
            render_ui_phase(
                &mut ctx,
                world,
                attachment,
                camera,
                transparent_phase,
                ui_view_entity,
            );
        }
        UiRenderTarget::EncodedView(_) => {
            let Some(encoded_view) = target.main_texture_encoded_view() else {
                return;
            };
            let attachment = target.get_unsampled_color_attachment_with_view(&encoded_view);
            render_ui_phase(
                &mut ctx,
                world,
                attachment,
                camera,
                transparent_phase,
                ui_view_entity,
            );
        }
        UiRenderTarget::Convert {
            to_srgb, from_srgb, ..
        } => {
            let Some(to_srgb) = pipeline_cache.get_render_pipeline(to_srgb) else {
                return;
            };
            // Without a pipeline back, upscaling decodes from sRGB instead.
            let from_srgb = match from_srgb {
                Some(id) => {
                    let Some(pipeline) = pipeline_cache.get_render_pipeline(id) else {
                        return;
                    };
                    Some(pipeline)
                }
                None => None,
            };
            run_srgb_conversion(
                &mut ctx,
                target,
                &conversion_pipeline,
                to_srgb,
                "ui_to_srgb",
            );
            let attachment = target.get_unsampled_color_attachment();
            render_ui_phase(
                &mut ctx,
                world,
                attachment,
                camera,
                transparent_phase,
                ui_view_entity,
            );
            if let Some(from_srgb) = from_srgb {
                run_srgb_conversion(
                    &mut ctx,
                    target,
                    &conversion_pipeline,
                    from_srgb,
                    "ui_from_srgb",
                );
            }
        }
    }
}

fn render_ui_phase(
    ctx: &mut RenderContext,
    world: &World,
    attachment: RenderPassColorAttachment,
    camera: &ExtractedCamera,
    transparent_phase: &SortedRenderPhase<TransparentUi>,
    ui_view_entity: Entity,
) {
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "ui");

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = transparent_phase.render(&mut render_pass, world, ui_view_entity) {
        error!("Error encountered while rendering the ui phase {err:?}");
    }

    pass_span.end(&mut render_pass);
}

#[derive(Debug)]
pub struct TransparentUi {
    pub sort_key: FloatOrd,
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub indexed: bool,
}

impl PhaseItem for TransparentUi {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TransparentUi {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut IndexMap<(Entity, MainEntity), TransparentUi, EntityHash>) {
        items.sort_by_key(|_, value| value.sort_key());
    }

    fn recalculate_sort_keys(
        _: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _: &ExtractedView,
    ) {
        // Sort keys are precalculated for UI phase items.
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub type DrawUi = (
    SetItemPipeline,
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<1>,
    DrawUiNode,
);

pub struct SetUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<I> {
    type Param = SRes<UiMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
pub struct SetUiTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<I> {
    type Param = SRes<ImageNodeBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, image_bind_groups.values.get(&batch.image).unwrap(), &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawUiNode;
impl<P: PhaseItem> RenderCommand<P> for DrawUiNode {
    type Param = SRes<UiMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));
        // Define how to "connect" the vertices
        pass.set_index_buffer(
            indices.slice(..),
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
