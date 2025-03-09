use super::{Dlss, DlssSdk};
use crate::core_3d::{Camera3d, MainPassViewportOverride};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_math::{UVec2, Vec4Swizzles};
use bevy_render::{
    camera::{CameraMainTextureUsages, ExtractedCamera, MipBias, TemporalJitter, Viewport},
    render_resource::TextureUsages,
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};
use dlss_wgpu::{DlssContext, DlssFeatureFlags, DlssPerfQualityMode};
use std::{mem, sync::Mutex};

pub fn prepare_dlss(
    mut query: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &Dlss,
        &mut TemporalJitter,
        &mut MipBias,
        Option<&mut ViewDlssContext>,
    )>,
    dlss_sdk: Res<DlssSdk>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (entity, view, camera, dlss, mut temporal_jitter, mut mip_bias, dlss_context) in &mut query
    {
        let upscaled_resolution = view.viewport.zw();

        let mut dlss_feature_flags = DlssFeatureFlags::LowResolutionMotionVectors
            | DlssFeatureFlags::InvertedDepth
            | DlssFeatureFlags::AutoExposure; // TODO
        if view.hdr {
            dlss_feature_flags |= DlssFeatureFlags::HighDynamicRange;
        }

        let changed = match dlss_context {
            Some(context) => {
                !(upscaled_resolution == context.context.upscaled_resolution()
                    && dlss.perf_quality_mode == context.perf_quality_mode
                    && view.hdr == context.hdr)
            }
            None => true,
        };

        if changed {
            let dlss_context = DlssContext::new(
                upscaled_resolution,
                dlss.perf_quality_mode,
                dlss_feature_flags,
                dlss_sdk,
                render_device.wgpu_device(),
                &render_queue,
            )
            .expect("Failed to create DlssContext");
        }

        let render_resolution = dlss_context.render_resolution();
        temporal_jitter.offset = dlss_context.suggested_jitter(frame_count.0, render_resolution);
        mip_bias.0 = dlss_context.suggested_mip_bias(render_resolution);

        commands
            .entity(entity)
            .insert(MainPassViewportOverride(Viewport {
                physical_position: view.viewport.xy(),
                physical_size: render_resolution,
                depth: camera.viewport.clone().map(|v| v.depth).unwrap_or(0.0..1.0),
            }));
    }
}

#[derive(Component)]
pub struct ViewDlssContext {
    context: DlssContext,
    perf_quality_mode: DlssPerfQualityMode,
    hdr: bool,
}

pub fn configure_dlss_view_targets(
    mut view_targets: Query<(&mut Camera3d, &mut CameraMainTextureUsages), With<Dlss>>,
) {
    for (mut camera_3d, mut camera_main_texture_usages) in view_targets.iter_mut() {
        camera_main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;

        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();
    }
}
