use super::{Dlss, DlssResource};
use crate::core_3d::{Camera3d, MainPassViewportOverride};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Commands, NonSendMut, Query, Res},
};
use bevy_math::Vec4Swizzles;
use bevy_render::{
    camera::{CameraMainTextureUsages, ExtractedCamera, MipBias, TemporalJitter, Viewport},
    render_resource::TextureUsages,
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};
use dlss_wgpu::{DlssContext, DlssFeatureFlags};
use std::{mem, sync::Mutex};

pub fn prepare_dlss(
    mut query: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &Dlss,
        &mut TemporalJitter,
        &mut MipBias,
    )>,
    mut dlss_resource: NonSendMut<DlssResource>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (entity, view, camera, dlss, mut temporal_jitter, mut mip_bias) in &mut query {
        let upscaled_resolution = view.viewport.zw();

        let mut dlss_feature_flags = DlssFeatureFlags::LowResolutionMotionVectors
            | DlssFeatureFlags::InvertedDepth
            | DlssFeatureFlags::AutoExposure; // TODO
        if view.hdr {
            dlss_feature_flags |= DlssFeatureFlags::HighDynamicRange;
        }

        let dlss_sdk = dlss_resource.sdk.clone();
        let (dlss_context, context_used_last_frame) = dlss_resource
            .context_cache
            .entry((
                upscaled_resolution,
                dlss.perf_quality_mode,
                dlss_feature_flags,
            ))
            .or_insert_with(|| {
                let dlss_context = DlssContext::new(
                    upscaled_resolution,
                    dlss.perf_quality_mode,
                    dlss_feature_flags,
                    dlss_sdk,
                    render_device.wgpu_device(),
                    &render_queue,
                )
                .expect("Failed to create DlssContext");
                (Mutex::new(dlss_context), true)
            });
        *context_used_last_frame = true;

        let dlss_context = dlss_context.lock().unwrap();
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

    dlss_resource
        .context_cache
        .retain(|_, (_, in_use)| mem::take(in_use));
}

pub fn configure_dlss_view_targets(
    mut view_targets: Query<(&mut Camera3d, &mut CameraMainTextureUsages), With<Dlss>>,
) {
    for (mut camera_3d, mut camera_main_texture_usages) in view_targets.iter_mut() {
        camera_main_texture_usages.0 |= TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST;

        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();
    }
}
