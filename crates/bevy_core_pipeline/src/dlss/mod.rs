mod extract;
mod prepare;

use crate::{DepthPrepass, MotionVectorPrepass};
use bevy_app::{App, Plugin};
use bevy_asset::uuid::Uuid;
use bevy_ecs::{
    component::{require, Component},
    prelude::ReflectComponent,
    schedule::IntoSystemConfigs,
};
use bevy_math::UVec2;
use bevy_platform_support::collections::HashMap;
use bevy_reflect::{prelude::ReflectDefault, reflect_remote, Reflect};
use bevy_render::{
    camera::TemporalJitter,
    renderer::RenderDevice,
    view::{prepare_view_targets, prepare_view_uniforms},
    DlssProjectId, ExtractSchedule, Render, RenderApp, RenderSet,
};
use dlss_wgpu::{DlssContext, DlssFeatureFlags, DlssSdk};
use std::{rc::Rc, sync::Mutex};
use tracing::info;

pub use bevy_render::DlssSupported;
pub use dlss_wgpu::DlssPerfQualityMode;

pub struct DlssPlugin {
    pub project_id: Uuid,
}

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DlssProjectId(self.project_id))
            .register_type::<Dlss>();
    }

    fn finish(&self, app: &mut App) {
        if app.world().get_resource::<DlssSupported>().is_none() {
            info!("DLSS is not supported on this system");
            return;
        }

        let render_app = app.get_sub_app_mut(RenderApp).unwrap();
        let render_device = render_app.world().resource::<RenderDevice>().clone();

        let dlss_sdk = DlssSdk::new(self.project_id, render_device.wgpu_device().clone());
        if dlss_sdk.is_err() {
            app.world_mut().remove_resource::<DlssSupported>();
            info!("DLSS is not supported on this system");
            return;
        }

        render_app
            .world_mut()
            .insert_non_send_resource(DlssResource {
                sdk: dlss_sdk.unwrap(),
                context_cache: HashMap::default(),
            });

        render_app
            .add_systems(ExtractSchedule, extract::extract_dlss)
            .add_systems(
                Render,
                prepare::configure_dlss_view_targets
                    .in_set(RenderSet::ManageViews)
                    .after(prepare_view_targets),
            )
            .add_systems(
                Render,
                prepare::prepare_dlss
                    .in_set(RenderSet::PrepareResources)
                    .before(prepare_view_uniforms),
            )
    }
}

#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default)]
#[require(TemporalJitter, DepthPrepass, MotionVectorPrepass)]
pub struct Dlss {
    #[reflect(remote = DlssPerfQualityModeRemoteReflect)]
    pub perf_quality_mode: DlssPerfQualityMode,
    pub reset: bool,
}

#[reflect_remote(DlssPerfQualityMode)]
#[derive(Default)]
enum DlssPerfQualityModeRemoteReflect {
    #[default]
    Auto,
    Dlaa,
    UltraQuality,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

struct DlssResource {
    sdk: Rc<DlssSdk>,
    context_cache: HashMap<
        (UpscaledResolution, DlssPerfQualityMode, DlssFeatureFlags),
        (Mutex<DlssContext>, ContextUsedLastFrame),
    >,
}

type UpscaledResolution = UVec2;
type ContextUsedLastFrame = bool;
