pub use bevy_render::DlssAvailable;
pub use dlss_wgpu::DlssPreset;

use crate::{DepthPrepass, MotionVectorPrepass};
use bevy_app::{App, Plugin};
use bevy_asset::uuid::Uuid;
use bevy_ecs::component::{require, Component};
use bevy_render::{camera::TemporalJitter, renderer::RenderDevice, DlssProjectId, RenderApp};
use dlss_wgpu::DlssSdk;
use tracing::info;

pub struct DlssPlugin {
    pub project_id: Uuid,
}

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DlssProjectId(self.project_id));
    }

    fn finish(&self, app: &mut App) {
        if app.world().get_resource::<DlssAvailable>().is_none() {
            info!("DLSS not available");
            return;
        }

        let render_app = app.get_sub_app_mut(RenderApp).unwrap();
        let render_device = render_app.world().resource::<RenderDevice>().clone();

        let dlss_sdk = DlssSdk::new(self.project_id, render_device.wgpu_device().clone());
        if dlss_sdk.is_err() {
            app.world_mut().remove_resource::<DlssAvailable>();
            info!("DLSS not available");
            return;
        }

        render_app
            .world_mut()
            .insert_non_send_resource(dlss_sdk.unwrap());
    }
}

#[derive(Component, Clone, Default)]
#[require(TemporalJitter, DepthPrepass, MotionVectorPrepass)]
pub struct Dlss {
    pub preset: DlssPreset,
    pub reset: bool,
}
