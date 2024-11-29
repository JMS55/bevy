//! Ray traced lighting.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    core_pipeline::{
        experimental::taa::{TemporalAntiAliasPlugin, TemporalAntiAliasing},
        prepass::DeferredPrepass,
    },
    pbr::experimental::solari::{Solari, SolariEnabled, SolariPlugin, SolariSupported},
    prelude::*,
    render::{
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::f32::consts::PI;

fn main() {
    let render_plugin = RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings {
            features: WgpuFeatures::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                | WgpuFeatures::EXPERIMENTAL_RAY_QUERY,
            ..default()
        }),
        ..default()
    };

    App::new()
        .add_plugins((
            DefaultPlugins.set(render_plugin),
            TemporalAntiAliasPlugin,
            SolariPlugin,
            CameraControllerPlugin,
        ))
        .insert_resource(AmbientLight::NONE)
        .add_systems(
            Startup,
            (
                setup.run_if(resource_exists::<SolariSupported>),
                solari_not_supported.run_if(not(resource_exists::<SolariSupported>)),
            ),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SolariEnabled);

    commands.spawn(SceneRoot(asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/CornellBox/CornellBox.glb"),
    )));

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, PI * -0.43, PI * -0.08, 0.0)),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Solari {
            debug_path_tracer: true,
        },
        Transform::from_xyz(-278.0, 273.0, 800.0),
        TemporalAntiAliasing::default(),
        Msaa::Off,
        DeferredPrepass,
        CameraController::default(),
    ));
}

fn solari_not_supported(mut commands: Commands) {
    commands.spawn((
        Text::new("Current GPU does not support Solari"),
        TextFont::from_font_size(36.0),
        Node {
            margin: UiRect::all(Val::Auto),
            ..default()
        },
    ));

    commands.spawn(Camera2d);
}
