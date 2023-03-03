//! This example compares different anti-aliasing methods.

#[cfg(feature = "dlss")]
use bevy::core_pipeline::dlss::{
    DlssAvailable, DlssBundle, DlssPlugin, DlssProjectId, DlssSettings,
};

use bevy::{
    core_pipeline::{
        experimental::taa::{
            TemporalAntialiasBundle, TemporalAntialiasPlugin, TemporalAntialiasSettings,
        },
        fxaa::{Fxaa, Sensitivity},
    },
    pbr::CascadeShadowConfigBuilder,
    prelude::*,
    render::{
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, SamplerDescriptor, TextureDimension, TextureFormat},
        texture::ImageSampler,
    },
};

use std::f32::consts::PI;

fn main() {
    let mut app = App::new();

    #[cfg(feature = "dlss")]
    app.insert_resource(DlssProjectId(uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96"
    )));

    app.insert_resource(Msaa::Off)
        // TODO: Get pipelined rendering working with DLSS
        .add_plugins(DefaultPlugins.build().disable::<PipelinedRenderingPlugin>())
        .add_plugin(TemporalAntialiasPlugin)
        .add_startup_system(setup)
        .add_system(modify_aa)
        .add_system(update_ui);

    #[cfg(feature = "dlss")]
    app.add_plugin(DlssPlugin);

    app.run();
}

fn modify_aa(
    keys: Res<Input<KeyCode>>,
    #[cfg(feature = "dlss")] mut camera: Query<
        (
            Entity,
            Option<&mut Fxaa>,
            Option<&TemporalAntialiasSettings>,
            Option<&mut DlssSettings>,
        ),
        With<Camera>,
    >,
    #[cfg(not(feature = "dlss"))] mut camera: Query<
        (
            Entity,
            Option<&mut Fxaa>,
            Option<&TemporalAntialiasSettings>,
        ),
        With<Camera>,
    >,
    mut msaa: ResMut<Msaa>,
    #[cfg(feature = "dlss")] dlss_available: Option<Res<DlssAvailable>>,
    mut commands: Commands,
) {
    #[cfg(feature = "dlss")]
    let (camera_entity, fxaa, taa, dlss) = camera.single_mut();
    #[cfg(not(feature = "dlss"))]
    let (camera_entity, fxaa, taa) = camera.single_mut();
    let mut camera = commands.entity(camera_entity);

    // No AA
    if keys.just_pressed(KeyCode::Key1) {
        *msaa = Msaa::Off;
        camera.remove::<Fxaa>();
        camera.remove::<TemporalAntialiasBundle>();
        #[cfg(feature = "dlss")]
        camera.remove::<DlssBundle>();
    }

    // MSAA
    if keys.just_pressed(KeyCode::Key2) && *msaa == Msaa::Off {
        camera.remove::<Fxaa>();
        camera.remove::<TemporalAntialiasBundle>();
        #[cfg(feature = "dlss")]
        camera.remove::<DlssBundle>();

        *msaa = Msaa::Sample4;
    }

    // MSAA Sample Count
    if *msaa != Msaa::Off {
        if keys.just_pressed(KeyCode::Q) {
            *msaa = Msaa::Sample2;
        }
        if keys.just_pressed(KeyCode::W) {
            *msaa = Msaa::Sample4;
        }
        if keys.just_pressed(KeyCode::E) {
            *msaa = Msaa::Sample8;
        }
    }

    // FXAA
    if keys.just_pressed(KeyCode::Key3) && fxaa.is_none() {
        *msaa = Msaa::Off;
        camera.remove::<TemporalAntialiasBundle>();
        #[cfg(feature = "dlss")]
        camera.remove::<DlssBundle>();

        camera.insert(Fxaa::default());
    }

    // FXAA Settings
    if let Some(mut fxaa) = fxaa {
        if keys.just_pressed(KeyCode::Q) {
            fxaa.edge_threshold = Sensitivity::Low;
            fxaa.edge_threshold_min = Sensitivity::Low;
        }
        if keys.just_pressed(KeyCode::W) {
            fxaa.edge_threshold = Sensitivity::Medium;
            fxaa.edge_threshold_min = Sensitivity::Medium;
        }
        if keys.just_pressed(KeyCode::E) {
            fxaa.edge_threshold = Sensitivity::High;
            fxaa.edge_threshold_min = Sensitivity::High;
        }
        if keys.just_pressed(KeyCode::R) {
            fxaa.edge_threshold = Sensitivity::Ultra;
            fxaa.edge_threshold_min = Sensitivity::Ultra;
        }
        if keys.just_pressed(KeyCode::T) {
            fxaa.edge_threshold = Sensitivity::Extreme;
            fxaa.edge_threshold_min = Sensitivity::Extreme;
        }
    }

    // TAA
    if keys.just_pressed(KeyCode::Key4) && taa.is_none() {
        *msaa = Msaa::Off;
        camera.remove::<Fxaa>();
        #[cfg(feature = "dlss")]
        camera.remove::<DlssBundle>();

        camera.insert(TemporalAntialiasBundle::default());
    }

    // DLSS
    #[cfg(feature = "dlss")]
    if keys.just_pressed(KeyCode::Key5) && dlss.is_none() && dlss_available.is_some() {
        *msaa = Msaa::Off;
        camera.remove::<Fxaa>();
        camera.remove::<TemporalAntialiasBundle>();

        camera.insert(DlssBundle::default());
    }
}

fn update_ui(
    #[cfg(feature = "dlss")] camera: Query<
        (
            Option<&Fxaa>,
            Option<&TemporalAntialiasSettings>,
            Option<&DlssSettings>,
        ),
        With<Camera>,
    >,
    #[cfg(not(feature = "dlss"))] camera: Query<
        (Option<&Fxaa>, Option<&TemporalAntialiasSettings>),
        With<Camera>,
    >,
    msaa: Res<Msaa>,
    #[cfg(feature = "dlss")] dlss_available: Option<Res<DlssAvailable>>,
    mut ui: Query<&mut Text>,
) {
    #[cfg(feature = "dlss")]
    let (fxaa, taa, dlss) = camera.single();
    #[cfg(not(feature = "dlss"))]
    let (fxaa, taa) = camera.single();

    let mut ui = ui.single_mut();
    let ui = &mut ui.sections[0].value;

    *ui = "Antialias Method\n".to_string();

    #[cfg(feature = "dlss")]
    let dlss_off = dlss.is_none();
    #[cfg(not(feature = "dlss"))]
    let dlss_off = true;
    if *msaa == Msaa::Off && fxaa.is_none() && taa.is_none() && dlss_off {
        ui.push_str("(1) *No AA*\n");
    } else {
        ui.push_str("(1) No AA\n");
    }

    if *msaa != Msaa::Off {
        ui.push_str("(2) *MSAA*\n");
    } else {
        ui.push_str("(2) MSAA\n");
    }

    if fxaa.is_some() {
        ui.push_str("(3) *FXAA*\n");
    } else {
        ui.push_str("(3) FXAA\n");
    }

    if taa.is_some() {
        ui.push_str("(4) *TAA*");
    } else {
        ui.push_str("(4) TAA");
    }

    #[cfg(not(feature = "dlss"))]
    ui.push_str("\n(X) DLSS (dlss feature disabled)");
    #[cfg(feature = "dlss")]
    if dlss_available.is_none() {
        ui.push_str("\n(X) DLSS (not available)");
    } else if dlss.is_some() {
        ui.push_str("\n(5) *DLSS*");
    } else {
        ui.push_str("\n(5) DLSS");
    }

    if *msaa != Msaa::Off {
        ui.push_str("\n\n----------\n\nSample Count\n");

        if *msaa == Msaa::Sample2 {
            ui.push_str("(Q) *2*\n");
        } else {
            ui.push_str("(Q) 2\n");
        }
        if *msaa == Msaa::Sample4 {
            ui.push_str("(W) *4*\n");
        } else {
            ui.push_str("(W) 4\n");
        }
        if *msaa == Msaa::Sample8 {
            ui.push_str("(E) *8*");
        } else {
            ui.push_str("(E) 8");
        }
    }

    if let Some(fxaa) = fxaa {
        ui.push_str("\n\n----------\n\nSensitivity\n");

        if fxaa.edge_threshold == Sensitivity::Low {
            ui.push_str("(Q) *Low*\n");
        } else {
            ui.push_str("(Q) Low\n");
        }

        if fxaa.edge_threshold == Sensitivity::Medium {
            ui.push_str("(W) *Medium*\n");
        } else {
            ui.push_str("(W) Medium\n");
        }

        if fxaa.edge_threshold == Sensitivity::High {
            ui.push_str("(E) *High*\n");
        } else {
            ui.push_str("(E) High\n");
        }

        if fxaa.edge_threshold == Sensitivity::Ultra {
            ui.push_str("(R) *Ultra*\n");
        } else {
            ui.push_str("(R) Ultra\n");
        }

        if fxaa.edge_threshold == Sensitivity::Extreme {
            ui.push_str("(T) *Extreme*");
        } else {
            ui.push_str("(T) Extreme");
        }
    }
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    // Plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });

    let cube_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });

    // Cubes
    for i in 0..5 {
        commands.spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 0.25 })),
            material: cube_material.clone(),
            transform: Transform::from_xyz(i as f32 * 0.25 - 1.0, 0.125, -i as f32 * 0.5),
            ..default()
        });
    }

    // Flight Helmet
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        ..default()
    });

    // Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            PI * -0.15,
            PI * -0.15,
        )),
        cascade_shadow_config: CascadeShadowConfigBuilder {
            maximum_distance: 3.0,
            first_cascade_far_bound: 0.9,
            ..default()
        }
        .into(),
        ..default()
    });

    // Camera
    commands.spawn(Camera3dBundle {
        camera: Camera {
            hdr: true,
            ..default()
        },
        transform: Transform::from_xyz(0.7, 0.7, 1.0).looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
        ..default()
    });

    // UI
    commands.spawn(
        TextBundle::from_section(
            "",
            TextStyle {
                font: asset_server.load("fonts/FiraMono-Medium.ttf"),
                font_size: 20.0,
                color: Color::BLACK,
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            position: UiRect {
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            },
            ..default()
        }),
    );
}

/// Creates a colorful test pattern
fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    let mut img = Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
    );
    img.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor::default());
    img
}
