//! Demonstrates realtime dynamic raytraced lighting using Bevy Solari.

use argh::FromArgs;
use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore},
    gltf::GltfMaterialName,
    image::{ImageAddressMode, ImageLoaderSettings},
    mesh::{Indices, VertexAttributeValues},
    post_process::bloom::Bloom,
    prelude::*,
    render::{diagnostic::RenderDiagnosticsPlugin, render_resource::TextureUsages},
    solari::{
        pathtracer::{Pathtracer, PathtracingPlugin},
        prelude::{RaytracingMesh3d, SolariDebugView, SolariLighting, SolariPlugins},
        realtime::SOLARI_DEBUG_COUNTERS,
    },
    world_serialization::WorldInstanceReady,
};
use chacha20::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use std::f32::consts::PI;

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::{
    anti_alias::dlss::{
        Dlss, DlssProjectId, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
    },
    render::camera::{MipBias, TemporalJitter},
};

/// `bevy_solari` demo.
#[derive(FromArgs, Resource, Clone)]
struct Args {
    /// use the reference pathtracer instead of the realtime lighting system.
    #[argh(switch)]
    pathtracer: Option<bool>,
    /// stress test a scene with many lights.
    #[argh(switch)]
    many_lights: Option<bool>,
    /// drive a fixed camera path and write a per-frame noise trace to this CSV
    /// path, then exit. Use to compare noise under camera motion between builds.
    #[argh(option)]
    noise_trace: Option<String>,
    /// override `SolariLighting::specular_confidence_weight_cap`, to A/B how much
    /// of the specular noise is stale temporal history.
    #[argh(option)]
    specular_confidence_cap: Option<f32>,
}

fn solari_lighting_from_args(args: &Args) -> SolariLighting {
    let mut solari_lighting = SolariLighting::default();
    if let Some(cap) = args.specular_confidence_cap {
        solari_lighting.specular_confidence_weight_cap = cap;
    }
    solari_lighting
}

fn main() {
    let args: Args = argh::from_env();

    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "5417916c-0291-4e3f-8f65-326c1858ab96" // Don't copy paste this - generate your own UUID!
    )));

    app.add_plugins((
        DefaultPlugins,
        SolariPlugins,
        FreeCameraPlugin,
        RenderDiagnosticsPlugin,
    ))
    .insert_resource(args.clone());

    if args.many_lights == Some(true) {
        app.add_systems(Startup, setup_many_lights);
    } else {
        app.add_systems(Startup, setup_pica_pica);
    }

    if args.pathtracer == Some(true) {
        app.add_plugins(PathtracingPlugin);
    } else {
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        app.add_systems(Update, toggle_dlss_rr);

        if args.many_lights != Some(true) {
            app.add_systems(Update, (pause_scene, toggle_lights, patrol_path));
        }
        app.init_resource::<SavedPostProcessing>()
            .add_systems(Startup, spawn_debug_text)
            .add_systems(Update, select_debug_view)
            .add_systems(
                PostUpdate,
                (
                    update_control_text,
                    update_performance_text,
                    update_debug_text,
                ),
            );

        if args.noise_trace.is_some() {
            app.init_resource::<NoiseTrace>()
                .add_systems(Update, drive_noise_trace);
        }
    }

    app.run();
}

fn setup_pica_pica(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<Args>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    commands
        .spawn((
            WorldAssetRoot(
                asset_server.load(
                    GltfAssetLabel::Scene(0)
                        .from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/mini_diorama_01.glb")
                ),
            ),
            Transform::from_scale(Vec3::splat(10.0)),
        ))
        .observe(add_raytracing_meshes_on_scene_load);

    commands
        .spawn((
            WorldAssetRoot(asset_server.load(
                GltfAssetLabel::Scene(0).from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/robot_01.glb")
            )),
            Transform::from_scale(Vec3::splat(2.0))
                .with_translation(Vec3::new(-2.0, 0.05, -2.1))
                .with_rotation(Quat::from_rotation_y(PI / 2.0)),
            PatrolPath {
                path: vec![
                    (Vec3::new(-2.0, 0.05, -2.1), Quat::from_rotation_y(PI / 2.0)),
                    (Vec3::new(2.2, 0.05, -2.1), Quat::from_rotation_y(0.0)),
                    (
                        Vec3::new(2.2, 0.05, 2.1),
                        Quat::from_rotation_y(3.0 * PI / 2.0),
                    ),
                    (Vec3::new(-2.0, 0.05, 2.1), Quat::from_rotation_y(PI)),
                ],
                i: 0,
            },
        ))
        .observe(add_raytracing_meshes_on_scene_load);

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false, // Solari replaces shadow mapping
            ..default()
        },
        Transform::from_rotation(Quat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704)).with_rotation(
            Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087),
        ),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));

    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        camera.insert(solari_lighting_from_args(&args));
    }

    // Using DLSS Ray Reconstruction for denoising (and cheaper rendering via upscaling) is _highly_ recommended when using Solari
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: Default::default(),
            reset: Default::default(),
            _phantom_data: Default::default(),
        });
    }

    commands.spawn((
        ControlText,
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12.0),
            left: px(12.0),
            ..default()
        },
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_left(Val2::all(px(4.0))),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            PerformanceText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

fn setup_many_lights(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<Args>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut plane_mesh = Plane3d::default()
        .mesh()
        .size(400.0, 400.0)
        .build()
        .with_generated_tangents()
        .unwrap();
    match plane_mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0).unwrap() {
        VertexAttributeValues::Float32x2(items) => {
            items.iter_mut().flatten().for_each(|x| *x *= 3.0);
        }
        _ => unreachable!(),
    }
    let plane_mesh = meshes.add(plane_mesh);
    let cube_mesh = meshes.add(
        Cuboid::default()
            .mesh()
            .build()
            .with_generated_tangents()
            .unwrap(),
    );
    let sphere_mesh = meshes.add(
        Sphere::new(1.0)
            .mesh()
            .build()
            .with_generated_tangents()
            .unwrap(),
    );

    commands
        .spawn((
            RaytracingMesh3d(plane_mesh.clone()),
            MeshMaterial3d(
                materials.add(StandardMaterial {
                    base_color_texture: Some(
                        asset_server
                            .load_builder()
                            .with_settings::<ImageLoaderSettings>(|settings| {
                                settings
                                    .sampler
                                    .get_or_init_descriptor()
                                    .set_address_mode(ImageAddressMode::Repeat);
                            })
                            .load("textures/uv_checker_bw.png"),
                    ),
                    perceptual_roughness: 0.0,
                    ..default()
                }),
            ),
        ))
        .insert_if(Mesh3d(plane_mesh), || args.pathtracer != Some(true));

    for _ in 0..8000 {
        commands
            .spawn((
                RaytracingMesh3d(cube_mesh.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(rng.random(), rng.random(), rng.random()),
                    perceptual_roughness: rng.random(),
                    ..default()
                })),
                Transform::default()
                    .with_scale(Vec3 {
                        x: rng.random_range(0.2..=2.0),
                        y: rng.random_range(0.2..=2.0),
                        z: rng.random_range(0.2..=2.0),
                    })
                    .with_translation(Vec3::new(
                        rng.random_range(-180.0..=180.0),
                        0.2,
                        rng.random_range(-180.0..=180.0),
                    )),
            ))
            .insert_if(Mesh3d(cube_mesh.clone()), || args.pathtracer != Some(true));
    }

    for x in -10..=10 {
        for y in -10..=10 {
            commands
                .spawn((
                    RaytracingMesh3d(sphere_mesh.clone()),
                    MeshMaterial3d(
                        materials.add(StandardMaterial {
                            emissive: Color::linear_rgb(
                                rng.random::<f32>() * 60000.0,
                                rng.random::<f32>() * 60000.0,
                                rng.random::<f32>() * 60000.0,
                            )
                            .into(),
                            ..default()
                        }),
                    ),
                    Transform::default().with_translation(Vec3::new(
                        (x * 20) as f32,
                        7.0,
                        (y * 20) as f32,
                    )),
                ))
                .insert_if(Mesh3d(sphere_mesh.clone()), || {
                    args.pathtracer != Some(true)
                });
        }
    }

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(6.11329, 166.74896, 451.8226)).with_rotation(
            Quat::from_xyzw(-0.183938, 0.009093744, 0.0017017953, 0.9828943),
        ),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        Bloom {
            intensity: 0.1,
            ..Bloom::NATURAL
        },
    ));

    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        camera.insert(solari_lighting_from_args(&args));
    }

    // Using DLSS Ray Reconstruction for denoising (and cheaper rendering via upscaling) is _highly_ recommended when using Solari
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: Default::default(),
            reset: Default::default(),
            _phantom_data: Default::default(),
        });
    }

    commands.spawn((
        ControlText,
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12.0),
            left: px(12.0),
            ..default()
        },
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_left(Val2::all(px(4.0))),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            PerformanceText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

fn add_raytracing_meshes_on_scene_load(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        Option<&GltfMaterialName>,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    args: Res<Args>,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((Mesh3d(mesh_handle), MeshMaterial3d(material_handle), material_name)) =
            mesh_query.get(descendant)
        {
            // Add raytracing mesh component
            commands
                .entity(descendant)
                .insert(RaytracingMesh3d(mesh_handle.clone()));

            // Ensure meshes are Solari compatible
            let mut mesh = meshes.get_mut(mesh_handle).unwrap();
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                let vertex_count = mesh.count_vertices();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertex_count]);
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_TANGENT,
                    vec![[0.0, 0.0, 0.0, 0.0]; vertex_count],
                );
            }
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
                mesh.generate_tangents().unwrap();
            }
            if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
                mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
            }
            if let Some(indices) = mesh.indices_mut()
                && let Indices::U16(_) = indices
            {
                *indices = Indices::U32(indices.iter().map(|i| i as u32).collect());
            }

            // Prevent rasterization if using pathtracer
            if args.pathtracer == Some(true) {
                commands.entity(descendant).remove::<Mesh3d>();
            }

            // Adjust scene materials to better demo Solari features
            if material_name.map(|s| s.0.as_str()) == Some("material") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.emissive = LinearRgba::BLACK;
            }
            if material_name.map(|s| s.0.as_str()) == Some("Lights") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.emissive =
                    LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;

                commands.insert_resource(RobotLightMaterial(material_handle.clone()));
            }
            if material_name.map(|s| s.0.as_str()) == Some("Glass_Dark_01") {
                let mut material = materials.get_mut(material_handle).unwrap();
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;
            }
        }
    }
}

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
fn toggle_dlss_rr(
    key_input: Res<ButtonInput<KeyCode>>,
    camera: Single<(Entity, Has<Dlss<DlssRayReconstructionFeature>>), With<SolariLighting>>,
    dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
    mut commands: Commands,
) {
    if key_input.just_pressed(KeyCode::Digit3) && dlss_rr_supported.is_some() {
        let (entity, dlss) = *camera;
        if dlss {
            commands
                .entity(entity)
                .remove::<(Dlss<DlssRayReconstructionFeature>, TemporalJitter, MipBias)>();
        } else {
            commands
                .entity(entity)
                .insert(Dlss::<DlssRayReconstructionFeature> {
                    perf_quality_mode: Default::default(),
                    reset: Default::default(),
                    _phantom_data: Default::default(),
                });
        }
    }
}

fn pause_scene(mut time: ResMut<Time<Virtual>>, key_input: Res<ButtonInput<KeyCode>>) {
    if key_input.just_pressed(KeyCode::Space) {
        time.toggle();
    }
}

/// Post-processing removed while a debug view is up, so it can be put back
/// afterwards.
#[derive(Resource, Default)]
struct SavedPostProcessing {
    tonemapping: Option<Tonemapping>,
    bloom: Option<Bloom>,
}

/// One key per debug view, so any of them is a single press. These avoid the keys
/// the free camera (WASD/QE/M/shift/numpad) and the scene toggles (space, 1-3)
/// already use, and function keys, which macOS eats by default.
const DEBUG_VIEW_KEYS: [(KeyCode, &str, SolariDebugView); 18] = [
    (KeyCode::KeyZ, "Z", SolariDebugView::None),
    (KeyCode::KeyX, "X", SolariDebugView::NoiseRelativeStdDev),
    (KeyCode::KeyY, "Y", SolariDebugView::NoiseResampledStdDev),
    (KeyCode::KeyC, "C", SolariDebugView::NoiseNonResampledShare),
    (KeyCode::KeyV, "V", SolariDebugView::NonResampledShare),
    (KeyCode::KeyB, "B", SolariDebugView::NonResampledOnly),
    (KeyCode::KeyN, "N", SolariDebugView::ResampledOnly),
    (KeyCode::KeyF, "F", SolariDebugView::SampleProvenance),
    (KeyCode::KeyR, "R", SolariDebugView::SampleAge),
    (KeyCode::KeyT, "T", SolariDebugView::SampleDuplication),
    (KeyCode::KeyG, "G", SolariDebugView::ConfidenceWeight),
    (KeyCode::KeyH, "H", SolariDebugView::TemporalRejectReason),
    (KeyCode::KeyJ, "J", SolariDebugView::SpatialReuseFailure),
    (KeyCode::KeyK, "K", SolariDebugView::JacobianRejection),
    (KeyCode::KeyL, "L", SolariDebugView::ContributionWeight),
    (KeyCode::KeyO, "O", SolariDebugView::WorldCacheSampleCount),
    (KeyCode::KeyP, "P", SolariDebugView::WorldCacheProbeFailure),
    (KeyCode::KeyI, "I", SolariDebugView::WorldCache),
];

fn select_debug_view(
    key_input: Res<ButtonInput<KeyCode>>,
    camera: Single<(
        Entity,
        &mut SolariLighting,
        Option<&Tonemapping>,
        Option<&Bloom>,
    )>,
    mut saved: ResMut<SavedPostProcessing>,
    mut commands: Commands,
) {
    let (entity, mut solari_lighting, tonemapping, bloom) = camera.into_inner();

    if key_input.just_pressed(KeyCode::Digit4) {
        solari_lighting.debug_counters = !solari_lighting.debug_counters;
    }

    let Some((_, _, selected)) = DEBUG_VIEW_KEYS
        .iter()
        .find(|(key, _, _)| key_input.just_pressed(*key))
    else {
        return;
    };

    let was_active = solari_lighting.debug_view != SolariDebugView::None;
    solari_lighting.debug_view = *selected;
    let is_active = solari_lighting.debug_view != SolariDebugView::None;

    // Debug views write display-ready colour straight to the view target, so the
    // post-processing chain has to get out of the way or it will remap the
    // heatmaps and smear the categorical ones.
    if is_active && !was_active {
        saved.tonemapping = tonemapping.copied();
        saved.bloom = bloom.cloned();
        commands.entity(entity).insert(Tonemapping::None);
        commands.entity(entity).remove::<Bloom>();
    } else if !is_active && was_active {
        commands
            .entity(entity)
            .insert(saved.tonemapping.take().unwrap_or_default());
        if let Some(bloom) = saved.bloom.take() {
            commands.entity(entity).insert(bloom);
        }
    }
}

#[derive(Resource)]
struct RobotLightMaterial(Handle<StandardMaterial>);

fn toggle_lights(
    key_input: Res<ButtonInput<KeyCode>>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    mut commands: Commands,
) {
    if key_input.just_pressed(KeyCode::Digit1) {
        if let Ok(directional_light) = directional_light.single() {
            commands.entity(directional_light).despawn();
        } else {
            commands.spawn((
                DirectionalLight {
                    illuminance: light_consts::lux::FULL_DAYLIGHT,
                    shadow_maps_enabled: false, // Solari replaces shadow mapping
                    ..default()
                },
                Transform::from_rotation(Quat::from_xyzw(
                    -0.13334629,
                    -0.86597735,
                    -0.3586996,
                    0.3219264,
                )),
            ));
        }
    }

    if key_input.just_pressed(KeyCode::Digit2)
        && let Some(robot_light_material) = robot_light_material
    {
        let mut material = materials.get_mut(&robot_light_material.0).unwrap();
        if material.emissive == LinearRgba::BLACK {
            material.emissive = LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
        } else {
            material.emissive = LinearRgba::BLACK;
        }
    }
}

#[derive(Component)]
struct PatrolPath {
    path: Vec<(Vec3, Quat)>,
    i: usize,
}

fn patrol_path(mut query: Query<(&mut PatrolPath, &mut Transform)>, time: Res<Time<Virtual>>) {
    for (mut path, mut transform) in query.iter_mut() {
        let (mut target_position, mut target_rotation) = path.path[path.i];
        let mut distance_to_target = transform.translation.distance(target_position);
        if distance_to_target < 0.01 {
            transform.translation = target_position;
            transform.rotation = target_rotation;

            path.i = (path.i + 1) % path.path.len();
            (target_position, target_rotation) = path.path[path.i];
            distance_to_target = transform.translation.distance(target_position);
        }

        let direction = (target_position - transform.translation).normalize();
        let movement = direction * time.delta_secs();

        if movement.length() > distance_to_target {
            transform.translation = target_position;
            transform.rotation = target_rotation;
        } else {
            transform.translation += movement;
        }
    }
}

#[derive(Component)]
struct ControlText;

fn update_control_text(
    mut text: Single<&mut Text, With<ControlText>>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    materials: Res<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    time: Res<Time<Virtual>>,
    args: Res<Args>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_camera: Query<
        Has<Dlss<DlssRayReconstructionFeature>>,
        With<SolariLighting>,
    >,
) {
    text.0.clear();

    if args.many_lights != Some(true) {
        if time.is_paused() {
            text.0.push_str("(Space): Resume");
        } else {
            text.0.push_str("(Space): Pause");
        }

        if directional_light.single().is_ok() {
            text.0.push_str("\n(1): Disable directional light");
        } else {
            text.0.push_str("\n(1): Enable directional light");
        }

        match robot_light_material.and_then(|m| materials.get(&m.0)) {
            Some(robot_light_material) if robot_light_material.emissive != LinearRgba::BLACK => {
                text.0.push_str("\n(2): Disable robot emissive light");
            }
            _ => {
                text.0.push_str("\n(2): Enable robot emissive light");
            }
        }
    }

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        if matches!(dlss_camera.single(), Ok(true)) {
            text.0.push_str("\n(3): Disable DLSS Ray Reconstruction");
        } else {
            text.0.push_str("\n(3): Enable DLSS Ray Reconstruction");
        }
    } else {
        text.0
            .push_str("\nDenoising: DLSS Ray Reconstruction not supported");
    }

    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
    text.0
        .push_str("\nDenoising: App not compiled with DLSS support");
}

#[derive(Component)]
struct PerformanceText;

fn update_performance_text(
    mut text: Single<&mut Text, With<PerformanceText>>,
    diagnostics: Res<DiagnosticsStore>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_camera: Query<
        Has<Dlss<DlssRayReconstructionFeature>>,
        With<SolariLighting>,
    >,
) {
    text.0.clear();

    let mut total = 0.0;
    let mut add_diagnostic = |name: &str, path: &'static str| {
        let path = DiagnosticPath::new(path);
        if let Some(value) = diagnostics.get(&path).and_then(Diagnostic::smoothed) {
            text.push_str(&format!("{name:17}  {value:.2} ms\n"));
            total += value;
        }
    };

    (add_diagnostic)(
        "Light tiles",
        "render/solari_lighting/presample_light_tiles/elapsed_gpu",
    );
    (add_diagnostic)(
        "World cache",
        "render/solari_lighting/world_cache/elapsed_gpu",
    );
    (add_diagnostic)("Lighting", "render/solari_lighting/lighting/elapsed_gpu");
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if matches!(dlss_camera.single(), Ok(true)) {
        (add_diagnostic)("DLSS-RR", "render/dlss_ray_reconstruction/elapsed_gpu");
    }
    text.push_str(&format!("{:17}  {total:.2} ms\n", "Total"));

    if let Some(world_cache_active_cells_count) = diagnostics
        .get(&DiagnosticPath::new(
            "render/solari_lighting/world_cache_active_cells_count",
        ))
        .and_then(Diagnostic::smoothed)
    {
        text.push_str(&format!(
            "\nWorld cache cells {} ({:.0}%)",
            world_cache_active_cells_count as u32,
            (world_cache_active_cells_count * 100.0) / (2u64.pow(20) as f64)
        ));
    }
}

#[derive(Component)]
struct DebugText;

fn spawn_debug_text(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(0.0),
            left: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_right(Val2::all(px(4.0))),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            DebugText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

fn update_debug_text(
    mut text: Single<&mut Text, With<DebugText>>,
    solari_lighting: Single<&SolariLighting>,
    diagnostics: Res<DiagnosticsStore>,
) {
    text.0.clear();

    for (_, key, view) in DEBUG_VIEW_KEYS {
        let marker = if view == solari_lighting.debug_view {
            ">"
        } else {
            " "
        };
        text.push_str(&format!("{marker} ({key}) {}\n", view.name()));
    }

    text.push_str(&format!(
        "\n(4): Debug counters: {}\n",
        if solari_lighting.debug_counters {
            "on (slow)"
        } else {
            "off"
        },
    ));

    if let Some(legend) = debug_view_legend(solari_lighting.debug_view) {
        text.push_str(&format!("\n{legend}\n"));
    }

    if !solari_lighting.debug_counters {
        return;
    }

    let counter = |name: &str| {
        diagnostics
            .get(&DiagnosticPath::new(format!(
                "render/solari_lighting/debug/{name}"
            )))
            .and_then(Diagnostic::smoothed)
    };

    let Some(pixels) = counter("pixels_shaded").filter(|p| *p > 0.0) else {
        return;
    };
    let queries = counter("world_cache_queries").unwrap_or(0.0);

    text.push_str("\nPer-frame rates\n");
    for name in SOLARI_DEBUG_COUNTERS.iter().skip(1) {
        let Some(value) = counter(name) else { continue };

        // Each counter is a ratio against whichever total makes it meaningful.
        let line = match *name {
            "world_cache_queries" => format!("{:31}  {:.2} /px", name, value / pixels),
            "world_cache_probe_exhausted" => {
                if queries > 0.0 {
                    format!("{:31}  {:.1}% of queries", name, value * 100.0 / queries)
                } else {
                    continue;
                }
            }
            // Accumulated per-pixel percent, so this is a mean share.
            "non_resampled_energy_percent" | "noise_relative_std_dev_percent" => {
                format!("{:31}  {:.1}% mean", name, value / pixels)
            }
            "spatial_candidates_rejected" => {
                format!("{:31}  {:.2} /px", name, value / pixels)
            }
            _ => format!("{:31}  {:.1}%", name, value * 100.0 / pixels),
        };
        text.push_str(&line);
        text.push('\n');
    }
}

/// How the camera moves during one segment of the noise trace.
enum TraceMotion {
    /// Hold still. Following a motion segment, this measures how fast the
    /// estimator recovers.
    Hold,
    /// Yaw in place. Large motion vectors with no parallax, so almost nothing
    /// disoccludes but every specular lobe rotates and its history goes stale.
    /// This is the segment that isolates specular temporal reuse.
    Yaw { degrees_per_frame: f32 },
    /// Translate sideways. Parallax, so geometry genuinely disoccludes and
    /// `temporal_rejected_dissimilar` should climb.
    Strafe { units_per_frame: f32 },
}

struct TraceSegment {
    label: &'static str,
    frames: u32,
    motion: TraceMotion,
}

/// Fixed camera path for the noise trace. Everything is indexed by frame number
/// rather than elapsed time, so the same run happens regardless of framerate.
///
/// Pan and strafe are separated deliberately: pan isolates stale specular history
/// with no disocclusion, strafe isolates disocclusion. Each is followed by a hold
/// so the recovery rate is measurable too.
///
/// The yaw segments alternate direction and keep cumulative yaw inside roughly
/// +/- 15 degrees. Panning far enough to leave the diorama would point the camera
/// at empty space, where nothing is shaded and every tally reads zero.
const NOISE_TRACE_PATH: &[TraceSegment] = &[
    TraceSegment {
        label: "settle",
        frames: 90,
        motion: TraceMotion::Hold,
    },
    TraceSegment {
        label: "pan_slow_right",
        frames: 90,
        motion: TraceMotion::Yaw {
            degrees_per_frame: 0.15,
        },
    },
    TraceSegment {
        label: "hold_after_pan_slow",
        frames: 60,
        motion: TraceMotion::Hold,
    },
    TraceSegment {
        label: "pan_fast_left",
        frames: 30,
        motion: TraceMotion::Yaw {
            degrees_per_frame: -0.9,
        },
    },
    TraceSegment {
        label: "hold_after_pan_fast",
        frames: 60,
        motion: TraceMotion::Hold,
    },
    TraceSegment {
        label: "strafe",
        frames: 90,
        motion: TraceMotion::Strafe {
            units_per_frame: 0.02,
        },
    },
    TraceSegment {
        label: "hold_after_strafe",
        frames: 60,
        motion: TraceMotion::Hold,
    },
    // Three times the speed of the segment above, and back the way it came. The
    // gentle strafe only disoccludes ~3% of pixels per frame, which is not
    // representative of how fast a camera actually moves.
    TraceSegment {
        label: "strafe_fast_back",
        frames: 60,
        motion: TraceMotion::Strafe {
            units_per_frame: -0.06,
        },
    },
    TraceSegment {
        label: "hold_after_strafe_fast",
        frames: 60,
        motion: TraceMotion::Hold,
    },
];

#[derive(Resource, Default)]
struct NoiseTrace {
    started: bool,
    frame: u32,
    rows: Vec<String>,
    /// Consecutive ticks that reported nothing shaded, so the path is holding
    /// rather than advancing.
    stalled: u32,
}

/// How long to wait for a measurable frame before giving up. Generous because
/// ticks run fast while there is no geometry yet, so this has to cover the scene
/// download and BLAS build as well as readback warmup and any stretch where the
/// window is not being rendered. Exceeding it means the camera path is pointing
/// off the scene.
const NOISE_TRACE_STALL_LIMIT: u32 = 5000;

fn noise_trace_total_frames() -> u32 {
    NOISE_TRACE_PATH.iter().map(|s| s.frames).sum()
}

/// The segment a frame falls in, and how far into it.
fn noise_trace_segment(frame: u32) -> Option<(&'static TraceSegment, u32)> {
    let mut start = 0;
    for segment in NOISE_TRACE_PATH {
        if frame < start + segment.frames {
            return Some((segment, frame - start));
        }
        start += segment.frames;
    }
    None
}

/// Camera yaw and position at a frame, integrated from the path. Recomputed from
/// scratch each frame so the pose is a pure function of the frame index and
/// cannot drift between runs.
fn noise_trace_pose(frame: u32, start_translation: Vec3) -> (Vec3, f32) {
    let mut yaw = 0.0;
    let mut translation = start_translation;
    for f in 0..frame {
        let Some((segment, _)) = noise_trace_segment(f) else {
            break;
        };
        match segment.motion {
            TraceMotion::Hold => {}
            TraceMotion::Yaw { degrees_per_frame } => yaw += degrees_per_frame.to_radians(),
            TraceMotion::Strafe { units_per_frame } => {
                translation += Quat::from_rotation_y(yaw) * Vec3::X * units_per_frame;
            }
        }
    }
    (translation, yaw)
}

fn drive_noise_trace(
    args: Res<Args>,
    mut trace: ResMut<NoiseTrace>,
    camera: Single<(Entity, &mut Transform, &mut SolariLighting)>,
    diagnostics: Res<DiagnosticsStore>,
    mut time: ResMut<Time<Virtual>>,
    mut start_transform: Local<Option<Transform>>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    let (entity, mut camera_transform, mut solari_lighting) = camera.into_inner();
    let start = *start_transform.get_or_insert(*camera_transform);

    if !trace.started {
        trace.started = true;

        // Setup lives here rather than in a Startup system so it cannot race the
        // system that spawns the camera.
        //
        // Hand control of the camera to the path, and freeze the scene animation so
        // camera motion is the only variable.
        commands.entity(entity).remove::<FreeCamera>();
        time.pause();
        solari_lighting.debug_counters = true;
        // Start every run from an empty temporal history.
        solari_lighting.reset = true;

        info!(
            "Noise trace: driving {} frames over {} segments",
            noise_trace_total_frames(),
            NOISE_TRACE_PATH.len()
        );

        let mut header = String::from("frame,segment");
        for name in SOLARI_DEBUG_COUNTERS {
            header.push(',');
            header.push_str(name);
        }
        trace.rows.push(header);
    }

    let path = args.noise_trace.as_deref().unwrap_or("noise_trace.csv");

    let Some((segment, _)) = noise_trace_segment(trace.frame) else {
        match std::fs::write(path, trace.rows.join("\n") + "\n") {
            Ok(()) => info!("Wrote {} noise trace rows to {path}", trace.rows.len() - 1),
            Err(error) => error!("Failed to write noise trace to {path}: {error}"),
        }
        exit.write(AppExit::Success);
        return;
    };

    // Counter readback lags the GPU by a frame or two, so a row is attributed to
    // the frame it was read on, not the frame it was recorded on. Segments are
    // long enough that this does not matter.
    let counter = |name: &str| {
        diagnostics
            .get(&DiagnosticPath::new(format!(
                "render/solari_lighting/debug/{name}"
            )))
            .and_then(Diagnostic::value)
            .unwrap_or(0.0)
    };

    // Hold the path in place until a frame actually gets measured, so a stretch of
    // unrendered frames cannot silently consume the trace. Without this the camera
    // walks the whole path while every row reads zero.
    if counter("pixels_shaded") == 0.0 {
        camera_transform.translation = start.translation;
        trace.stalled += 1;
        if trace.stalled > NOISE_TRACE_STALL_LIMIT {
            error!(
                "Noise trace stalled at frame {} of {}: nothing was shaded for {} ticks. \
                 Either the window is not rendering, or the camera path points off the scene.",
                trace.frame,
                noise_trace_total_frames(),
                trace.stalled
            );
            exit.write(AppExit::Success);
        }
        return;
    }
    if trace.frame == 0 {
        info!("Noise trace: scene is up, starting the path");
    }
    trace.stalled = 0;

    let mut row = format!("{},{}", trace.frame, segment.label);
    for name in SOLARI_DEBUG_COUNTERS {
        row.push_str(&format!(",{}", counter(name) as u64));
    }
    trace.rows.push(row);

    // Flush periodically so a run cut short (window closed, display change) still
    // leaves usable rows behind.
    if trace.frame.is_multiple_of(60) {
        let _ = std::fs::write(path, trace.rows.join("\n") + "\n");
    }

    let (translation, yaw) = noise_trace_pose(trace.frame, start.translation);
    camera_transform.translation = translation;
    camera_transform.rotation = Quat::from_rotation_y(yaw) * start.rotation;

    trace.frame += 1;
}

/// What the colours in each debug view mean.
fn debug_view_legend(view: SolariDebugView) -> Option<&'static str> {
    match view {
        SolariDebugView::None => None,
        SolariDebugView::SampleProvenance => Some(
            "grey none   yellow NEE direct   pink emissive direct\n\
             green reconnected NEE   orange reconnected emissive\n\
             blue world cache",
        ),
        SolariDebugView::TemporalRejectReason => Some(
            "black accepted   blue off-screen   red surface mismatch\n\
             magenta light despawned   grey no history",
        ),
        SolariDebugView::JacobianRejection => Some(
            "sample discarded: green temporal  blue spatial  red both\n\
             grey: MIS partner dropped only",
        ),
        SolariDebugView::NonResampledOnly
        | SolariDebugView::ResampledOnly
        | SolariDebugView::WorldCache => Some("tonemapped radiance"),
        SolariDebugView::WorldCacheProbeFailure => Some("red probe steps exhausted"),
        SolariDebugView::SampleAge => Some("blue independent  ->  red stale, correlated in time"),
        SolariDebugView::SampleDuplication => {
            Some("blue unique sample  ->  red neighbours share it")
        }
        // Everything else is the shared heatmap ramp.
        _ => Some("blue low  ->  green  ->  yellow  ->  red high"),
    }
}
