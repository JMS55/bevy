//! Demonstrates realtime dynamic raytraced lighting using Bevy Solari.

use argh::FromArgs;
use bevy::{
    camera::{CameraMainTextureUsages, Exposure},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore},
    gltf::GltfMaterialName,
    image::{ImageAddressMode, ImageLoaderSettings},
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::{diagnostic::RenderDiagnosticsPlugin, render_resource::TextureUsages},
    solari::{
        pathtracer::{Pathtracer, PathtracingPlugin},
        prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins, SolariVarianceDebug},
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
#[derive(FromArgs, Resource, Clone, Copy)]
struct Args {
    /// use the reference pathtracer instead of the realtime lighting system.
    #[argh(switch)]
    pathtracer: Option<bool>,
    /// stress test a scene with many lights.
    #[argh(switch)]
    many_lights: Option<bool>,
    /// a scene with a translating + rotating perfect mirror, for checking that specular motion
    /// vectors (primary surface replacement) track the mirror's own motion. Enable DLSS-RR (key 3)
    /// and watch the reflection for ghosting/smearing as the mirror moves.
    #[argh(switch)]
    moving_mirror: Option<bool>,
    /// minimal repro scene for the specular GI double-count bug. A satin-metal
    /// floor inlay (roughness in [0.2, 0.6), metallic=1) should not match the
    /// `--pathtracer` reference until the reservoir-merge fix is applied.
    #[argh(switch)]
    gi_double_count: Option<bool>,
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
    .insert_resource(args);

    if args.gi_double_count == Some(true) {
        app.add_systems(Startup, setup_gi_double_count);
    } else if args.moving_mirror == Some(true) {
        app.add_systems(Startup, setup_moving_mirror);
    } else if args.many_lights == Some(true) {
        app.add_systems(Startup, setup_many_lights);
    } else {
        app.add_systems(Startup, setup_pica_pica);
    }

    if args.pathtracer == Some(true) {
        app.add_plugins(PathtracingPlugin);
    } else {
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        app.add_systems(Update, toggle_dlss_rr);

        if args.moving_mirror == Some(true) {
            app.add_systems(Update, (pause_scene, move_mirror));
        } else if args.many_lights == Some(true) {
            app.add_systems(Update, toggle_color_noise_reduction);
        } else if args.gi_double_count != Some(true) {
            app.add_systems(Update, (pause_scene, toggle_lights, patrol_path));
        }
        app.add_systems(PostUpdate, (update_control_text, update_performance_text));
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
        camera.insert(SolariLighting::default());
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
            border_radius: BorderRadius::bottom_left(px(4.0)),
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
    ));

    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        // SolariVarianceDebug carries the `disable_color_noise_reduction` toggle (key C).
        // Its presence also enables per-frame variance accumulation, a small overhead.
        camera.insert((SolariLighting::default(), SolariVarianceDebug::default()));
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
            border_radius: BorderRadius::bottom_left(px(4.0)),
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

/// Minimal repro for the specular GI double-count bug.
///
/// A closed, fully-diffuse Cornell-style box (red left wall, green right wall) is the GI source:
/// light from the ceiling panel bounces off the coloured walls onto the floor, so every floor
/// pixel carries a strong, coloured reconnection-GI sample in its reservoir.
///
/// Two coplanar inlays sit in the floor:
///   * a **satin metal** patch (`metallic = 1.0`, `perceptual_roughness = 0.35`) — inside the
///     `[SPECULAR_DOMINANCE_SKIP_RESAMPLING_THRESHOLD (0.2), RECONNECTION_ROUGHNESS_MIN (0.6))`
///     window. It runs temporal+spatial merges (roughness >= 0.2) but can never publish a
///     reconnection sample (`x1_lobe_ok` fails, so all of its own GI goes to
///     `non_resampled_radiance`). Its diffuse floor neighbours pass `pixel_dissimilar` (coplanar,
///     same normal), so their GI reconnection samples get resampled into its reservoir and shaded
///     on top of the already-complete GI — an overcount.
///   * a **diffuse** control patch that behaves normally and should always match the reference.
///
/// Verify: run once with `--gi-double-count` and once with `--gi-double-count --pathtracer` and
/// compare. Before the fix the metal patch reads too bright (and tinted by the floor bounce) while
/// the diffuse patch matches. After the fix, both should converge to the pathtraced reference.
/// Keep the scene/camera static and let it converge (~a second); do not enable DLSS-RR (key 3),
/// which would denoise away the pixel-level comparison.
fn setup_gi_double_count(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<Args>,
) {
    // Emissive radiance of the ceiling panel. Tune until the box is well-exposed (floor ~mid-grey).
    const LIGHT_STRENGTH: f32 = 2000.0;
    // Satin-metal roughness. Must satisfy 0.2 <= r < 0.6 to hit the bug.
    const METAL_ROUGHNESS: f32 = 0.35;

    let mut cuboid = |x: f32, y: f32, z: f32| {
        meshes.add(
            Cuboid::new(x, y, z)
                .mesh()
                .build()
                .with_generated_tangents()
                .unwrap(),
        )
    };
    let slab_mesh = cuboid(8.0, 0.2, 8.0); // floor / ceiling
    let back_mesh = cuboid(8.0, 8.0, 0.2);
    let side_mesh = cuboid(0.2, 8.0, 8.0);
    let light_mesh = cuboid(3.0, 0.1, 3.0);
    let patch_mesh = cuboid(1.6, 0.2, 1.6);

    let diffuse = |c: Color| StandardMaterial {
        base_color: c,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    };
    let gray = materials.add(diffuse(Color::srgb(0.8, 0.8, 0.8)));
    let red = materials.add(diffuse(Color::srgb(0.63, 0.065, 0.05)));
    let green = materials.add(diffuse(Color::srgb(0.14, 0.45, 0.091)));
    let light = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(1.0, 1.0, 1.0) * LIGHT_STRENGTH,
        ..default()
    });
    let metal = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.9),
        perceptual_roughness: METAL_ROUGHNESS,
        metallic: 1.0,
        ..default()
    });
    let diffuse_patch = materials.add(diffuse(Color::srgb(0.9, 0.9, 0.9)));

    // (mesh, material, transform) for every surface. Mesh3d is only added for the realtime path;
    // the pathtracer uses RaytracingMesh3d alone (rasterization is skipped as in the other scenes).
    let surfaces = [
        (
            slab_mesh.clone(),
            gray.clone(),
            Transform::from_xyz(0.0, -0.1, 0.0),
        ), // floor
        (
            slab_mesh.clone(),
            gray.clone(),
            Transform::from_xyz(0.0, 8.1, 0.0),
        ), // ceiling
        (back_mesh, gray, Transform::from_xyz(0.0, 4.0, -4.1)), // back wall
        (side_mesh.clone(), red, Transform::from_xyz(-4.1, 4.0, 0.0)), // left wall (red)
        (side_mesh, green, Transform::from_xyz(4.1, 4.0, 0.0)), // right wall (green)
        (light_mesh, light, Transform::from_xyz(0.0, 7.95, 0.0)), // ceiling light
        // Coplanar floor inlays, raised 5mm so they read as flush inlays without z-fighting.
        (
            patch_mesh.clone(),
            metal,
            Transform::from_xyz(-1.6, -0.095, 0.0),
        ), // BUGGY satin metal
        (
            patch_mesh,
            diffuse_patch,
            Transform::from_xyz(1.6, -0.095, 0.0),
        ), // control diffuse
    ];
    for (mesh, material, transform) in surfaces {
        commands
            .spawn((
                RaytracingMesh3d(mesh.clone()),
                MeshMaterial3d(material),
                transform,
            ))
            .insert_if(Mesh3d(mesh), || args.pathtracer != Some(true));
    }

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Exposure::INDOOR,
        Transform::from_xyz(0.0, 6.0, 10.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));
    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        // Accumulate to a converged reference (no DLSS) so the realtime result can be compared
        // against the pathtracer. Accumulation resets automatically when the camera moves.
        camera.insert(SolariLighting {
            // accumulate: true,
            ..default()
        });
    }

    // Text entities the shared PostUpdate systems query for (kept minimal; DLSS is left off).
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
            border_radius: BorderRadius::bottom_left(px(4.0)),
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

/// A perfect mirror that translates and rotates, in front of a handful of coloured objects.
///
/// The primary-surface-replacement path writes specular motion vectors for mirror pixels by
/// reflecting the hit behind the mirror into virtual space. If those motion vectors only track the
/// reflected geometry (and not the mirror's own motion), DLSS Ray Reconstruction ghosts/smears the
/// reflection whenever the mirror moves. Enable DLSS-RR (key 3) and watch the reflected objects as
/// the mirror slides and rocks; a correct fix keeps the reflection crisp.
fn setup_moving_mirror(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<Args>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    let cuboid = |x: f32, y: f32, z: f32, meshes: &mut Assets<Mesh>| {
        meshes.add(
            Cuboid::new(x, y, z)
                .mesh()
                .build()
                .with_generated_tangents()
                .unwrap(),
        )
    };
    let sphere_mesh = meshes.add(
        Sphere::new(0.6)
            .mesh()
            .build()
            .with_generated_tangents()
            .unwrap(),
    );

    let diffuse = |c: Color| StandardMaterial {
        base_color: c,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    };

    let mirror_origin = Vec3::new(0.0, 1.6, -3.0);

    // Static scene geometry: floor, then coloured objects in front of the mirror (on the camera
    // side) so their reflection is visible. Mesh3d is only added for the realtime path.
    let surfaces = [
        (
            cuboid(24.0, 0.2, 24.0, &mut meshes),
            materials.add(diffuse(Color::srgb(0.5, 0.5, 0.5))),
            Transform::from_xyz(0.0, -0.1, 0.0),
        ),
        (
            cuboid(1.0, 1.0, 1.0, &mut meshes),
            materials.add(diffuse(Color::srgb(0.9, 0.1, 0.1))),
            Transform::from_xyz(-1.8, 0.5, 0.5),
        ),
        (
            cuboid(1.0, 2.0, 1.0, &mut meshes),
            materials.add(diffuse(Color::srgb(0.1, 0.7, 0.2))),
            Transform::from_xyz(1.6, 1.0, -0.3),
        ),
        (
            sphere_mesh.clone(),
            materials.add(diffuse(Color::srgb(0.15, 0.35, 0.9))),
            Transform::from_xyz(0.2, 0.6, 1.3),
        ),
        // An emissive sphere gives the reflection a bright feature that ghosting is easy to spot on.
        (
            sphere_mesh,
            materials.add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::rgb(1.0, 0.85, 0.3) * 200.0,
                ..default()
            }),
            Transform::from_xyz(-0.6, 2.2, -0.5),
        ),
    ];
    for (mesh, material, transform) in surfaces {
        commands
            .spawn((
                RaytracingMesh3d(mesh.clone()),
                MeshMaterial3d(material),
                transform,
            ))
            .insert_if(Mesh3d(mesh), || args.pathtracer != Some(true));
    }

    // The moving mirror: a perfect metallic reflector (metallic = 1, roughness = 0).
    let mirror_mesh = cuboid(5.0, 3.0, 0.2, &mut meshes);
    commands
        .spawn((
            RaytracingMesh3d(mirror_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.0,
                metallic: 1.0,
                ..default()
            })),
            Transform::from_translation(mirror_origin),
            MovingMirror {
                origin: mirror_origin,
            },
        ))
        .insert_if(Mesh3d(mirror_mesh), || args.pathtracer != Some(true));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false, // Solari replaces shadow mapping
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.5, 0.0)),
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
        Transform::from_xyz(0.0, 2.4, 7.0).looking_at(Vec3::new(0.0, 1.2, -3.0), Vec3::Y),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));
    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        camera.insert(SolariLighting::default());
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
            border_radius: BorderRadius::bottom_left(px(4.0)),
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

/// `C` toggles chroma-marginalized (color-noise-reduced) shading, to A/B the
/// vector-valued shading in the spatial merge and light RIS (many-lights scene).
fn toggle_color_noise_reduction(
    key_input: Res<ButtonInput<KeyCode>>,
    mut variance: Query<&mut SolariVarianceDebug, With<SolariLighting>>,
) {
    if key_input.just_pressed(KeyCode::KeyC)
        && let Ok(mut variance) = variance.single_mut()
    {
        variance.disable_color_noise_reduction = !variance.disable_color_noise_reduction;
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
struct MovingMirror {
    /// Position the mirror oscillates around.
    origin: Vec3,
}

/// Slide the mirror side-to-side and rock it back and forth, so both the translation and the
/// rotation of the mirror plane exercise the specular motion vectors.
fn move_mirror(mut query: Query<(&MovingMirror, &mut Transform)>, time: Res<Time<Virtual>>) {
    let t = time.elapsed_secs();
    for (mirror, mut transform) in query.iter_mut() {
        let slide = ops::sin(t * 0.9) * 2.0;
        let rock = ops::sin(t * 0.6) * 0.35;
        transform.translation = mirror.origin + Vec3::new(slide, 0.0, 0.0);
        transform.rotation = Quat::from_rotation_y(rock);
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
    variance: Query<&SolariVarianceDebug, With<SolariLighting>>,
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

    if let Ok(variance) = variance.single() {
        if variance.disable_color_noise_reduction {
            text.0.push_str("\n(C): Enable color-noise reduction");
        } else {
            text.0.push_str("\n(C): Disable color-noise reduction");
        }
    }
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
