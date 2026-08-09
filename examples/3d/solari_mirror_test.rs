//! Test scene for Solari's mirror handling and primary surface replacement (PSR).
//!
//! PSR means that when the camera looks at a perfect mirror, the denoiser is told about the surface
//! seen *through* the mirror rather than the mirror itself — its albedo, normal, depth and motion.
//! Otherwise the denoiser tries to reproject a reflection as though it were painted on the glass,
//! which smears whenever the camera or the reflected object moves.
//!
//! The scene isolates the four cases that behave differently:
//!
//! - **A single flat mirror.** The control case. Unfolding along the camera ray and reflecting the
//!   hit about the mirror plane are provably identical for one flat reflector, so anything wrong
//!   here is the path-length accumulation rather than the geometry.
//! - **Two mirrors with skew normals.** Where the constructions actually diverge, and where the
//!   order the reflections compose in matters. Deliberately *not* parallel — parallel mirrors
//!   commute, so a corridor of facing mirrors would pass either way and prove nothing.
//! - **Two chrome spheres of very different radii.** Curvature is knowingly not corrected for, so
//!   the tight one should be refused replacement while the gentle dome is still accepted. The gate
//!   scales with reflection length as well, which is what the dolly is for: a convex mirror's
//!   camera-facing side reflects back past the camera, so distance is what varies that term.
//! - **A moving object** reflected in all of the above, which is what exercises the motion side.
//! - **A directional light seen through a mirror.** The sun reaches only a hard-edged strip of floor,
//!   and one mirror is tilted down to look at that strip. Switching the sun off must darken the strip
//!   and its reflection by the same amount: an analytic light is delivered by next-event estimation at
//!   the *reflected* surface, which is the part a future move of mirror shading out of ReSTIR would
//!   most easily drop.
//! - **Two polished-but-not-mirror surfaces**, a brushed floor strip and a satin sphere. Neither is
//!   ever replaced; both only gain a specular motion vector saying their reflection moves differently
//!   from the surface carrying it. The sphere is the curved-and-glossy case the curvature refusal
//!   used to take a motion vector away from, despite it writing no virtual depth to shatter.
//!
//! The camera sways sideways on its own, because lateral parallax is what makes a wrong virtual
//! image obvious — a still frame looks fine even when the motion vectors are badly wrong.
//!
//! Controls: `3` DLSS Ray Reconstruction, `Space` pauses the camera, `M` pauses the moving objects,
//! `[` and `]` dolly in and out.
//!
//! PSR behaviour is not runtime-switchable — the individual pieces are exercised by the geometry
//! above rather than by keys. Since it exists to feed DLSS Ray Reconstruction, none of it changes
//! anything visible without `--features dlss` on a supported GPU; the overlay says so when that is
//! the case.

use bevy::{
    camera::{CameraMainTextureUsages, Exposure},
    ecs::relationship::DescendantIter,
    math::ops,
    mesh::Indices,
    prelude::*,
    render::render_resource::TextureUsages,
    solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
    world_serialization::{WorldAssetRoot, WorldInstanceReady},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::{
    anti_alias::dlss::{
        Dlss, DlssProjectId, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
    },
    render::camera::{MipBias, TemporalJitter},
};

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "b0a2b4d6-1c3e-4f58-9a7b-2d6e8f0a1c34"
    )));

    app.add_plugins((DefaultPlugins, SolariPlugins))
        .init_resource::<DemoState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_input,
                sway_camera,
                move_object,
                move_mirrors,
                update_hud,
            ),
        );

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.add_systems(Update, toggle_dlss_rr);

    app.run();
}

#[derive(Resource)]
struct DemoState {
    camera_paused: bool,
    object_paused: bool,
    /// Accumulated camera phase, so pausing holds position instead of snapping on resume.
    camera_phase: f32,
    object_phase: f32,
    /// Multiplier on the camera's distance from the scene. The curvature gate scales with reflection
    /// length, and a convex mirror's camera-facing side reflects back past the camera — so dollying
    /// is what varies the term the gate is actually testing.
    camera_distance: f32,
}

impl Default for DemoState {
    fn default() -> Self {
        Self {
            camera_paused: false,
            object_paused: false,
            camera_phase: 0.0,
            object_phase: 0.0,
            camera_distance: 1.0,
        }
    }
}

/// Solari only accepts meshes carrying exactly `{POSITION, NORMAL, UV_0, TANGENT}`, and none of the
/// primitive shapes generate tangents. A mesh without them is skipped silently — no warning, it just
/// never reaches the acceleration structure — so every mesh here goes through this.
fn raytraced(mesh: impl Into<Mesh>) -> Mesh {
    mesh.into().with_generated_tangents().unwrap()
}

#[derive(Component)]
struct MovingObject;

/// A mirror sliding along its own normal. Moves the virtual world by twice its displacement, which
/// nothing in the motion vector accounts for — the delta is transformed by the reflection chain on the
/// assumption the reflectors are static.
#[derive(Component)]
struct SlidingMirror {
    base: Vec3,
    normal: Vec3,
}

/// A mirror turning about an axis lying in its own plane. Rotates the virtual world by twice the
/// angle, also unaccounted for. Rotation about the *normal* would move nothing at all, which is worth
/// knowing but needs no scene of its own.
#[derive(Component)]
struct TurningMirror {
    base: Quat,
}

#[derive(Component)]
struct Hud;

const CAMERA_BASE: Vec3 = Vec3::new(0.0, 2.2, 7.5);
const LOOK_AT: Vec3 = Vec3::new(0.0, 1.6, -1.0);
const SUN_ILLUMINANCE: f32 = light_consts::lux::FULL_DAYLIGHT * 0.5;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    // A perfect mirror. `metallic` has to survive 8-bit G-buffer quantization as exactly 255 and
    // `perceptual_roughness` as a byte <= 8, so 1.0 / 0.0 rather than anything merely close.
    let mirror = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.97),
        metallic: 1.0,
        perceptual_roughness: 0.0,
        ..default()
    });
    let floor_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.38),
        perceptual_roughness: 0.85,
        ..default()
    });
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.5, 0.45),
        perceptual_roughness: 0.9,
        ..default()
    });

    let floor = meshes.add(raytraced(Plane3d::new(Vec3::Y, Vec2::splat(12.0))));
    commands.spawn((
        RaytracingMesh3d(floor.clone()),
        Mesh3d(floor),
        MeshMaterial3d(floor_material),
    ));

    // Back wall, so the mirrors have something with structure to reflect.
    let wall = meshes.add(raytraced(Plane3d::new(Vec3::Z, Vec2::new(12.0, 4.0))));
    commands.spawn((
        RaytracingMesh3d(wall.clone()),
        Mesh3d(wall),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 4.0, -9.0),
    ));

    // Enclose the room. Without a front wall and ceiling the spheres' camera-facing reflections fly
    // off into nothing, the chain misses, and those pixels quietly fall back to no replacement — a
    // hit/miss mosaic across a curved surface that muddles the curvature test with a second effect.
    let front_wall = meshes.add(raytraced(Plane3d::new(Vec3::NEG_Z, Vec2::new(12.0, 4.0))));
    commands.spawn((
        RaytracingMesh3d(front_wall.clone()),
        Mesh3d(front_wall),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 4.0, 12.0),
    ));
    // Ceiling in two halves with a strip open between them, so a directional light can reach the
    // mirrors. A sealed room blocks every sun shadow ray, which is what silently made this scene almost
    // black the first time; the gap is narrow enough that most reflections still land on something.
    let ceiling = meshes.add(raytraced(Plane3d::new(Vec3::NEG_Y, Vec2::new(5.25, 12.0))));
    for x in [-6.75, 6.75] {
        commands.spawn((
            RaytracingMesh3d(ceiling.clone()),
            Mesh3d(ceiling.clone()),
            MeshMaterial3d(wall_material.clone()),
            Transform::from_xyz(x, 7.0, 0.0),
        ));
    }

    // The sun, and the reason the ceiling has a gap at all. Its shaft lands on a strip of floor at
    // roughly x in [0.25, 3.25], z in (-1.5, 3.2) — the only sunlit ground in the room, with edges hard
    // enough that its presence is never in doubt.
    //
    // Aimed steeply along +z deliberately. The lamp quad hangs at y = 6.8 spanning z in [-9, 7], *below*
    // the ceiling gap at y = 7, so a shallower sun is swallowed by the lamp before it ever reaches the
    // opening. That is the trap here: the light looks correctly placed, the gap looks open, and the
    // contribution is nevertheless exactly zero. Only rays climbing fast enough in +z to clear the
    // lamp's far edge get out.
    //
    // `shadow_maps_enabled` is irrelevant either way — Solari traces its shadow rays and ignores it.
    commands.spawn((
        DirectionalLight {
            illuminance: SUN_ILLUMINANCE,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-3.0, 12.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Control case: one flat mirror. Both virtual-image constructions agree exactly here, so anything
    // wrong in this half of the frame is the path-length accumulation rather than the geometry.
    let panel = meshes.add(raytraced(Plane3d::new(
        Vec3::new(0.45, 0.0, 1.0).normalize(),
        Vec2::new(2.0, 1.75),
    )));
    commands.spawn((
        RaytracingMesh3d(panel.clone()),
        Mesh3d(panel),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(-3.6, 1.75, -3.2),
    ));

    // A strongly coloured mirror, so the chain tint has something to show. Every other mirror here is
    // near-white, which makes tinting by chain reflectance almost a no-op — the effect is only
    // legible on a metal that actually colours what it reflects.
    let gold_panel = meshes.add(raytraced(Plane3d::new(
        Vec3::new(0.25, 0.0, 1.0).normalize(),
        Vec2::new(1.2, 1.5),
    )));
    commands.spawn((
        RaytracingMesh3d(gold_panel.clone()),
        Mesh3d(gold_panel),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.72, 0.25),
            metallic: 1.0,
            perceptual_roughness: 0.0,
            ..default()
        })),
        Transform::from_xyz(-2.9, 1.5, -3.2),
    ));

    // The case that matters: two mirrors facing each other but tilted, so their normals are skew.
    // A ray entering the gap bounces off both, which is exactly when folding the whole chain about
    // the first mirror stops being correct.
    let skew_left = meshes.add(raytraced(Plane3d::new(
        Vec3::new(1.0, 0.0, 0.34).normalize(),
        Vec2::new(1.6, 1.5),
    )));
    commands.spawn((
        RaytracingMesh3d(skew_left.clone()),
        Mesh3d(skew_left),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(2.1, 1.5, -3.4),
    ));
    let skew_right = meshes.add(raytraced(Plane3d::new(
        Vec3::new(-1.0, 0.0, 0.34).normalize(),
        Vec2::new(1.6, 1.5),
    )));
    commands.spawn((
        RaytracingMesh3d(skew_right.clone()),
        Mesh3d(skew_right),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(5.3, 1.5, -3.4),
    ));

    // A textured, geometrically detailed subject, sitting in front of the mirror plane so it is what
    // the panels reflect.
    //
    // Every other material here is a flat colour, which makes the albedo guides constant — and a
    // constant swapped for a different constant gives the denoiser nothing to get wrong, which is
    // why deliberately corrupting those guides changed nothing at all. Detailed albedo and normal
    // maps are the only way that half of the pass can be observed.
    commands
        .spawn((
            WorldAssetRoot(
                asset_server
                    .load(GltfAssetLabel::Scene(0).from_asset("models/FlightHelmet/FlightHelmet.gltf")),
            ),
            Transform::from_xyz(1.1, 0.0, 1.4).with_scale(Vec3::splat(4.5)),
        ))
        .observe(add_raytracing_meshes_on_scene_load);

    // The two dielectric cases. Neither is a metal, so neither got any replacement at all under the
    // old `metallic > 0.9999` gate — a smooth surface is a mirror because its specular lobe carries
    // the energy, not because of what it is made of.
    //
    // Black: almost nothing left to reflect diffusely, so the specular fraction is near 1 and this
    // behaves like a mirror. Blue: `F0` is only 0.04 against a coloured diffuse lobe, so the specular
    // fraction is a few percent — it keeps its own depth, motion and normal, and the reflection is
    // mixed into its albedo rather than taking the pixel over.
    let dielectric_panel = meshes.add(raytraced(Plane3d::new(
        Vec3::new(0.55, 0.0, 1.0).normalize(),
        Vec2::new(0.9, 0.8),
    )));
    commands.spawn((
        RaytracingMesh3d(dielectric_panel.clone()),
        Mesh3d(dielectric_panel.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.01, 0.01, 0.012),
            metallic: 0.0,
            perceptual_roughness: 0.0,
            ..default()
        })),
        Transform::from_xyz(-5.6, 0.75, 0.4),
    ));
    commands.spawn((
        RaytracingMesh3d(dielectric_panel.clone()),
        Mesh3d(dielectric_panel),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.28, 0.5),
            metallic: 0.0,
            perceptual_roughness: 0.0,
            ..default()
        })),
        Transform::from_xyz(-4.0, 0.75, 0.4),
    ));

    // Curvature is deliberately not corrected for, so this one is expected to misbehave under
    // motion. Kept in frame so that stays a known limitation rather than a surprise.
    // Default UV sphere rather than an icosphere: tangent generation needs sane UVs, and an
    // icosphere's are seamed enough that it can fail outright.
    let sphere = meshes.add(raytraced(Sphere::new(0.8).mesh().build()));
    commands.spawn((
        RaytracingMesh3d(sphere.clone()),
        Mesh3d(sphere),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(-0.4, 0.8, -1.2),
    ));

    // The directional light test. A flat mirror hung face-down and tilted back, so what the camera sees
    // in it is the sunlit floor rather than anything at eye level.
    //
    // It has to be aimed, not just placed. Sunlight only reaches surfaces facing roughly +z and up,
    // because that is the only direction that clears the lamp — and a mirror the camera can see must
    // itself face +z, so it can never show the lit *face* of a vertical surface. The floor is the way
    // out: it faces straight up, it is squarely in the shaft, and a mirror tilted down can look at it
    // while still facing the camera.
    //
    // Geometry, for anyone moving this: from the camera the reflected ray leaves at about
    // (0.20, -0.82, 0.53) and meets the floor near (2.4, 0, 0.7), comfortably inside the lit strip. The
    // camera's sway walks that point from x ~ 1.4 to ~ 3.3, so the shaft's hard edge sweeps across the
    // panel rather than sitting still — which is what makes it a reprojection test and not just a
    // brightness test.
    let sun_mirror = meshes.add(raytraced(Plane3d::new(
        Vec3::new(0.0, -0.5, 0.866),
        Vec2::new(0.7, 0.55),
    )));
    commands.spawn((
        RaytracingMesh3d(sun_mirror.clone()),
        Mesh3d(sun_mirror),
        MeshMaterial3d(mirror.clone()),
        Transform::from_xyz(1.75, 2.6, -1.0),
    ));

    // Same material, an eighth of the curvature, mostly sunk into the floor so only a gentle cap
    // shows. Its virtual image is just as wrong as the small sphere's, but it varies slowly enough
    // across the screen that the denoiser can still reproject it — so the gate should accept this one
    // and reject the small sphere. Two spheres rather than one is what separates "too curved" from
    // "curved at all".
    let dome = meshes.add(raytraced(Sphere::new(6.0).mesh().build()));
    commands.spawn((
        RaytracingMesh3d(dome.clone()),
        Mesh3d(dome),
        MeshMaterial3d(mirror),
        Transform::from_xyz(3.4, -5.35, 0.6),
    ));

    // The only thing with object motion of its own, so it is what separates a correct motion vector
    // from one that merely handles the camera. Emissive, because a dielectric reflects about 4% of what
    // it sees and a diffuse object at that level is not identifiable enough to judge anything by.
    //
    // It orbits *in front* of the mirror plane. Every panel here faces roughly towards the camera, so
    // they reflect what is in front of them — an object behind them, which is where this used to sit,
    // appears in no reflection at all.
    let cube = meshes.add(raytraced(Cuboid::from_length(0.7)));
    commands.spawn((
        RaytracingMesh3d(cube.clone()),
        Mesh3d(cube),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.15, 0.1),
            emissive: LinearRgba::rgb(900.0, 60.0, 30.0),
            perceptual_roughness: 0.6,
            ..default()
        })),
        Transform::from_xyz(0.0, 1.3, 1.5),
        MovingObject,
    ));

    // Moving reflectors, which the motion vector does not handle. Tinted, because a white mirror
    // reflecting a white room cannot be picked out at all. Near foreground flanking centre: at z = 3.5
    // the camera is about 4 m away and the visible half-width is ~2.8, so x = ±2.2 is comfortably
    // inside the frustum.
    let moving_mirror_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.85, 0.95),
        metallic: 1.0,
        perceptual_roughness: 0.0,
        ..default()
    });
    let moving_mirror_normal = Vec3::new(0.0, 0.0, 1.0);
    let moving_panel = meshes.add(raytraced(Plane3d::new(
        moving_mirror_normal,
        Vec2::new(0.8, 0.65),
    )));
    let sliding_base = Vec3::new(-2.2, 1.2, 3.5);
    commands.spawn((
        RaytracingMesh3d(moving_panel.clone()),
        Mesh3d(moving_panel.clone()),
        MeshMaterial3d(moving_mirror_material.clone()),
        Transform::from_translation(sliding_base),
        SlidingMirror {
            base: sliding_base,
            normal: moving_mirror_normal,
        },
    ));
    commands.spawn((
        RaytracingMesh3d(moving_panel.clone()),
        Mesh3d(moving_panel),
        MeshMaterial3d(moving_mirror_material),
        Transform::from_xyz(2.2, 1.2, 3.5),
        TurningMirror {
            base: Quat::IDENTITY,
        },
    ));

    // The glossy cases. Nothing else in this scene lives in the band that `9` governs: every surface
    // here is either a mirror at perceptual roughness 0 or a matte one at 0.85, and the interesting
    // range is 0.032 to 0.25 — wide enough that the lobe is not a direction any more, tight enough
    // that the reflection is still recognisably a reflection.
    //
    // 0.16 is chosen to survive the G-buffer's byte quantization with room to spare: it lands on
    // 41/255, whose alpha of 0.026 sits well inside the 0.001 to 0.0625 the shader tests.
    let satin = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.86, 0.9),
        metallic: 1.0,
        perceptual_roughness: 0.16,
        ..default()
    });

    // A brushed strip laid over the floor in the near foreground, where the orbiting cube passes
    // straight through its reflection field. This is the smear test: without a specular motion vector
    // the denoiser believes the cube's reflection is painted onto the floor and drags it.
    //
    // Sited at z ~ 4.6 on purpose. Reflected rays leave a floor this close to the camera steeply
    // enough to reach the ceiling gap at x in [-1.5, 1.5] and escape, so part of the strip has no
    // reflection to find at all — which is the only place in this room that exercises the
    // nothing-out-there path. The camera's sway swings that region across the strip.
    let glossy_strip = meshes.add(raytraced(Plane3d::new(Vec3::Y, Vec2::new(2.2, 1.6))));
    commands.spawn((
        RaytracingMesh3d(glossy_strip.clone()),
        Mesh3d(glossy_strip),
        MeshMaterial3d(satin.clone()),
        // Lifted clear of the floor rather than replacing it, so the surrounding matte floor stays as
        // a side-by-side reference for how much of the smear is the glossy handling.
        Transform::from_xyz(0.0, 0.02, 4.6),
    ));

    // Curved *and* glossy, which is the combination the curvature refusal was over-applying to. It
    // writes no virtual depth, so there is no depth field for its curvature to shatter — but the gate
    // did not distinguish, and took its motion vector away along with the mirrors'.
    let satin_sphere = meshes.add(raytraced(Sphere::new(0.8).mesh().build()));
    commands.spawn((
        RaytracingMesh3d(satin_sphere.clone()),
        Mesh3d(satin_sphere),
        MeshMaterial3d(satin),
        Transform::from_xyz(-2.2, 0.8, 1.0),
    ));

    // Solari takes its light from emissive geometry, so the "lamp" is a real quad above the scene.
    //
    // It has to be large, not just bright. Total power is radiance times area, and a small panel that
    // reads as white still leaves a room this size at a few percent of its brightness. Widening it
    // lights the room without blowing the lamp out further.
    let lamp = meshes.add(raytraced(Plane3d::new(Vec3::NEG_Y, Vec2::new(8.0, 8.0))));
    commands.spawn((
        RaytracingMesh3d(lamp.clone()),
        Mesh3d(lamp),
        // Emissive is radiance, and it is competing with a daylight-strength sun, so this has to be
        // in the thousands to register at all. Values that look reasonable next to an LDR colour
        // picker render as black here.
        MeshMaterial3d(materials.add(StandardMaterial {
            emissive: LinearRgba::rgb(4000.0, 3600.0, 3000.0),
            ..default()
        })),
        Transform::from_xyz(0.0, 6.8, -1.0),
    ));

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Transform::from_translation(CAMERA_BASE).looking_at(LOOK_AT, Vec3::Y),
        // Lit entirely by one emissive panel, so meter it as an interior rather than leaving the
        // default, which is calibrated for outdoor levels and renders this room dim.
        Exposure::INDOOR,
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariLighting::default(),
    ));

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: Default::default(),
            reset: Default::default(),
            _phantom_data: Default::default(),
        });
    }
    let _ = &mut camera;

    commands.spawn((
        Hud,
        Text::default(),
        // The room is lit to interior levels and mostly pale, so white text washed out wherever it
        // crossed a wall or the lit floor.
        TextColor(Color::BLACK),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn handle_input(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<DemoState>) {
    if keys.pressed(KeyCode::BracketLeft) {
        state.camera_distance = (state.camera_distance - 0.01).max(0.12);
    }
    if keys.pressed(KeyCode::BracketRight) {
        // Capped at 1.0 so dollying out cannot push the camera through the front wall.
        state.camera_distance = (state.camera_distance + 0.01).min(1.0);
    }
    if keys.just_pressed(KeyCode::Space) {
        state.camera_paused = !state.camera_paused;
    }
    if keys.just_pressed(KeyCode::KeyM) {
        state.object_paused = !state.object_paused;
    }
}

fn sway_camera(
    time: Res<Time>,
    mut state: ResMut<DemoState>,
    mut camera: Single<&mut Transform, With<Camera3d>>,
) {
    if !state.camera_paused {
        state.camera_phase += time.delta_secs() * 0.6;
    }
    // Lateral motion specifically: parallax across the mirror plane is what makes a wrong virtual
    // image visible. Dollying in and out barely moves the reflection at all.
    let offset = Vec3::new(ops::sin(state.camera_phase) * 2.6 * state.camera_distance, 0.0, 0.0);
    let base = LOOK_AT + (CAMERA_BASE - LOOK_AT) * state.camera_distance;
    **camera = Transform::from_translation(base + offset).looking_at(LOOK_AT, Vec3::Y);
}

fn move_object(
    time: Res<Time>,
    mut state: ResMut<DemoState>,
    mut object: Single<&mut Transform, With<MovingObject>>,
) {
    if !state.object_paused {
        state.object_phase += time.delta_secs() * 0.9;
    }
    // Orbiting rather than sliding: it sweeps through several mirrors' reflection fields in turn, and
    // a reflection appearing and disappearing is far easier to judge than one that merely shifts.
    object.translation = Vec3::new(
        ops::sin(state.object_phase) * 3.5,
        1.3,
        1.5 + ops::cos(state.object_phase) * 3.0,
    );
    object.rotation = Quat::from_rotation_y(state.object_phase * 0.7);
}

fn move_mirrors(
    state: Res<DemoState>,
    mut sliding: Query<(&mut Transform, &SlidingMirror)>,
    mut turning: Query<(&mut Transform, &TurningMirror), Without<SlidingMirror>>,
) {
    for (mut transform, mirror) in &mut sliding {
        transform.translation = mirror.base + mirror.normal * ops::sin(state.object_phase) * 0.55;
    }
    for (mut transform, mirror) in &mut turning {
        // About Y, which for a roughly forward-facing panel is an axis lying in its own plane — the
        // component that doubles. A spin about the normal would move the virtual world not at all.
        transform.rotation = mirror.base * Quat::from_rotation_y(ops::sin(state.object_phase) * 0.16);
    }
}

fn update_hud(state: Res<DemoState>, mut hud: Single<&mut Text, With<Hud>>) {
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    let denoiser_note = "";
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
    let denoiser_note =
        "\n\nbuilt without `--features dlss`: primary surface replacement feeds DLSS Ray\nReconstruction only, so it will not change anything on screen right now";

    hud.0 = format!(
        "3 DLSS-RR    Space camera {}    M objects {}    [ ] dolly {:.2}x{}",
        if state.camera_paused { "paused" } else { "swaying" },
        if state.object_paused { "paused" } else { "moving" },
        state.camera_distance,
        denoiser_note,
    );
}

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
fn toggle_dlss_rr(
    keys: Res<ButtonInput<KeyCode>>,
    camera: Single<(Entity, Has<Dlss<DlssRayReconstructionFeature>>), With<SolariLighting>>,
    dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::Digit3) && dlss_rr_supported.is_some() {
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

/// Loaded glTF meshes are not Solari-compatible as authored: the acceleration structure needs exactly
/// `{POSITION, NORMAL, UV_0, TANGENT}` with 32-bit indices, and silently skips anything else. Same
/// fixups as the main Solari example.
fn add_raytracing_meshes_on_scene_load(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    for descendant in DescendantIter::new(&children, scene_ready.entity) {
        if let Ok(Mesh3d(mesh_handle)) = mesh_query.get(descendant) {
            commands
                .entity(descendant)
                .insert(RaytracingMesh3d(mesh_handle.clone()));

            let Some(mut mesh) = meshes.get_mut(mesh_handle) else {
                continue;
            };
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
        }
    }
}
