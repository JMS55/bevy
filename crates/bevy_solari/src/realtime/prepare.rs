use super::SolariLighting;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, ViewDlssRayReconstructionTextures,
};
use bevy_camera::MainPassResolutionOverride;
use bevy_diagnostic::FrameCount;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bytemuck::{Pod, Zeroable};

/// Size of the `LightSample` shader struct in bytes.
const LIGHT_SAMPLE_STRUCT_SIZE: u64 = 8;

/// Size of the `ResolvedLightSamplePacked` shader struct in bytes.
const RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE: u64 = 24;

/// Size of the `Reservoir` shader struct in bytes.
const RESERVOIR_STRUCT_SIZE: u64 = 48;

pub const LIGHT_TILE_BLOCKS: u64 = 128;
pub const LIGHT_TILE_SAMPLES_PER_BLOCK: u64 = 1024;

/// Amount of entries in the world cache (must be a power of 2, and >= 2^10)
pub const WORLD_CACHE_SIZE: u64 = 2u64.pow(20);
/// Sum of per-cell field sizes in `WorldCache`. Keep in sync with `realtime_bindings.wgsl`.
const WORLD_CACHE_ENTRY_SIZE: u64 = 84;
/// Size of the fixed `b` array (`array<u32, WORLD_CACHE_SIZE / 1024>`).
const WORLD_CACHE_B_SIZE: u64 = (WORLD_CACHE_SIZE / 1024) * size_of::<u32>() as u64;
/// Offset of `active_cells_count`.
pub const WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET: u64 =
    WORLD_CACHE_SIZE * WORLD_CACHE_ENTRY_SIZE + WORLD_CACHE_B_SIZE;
/// Must stay under wgpu's default `max_storage_buffer_binding_size` (128 MiB or 2^27 bytes).
pub const WORLD_CACHE_BUFFER_SIZE: u64 =
    (WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET + size_of::<u32>() as u64).next_multiple_of(16);

/// Names of the scene-wide tallies in the debug counter buffer, indexed by their
/// slot in it. Keep in sync with the `DEBUG_COUNTER_*` constants in `debug.wgsl`.
///
/// Rates are read as a ratio against another counter rather than in isolation:
/// most are per-pixel events to divide by `pixels_shaded`, while
/// `world_cache_probe_exhausted` divides by `world_cache_queries` and
/// `noise_specular_percent` divides by `specular_pixels`.
///
/// The two jacobian tallies per merge are separated because they fail
/// differently. `discard_neighbor` means the neighbour's sample could not be
/// selected at all, costing variance. `inflate_canonical` means only the
/// neighbour's ability to have produced the canonical sample was zeroed, which
/// snaps the canonical MIS weight to one and biases instead.
pub const SOLARI_DEBUG_COUNTERS: [&str; 34] = [
    "pixels_shaded",
    "specular_pixels",
    "temporal_reprojected_offscreen",
    "temporal_rejected_dissimilar",
    "temporal_rejected_light_despawned",
    "temporal_no_history",
    "x2_not_reusable",
    "spatial_no_neighbor_found",
    "spatial_candidates_rejected",
    "jacobian_temporal_discard_neighbor",
    "jacobian_temporal_inflate_canonical",
    "jacobian_spatial_discard_neighbor",
    "jacobian_spatial_inflate_canonical",
    "world_cache_probe_exhausted",
    "world_cache_queries",
    "path_terminated_into_cache",
    "path_killed_by_russian_roulette",
    "non_resampled_energy_percent",
    "noise_relative_std_dev_percent",
    "noise_specular_percent",
    "noise_diffuse_percent",
    "noise_resampled_percent",
    "noise_resampled_specular_percent",
    "noise_non_resampled_share_percent",
    // Perception tracks the worst regions, not the mean, so the noise tallies also
    // come split by category and as a tail count. A mean over every pixel buries a
    // small number of very bad pixels, which is what disocclusion and the specular
    // bypass produce.
    "history_rejected_pixels",
    "noise_history_rejected_percent",
    "noise_bypass_percent",
    "noise_over_100pct_pixels",
    "noise_over_200pct_pixels",
    "confidence_weight_x10",
    // Correlation, not variance. Reuse trades independence for variance reduction,
    // and a denoiser can only remove noise that is roughly independent per pixel.
    // Where valid samples are scarce, reuse concentrates a few samples across whole
    // neighbourhoods and many frames, which reads as structure and survives
    // denoising as blotches. These two make that cost visible.
    "sample_age_frames",
    "sample_duplication_percent",
    "sample_duplication_specular_percent",
    "sample_duplication_over_25pct_pixels",
];

/// GPU representation of the user-configurable [`SolariLighting`] settings, plus
/// per-frame state.
///
/// Field order and types must match the `SolariLightingSettings` struct in
/// `realtime_bindings.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolariLightingUniforms {
    confidence_weight_cap: f32,
    specular_confidence_weight_cap: f32,
    primary_di_samples: u32,
    secondary_di_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_direct_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_cell_updates_soft_target: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
    debug_view: u32,
    debug_counters: u32,
    /// Set for the frame the debug buffers were (re)allocated on, so the noise
    /// estimator discards the uninitialized history instead of decaying it.
    debug_reset: u32,
}

impl SolariLightingUniforms {
    fn new(settings: &SolariLighting, frame_count: u32, debug_reset: bool) -> Self {
        Self {
            confidence_weight_cap: settings.confidence_weight_cap,
            specular_confidence_weight_cap: settings.specular_confidence_weight_cap,
            primary_di_samples: settings.primary_di_samples,
            secondary_di_samples: settings.secondary_di_samples,
            max_bounces: settings.max_bounces,
            world_cache_max_temporal_samples: settings.world_cache_max_temporal_samples,
            world_cache_direct_light_sample_count: settings.world_cache_direct_light_sample_count,
            world_cache_max_gi_ray_distance: settings.world_cache_max_gi_ray_distance,
            world_cache_cell_updates_soft_target: settings.world_cache_cell_updates_soft_target,
            world_cache_position_base_cell_size: settings.world_cache_position_base_cell_size,
            world_cache_position_lod_scale: settings.world_cache_position_lod_scale,
            frame_rng: frame_count.wrapping_mul(5782582),
            reset: settings.reset as u32,
            debug_view: settings.debug_view as u32,
            debug_counters: settings.debug_counters as u32,
            debug_reset: debug_reset as u32,
        }
    }
}

/// Internal rendering resources used for Solari lighting.
#[derive(Component)]
pub struct SolariLightingResources {
    pub constants: Buffer,
    pub light_tile_samples: Buffer,
    pub light_tile_resolved_samples: Buffer,
    pub reservoirs_a: Buffer,
    pub reservoirs_b: Buffer,
    pub world_cache: Buffer,
    pub world_cache_active_cells_dispatch: Buffer,
    /// Scene-wide tallies, one `u32` per entry in [`SOLARI_DEBUG_COUNTERS`].
    /// Cleared and read back every frame.
    pub debug_counters: Buffer,
    /// Per-pixel debug bitfield, carrying signals from the initial/temporal pass
    /// through to the shading pass that emits the debug view.
    pub debug_flags: Buffer,
    /// Ping-ponged first and second moments of the shaded output, for the
    /// per-pixel noise estimate.
    pub noise_moments: [CachedTexture; 2],
    /// Whether the debug resources above are allocated at full size.
    pub debug_enabled: bool,
    pub view_size: UVec2,
}

pub fn prepare_solari_lighting_resources(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&SolariLightingResources>,
        Option<&MainPassResolutionOverride>,
    )>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&SolariLightingResources>,
        Option<&MainPassResolutionOverride>,
        Has<Dlss<DlssRayReconstructionFeature>>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for query_item in &query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (entity, camera, solari_lighting, solari_lighting_resources, resolution_override) =
            query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (
            entity,
            camera,
            solari_lighting,
            solari_lighting_resources,
            resolution_override,
            has_dlss_rr,
        ) = query_item;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        let debug_enabled =
            solari_lighting.debug_view.needs_debug_resources() || solari_lighting.debug_counters;

        if let Some(solari_lighting_resources) = solari_lighting_resources
            && solari_lighting_resources.view_size == view_size
            && solari_lighting_resources.debug_enabled == debug_enabled
        {
            // The constants uniform can change every frame, so always upload it.
            render_queue.write_buffer(
                &solari_lighting_resources.constants,
                0,
                bytemuck::bytes_of(&SolariLightingUniforms::new(
                    solari_lighting,
                    frame_count.0,
                    false,
                )),
            );
            continue;
        }

        // Everything below reallocates, so the noise history starts out garbage.
        let uniforms = SolariLightingUniforms::new(solari_lighting, frame_count.0, true);

        let constants = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("solari_lighting_constants"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let light_tile_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_samples"),
            size: LIGHT_TILE_BLOCKS * LIGHT_TILE_SAMPLES_PER_BLOCK * LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let light_tile_resolved_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_resolved_samples"),
            size: LIGHT_TILE_BLOCKS
                * LIGHT_TILE_SAMPLES_PER_BLOCK
                * RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let reservoirs_buffer = |name| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: (view_size.x * view_size.y) as u64 * RESERVOIR_STRUCT_SIZE,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let reservoirs_a = reservoirs_buffer("solari_lighting_reservoirs_a");
        let reservoirs_b = reservoirs_buffer("solari_lighting_reservoirs_b");

        let world_cache = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache"),
            size: WORLD_CACHE_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let world_cache_active_cells_dispatch = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache_active_cells_dispatch"),
            size: size_of::<[u32; 3]>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let debug_counters = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_debug_counters"),
            size: (SOLARI_DEBUG_COUNTERS.len() * size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The per-pixel debug resources are only worth their memory while a debug
        // view or the counters are on, but the bind group still needs something
        // bound, so shrink them to the minimum instead of dropping them.
        let debug_flags = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_debug_flags"),
            size: if debug_enabled {
                (view_size.x * view_size.y) as u64 * size_of::<u32>() as u64
            } else {
                size_of::<u32>() as u64
            },
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let noise_moments_size = if debug_enabled { view_size } else { UVec2::ONE };
        let noise_moments = [0, 1].map(|i| {
            // Rgba32Float because the second moment of HDR luminance overflows f16.
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_noise_moments"),
                size: noise_moments_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let default_view = texture.create_view(&TextureViewDescriptor {
                label: Some(if i == 0 {
                    "solari_lighting_noise_moments_0"
                } else {
                    "solari_lighting_noise_moments_1"
                }),
                ..Default::default()
            });
            CachedTexture {
                texture,
                default_view,
            }
        });

        commands.entity(entity).insert(SolariLightingResources {
            constants,
            light_tile_samples,
            light_tile_resolved_samples,
            reservoirs_a,
            reservoirs_b,
            world_cache,
            world_cache_active_cells_dispatch,
            debug_counters,
            debug_flags,
            noise_moments,
            debug_enabled,
            view_size,
        });

        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        if has_dlss_rr {
            let diffuse_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_diffuse_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let diffuse_albedo_view = diffuse_albedo.create_view(&TextureViewDescriptor::default());

            let specular_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_albedo_view =
                specular_albedo.create_view(&TextureViewDescriptor::default());

            let normal_roughness = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_normal_roughness"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let normal_roughness_view =
                normal_roughness.create_view(&TextureViewDescriptor::default());

            let specular_motion_vectors = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_motion_vectors"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_motion_vectors_view =
                specular_motion_vectors.create_view(&TextureViewDescriptor::default());

            commands
                .entity(entity)
                .insert(ViewDlssRayReconstructionTextures {
                    diffuse_albedo: CachedTexture {
                        texture: diffuse_albedo,
                        default_view: diffuse_albedo_view,
                    },
                    specular_albedo: CachedTexture {
                        texture: specular_albedo,
                        default_view: specular_albedo_view,
                    },
                    normal_roughness: CachedTexture {
                        texture: normal_roughness,
                        default_view: normal_roughness_view,
                    },
                    specular_motion_vectors: CachedTexture {
                        texture: specular_motion_vectors,
                        default_view: specular_motion_vectors_view,
                    },
                });
        }
    }
}
