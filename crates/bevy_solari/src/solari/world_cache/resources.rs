use super::WORLD_CACHE_SIZE;
use bevy_ecs::{
    system::Resource,
    world::{FromWorld, World},
};
use bevy_render::{render_resource::*, renderer::RenderDevice};
use std::num::NonZeroU64;

#[derive(Resource)]
pub struct SolariWorldCacheResources {
    pub bind_group_layout: BindGroupLayout,
    pub bind_group_layout_no_dispatch: BindGroupLayout,
    pub bind_group: BindGroup,
    pub bind_group_no_dispatch: BindGroup,
    pub active_cells_dispatch_buffer: Buffer,
}

impl FromWorld for SolariWorldCacheResources {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout_entries = &[
            // Checksums
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // Life
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // Irradiance
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(16) }),
                },
                count: None,
            },
            // Cell data
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(32) }),
                },
                count: None,
            },
            // Active cells new irradiance
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(16) }),
                },
                count: None,
            },
            // B1
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // B2
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // Active cell indices
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // Active cells count
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
                },
                count: None,
            },
            // Active cells dispatch
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(12) }),
                },
                count: None,
            },
        ];

        let checksums = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_checksums"),
            size: 4 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let life = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_life"),
            size: 4 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let irradiance = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_irradiance"),
            size: 16 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cell_data = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_cell_data"),
            size: 32 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_cells_new_irradiance = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_active_cells_new_irradiance"),
            size: 16 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b1 = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_b1"),
            size: 4 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b2 = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_b2"),
            size: 4 * 1024,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_cell_indices = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_active_cell_indices"),
            size: 4 * WORLD_CACHE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_cells_count = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_active_cells_count"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_cells_dispatch_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_solari_world_cache_active_cells_dispatch_buffer"),
            size: 12,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let bind_group_entries = &[
            BindGroupEntry {
                binding: 0,
                resource: checksums.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: life.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: irradiance.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: cell_data.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: active_cells_new_irradiance.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: b1.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: b2.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: active_cell_indices.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: active_cells_count.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: active_cells_dispatch_buffer.as_entire_binding(),
            },
        ];

        let bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("solari_world_cache_bind_group_layout"),
                entries: bind_group_layout_entries,
            });

        let bind_group_layout_no_dispatch =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("solari_world_cache_bind_group_layout_no_dispatch"),
                entries: &bind_group_layout_entries[0..bind_group_entries.len() - 1],
            });

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("solari_world_cache_bind_group"),
            layout: &bind_group_layout,
            entries: bind_group_entries,
        });

        let bind_group_no_dispatch = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("solari_world_cache_bind_group_no_dispatch"),
            layout: &bind_group_layout_no_dispatch,
            entries: &bind_group_entries[0..bind_group_entries.len() - 1],
        });

        Self {
            bind_group_layout,
            bind_group_layout_no_dispatch,
            bind_group,
            bind_group_no_dispatch,
            active_cells_dispatch_buffer,
        }
    }
}
