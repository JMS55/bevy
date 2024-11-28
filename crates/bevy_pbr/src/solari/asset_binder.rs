use super::util::is_mesh_solari_compatible;
use bevy_asset::{AssetId, Handle};
use bevy_ecs::{
    system::{Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_image::Image;
use bevy_render::{
    mesh::{allocator::MeshAllocator, Mesh, RenderMesh},
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::*,
    renderer::RenderDevice,
    texture::{FallbackImage, GpuImage},
};
use bevy_utils::HashMap;
use std::{num::NonZeroU32, ops::Deref};

#[derive(Resource)]
pub struct AssetBindings {
    pub bind_group_layout: BindGroupLayout,
    pub mesh_indices: HashMap<AssetId<Mesh>, [u32; 4]>,
    pub image_indices: HashMap<AssetId<Image>, u32>,
    pub bind_group: Option<BindGroup>,
    pub extracted_images: Vec<AssetId<Image>>,
}

impl AssetBindings {
    pub fn get_image_index(&self, handle: Option<Handle<Image>>) -> u32 {
        match handle {
            Some(handle) => *self.image_indices.get(&handle.id()).unwrap_or(&u32::MAX),
            None => u32::MAX,
        }
    }
}

impl FromWorld for AssetBindings {
    fn from_world(world: &mut World) -> Self {
        Self {
            bind_group_layout: world.resource::<RenderDevice>().create_bind_group_layout(
                "solari_assets_bind_group_layout",
                &bind_group_layout_entries(),
            ),
            mesh_indices: HashMap::new(),
            image_indices: HashMap::new(),
            bind_group: None,
            extracted_images: Vec::new(),
        }
    }
}

pub fn copy_extracted_image_ids(
    mut asset_bindings: ResMut<AssetBindings>,
    extracted_images: Res<ExtractedAssets<GpuImage>>,
) {
    asset_bindings.extracted_images.clear();
    asset_bindings.extracted_images.extend(
        extracted_images
            .extracted
            .iter()
            .map(|(asset_id, _)| *asset_id),
    );
}

pub fn prepare_asset_binding_arrays(
    mut asset_bindings: ResMut<AssetBindings>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    render_device: Res<RenderDevice>,
) {
    // TODO: Detect new mesh allocator slabs instead of new meshes
    // TODO: Separate mesh and image bind groups
    if extracted_meshes.extracted.is_empty() && asset_bindings.extracted_images.is_empty() {
        return;
    }

    // Clear existing binding array indices
    asset_bindings.mesh_indices.clear();
    asset_bindings.image_indices.clear();

    // Build binding arrays of vertex and inder buffers
    let mut vertex_buffers = Vec::new();
    let mut index_buffers = Vec::new();
    let mut vertex_buffers_seen = HashMap::new();
    let mut index_buffers_seen = HashMap::new();
    for (asset_id, _) in extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_solari_compatible(mesh))
    {
        let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
        let vertex_buffer_index = match vertex_buffers_seen.get(&vertex_slice.buffer.id()) {
            Some(vertex_buffer_index) => *vertex_buffer_index,
            None => {
                let vertex_buffer_index = vertex_buffers.len() as u32;
                vertex_buffers.push(vertex_slice.buffer.as_entire_buffer_binding());
                vertex_buffers_seen.insert(vertex_slice.buffer.id(), vertex_buffer_index);
                vertex_buffer_index
            }
        };

        let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();
        let index_buffer_index = match index_buffers_seen.get(&index_slice.buffer.id()) {
            Some(index_buffer_index) => *index_buffer_index,
            None => {
                let index_buffer_index = index_buffers.len() as u32;
                index_buffers.push(index_slice.buffer.as_entire_buffer_binding());
                index_buffers_seen.insert(index_slice.buffer.id(), index_buffer_index);
                index_buffer_index
            }
        };

        asset_bindings.mesh_indices.insert(
            *asset_id,
            [
                vertex_buffer_index,
                vertex_slice.range.start,
                index_buffer_index,
                index_slice.range.start,
            ],
        );
    }

    // Build binding arrays of images and samplers
    let device_features = Some(render_device.features());
    let (mut images, mut samplers) = gpu_images
        .iter()
        .filter(|(_, image)| {
            image.texture_format.sample_type(None, device_features)
                == Some(TextureSampleType::Float { filterable: true })
                && image.texture.dimension() == TextureDimension::D2
                && image.texture.sample_count() == 1
        })
        .enumerate()
        .map(|(i, (asset_id, image))| {
            asset_bindings.image_indices.insert(asset_id, i as u32);
            (image.texture_view.deref(), image.sampler.deref())
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    images.push(&fallback_image.d2.texture_view);
    samplers.push(&fallback_image.d2.sampler);

    // Build the new binding group
    asset_bindings.bind_group = Some(render_device.create_bind_group(
        "solari_assets_bind_group",
        &asset_bindings.bind_group_layout,
        &BindGroupEntries::sequential((
            vertex_buffers.as_slice(),
            index_buffers.as_slice(),
            images.as_slice(),
            samplers.as_slice(),
        )),
    ));
}

fn bind_group_layout_entries() -> [BindGroupLayoutEntry; 4] {
    [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: NonZeroU32::new(1000),
        },
    ]
}
