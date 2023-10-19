use super::scene_types::{GpuSolariMaterial, SolariUniforms};
use bevy_ecs::{
    system::Resource,
    world::{FromWorld, World},
};
use bevy_math::Mat4;
use bevy_render::{render_resource::*, renderer::RenderDevice};
use std::num::NonZeroU32;

use crate::bind_group_layout_entries::*;

#[derive(Resource)]
pub struct SolariSceneBindGroupLayout(pub BindGroupLayout);

impl FromWorld for SolariSceneBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self(render_device.create_bind_group_layout_ext(
            "solari_scene_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // TLAS
                    acceleration_structure(),
                    // Mesh material indices buffer
                    storage_buffer_read_only(false, Some(u32::min_size())),
                    // Index buffers
                    BindGroupLayoutEntryExt {
                        ty: storage_buffer_read_only(false, None), // TODO min_binding_size
                        visibility: Some(ShaderStages::COMPUTE),
                        count: Some(unsafe { NonZeroU32::new_unchecked(10_000) }),
                    },
                    // Vertex buffers
                    BindGroupLayoutEntryExt {
                        ty: storage_buffer_read_only(false, None), // TODO min_binding_size
                        visibility: Some(ShaderStages::COMPUTE),
                        count: Some(unsafe { NonZeroU32::new_unchecked(10_000) }),
                    },
                    // Transforms buffer
                    storage_buffer_read_only(false, Some(Mat4::min_size())),
                    // Material buffer
                    storage_buffer_read_only(false, Some(GpuSolariMaterial::min_size())),
                    // Texture maps
                    BindGroupLayoutEntryExt {
                        visibility: Some(ShaderStages::COMPUTE),
                        ty: texture_2d_f32(),
                        count: Some(unsafe { NonZeroU32::new_unchecked(10_000) }),
                    },
                    // Texture map samplers
                    BindGroupLayoutEntryExt {
                        visibility: Some(ShaderStages::COMPUTE),
                        ty: sampler(SamplerBindingType::Filtering),
                        count: Some(unsafe { NonZeroU32::new_unchecked(10_000) }),
                    },
                    // Emissive object mesh material indices buffer
                    storage_buffer_read_only(false, Some(u32::min_size())),
                    // Emissive object triangle counts buffer
                    storage_buffer_read_only(false, Some(u32::min_size())),
                    // Uniforms
                    uniform_buffer(false, Some(SolariUniforms::min_size())),
                ),
            ),
        ))
    }
}
