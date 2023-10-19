use bevy_render::{
    render_resource::{
        BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType,
        SamplerBindingType, TextureSampleType, TextureViewDimension,
    },
    renderer::RenderDevice,
};
use bevy_utils::all_tuples_with_size;
use bevy_utils::tracing::warn;
use std::num::{NonZeroU32, NonZeroU64};
use wgpu::{BindingType, ShaderStages, StorageTextureAccess, TextureFormat};

pub struct BindGroupLayoutEntryExt {
    pub ty: BindingType,
    pub visibility: Option<ShaderStages>,
    pub count: Option<NonZeroU32>,
}

pub struct BindGroupLayoutEntries<const N: usize> {
    entries: [BindGroupLayoutEntry; N],
}

impl<const N: usize> BindGroupLayoutEntries<N> {
    #[inline]
    #[allow(unused)]
    pub fn sequential(
        default_visibility: ShaderStages,
        entries_ext: impl IntoBindGroupLayoutEntryExtArray<N>,
    ) -> Self {
        let mut i = 0;
        Self {
            entries: entries_ext.into_array().map(|entry| {
                let binding = i;
                i += 1;
                BindGroupLayoutEntry {
                    binding,
                    ty: entry.ty,
                    visibility: entry.visibility.unwrap_or(default_visibility),
                    count: entry.count,
                }
            }),
        }
    }

    #[inline]
    #[allow(unused)]
    pub fn with_indices(
        default_visibility: ShaderStages,
        indexed_entries: impl IntoIndexedBindGroupLayoutEntryExtArray<N>,
    ) -> Self {
        Self {
            entries: indexed_entries
                .into_array()
                .map(|(binding, entry)| BindGroupLayoutEntry {
                    binding,
                    ty: entry.ty,
                    visibility: entry.visibility.unwrap_or(default_visibility),
                    count: entry.count,
                }),
        }
    }
}

impl<const N: usize> std::ops::Deref for BindGroupLayoutEntries<N> {
    type Target = [BindGroupLayoutEntry];
    fn deref(&self) -> &[BindGroupLayoutEntry] {
        &self.entries
    }
}

pub trait IntoBindGroupLayoutEntryExt {
    fn into_bind_group_layout_entry(self) -> BindGroupLayoutEntryExt;
}

impl IntoBindGroupLayoutEntryExt for BindingType {
    fn into_bind_group_layout_entry(self) -> BindGroupLayoutEntryExt {
        BindGroupLayoutEntryExt {
            ty: self,
            visibility: None,
            count: None,
        }
    }
}

impl IntoBindGroupLayoutEntryExt for BindGroupLayoutEntry {
    fn into_bind_group_layout_entry(self) -> BindGroupLayoutEntryExt {
        if self.binding != 0 {
            warn!("BindGroupLayoutEntries ignores binding numbers");
        }
        BindGroupLayoutEntryExt {
            ty: self.ty,
            visibility: Some(self.visibility),
            count: self.count,
        }
    }
}

impl IntoBindGroupLayoutEntryExt for BindGroupLayoutEntryExt {
    fn into_bind_group_layout_entry(self) -> BindGroupLayoutEntryExt {
        self
    }
}

pub trait IntoBindGroupLayoutEntryExtArray<const N: usize> {
    fn into_array(self) -> [BindGroupLayoutEntryExt; N];
}
macro_rules! impl_to_binding_type_slice {
    ($N: expr, $(($T: ident, $I: ident)),*) => {
        impl<$($T: IntoBindGroupLayoutEntryExt),*> IntoBindGroupLayoutEntryExtArray<$N> for ($($T,)*) {
            #[inline]
            fn into_array(self) -> [BindGroupLayoutEntryExt; $N] {
                let ($($I,)*) = self;
                [$($I.into_bind_group_layout_entry(), )*]
            }
        }
    }
}
all_tuples_with_size!(impl_to_binding_type_slice, 1, 32, T, s);

pub trait IntoIndexedBindGroupLayoutEntryExtArray<const N: usize> {
    fn into_array(self) -> [(u32, BindGroupLayoutEntryExt); N];
}
macro_rules! impl_to_indexed_binding_type_slice {
    ($N: expr, $(($T: ident, $S: ident, $I: ident)),*) => {
        impl<$($T: IntoBindGroupLayoutEntryExt),*> IntoIndexedBindGroupLayoutEntryExtArray<$N> for ($((u32, $T),)*) {
            #[inline]
            fn into_array(self) -> [(u32, BindGroupLayoutEntryExt); $N] {
                let ($(($S, $I),)*) = self;
                [$(($S, $I.into_bind_group_layout_entry())), *]
            }
        }
    }
}
all_tuples_with_size!(impl_to_indexed_binding_type_slice, 1, 32, T, n, s);

#[allow(unused)]
pub fn storage_buffer(
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZeroU64>,
) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset,
        min_binding_size,
    }
}

#[allow(unused)]
pub fn storage_buffer_read_only(
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZeroU64>,
) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset,
        min_binding_size,
    }
}

#[allow(unused)]
pub fn uniform_buffer(
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZeroU64>,
) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset,
        min_binding_size,
    }
}

#[allow(unused)]
pub fn texture_2d(sample_type: TextureSampleType) -> BindingType {
    BindingType::Texture {
        sample_type,
        view_dimension: TextureViewDimension::D2,
        multisampled: false,
    }
}

#[allow(unused)]
pub fn texture_2d_multisampled(sample_type: TextureSampleType) -> BindingType {
    BindingType::Texture {
        sample_type,
        view_dimension: TextureViewDimension::D2,
        multisampled: true,
    }
}

#[allow(unused)]
pub fn texture_2d_array(sample_type: TextureSampleType) -> BindingType {
    BindingType::Texture {
        sample_type,
        view_dimension: TextureViewDimension::D2Array,
        multisampled: false,
    }
}

#[allow(unused)]
pub fn texture_2d_array_multisampled(sample_type: TextureSampleType) -> BindingType {
    BindingType::Texture {
        sample_type,
        view_dimension: TextureViewDimension::D2Array,
        multisampled: true,
    }
}

#[allow(unused)]
pub fn texture_2d_f32() -> BindingType {
    texture_2d(TextureSampleType::Float { filterable: true })
}

#[allow(unused)]
pub fn texture_2d_multisampled_f32() -> BindingType {
    texture_2d_multisampled(TextureSampleType::Float { filterable: true })
}

#[allow(unused)]
pub fn texture_2d_i32() -> BindingType {
    texture_2d(TextureSampleType::Sint)
}

#[allow(unused)]
pub fn texture_2d_multisampled_i32() -> BindingType {
    texture_2d_multisampled(TextureSampleType::Sint)
}

#[allow(unused)]
pub fn texture_2d_u32() -> BindingType {
    texture_2d(TextureSampleType::Uint)
}

#[allow(unused)]
pub fn texture_2d_multisampled_u32() -> BindingType {
    texture_2d_multisampled(TextureSampleType::Uint)
}

#[allow(unused)]
pub fn texture_depth_2d() -> BindingType {
    texture_2d(TextureSampleType::Depth)
}

#[allow(unused)]
pub fn texture_depth_2d_multisampled() -> BindingType {
    texture_2d_multisampled(TextureSampleType::Depth)
}

#[allow(unused)]
pub fn acceleration_structure() -> BindingType {
    BindingType::AccelerationStructure
}

#[allow(unused)]
pub fn sampler(sampler_binding_type: SamplerBindingType) -> BindingType {
    BindingType::Sampler(sampler_binding_type)
}

#[allow(unused)]
pub fn texture_storage_2d(format: TextureFormat, access: StorageTextureAccess) -> BindingType {
    BindingType::StorageTexture {
        access,
        format,
        view_dimension: TextureViewDimension::D2,
    }
}

#[allow(unused)]
pub fn texture_storage_2d_array(
    format: TextureFormat,
    access: StorageTextureAccess,
) -> BindingType {
    BindingType::StorageTexture {
        access,
        format,
        view_dimension: TextureViewDimension::D2Array,
    }
}

pub trait RenderDeviceExt {
    fn create_bind_group_layout_ext<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        entries: &[BindGroupLayoutEntry],
    ) -> BindGroupLayout;
}

impl RenderDeviceExt for RenderDevice {
    fn create_bind_group_layout_ext<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        entries: &[BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        self.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: label.into(),
            entries,
        })
    }
}
