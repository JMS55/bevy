//! Run a prepass before the main pass to generate depth, normals, and/or motion vectors textures, sometimes called a thin g-buffer.
//! These textures are useful for various screen-space effects and reducing overdraw in the main pass.
//!
//! The prepass only runs for opaque meshes or meshes with an alpha mask. Transparent meshes are ignored.
//!
//! To enable the prepass, you need to add a prepass component to a [`crate::prelude::Camera3d`].
//!
//! [`DepthPrepass`]
//! [`NormalPrepass`]
//! [`MotionVectorPrepass`]
//!
//! The textures are automatically added to the default mesh view bindings. You can also get the raw textures
//! by querying the [`ViewPrepassTextures`] component on any camera with a prepass component.
//!
//! The depth prepass will always run and generate the depth buffer as a side effect, but it won't copy it
//! to a separate texture unless the [`DepthPrepass`] is activated. This means that if any prepass component is present
//! it will always create a depth buffer that will be used by the main pass.
//!
//! When using the default mesh view bindings you should be able to use `prepass_depth()`,
//! `prepass_normal()`, and `prepass_motion_vector()` to load the related textures.
//! These functions are defined in `bevy_pbr::prepass_utils`. See the `shader_prepass` example that shows how to use them.
//!
//! The prepass runs for each `Material`. You can control if the prepass should run per-material by setting the `prepass_enabled`
//! flag on the `MaterialPlugin`.
//!
//! Currently only works for 3D.

pub mod node;

use std::ops::Range;

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_render::{
    mesh::Mesh,
    render_phase::{
        BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
        PhaseItemExtraIndex,
    },
    render_resource::{BindGroupId, CachedRenderPipelineId, Extent3d, TextureFormat, TextureView},
    texture::ColorAttachment,
};

pub const NORMAL_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgb10a2Unorm;
pub const MOTION_VECTOR_PREPASS_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// If added to a [`crate::prelude::Camera3d`] then depth values will be copied to a separate texture available to the main pass.
#[derive(Component, Default, Reflect, Clone)]
pub struct DepthPrepass;

/// If added to a [`crate::prelude::Camera3d`] then vertex world normals will be copied to a separate texture available to the main pass.
/// Normals will have normal map textures already applied.
#[derive(Component, Default, Reflect, Clone)]
pub struct NormalPrepass;

/// If added to a [`crate::prelude::Camera3d`] then screen space motion vectors will be copied to a separate texture available to the main pass.
#[derive(Component, Default, Reflect, Clone)]
pub struct MotionVectorPrepass;

/// If added to a [`crate::prelude::Camera3d`] then deferred materials will be rendered to the deferred gbuffer texture and will be available to subsequent passes.
/// Note the default deferred lighting plugin also requires `DepthPrepass` to work correctly.
#[derive(Component, Default, Reflect)]
pub struct DeferredPrepass;

/// Textures that are written to by the prepass.
///
/// This component will only be present if any of the relevant prepass components are also present.
#[derive(Component)]
pub struct ViewPrepassTextures {
    /// The depth texture generated by the prepass.
    /// Exists only if [`DepthPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
    pub depth: Option<ColorAttachment>,
    /// The normals texture generated by the prepass.
    /// Exists only if [`NormalPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
    pub normal: Option<ColorAttachment>,
    /// The motion vectors texture generated by the prepass.
    /// Exists only if [`MotionVectorPrepass`] is added to the `ViewTarget`
    pub motion_vectors: Option<ColorAttachment>,
    /// The deferred gbuffer generated by the deferred pass.
    /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
    pub deferred: Option<ColorAttachment>,
    /// A texture that specifies the deferred lighting pass id for a material.
    /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
    pub deferred_lighting_pass_id: Option<ColorAttachment>,
    /// The size of the textures.
    pub size: Extent3d,
}

impl ViewPrepassTextures {
    pub fn depth_view(&self) -> Option<&TextureView> {
        self.depth.as_ref().map(|t| &t.texture.default_view)
    }

    pub fn normal_view(&self) -> Option<&TextureView> {
        self.normal.as_ref().map(|t| &t.texture.default_view)
    }

    pub fn motion_vectors_view(&self) -> Option<&TextureView> {
        self.motion_vectors
            .as_ref()
            .map(|t| &t.texture.default_view)
    }

    pub fn deferred_view(&self) -> Option<&TextureView> {
        self.deferred.as_ref().map(|t| &t.texture.default_view)
    }
}

/// Opaque phase of the 3D prepass.
///
/// Sorted by pipeline, then by mesh to improve batching.
///
/// Used to render all 3D meshes with materials that have no transparency.
pub struct Opaque3dPrepass {
    /// Information that separates items into bins.
    pub key: OpaqueNoLightmap3dBinKey,

    /// An entity from which Bevy fetches data common to all instances in this
    /// batch, such as the mesh.
    pub representative_entity: Entity,

    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

// TODO: Try interning these.
/// The data used to bin each opaque 3D mesh in the prepass and deferred pass.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpaqueNoLightmap3dBinKey {
    /// The ID of the GPU pipeline.
    pub pipeline: CachedRenderPipelineId,

    /// The function used to draw the mesh.
    pub draw_function: DrawFunctionId,

    /// The ID of the mesh.
    pub asset_id: AssetId<Mesh>,

    /// The ID of a bind group specific to the material.
    ///
    /// In the case of PBR, this is the `MaterialBindGroupId`.
    pub material_bind_group_id: Option<BindGroupId>,
}

impl PhaseItem for Opaque3dPrepass {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for Opaque3dPrepass {
    type BinKey = OpaqueNoLightmap3dBinKey;

    #[inline]
    fn new(
        key: Self::BinKey,
        representative_entity: Entity,
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Opaque3dPrepass {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for Opaque3dPrepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

/// Alpha mask phase of the 3D prepass.
///
/// Sorted by pipeline, then by mesh to improve batching.
///
/// Used to render all meshes with a material with an alpha mask.
pub struct AlphaMask3dPrepass {
    pub key: OpaqueNoLightmap3dBinKey,
    pub representative_entity: Entity,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for AlphaMask3dPrepass {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for AlphaMask3dPrepass {
    type BinKey = OpaqueNoLightmap3dBinKey;

    #[inline]
    fn new(
        key: Self::BinKey,
        representative_entity: Entity,
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for AlphaMask3dPrepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}
