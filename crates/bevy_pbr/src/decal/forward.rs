use crate::{environment_map::EnvironmentMapLight, irradiance_volume::IrradianceVolume, *};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::AssetServer;
use bevy_core_pipeline::{
    core_3d::{Camera3d, Transparent3d},
    oit::OrderIndependentTransparencySettings,
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, ROQueryItem, With},
    schedule::IntoSystemConfigs,
    system::{Query, Res, ResMut, SystemParamItem},
};
use bevy_reflect::Reflect;
use bevy_render::{
    alpha::AlphaMode,
    camera::{Projection, TemporalJitter},
    extract_component::ExtractComponent,
    mesh::{
        MeshVertexBufferLayout, MeshVertexBufferLayoutRef, MeshVertexBufferLayouts,
        VertexBufferLayout,
    },
    render_asset::RenderAssets,
    render_phase::{
        DrawFunctions, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
        TrackedRenderPass, ViewSortedRenderPhases,
    },
    render_resource::{
        CompareFunction, PipelineCache, ShaderRef, SpecializedMeshPipeline, VertexState,
        VertexStepMode,
    },
    sync_component::SyncComponentPlugin,
    view::{
        check_visibility, ExtractedView, Msaa, RenderVisibilityRanges, RenderVisibleEntities,
        Visibility, VisibilitySystems,
    },
};
use bevy_transform::components::Transform;
use bevy_utils::tracing::error;
use core::hash::Hash;

/// TODO: Docs, used with MeshMaterial3d
#[derive(Component, ExtractComponent, Reflect, Clone, Copy, Default)]
#[require(Transform, Visibility)]
pub struct ForwardDecal;

/// TODO: Docs
pub struct ForwardDecalPlugin;

impl Plugin for ForwardDecalPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ForwardDecal>()
            .add_systems(
                PostUpdate,
                check_visibility::<With<ForwardDecal>>.in_set(VisibilitySystems::CheckVisibility),
            )
            .add_plugins(SyncComponentPlugin::<ForwardDecal>::default());
    }
}

struct DrawForwardDecalQuad;
impl RenderCommand<Transparent3d> for DrawForwardDecalQuad {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &Transparent3d,
        _view: (),
        _: Option<ROQueryItem<'w, Self::ItemQuery>>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        todo!()
    }
}

type DrawForwardDecal<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    // TODO: Need to _unset_ MeshBindGroup at group 1
    SetMaterialBindGroup<M, 2>,
    DrawForwardDecalQuad,
);

#[allow(clippy::too_many_arguments)]
pub fn queue_forward_decals<M: Material>(
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    material_pipeline: Res<MaterialPipeline<M>>,
    pipeline_cache: Res<PipelineCache>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    render_lightmaps: Res<RenderLightmaps>,
    render_visibility_ranges: Res<RenderVisibilityRanges>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    asset_server: Res<AssetServer>,
    mut mesh_vertex_buffer_layouts: ResMut<MeshVertexBufferLayouts>,
    views: Query<(
        Entity,
        &ExtractedView,
        &RenderVisibleEntities,
        &Msaa,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ShadowFilteringMethod>,
        Has<ScreenSpaceAmbientOcclusion>,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        Option<&Camera3d>,
        Has<TemporalJitter>,
        Option<&Projection>,
        (
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<RenderViewLightProbes<IrradianceVolume>>,
        ),
        Has<OrderIndependentTransparencySettings>,
    )>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let fake_vertex_buffer_layout = &fake_vertex_buffer_layout(&mut mesh_vertex_buffer_layouts);

    for (
        view_entity,
        view,
        visible_entities,
        msaa,
        tonemapping,
        dither,
        shadow_filter_method,
        ssao,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        camera_3d,
        temporal_jitter,
        projection,
        (has_environment_maps, has_irradiance_volumes),
        has_oit,
    ) in &views
    {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let draw_transparent_pbr = transparent_draw_functions
            .read()
            .id::<DrawForwardDecal<M>>();

        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }

        if temporal_jitter {
            view_key |= MeshPipelineKey::TEMPORAL_JITTER;
        }

        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        if has_oit {
            view_key |= MeshPipelineKey::OIT_ENABLED;
        }

        if let Some(projection) = projection {
            view_key |= match projection {
                Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
            };
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= tonemapping_pipeline_key(*tonemapping);
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }
        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if let Some(camera_3d) = camera_3d {
            view_key |= screen_space_specular_transmission_pipeline_key(
                camera_3d.screen_space_specular_transmission_quality,
            );
        }

        for (render_entity, visible_entity) in visible_entities.iter::<With<ForwardDecal>>() {
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };

            let mut mesh_pipeline_key_bits = material.properties.mesh_pipeline_key_bits;
            mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(AlphaMode::Blend, msaa));
            let mut mesh_key = view_key | mesh_pipeline_key_bits;

            let lightmap_image = render_lightmaps
                .render_lightmaps
                .get(visible_entity)
                .map(|lightmap| lightmap.image);
            if lightmap_image.is_some() {
                mesh_key |= MeshPipelineKey::LIGHTMAPPED;
            }

            if render_visibility_ranges.entity_has_crossfading_visibility_ranges(*visible_entity) {
                mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
            }

            let mut pipeline_descriptor = match material_pipeline.specialize(
                MaterialPipelineKey {
                    mesh_key,
                    bind_group_data: material.key.clone(),
                },
                fake_vertex_buffer_layout,
            ) {
                Ok(pipeline_descriptor) => pipeline_descriptor,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            let (depth_stencil, fragment) = (
                pipeline_descriptor.depth_stencil.as_mut().unwrap(),
                pipeline_descriptor.fragment.as_mut().unwrap(),
            );

            pipeline_descriptor.layout.remove(1); // Remove mesh bind group, TODO: need to replace slot 1 with None, not remove
            depth_stencil.depth_compare = CompareFunction::Always;
            fragment.shader = match M::forward_decal_fragment_shader() {
                ShaderRef::Default => todo!(),
                ShaderRef::Handle(handle) => handle,
                ShaderRef::Path(path) => asset_server.load(path),
            };
            pipeline_descriptor.vertex = VertexState {
                shader: todo!(),
                shader_defs: pipeline_descriptor.vertex.shader_defs,
                entry_point: todo!(),
                buffers: Vec::new(),
            };

            let pipeline = pipeline_cache.queue_render_pipeline(pipeline_descriptor);

            transparent_phase.add(Transparent3d {
                entity: (*render_entity, *visible_entity),
                draw_function: draw_transparent_pbr,
                pipeline,
                // Transparent items are rendered back to front, so force forward decals to render first,
                // so that they're correctly visible through transparent objects
                distance: f32::MAX,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}

// Forward decal materials don't use a traditional vertex buffer, but the material specialization requires one.
fn fake_vertex_buffer_layout(layouts: &mut MeshVertexBufferLayouts) -> MeshVertexBufferLayoutRef {
    layouts.insert(MeshVertexBufferLayout::new(
        vec![],
        VertexBufferLayout {
            array_stride: 0,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![],
        },
    ))
}
