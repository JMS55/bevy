use super::{
    asset_binder::AssetBindings, scene_binder::SceneBindings, Solari,
    SOLARI_PATH_TRACER_SHADER_HANDLE,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::QueryItem,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    texture::{CachedTexture, TextureCache},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use binding_types::{texture_storage_2d, uniform_buffer};

pub struct PathTracerNode {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl ViewNode for PathTracerNode {
    type ViewQuery = (
        &'static PathTracerAccumulationTexture,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (accumulation_texture, camera, view_target, view_uniform_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_bindings = world.resource::<AssetBindings>();
        let scene_bindings = world.resource::<SceneBindings>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let (
            Some(pipeline),
            Some(asset_bindings),
            Some(scene_bindings),
            Some(viewport),
            Some(view_uniforms),
        ) = (
            pipeline_cache.get_compute_pipeline(self.pipeline),
            &asset_bindings.bind_group,
            &scene_bindings.bind_group,
            camera.physical_viewport_size,
            view_uniforms.uniforms.binding(),
        )
        else {
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "solari_path_tracer_bind_group",
            &self.bind_group_layout,
            &BindGroupEntries::sequential((
                &accumulation_texture.0.default_view,
                view_target.get_color_attachment().view,
                view_uniforms,
            )),
        );

        let command_encoder = render_context.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari_path_tracer"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, asset_bindings, &[]);
        pass.set_bind_group(1, scene_bindings, &[]);
        pass.set_bind_group(2, &bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups((viewport.x + 7) / 8, (viewport.y + 7) / 8, 1);

        Ok(())
    }
}

impl FromWorld for PathTracerNode {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_bindings = world.resource::<AssetBindings>();
        let scene_bindings = world.resource::<SceneBindings>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "solari_path_tracer_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(
                        ViewTarget::TEXTURE_FORMAT_HDR,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("solari_path_tracer_pipeline".into()),
            layout: vec![
                asset_bindings.bind_group_layout.clone(),
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SOLARI_PATH_TRACER_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "path_trace".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}

#[derive(Component)]
pub struct PathTracerAccumulationTexture(CachedTexture);

pub fn prepare_path_tracer_accumulation_texture(
    query: Query<(Entity, &Solari, &ExtractedCamera)>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, solari_settings, camera) in &query {
        if !solari_settings.debug_path_tracer {
            continue;
        }
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let descriptor = TextureDescriptor {
            label: Some("solari_path_tracer_accumulation_texture"),
            size: Extent3d {
                width: viewport.x,
                height: viewport.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        commands
            .entity(entity)
            .insert(PathTracerAccumulationTexture(
                texture_cache.get(&render_device, descriptor),
            ));
    }
}
