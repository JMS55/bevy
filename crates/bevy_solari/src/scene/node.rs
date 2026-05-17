use crate::scene::RaytracingSceneBindings;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::{
    render_resource::{
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    },
    renderer::RenderContext,
};
use bevy_utils::default;

#[derive(Resource)]
pub struct RaytracingScenePipelines {
    simplify_materials_pipeline: CachedComputePipelineId,
}

pub fn raytracing_scene_setup(
    scene_bindings: Res<RaytracingSceneBindings>,
    raytracing_scene_pipelines: Option<Res<RaytracingScenePipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (Some(tlas), Some(scene_bind_group), Some(scene_pipelines)) = (
        &scene_bindings.tlas,
        &scene_bindings.bind_group,
        raytracing_scene_pipelines,
    ) else {
        return;
    };
    let Some(simplify_materials_pipeline) =
        pipeline_cache.get_compute_pipeline(scene_pipelines.simplify_materials_pipeline)
    else {
        return;
    };

    let command_encoder = ctx.command_encoder();

    command_encoder.build_acceleration_structures(&[], [tlas]);

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("raytracing_scene_simplify_materials"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_pipeline(simplify_materials_pipeline);
    pass.dispatch_workgroups(scene_bindings.material_count.div_ceil(64), 1, 1);
}

pub fn init_raytracing_scene_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    asset_server: Res<AssetServer>,
) {
    let simplify_materials_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("raytracing_scene_simplify_materials_pipeline".into()),
            layout: vec![scene_bindings.bind_group_layout.clone()],
            shader: load_embedded_asset!(asset_server.as_ref(), "simplify_materials.wgsl"),
            entry_point: Some("simplify_materials".into()),
            ..default()
        });

    commands.insert_resource(RaytracingScenePipelines {
        simplify_materials_pipeline,
    });
}
