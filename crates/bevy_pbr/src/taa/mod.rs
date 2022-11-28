use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, HandleUntyped};
use bevy_core_pipeline::{
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    prelude::Camera3d,
    prepass::{PrepassSettings, ViewPrepassTextures},
};
use bevy_ecs::{
    prelude::{Bundle, Component, Entity},
    query::{QueryState, With},
    system::{Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_reflect::{Reflect, TypeUuid};
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    prelude::Camera,
    render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType},
    render_phase::TrackedRenderPass,
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, Extent3d, FilterMode, FragmentState, MultisampleState,
        Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, Shader,
        ShaderStages, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsages, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::{BevyDefault, CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, ViewTarget},
    Extract, RenderApp, RenderStage,
};
use bevy_utils::HashMap;

use crate::AmbientOcclusionTextures;

mod draw_3d_graph {
    pub mod node {
        /// Label for the TAA render node.
        pub const TAA: &str = "taa";
    }
}

const TAA_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 656865235226276);

pub struct TemporalAntialiasPlugin;

impl Plugin for TemporalAntialiasPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, TAA_SHADER_HANDLE, "taa.wgsl", Shader::from_wgsl);

        app.insert_resource(Msaa { samples: 1 })
            .register_type::<TemporalAntialiasSettings>();

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else { return };

        render_app
            .init_resource::<TAAPipelines>()
            .add_system_to_stage(RenderStage::Extract, extract_taa_settings)
            .add_system_to_stage(RenderStage::Prepare, prepare_taa_textures)
            .add_system_to_stage(RenderStage::Queue, queue_taa_blit_bind_groups);

        let taa_node = TAANode::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        let draw_3d_graph = graph
            .get_sub_graph_mut(bevy_core_pipeline::core_3d::graph::NAME)
            .unwrap();
        draw_3d_graph.add_node(draw_3d_graph::node::TAA, taa_node);
        draw_3d_graph.add_slot_edge(
            draw_3d_graph.input_node().id,
            bevy_core_pipeline::core_3d::graph::input::VIEW_ENTITY,
            draw_3d_graph::node::TAA,
            TAANode::IN_VIEW,
        );
        // MAIN_PASS -> TAA -> TONEMAPPING
        draw_3d_graph.add_node_edge(
            bevy_core_pipeline::core_3d::graph::node::MAIN_PASS,
            draw_3d_graph::node::TAA,
        );
        draw_3d_graph.add_node_edge(
            draw_3d_graph::node::TAA,
            bevy_core_pipeline::core_3d::graph::node::TONEMAPPING,
        );
    }
}

#[derive(Bundle, Default)]
pub struct TemporalAntialiasBundle {
    pub settings: TemporalAntialiasSettings,
    pub jitter: TemporalJitter,
}

#[derive(Component, Reflect, Default, Clone)]
pub struct TemporalAntialiasSettings;

struct TAANode {
    view_query: QueryState<(
        &'static ExtractedCamera,
        &'static ExtractedView,
        &'static ViewTarget,
        &'static TAATextures,
        &'static ViewPrepassTextures,
        &'static TAABlitBindGroup,
        &'static AmbientOcclusionTextures,
    )>,
}

impl TAANode {
    const IN_VIEW: &'static str = "view";

    fn new(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for TAANode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        #[cfg(feature = "trace")]
        let _taa_span = info_span!("taa").entered();

        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (
            Ok((camera, view, view_target, taa_textures, prepass_textures, blit_bind_group, ao)),
            Some(pipelines),
            Some(pipeline_cache),
        ) = (
            self.view_query.get_manual(world, view_entity),
            world.get_resource::<TAAPipelines>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let (taa_pipeline, blit_pipeline) = match view.hdr {
            true => (pipelines.taa_hdr_pipeline, pipelines.blit_hdr_pipeline),
            false => (pipelines.taa_sdr_pipeline, pipelines.blit_sdr_pipeline),
        };
        let (Some(taa_pipeline), Some(blit_pipeline), Some(prepass_velocity_texture)) = (
            pipeline_cache.get_render_pipeline(taa_pipeline),
            pipeline_cache.get_render_pipeline(blit_pipeline),
            &prepass_textures.velocity,
        ) else {
            return Ok(());
        };
        let view_target = view_target.post_process_write();

        let taa_bind_group = render_context
            .render_device
            .create_bind_group(&BindGroupDescriptor {
                label: Some("taa_bind_group"),
                layout: &pipelines.taa_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &ao.ambient_occlusion_texture.default_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &taa_textures.accumulation.default_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &prepass_velocity_texture.default_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&pipelines.nearest_sampler),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::Sampler(&pipelines.linear_sampler),
                    },
                ],
            });

        {
            let mut taa_pass =
                TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                    &(RenderPassDescriptor {
                        label: Some("taa_pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &taa_textures.output.default_view,
                            resolve_target: None,
                            ops: Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                    }),
                ));
            taa_pass.set_render_pipeline(taa_pipeline);
            taa_pass.set_bind_group(0, &taa_bind_group, &[]);
            if let Some(viewport) = camera.viewport.as_ref() {
                taa_pass.set_camera_viewport(viewport);
            }
            taa_pass.draw(0..3, 0..1);
        }

        {
            let mut blit_pass =
                TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                    &(RenderPassDescriptor {
                        label: Some("taa_blit_pass"),
                        color_attachments: &[
                            Some(RenderPassColorAttachment {
                                view: &ao.ambient_occlusion_texture.default_view,
                                resolve_target: None,
                                ops: Operations::default(),
                            }),
                            Some(RenderPassColorAttachment {
                                view: &taa_textures.accumulation.default_view,
                                resolve_target: None,
                                ops: Operations::default(),
                            }),
                        ],
                        depth_stencil_attachment: None,
                    }),
                ));
            blit_pass.set_render_pipeline(blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group.bind_group, &[]);
            if let Some(viewport) = camera.viewport.as_ref() {
                blit_pass.set_camera_viewport(viewport);
            }
            blit_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

#[derive(Resource)]
struct TAAPipelines {
    taa_sdr_pipeline: CachedRenderPipelineId,
    taa_hdr_pipeline: CachedRenderPipelineId,

    blit_sdr_pipeline: CachedRenderPipelineId,
    blit_hdr_pipeline: CachedRenderPipelineId,

    taa_bind_group_layout: BindGroupLayout,
    blit_bind_group_layout: BindGroupLayout,

    nearest_sampler: Sampler,
    linear_sampler: Sampler,
}

impl FromWorld for TAAPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("taa_nearest_sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..SamplerDescriptor::default()
        });
        let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("taa_linear_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        });

        let taa_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("taa_bind_group_layout"),
                entries: &[
                    // View target
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // TAA Accumulation
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Velocity
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Nearest sampler
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // Linear sampler
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("taa_blit_bind_group_layout"),
                entries: &[
                    // TAA Output
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Linear sampler
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let taa_sdr_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("taa_sdr_pipeline".into()),
            layout: Some(vec![taa_bind_group_layout.clone()]),
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: TAA_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "taa".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::R32Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });

        let taa_hdr_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("taa_hdr_pipeline".into()),
            layout: Some(vec![taa_bind_group_layout.clone()]),
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: TAA_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec!["TONEMAP".into()],
                entry_point: "taa".into(),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });

        let blit_sdr_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("taa_blit_sdr_pipeline".into()),
            layout: Some(vec![blit_bind_group_layout.clone()]),
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: TAA_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "blit".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::R32Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::R32Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });

        let blit_hdr_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("taa_blit_hdr_pipeline".into()),
            layout: Some(vec![blit_bind_group_layout.clone()]),
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: TAA_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec!["TONEMAP".into()],
                entry_point: "blit".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });

        TAAPipelines {
            taa_sdr_pipeline,
            taa_hdr_pipeline,

            blit_sdr_pipeline,
            blit_hdr_pipeline,

            taa_bind_group_layout,
            blit_bind_group_layout,

            nearest_sampler,
            linear_sampler,
        }
    }
}
fn extract_taa_settings(
    mut commands: Commands,
    cameras_3d: Extract<
        Query<
            (Entity, &Camera, &TemporalAntialiasSettings),
            (With<Camera3d>, With<PrepassSettings>, With<TemporalJitter>),
        >,
    >,
) {
    for (entity, camera, taa_settings) in &cameras_3d {
        if camera.is_active {
            commands.get_or_spawn(entity).insert(taa_settings.clone());
        }
    }
}

#[derive(Component)]
struct TAATextures {
    accumulation: CachedTexture,
    output: CachedTexture,
}

fn prepare_taa_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<
        (Entity, &ExtractedCamera, &ExtractedView, &PrepassSettings),
        With<TemporalAntialiasSettings>,
    >,
) {
    let mut accumulation_textures = HashMap::default();
    let mut output_textures = HashMap::default();
    let views = views
        .iter()
        .filter(|(_, _, _, prepass_settings)| prepass_settings.velocity_enabled);
    for (entity, camera, view, _) in views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if view.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::R32Float
                },
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            };

            texture_descriptor.label = Some("taa_accumulation_texture");
            let accumulation = accumulation_textures
                .entry(camera.target.clone())
                .or_insert_with(|| texture_cache.get(&render_device, texture_descriptor.clone()))
                .clone();

            texture_descriptor.label = Some("taa_view_target_blit_texture");
            let output = output_textures
                .entry(camera.target.clone())
                .or_insert_with(|| texture_cache.get(&render_device, texture_descriptor))
                .clone();

            commands.entity(entity).insert(TAATextures {
                accumulation,
                output,
            });
        }
    }
}

#[derive(Component)]
struct TAABlitBindGroup {
    bind_group: BindGroup,
}

fn queue_taa_blit_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<TAAPipelines>,
    views: Query<(Entity, &TAATextures), With<TemporalAntialiasSettings>>,
) {
    for (entity, taa_textures) in &views {
        let blit_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("taa_blit_bind_group"),
            layout: &pipelines.blit_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&taa_textures.output.default_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&pipelines.nearest_sampler),
                },
            ],
        });

        commands.entity(entity).insert(TAABlitBindGroup {
            bind_group: blit_bind_group,
        });
    }
}
