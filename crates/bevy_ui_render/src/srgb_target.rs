//! Makes the hardware blend UI in sRGB space, as web browsers do, whatever space
//! the camera's main texture stores colors in.
//!
//! UI shaders encode their output to sRGB with `encode_output` from
//! `bevy_ui_render::ui`. For the hardware to blend those encoded values, the
//! render target must store sRGB-encoded values and must not convert them on
//! write. [`UiRenderTarget`] picks the cheapest way to get such a target for
//! each camera.

use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::{CameraOutputMode, ClearColorConfig, CompositingSpace};
use bevy_core_pipeline::{
    upscaling::{prepare_view_upscaling_pipelines, UpscalingSourceSpace},
    FullscreenShader,
};
use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::ViewSortedRenderPhases,
    render_resource::{
        binding_types::texture_2d, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntries, CachedRenderPipelineId, ColorTargetState, ColorWrites,
        FragmentState, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipeline, RenderPipelineDescriptor, ShaderStages, SpecializedRenderPipeline,
        SpecializedRenderPipelines, TextureFormat, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice},
    view::{
        main_texture_key, prepare_view_targets, ExtractedView, MainTextureKey, Msaa,
        ResolvedCompositingSpace, ViewTarget,
    },
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::default;

use crate::{TransparentUi, UiCameraView};

/// Adds [`UiRenderTarget`] to cameras that render UI.
pub struct UiRenderTargetPlugin;

impl Plugin for UiRenderTargetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "srgb_conversion.wesl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<SrgbConversionPipeline>>()
            .add_systems(RenderStartup, init_srgb_conversion_pipeline)
            .add_systems(
                Render,
                (
                    prepare_ui_render_targets
                        .in_set(RenderSystems::PrepareViews)
                        .after(prepare_view_targets),
                    prepare_ui_upscaling_source_spaces
                        .in_set(RenderSystems::Prepare)
                        .before(prepare_view_upscaling_pipelines),
                ),
            );
    }
}

/// How the UI pass gets a render target that blends in sRGB space. It lives on
/// the camera's render entity, and UI pipelines render to its
/// [`format`](Self::format).
#[derive(Component, Clone, Copy, Debug)]
pub enum UiRenderTarget {
    /// The main texture already stores sRGB-encoded values, because the camera
    /// stack composites in [`CompositingSpace::Srgb`]. UI renders straight into
    /// it.
    Direct(TextureFormat),
    /// The main texture has an sRGB format. UI renders through a view in
    /// [`ViewTarget::main_texture_encoded_view_format`], which blends the
    /// stored, encoded values. This costs nothing extra.
    EncodedView(TextureFormat),
    /// Every other main texture, such as an `Rgba16Float` one, or an sRGB one
    /// on a device that can't view textures in other formats. Before the UI
    /// renders, a fullscreen pass converts the main texture to sRGB-encoded
    /// values. After the UI renders, the main texture has to go back to the
    /// camera's compositing space.
    Convert {
        /// The main texture format.
        format: TextureFormat,
        /// The pipeline that converts the main texture to sRGB.
        to_srgb: CachedRenderPipelineId,
        /// The pipeline that converts the main texture back from sRGB, when the
        /// UI pass has to.
        ///
        /// `None` when the camera is the last to render to its main texture this
        /// frame, the stack's first camera clears it, and the camera writes to its
        /// output. Then nothing reads the main texture after upscaling, so the UI
        /// pass leaves it in sRGB and [`UpscalingSourceSpace`] makes upscaling
        /// decode it. That saves a fullscreen pass.
        from_srgb: Option<CachedRenderPipelineId>,
    },
}

impl UiRenderTarget {
    /// The format that UI pipelines render to.
    pub fn format(&self) -> TextureFormat {
        match *self {
            Self::Direct(format) | Self::EncodedView(format) | Self::Convert { format, .. } => {
                format
            }
        }
    }
}

/// The cameras that render to one main texture this frame, by
/// `sorted_camera_index_for_target`.
struct MainTextureStack {
    /// Whether the first camera clears the main texture.
    first_clears: bool,
    first_index: usize,
    last_index: usize,
}

/// Picks each UI camera's [`UiRenderTarget`].
pub fn prepare_ui_render_targets(
    mut commands: Commands,
    cameras: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &Msaa,
        &ViewTarget,
        Option<&ResolvedCompositingSpace>,
        Has<UiCameraView>,
    )>,
    cameras_without_target: Query<Entity, (With<UiCameraView>, Without<ViewTarget>)>,
    conversion_pipeline: Res<SrgbConversionPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SrgbConversionPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    for entity in &cameras_without_target {
        commands
            .entity(entity)
            .remove::<(UiRenderTarget, UpscalingSourceSpace)>();
    }

    let mut stacks: HashMap<MainTextureKey, MainTextureStack> = HashMap::default();
    for (_, camera, view, msaa, ..) in &cameras {
        let index = camera.sorted_camera_index_for_target;
        let clears = !matches!(camera.clear_color, ClearColorConfig::None);
        stacks
            .entry(main_texture_key(camera, view, *msaa))
            .and_modify(|stack| {
                if index < stack.first_index {
                    stack.first_index = index;
                    stack.first_clears = clears;
                }
                stack.last_index = stack.last_index.max(index);
            })
            .or_insert(MainTextureStack {
                first_clears: clears,
                first_index: index,
                last_index: index,
            });
    }

    for (entity, camera, view, msaa, view_target, compositing_space, has_ui) in &cameras {
        if !has_ui {
            continue;
        }

        let format = view_target.main_texture_format();
        let compositing_space = ResolvedCompositingSpace::space(compositing_space);
        let target = if compositing_space == Some(CompositingSpace::Srgb) {
            UiRenderTarget::Direct(format)
        } else if let Some(encoded_format) = view_target.main_texture_encoded_view_format() {
            UiRenderTarget::EncodedView(encoded_format)
        } else {
            let mut specialize = |to_srgb| {
                let id = pipelines.specialize(
                    &pipeline_cache,
                    &conversion_pipeline,
                    SrgbConversionPipelineKey {
                        format,
                        oklab: compositing_space == Some(CompositingSpace::Oklab),
                        to_srgb,
                    },
                );
                // Without the conversion the UI can't render, so wait for it like the
                // tonemapping pass does. This returns at once when it's already compiled.
                pipeline_cache.block_on_render_pipeline(id);
                id
            };
            let stack = &stacks[&main_texture_key(camera, view, *msaa)];
            let upscaling_decodes = stack.first_clears
                && stack.last_index == camera.sorted_camera_index_for_target
                && matches!(camera.output_mode, CameraOutputMode::Write { .. });
            UiRenderTarget::Convert {
                format,
                to_srgb: specialize(true),
                from_srgb: (!upscaling_decodes).then(|| specialize(false)),
            }
        };

        let mut entity_commands = commands.entity(entity);
        if matches!(
            target,
            UiRenderTarget::Convert {
                from_srgb: None,
                ..
            }
        ) {
            // `prepare_ui_upscaling_source_spaces` sets the space once it knows
            // whether the UI draws this frame.
            entity_commands.insert((target, UpscalingSourceSpace(compositing_space)));
        } else {
            entity_commands
                .insert(target)
                .remove::<UpscalingSourceSpace>();
        }
    }
}

/// Makes upscaling decode from sRGB on frames where the UI pass leaves the main
/// texture in sRGB. See [`UiRenderTarget::Convert::from_srgb`].
pub fn prepare_ui_upscaling_source_spaces(
    mut cameras: Query<(
        &UiCameraView,
        &UiRenderTarget,
        Option<&ResolvedCompositingSpace>,
        &mut UpscalingSourceSpace,
    )>,
    ui_views: Query<&ExtractedView>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
) {
    for (ui_camera_view, target, compositing_space, mut source_space) in &mut cameras {
        // The UI pass skips frames without UI, so it only leaves the main texture
        // in sRGB when there is some.
        let leaves_srgb = matches!(
            target,
            UiRenderTarget::Convert {
                from_srgb: None,
                ..
            }
        ) && ui_views
            .get(ui_camera_view.0)
            .ok()
            .and_then(|view| transparent_render_phases.get(&view.retained_view_entity))
            .is_some_and(|phase| !phase.items.is_empty());
        source_space.set_if_neq(UpscalingSourceSpace(if leaves_srgb {
            Some(CompositingSpace::Srgb)
        } else {
            ResolvedCompositingSpace::space(compositing_space)
        }));
    }
}

/// Converts a main texture between its compositing space and sRGB, for
/// [`UiRenderTarget::Convert`].
#[derive(Resource)]
pub struct SrgbConversionPipeline {
    layout: BindGroupLayout,
    layout_descriptor: BindGroupLayoutDescriptor,
    fullscreen_shader: FullscreenShader,
    shader: Handle<Shader>,
}

fn init_srgb_conversion_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let entries = BindGroupLayoutEntries::single(
        ShaderStages::FRAGMENT,
        texture_2d(TextureSampleType::Float { filterable: false }),
    );
    commands.insert_resource(SrgbConversionPipeline {
        layout: render_device.create_bind_group_layout("ui_srgb_conversion_layout", &entries),
        layout_descriptor: BindGroupLayoutDescriptor::new(
            "ui_srgb_conversion_layout",
            entries.to_vec().leak(),
        ),
        fullscreen_shader: fullscreen_shader.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "srgb_conversion.wesl"),
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SrgbConversionPipelineKey {
    /// The main texture format.
    pub format: TextureFormat,
    /// Whether the main texture stores Oklab instead of linear RGB.
    pub oklab: bool,
    /// Whether to convert to sRGB, or back from it.
    pub to_srgb: bool,
}

impl SpecializedRenderPipeline for SrgbConversionPipeline {
    type Key = SrgbConversionPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("ui_srgb_conversion_pipeline".into()),
            layout: vec![self.layout_descriptor.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: if key.oklab {
                    vec!["OKLAB".into()]
                } else {
                    Vec::new()
                },
                entry_point: Some(if key.to_srgb { "to_srgb" } else { "from_srgb" }.into()),
                targets: vec![Some(ColorTargetState {
                    format: key.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// Runs one conversion pass over the whole main texture.
pub fn run_srgb_conversion(
    ctx: &mut RenderContext,
    view_target: &ViewTarget,
    conversion_pipeline: &SrgbConversionPipeline,
    pipeline: &RenderPipeline,
    label: &'static str,
) {
    let post_process = view_target.post_process_write();
    let bind_group = ctx.render_device().create_bind_group(
        label,
        &conversion_pipeline.layout,
        &BindGroupEntries::single(post_process.source),
    );
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}
