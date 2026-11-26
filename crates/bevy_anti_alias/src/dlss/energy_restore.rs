//! Keeps DLSS Ray Reconstruction from shifting the image's brightness. See [`DlssEnergyRestore`].

use bevy_camera::Camera;
use bevy_ecs::{
    component::Component, entity::Entity, query::With, reflect::ReflectComponent, resource::Resource,
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent,
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
        BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder,
        ComputePassDescriptor, ComputePipeline, Extent3d, Origin3d, PipelineCompilationOptions,
        PipelineLayoutDescriptor, RawComputePipelineDescriptor, ShaderModuleDescriptor,
        ShaderSource, ShaderStages, StorageTextureAccess, TexelCopyTextureInfo, Texture,
        TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::{RenderDevice, RenderQueue},
    RenderApp,
};
use std::{collections::HashMap, sync::Mutex};

/// Keeps DLSS Ray Reconstruction from shifting the image's brightness.
///
/// DLSS-RR does not preserve the mean brightness of its noisy input. In Solari's Zero-Day scene its output runs
/// about 8% darker than its input while the camera moves, and climbs over several seconds to 5–20% brighter than
/// the input once the camera stops. This pass measures the mean luminance of DLSS-RR's input and output over
/// coarse screen blocks every frame, and rescales the output's low-frequency luminance to the input's. The
/// denoiser's inputs, and the detail it reconstructs, are left untouched.
///
/// The correction is one whole-frame gain, smoothed over time by [`Self::global_response`], times per-block
/// factors that vary slowly over the screen and over time ([`Self::block_size`], [`Self::block_response`]).
/// Pixels that DLSS-RR renders much brighter than the frame's mean (emitters and their highlights) are left out
/// of both measurements, see [`Self::emitter_threshold`].
///
/// It runs for every camera with DLSS Ray Reconstruction, using the defaults below when the camera does not have
/// this component. Add the component to tune it, or to turn it off with `enabled: false`.
#[derive(Component, Reflect, Clone, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[extract_component_filter(With<Camera>)]
#[extract_app(RenderApp)]
pub struct DlssEnergyRestore {
    /// Whether the correction runs.
    ///
    /// Defaults to `true`.
    pub enabled: bool,

    /// Size, in output pixels, of the screen blocks the correction varies over.
    ///
    /// Smaller blocks follow local brightness shifts more closely but flicker more.
    ///
    /// Defaults to `128`.
    pub block_size: u32,

    /// How quickly the whole-frame gain follows a change in DLSS-RR's brightness: the weight, per frame, of the
    /// newest measurement in an exponential moving average.
    ///
    /// Higher values track the input more tightly when the camera starts or stops moving (with `0.05`, the
    /// output stays within 1% of the input's brightness after the camera stops; with `0.03`, it settles within 2%)
    /// at the cost of slightly more low-frequency flicker.
    ///
    /// Defaults to `0.03`.
    pub global_response: f32,

    /// How quickly the per-block factors follow: the per-frame weight of the newest measurement.
    ///
    /// Kept much lower than [`Self::global_response`] because a single block's per-frame measurement is noisy.
    ///
    /// Defaults to `0.005`.
    pub block_response: f32,

    /// Lowest per-frame input/output brightness ratio taken into account. Limits how far a single noisy frame,
    /// such as one with a firefly, can move the correction.
    ///
    /// Defaults to `0.67`.
    pub min_ratio: f32,

    /// Highest per-frame input/output brightness ratio taken into account.
    ///
    /// Defaults to `1.5`.
    pub max_ratio: f32,

    /// Pixels whose DLSS-RR output is brighter than this multiple of the frame's mean output are left out of the
    /// measurement. DLSS-RR darkens emitters themselves while it brightens everything else, so including them
    /// would push the emitters' energy onto the rest of the image. `0` keeps every pixel.
    ///
    /// Defaults to `16.0`.
    pub emitter_threshold: f32,
}

impl Default for DlssEnergyRestore {
    fn default() -> Self {
        Self {
            enabled: true,
            block_size: 128,
            global_response: 0.03,
            block_response: 0.005,
            min_ratio: 0.67,
            max_ratio: 1.5,
            emitter_threshold: 16.0,
        }
    }
}

struct Pipelines {
    layout: BindGroupLayout,
    reduce_input: ComputePipeline,
    reduce_output: ComputePipeline,
    update_gain: ComputePipeline,
    apply: ComputePipeline,
}

struct ViewState {
    out_size: (u32, u32),
    block: u32,
    params: Buffer,
    input_mean: Buffer,
    output_mean: Buffer,
    gain: Buffer,
    state: Buffer,
    copy: Texture,
    copy_view: TextureView,
}

/// GPU resources of the energy restore pass, with each DLSS-RR view's measurements and gains.
#[derive(Resource, Default)]
pub(crate) struct EnergyRestore {
    pipelines: Mutex<Option<Pipelines>>,
    views: Mutex<HashMap<Entity, ViewState>>,
}

fn create_pipelines(device: &RenderDevice) -> Pipelines {
    let texture = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let buffer = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage = BufferBindingType::Storage { read_only: false };
    let entries = [
        buffer(0, BufferBindingType::Uniform),
        texture(1),
        texture(2),
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
        buffer(4, storage),
        buffer(5, storage),
        buffer(6, storage),
        buffer(7, storage),
    ];
    let layout = device.create_bind_group_layout("dlss_energy_restore_layout", &entries);
    let wgpu = device.wgpu_device();
    let module = wgpu.create_shader_module(ShaderModuleDescriptor {
        label: Some("dlss_energy_restore"),
        source: ShaderSource::Wgsl(include_str!("energy_restore.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dlss_energy_restore_pipeline_layout"),
        bind_group_layouts: &[Some(&*layout)],
        ..Default::default()
    });
    let pipeline = |entry_point: &str| -> ComputePipeline {
        wgpu.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
        .into()
    };
    Pipelines {
        reduce_input: pipeline("reduce_input"),
        reduce_output: pipeline("reduce_output"),
        update_gain: pipeline("update_gain"),
        apply: pipeline("apply"),
        layout,
    }
}

impl EnergyRestore {
    /// Records the energy restore. Call after DLSS-RR has written `output`; `input` must still hold its input.
    pub(crate) fn run(
        &self,
        view_entity: Entity,
        settings: &DlssEnergyRestore,
        device: &RenderDevice,
        queue: &RenderQueue,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        output_texture: &Texture,
        render_size: (u32, u32),
        reset: bool,
    ) {
        if output_texture.format() != TextureFormat::Rgba16Float {
            return;
        }
        let out_size = (output_texture.width(), output_texture.height());
        let block = settings.block_size.max(8);
        let grid = (out_size.0.div_ceil(block), out_size.1.div_ceil(block));

        let mut pipelines = self.pipelines.lock().unwrap();
        let pipelines = pipelines.get_or_insert_with(|| create_pipelines(device));
        let mut views = self.views.lock().unwrap();
        let stale = views
            .get(&view_entity)
            .is_none_or(|v| v.out_size != out_size || v.block != block);
        if stale {
            views.insert(view_entity, create_view_state(device, out_size, block, grid));
        }
        let v = &views[&view_entity];

        let words: [u32; 16] = [
            out_size.0,
            out_size.1,
            render_size.0,
            render_size.1,
            grid.0,
            grid.1,
            block,
            (reset || stale) as u32,
            settings.global_response.to_bits(),
            settings.block_response.to_bits(),
            settings.min_ratio.to_bits(),
            settings.max_ratio.to_bits(),
            settings.emitter_threshold.to_bits(),
            0,
            0,
            0,
        ];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        queue.write_buffer(&v.params, 0, &bytes);

        let copy_info = |texture| TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        };
        encoder.copy_texture_to_texture(
            copy_info(output_texture),
            copy_info(&v.copy),
            Extent3d {
                width: out_size.0,
                height: out_size.1,
                depth_or_array_layers: 1,
            },
        );

        let bind_group = bind_group(device, pipelines, v, input, output);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dlss_energy_restore"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &*bind_group, &[]);
        pass.set_pipeline(&pipelines.reduce_input);
        pass.dispatch_workgroups(grid.0, grid.1, 1);
        pass.set_pipeline(&pipelines.reduce_output);
        pass.dispatch_workgroups(grid.0, grid.1, 1);
        pass.set_pipeline(&pipelines.update_gain);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.apply);
        pass.dispatch_workgroups(out_size.0.div_ceil(8), out_size.1.div_ceil(8), 1);
    }
}

fn create_view_state(
    device: &RenderDevice,
    out_size: (u32, u32),
    block: u32,
    grid: (u32, u32),
) -> ViewState {
    let blocks = (grid.0 * grid.1) as u64;
    let storage = |label, count: u64| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: (count * 4).max(16),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let copy = device.create_texture(&TextureDescriptor {
        label: Some("dlss_energy_restore_output_copy"),
        size: Extent3d {
            width: out_size.0,
            height: out_size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let copy_view = copy.create_view(&TextureViewDescriptor::default());
    ViewState {
        out_size,
        block,
        params: device.create_buffer(&BufferDescriptor {
            label: Some("dlss_energy_restore_params"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        input_mean: storage("dlss_energy_restore_input_mean", blocks),
        output_mean: storage("dlss_energy_restore_output_mean", blocks),
        gain: storage("dlss_energy_restore_gain", blocks),
        // Per-block factors, then the whole-frame gain, then last frame's mean output luminance.
        state: storage("dlss_energy_restore_state", blocks + 2),
        copy,
        copy_view,
    }
}

fn bind_group(
    device: &RenderDevice,
    pipelines: &Pipelines,
    v: &ViewState,
    input: &TextureView,
    output: &TextureView,
) -> BindGroup {
    device.create_bind_group(
        "dlss_energy_restore_bind_group",
        &pipelines.layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: v.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(input),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&v.copy_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(output),
            },
            BindGroupEntry {
                binding: 4,
                resource: v.input_mean.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: v.output_mean.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: v.gain.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: v.state.as_entire_binding(),
            },
        ],
    )
}
