use super::{
    bind_group::BindGroupCacheState, instances::InstanceState, tlas_build, BlasManager,
    RaytracingSceneBindings,
};
use bevy_asset::load_embedded_asset;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        AccelerationStructureFlags, AccelerationStructureUpdateMode, BindGroup, BindGroupEntries,
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferDescriptor, BufferId,
        BufferUsages, CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, CreateTlasDescriptor, PipelineCache, ShaderStages, Tlas,
        TlasInstance,
    },
    renderer::{RenderContext, RenderDevice},
};
use bevy_utils::{default, once};
use tracing::{info_span, warn};
use wgpu::{BufferTransition, BufferUses};

/// Workgroup size of `tlas_instances.wgsl`. Has to match the shader.
const TLAS_INSTANCE_PACK_WORKGROUP_SIZE: u32 = 64;

/// Compute pipeline that packs per-slot transforms and BLAS addresses into TLAS descriptors.
#[derive(Resource)]
pub struct TlasInstancePackPipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub id: Option<CachedComputePipelineId>,
}

impl FromWorld for TlasInstancePackPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "tlas_instance_pack_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );

        if !tlas_build::supported(world.resource::<RenderDevice>()) {
            return Self { layout, id: None };
        }

        let shader = load_embedded_asset!(world, "tlas_instances.wgsl");
        let id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("tlas_instance_pack_pipeline".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("pack_tlas_instances".into()),
                    ..default()
                });

        Self {
            layout,
            id: Some(id),
        }
    }
}

/// Smallest TLAS instance capacity handed out, and the floor [`tlas_capacity_for`] grows from.
const TLAS_MIN_CAPACITY: u32 = 128;

/// Width of a TLAS instance's custom data in both Vulkan (`instanceCustomIndex`) and DXR
/// (`InstanceID`). `tlas_instances.wgsl` packs instance slots into that field, so they have to fit.
const TLAS_CUSTOM_DATA_BITS: u32 = 24;

/// Instance capacity to allocate to hold `instance_count` slots.
///
/// Geometric rather than a fixed step, so that the number of reallocations over a scene's lifetime
/// is logarithmic in its size rather than linear. A pure function of the count keeps the descriptor
/// buffer and both TLASes on the same capacity curve without coordinating their previous sizes.
fn tlas_capacity_for(instance_count: u32) -> u32 {
    let mut capacity = TLAS_MIN_CAPACITY;
    while capacity < instance_count {
        capacity = capacity.saturating_add(capacity.div_ceil(2));
    }
    capacity
}

/// TLAS parity and capacity state.
///
/// A parity is only bindable after it has been built. Keeping the BLASes a built parity points at
/// alive is [`BlasManager`]'s job, which is why nothing here owns one: it defers every retirement
/// until [`BlasManager::note_tlas_build`] has seen both parities rebuilt.
pub struct TlasState {
    /// Whether builds go through [`tlas_build`]'s raw path rather than `wgpu-core`.
    raw_build: bool,
    /// Alternating current/previous acceleration structures.
    pub structures: [Option<Tlas>; 2],
    capacity: [u32; 2],
    /// Whether each structure has had a build recorded since its latest allocation.
    pub built: [bool; 2],
    pub frame_parity: usize,
    pub instance_descriptors: Option<Buffer>,
    instance_descriptor_capacity: u32,
    pub scratch: Option<Buffer>,
    scratch_capacity: u64,
    scratch_sized_for: u32,
    pub instances_packed: bool,
    pub instance_pack_bind_group: Option<BindGroup>,
    instance_pack_buffer_ids: Option<[BufferId; 3]>,
}

impl TlasState {
    pub fn new(render_device: &RenderDevice) -> Self {
        Self {
            raw_build: tlas_build::supported(render_device),
            structures: [None, None],
            capacity: [0, 0],
            built: [false, false],
            frame_parity: 0,
            instance_descriptors: None,
            instance_descriptor_capacity: 0,
            scratch: None,
            scratch_capacity: 0,
            scratch_sized_for: 0,
            instances_packed: false,
            instance_pack_bind_group: None,
            instance_pack_buffer_ids: None,
        }
    }

    /// Whether builds go through [`tlas_build`]'s raw path rather than `wgpu-core`.
    pub fn uses_raw_build(&self) -> bool {
        self.raw_build
    }

    /// Moves to the next TLAS parity and brings it up to date with this frame's changes.
    ///
    /// The two acceleration structures alternate: this frame's is rebuilt, and last frame's stays
    /// intact so the shaders can trace against it.
    ///
    /// `build_ready` reports whether this frame will be able to record a build. When it is false
    /// nothing happens at all — not even the parity flip.
    pub fn advance(
        &mut self,
        instances: &InstanceState,
        bind_groups: &mut BindGroupCacheState,
        render_device: &RenderDevice,
        build_ready: bool,
    ) {
        let _span = info_span!("advance_tlas").entered();

        // An empty scene must not allocate an unbuilt TLAS that could resurface as a later
        // previous-frame entry.
        if !build_ready || instances.slots.len() == 0 {
            return;
        }

        debug_assert!(
            instances.slots.len() < 1 << TLAS_CUSTOM_DATA_BITS,
            "instance slot count {} does not fit in a TLAS instance's custom data",
            instances.slots.len()
        );

        // Secure every build input before committing the parity flip. Neither buffer exists on the
        // `wgpu-core` path, which builds from instances written into the TLAS itself.
        let instance_count = instances.slots.len();
        if self.raw_build {
            self.reserve_instance_descriptors(instance_count, render_device);
            self.reserve_tlas_scratch(render_device);
            if self.instance_descriptors.is_none() || self.scratch.is_none() {
                return;
            }
        }

        self.frame_parity ^= 1;
        let parity = self.frame_parity;
        self.reserve_tlas(parity, instance_count, render_device, bind_groups);
    }

    /// Makes sure the instance descriptor buffer covers every stable slot.
    fn reserve_instance_descriptors(&mut self, needed: u32, render_device: &RenderDevice) {
        if self.instance_descriptors.is_some() && needed <= self.instance_descriptor_capacity {
            return;
        }

        let capacity = tlas_capacity_for(needed);
        self.instance_descriptors = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_tlas_instance_descriptors"),
            size: u64::from(capacity) * tlas_build::INSTANCE_DESCRIPTOR_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::TLAS_INPUT,
            mapped_at_creation: false,
        }));
        self.instance_descriptor_capacity = capacity;
        self.instance_pack_bind_group = None;
    }

    /// Makes sure the build scratch buffer is big enough for this frame's descriptor capacity.
    fn reserve_tlas_scratch(&mut self, render_device: &RenderDevice) {
        let capacity = self.instance_descriptor_capacity;
        if self.scratch.is_some() && capacity <= self.scratch_sized_for {
            return;
        }

        let Some(instances) = self.instance_descriptors.as_ref() else {
            return;
        };
        let Some(needed) = tlas_build::tlas_scratch_size(render_device, instances, capacity) else {
            return;
        };
        self.scratch_sized_for = capacity;

        if self.scratch.is_some() && needed <= self.scratch_capacity {
            return;
        }

        // The pack transition lets wgpu retain an outgrown scratch buffer until in-flight work
        // releases it.
        self.scratch = tlas_build::create_scratch_buffer(render_device, needed);
        self.scratch_capacity = if self.scratch.is_some() { needed } else { 0 };
    }

    /// Makes sure one parity can hold every instance slot without disturbing the other parity.
    fn reserve_tlas(
        &mut self,
        parity: usize,
        needed: u32,
        render_device: &RenderDevice,
        bind_groups: &mut BindGroupCacheState,
    ) {
        if self.structures[parity].is_some() && needed <= self.capacity[parity] {
            return;
        }

        let capacity = tlas_capacity_for(needed);
        self.structures[parity] = Some(render_device.wgpu_device().create_tlas(
            &CreateTlasDescriptor {
                label: Some("tlas"),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
                max_instances: capacity,
            },
        ));
        self.capacity[parity] = capacity;
        self.built[parity] = false;
        bind_groups.invalid = true;
    }

    /// Rebuilds the pack shader's bind group only when one of its three buffers moves.
    pub fn update_instance_pack_bind_group(
        &mut self,
        instances: &InstanceState,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        pipeline: &TlasInstancePackPipeline,
    ) {
        let (Some(transforms), Some(blas_refs), Some(instances)) = (
            instances.transforms.buffer(),
            instances.blas_refs.buffer(),
            self.instance_descriptors.as_ref(),
        ) else {
            self.instance_pack_bind_group = None;
            return;
        };

        let ids = [transforms.id(), blas_refs.id(), instances.id()];
        if self.instance_pack_bind_group.is_some() && self.instance_pack_buffer_ids == Some(ids) {
            return;
        }

        let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
        self.instance_pack_bind_group = Some(render_device.create_bind_group(
            "tlas_instance_pack_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                transforms.as_entire_binding(),
                blas_refs.as_entire_binding(),
                instances.as_entire_binding(),
            )),
        ));
        self.instance_pack_buffer_ids = Some(ids);
    }
}

/// Packs this frame's TLAS instance descriptors on the GPU.
pub fn pack_raytracing_tlas_instances(
    mut bindings: ResMut<RaytracingSceneBindings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<TlasInstancePackPipeline>,
    mut render_context: RenderContext,
) {
    let bindings = &mut *bindings;
    bindings.tlas.instances_packed = false;

    if !bindings.tlas.raw_build {
        return;
    }
    if bindings.tlas.structures[bindings.tlas.frame_parity].is_none() {
        return;
    }

    let (Some(bind_group), Some(compute_pipeline), Some(instances), Some(scratch)) = (
        bindings.tlas.instance_pack_bind_group.as_ref(),
        pipeline
            .id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id)),
        bindings.tlas.instance_descriptors.as_ref(),
        bindings.tlas.scratch.as_ref(),
    ) else {
        return;
    };

    let slot_count = bindings.instances.slots.len();
    if slot_count == 0 {
        return;
    }

    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let command_encoder = render_context.command_encoder();
    let time_span = diagnostics.time_span(command_encoder, "pack_tlas_instances");
    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("pack_tlas_instances"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(slot_count.div_ceil(TLAS_INSTANCE_PACK_WORKGROUP_SIZE), 1, 1);
    }
    time_span.end(command_encoder);

    command_encoder.transition_resources(
        [
            BufferTransition {
                buffer: &**instances,
                state: BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            },
            BufferTransition {
                buffer: &**scratch,
                state: BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
            },
        ]
        .into_iter(),
        core::iter::empty(),
    );

    bindings.tlas.instances_packed = true;
}

/// Records this frame's TLAS build into the render graph's command encoder.
pub fn build_raytracing_tlas(
    mut bindings: ResMut<RaytracingSceneBindings>,
    mut blas_manager: ResMut<BlasManager>,
    mut render_context: RenderContext,
) {
    let bindings = &mut *bindings;
    let parity = bindings.tlas.frame_parity;

    let built = if bindings.tlas.raw_build {
        build_tlas_raw(bindings, &mut render_context)
    } else {
        build_tlas_through_wgpu_core(bindings, &blas_manager, &mut render_context)
    };

    if built {
        bindings.tlas.built[parity] = true;
        // This parity no longer points at whatever was retired before it, which is what lets the
        // oldest retirements go.
        blas_manager.note_tlas_build();
    }
}

/// Builds from the descriptors the pack pass wrote, straight through `wgpu_hal`.
fn build_tlas_raw(
    bindings: &mut RaytracingSceneBindings,
    render_context: &mut RenderContext,
) -> bool {
    let parity = bindings.tlas.frame_parity;
    let Some(tlas) = bindings.tlas.structures[parity].as_mut() else {
        return false;
    };

    let (true, Some(instances), Some(scratch)) = (
        bindings.tlas.instances_packed,
        bindings.tlas.instance_descriptors.as_ref(),
        bindings.tlas.scratch.as_ref(),
    ) else {
        once!(warn!(
            "TLAS allocated but not built: packed={}, descriptors={}, scratch={}",
            bindings.tlas.instances_packed,
            bindings.tlas.instance_descriptors.is_some(),
            bindings.tlas.scratch.is_some(),
        ));
        return false;
    };

    let render_device = render_context.render_device().clone();
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(render_context.command_encoder(), "tlas_build");

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tlas_build_command_encoder"),
    });
    let built = tlas_build::build_tlas(
        &mut command_encoder,
        tlas,
        instances,
        bindings.instances.slots.len(),
        scratch,
    );
    render_context.add_command_buffer(command_encoder.finish());

    time_span.end(render_context.command_encoder());
    if !built {
        once!(warn!(
            "TLAS build recorded nothing; the backend probe and the build disagree about hal \
             access."
        ));
    }
    built
}

/// Builds from instance descriptors filled in on the CPU, through `wgpu-core`.
///
/// The portable path, and the only one Metal can use. It costs a slot's worth of work per frame
/// for every instance rather than only the ones that moved, and `wgpu-core` then re-derives the
/// whole build from what it is handed here, which is what [`tlas_build`] exists to avoid.
fn build_tlas_through_wgpu_core(
    bindings: &mut RaytracingSceneBindings,
    blas_manager: &BlasManager,
    render_context: &mut RenderContext,
) -> bool {
    // An empty scene leaves the parity unflipped, so whatever is here belongs to an earlier frame
    // and is still being traced as the previous one.
    if bindings.instances.slots.len() == 0 {
        return false;
    }
    let parity = bindings.tlas.frame_parity;
    let Some(tlas) = bindings.tlas.structures[parity].as_mut() else {
        return false;
    };

    {
        let _span = info_span!("fill_tlas_instances").entered();

        // The TLAS outlives the frame, so a slot freed or deactivated since its last build still
        // holds that build's instance and has to be cleared before this frame's are written.
        let capacity = tlas.get().len();
        tlas[0..capacity].iter_mut().for_each(|entry| *entry = None);

        for (slot, mesh, transform) in bindings.instances.drawable() {
            // A mesh can lose its acceleration structure after the instance resolved against it,
            // which leaves the slot with nothing to point at for a frame.
            let Some(blas) = blas_manager.get(&mesh) else {
                continue;
            };
            tlas[slot as usize] = Some(TlasInstance::new(blas, transform, slot, 0xFF));
        }
    }

    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let command_encoder = render_context.command_encoder();
    let time_span = diagnostics.time_span(command_encoder, "tlas_build");
    command_encoder.build_acceleration_structures(&[], [&*tlas]);
    time_span.end(command_encoder);

    true
}

#[cfg(test)]
mod tests {
    use super::{tlas_capacity_for, TLAS_MIN_CAPACITY};

    #[test]
    fn tlas_capacity_grows_geometrically_at_boundaries() {
        assert_eq!(tlas_capacity_for(0), TLAS_MIN_CAPACITY);
        assert_eq!(tlas_capacity_for(TLAS_MIN_CAPACITY), TLAS_MIN_CAPACITY);
        assert_eq!(tlas_capacity_for(TLAS_MIN_CAPACITY + 1), 192);
        assert_eq!(tlas_capacity_for(192), 192);
        assert_eq!(tlas_capacity_for(193), 288);
    }
}
