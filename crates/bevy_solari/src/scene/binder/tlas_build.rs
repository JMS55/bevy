//! Building the binder's TLAS through `wgpu_hal` directly, bypassing `wgpu-core`.
//!
//! `wgpu-core` re-derives an entire TLAS build from its `Vec<Option<TlasInstance>>` on every call:
//! per instance it resolves and clones an `Arc<Blas>` three times, heap-allocates a `Vec<u8>` for
//! the descriptor bytes, and appends to a staging buffer that grows from empty each frame. At 100k
//! instances that dominates the frame regardless of how few instances actually moved.
//!
//! Solari already keeps every instance's transform on the GPU, so instead a compute shader packs
//! the descriptors into a buffer we own, and the build is recorded straight into the hal encoder.
//! The CPU cost per frame becomes one dispatch and one build call.
//!
//! Vulkan and DX12 only. Their instance descriptors (`VkAccelerationStructureInstanceKHR` and
//! `D3D12_RAYTRACING_INSTANCE_DESC`) are byte-identical, so one shader and one code path serve
//! both. Every other backend falls back to the slower but portable `wgpu-core` build, see
//! [`tlas::build_raytracing_tlas`](super::tlas::build_raytracing_tlas).
//!
//! Metal cannot work this way. `wgpu-core` makes the acceleration structures a TLAS points at
//! resident by collecting its dependency list into an `MTLResidencySet` attached to the submission,
//! and skips that when the list is empty. `mark_acceleration_structures_built` records an empty
//! list and clears whatever the TLAS had, so the BLASes are never made resident and traversal reads
//! memory Metal is free to have evicted. That hal method is a no-op on Vulkan and DX12, which is
//! why they are unaffected, and there is no way to supply the dependency list from outside `wgpu`.
//!
//! Rather than matching on [`Backend`](wgpu::Backend), every operation tries each compiled-in
//! backend and relies on `as_hal` returning `None` for a resource that doesn't belong to it. That
//! can't disagree with the adapter actually in use.
//!
//! Anything recorded here is invisible to `wgpu-core`'s state tracker, so it emits no barriers for
//! it and learns nothing about the states left behind. Rather than hand-rolling those, the buffers
//! are transitioned with `CommandEncoder::transition_resources` from the system that runs the pack
//! pass through the regular API. That keeps the tracker's view accurate, which matters on DX12
//! where a transition carries an explicit before-state, and holds the buffers alive for the
//! submission. The acceleration structures' own barriers have no such public spelling, and are
//! placed by hand in [`build_tlas`].

#![expect(
    unsafe_code,
    reason = "Building the TLAS without wgpu-core's per-instance cost requires wgpu_hal."
)]
// With neither backend compiled in, every arm of `first_supported_backend` is cfg'd away, so the
// per-backend helpers are never instantiated and the entry points ignore their arguments. They
// still have to exist, because Solari compiles everywhere and only declines to load at runtime.
#![cfg_attr(
    not(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    )),
    expect(
        dead_code,
        unused_variables,
        reason = "no backend with a raw TLAS build path is compiled in for this target"
    )
)]

use bevy_render::{
    render_resource::{Blas, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Tlas},
    renderer::RenderDevice,
};
use wgpu::{hal, BufferUses};

/// Size of one TLAS instance descriptor, on both Vulkan and DXR.
pub const INSTANCE_DESCRIPTOR_SIZE: u64 = 64;

/// Runs `$body` against each backend with a raw build path until one produces `Some<$ok>`.
///
/// The caller spells out the result type rather than letting `$body` infer it, because on a target
/// where every arm is cfg'd away the expansion is a bare `None` with nothing to infer from.
///
/// The cfgs mirror when `wgpu` actually exposes each `hal::api` type, which is not the same as
/// `wgpu-hal`'s own aliases. `wgpu-hal`'s `vulkan` alias covers every non-wasm target, but the
/// feature behind it only reaches `wgpu-hal` by way of `wgpu-core-deps-windows-linux-android`,
/// which is only a dependency on those targets. On Apple it would take `vulkan-portability`, and
/// there is no reason to reach for `MoltenVK` when Metal is right there. `bevy_render` enables the
/// `vulkan`, `dx12` and `metal` features unconditionally, so they need no check here.
///
/// Keep the cfgs in step with the module-level `dead_code` expectation above.
macro_rules! first_supported_backend {
    ($ok:ty, |$api:ident| $body:block) => {{
        // Shadowed rather than reassigned, so that a target with no arms at all doesn't leave an
        // unused `mut` behind
        let result: Option<$ok> = None;

        #[cfg(any(
            windows,
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        ))]
        let result = result.or_else(|| {
            type $api = hal::api::Vulkan;
            $body
        });

        #[cfg(windows)]
        let result = result.or_else(|| {
            type $api = hal::api::Dx12;
            $body
        });

        result
    }};
}

/// Whether this device's backend has a raw TLAS build path.
///
/// [`TlasState`](super::tlas::TlasState) builds through `wgpu-core` instead when this is false.
pub fn supported(render_device: &RenderDevice) -> bool {
    let device = render_device.wgpu_device();

    first_supported_backend!((), |A| {
        // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
        unsafe { device.as_hal::<A>() }.map(|_| ())
    })
    .is_some()
}

/// Scratch space a TLAS build over `instance_count` instances of `instances` needs.
pub fn tlas_scratch_size(
    render_device: &RenderDevice,
    instances: &Buffer,
    instance_count: u32,
) -> Option<u64> {
    first_supported_backend!(u64, |A| {
        tlas_scratch_size_impl::<A>(render_device, instances, instance_count)
    })
}

fn tlas_scratch_size_impl<A: hal::Api>(
    render_device: &RenderDevice,
    instances: &Buffer,
    instance_count: u32,
) -> Option<u64> {
    use hal::Device as _;

    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
    let hal_device = unsafe { render_device.wgpu_device().as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
    let hal_instances = unsafe { instances.as_hal::<A>() }?;

    // Vulkan and DXR both document buffers and addresses as ignored for sizing, and pass a null
    // one. `wgpu_hal`'s Metal path unwraps the buffer unconditionally though, so a real one has to
    // be handed over, which is why this can't be asked before the descriptor buffer exists.
    let entries =
        hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
            buffer: Some(&*hal_instances),
            offset: 0,
            count: instance_count,
        });

    // SAFETY: `entries` describes instances of a buffer belonging to backend `A`, which `as_hal`
    // just confirmed, and the sizing query records nothing and reads no memory through it
    let sizes = unsafe {
        hal_device.get_acceleration_structure_build_sizes(
            &hal::GetAccelerationStructureBuildSizesDescriptor {
                entries: &entries,
                flags: TLAS_BUILD_FLAGS,
            },
        )
    };

    Some(sizes.build_scratch_size)
}

/// Allocates a buffer usable as acceleration structure build scratch.
///
/// There is no [`BufferUsages`] for scratch, and a plain `STORAGE` buffer won't do: Vulkan only
/// adds `SHADER_DEVICE_ADDRESS` for acceleration structure usages, and scratch is addressed by
/// pointer. So the buffer is created through hal and handed straight to `wgpu-core`, which then
/// owns destroying it.
///
/// Being owned by `wgpu-core` is not by itself enough to make replacing this buffer safe. It defers
/// a free only for resources a submission's tracker references, and a buffer used exclusively
/// through `as_hal` is referenced by nothing. What earns the deferral is the caller transitioning
/// it with `CommandEncoder::transition_resources` each frame it is used.
pub fn create_scratch_buffer(render_device: &RenderDevice, size: u64) -> Option<Buffer> {
    first_supported_backend!(Buffer, |A| {
        create_scratch_buffer_impl::<A>(render_device, size)
    })
}

fn create_scratch_buffer_impl<A: hal::Api>(
    render_device: &RenderDevice,
    size: u64,
) -> Option<Buffer> {
    use hal::Device as _;

    let device = render_device.wgpu_device();
    let hal_buffer = {
        // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
        let hal_device = unsafe { device.as_hal::<A>() }?;
        // SAFETY: the descriptor is valid and the buffer is handed straight to `wgpu-core` below,
        // which takes over destroying it
        unsafe {
            hal_device.create_buffer(&hal::BufferDescriptor {
                label: Some("solari_tlas_scratch"),
                size,
                usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                memory_flags: hal::MemoryFlags::empty(),
            })
        }
        .ok()?
    };

    // The descriptor has to describe the buffer that was just created. `usage` is only consulted
    // for validating wgpu-level operations, and this buffer is only ever reached through
    // `as_hal`, so `STORAGE` stands in for a usage that has no public spelling.
    let descriptor = BufferDescriptor {
        label: Some("solari_tlas_scratch"),
        size,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    };

    // SAFETY: `hal_buffer` was created from this device, matches `descriptor`, and has nonzero
    // size (callers never ask for zero scratch)
    Some(unsafe { device.create_buffer_from_hal::<A>(hal_buffer, &descriptor) }.into())
}

/// Flags the TLAS is created with, which its build has to be given again.
const TLAS_BUILD_FLAGS: wgpu::AccelerationStructureFlags =
    wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;

/// Records a TLAS build reading instance descriptors straight out of `instances`.
///
/// Returns `false` on a backend without a raw path, having recorded nothing. That shouldn't happen,
/// as such a backend goes through `wgpu-core` instead and never reaches this.
///
/// The caller must already have transitioned `instances` to
/// [`BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT`] and `scratch` to
/// [`BufferUses::ACCELERATION_STRUCTURE_SCRATCH`] with `CommandEncoder::transition_resources`, on
/// an earlier regularly-encoded command buffer in the same submission. The scratch transition is
/// not optional even when its state is unchanged: scratch is an exclusive usage, so the
/// redundant-looking transition is what keeps consecutive frames' builds from overlapping in it.
///
/// `encoder` must not have had any wgpu command recorded into it, including timestamp writes, as
/// `wgpu-core` panics if an encoder mixes the wgpu and raw encoding APIs. [`RenderContext`] hands
/// each system its own encoder, so a system that calls only this is fine.
///
/// Marking the build clears the TLAS's dependency list, which is what `wgpu-core` otherwise uses to
/// keep the BLASes a TLAS points at alive. The caller has to retain them past every submission that
/// might still trace this TLAS.
///
/// [`BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT`]: wgpu::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT
/// [`BufferUses::ACCELERATION_STRUCTURE_SCRATCH`]: wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH
/// [`RenderContext`]: bevy_render::renderer::RenderContext
pub fn build_tlas(
    encoder: &mut CommandEncoder,
    tlas: &mut Tlas,
    instances: &Buffer,
    instance_count: u32,
    scratch: &Buffer,
) -> bool {
    let built = first_supported_backend!((), |A| {
        build_tlas_impl::<A>(encoder, tlas, instances, instance_count, scratch)
    })
    .is_some();

    if built {
        // Without this the build is invisible to `wgpu-core` and it rejects the TLAS when bound. It
        // is itself a raw call, so it doesn't conflict with the encoder exclusivity above.
        //
        // SAFETY: the TLAS was just built into this encoder, and every BLAS its instances point at
        // was built and submitted earlier in the frame by `prepare_raytracing_blas`.
        unsafe {
            encoder.mark_acceleration_structures_built(core::iter::empty::<&Blas>(), [&*tlas]);
        }
    }

    built
}

fn build_tlas_impl<A: hal::Api>(
    encoder: &mut CommandEncoder,
    tlas: &mut Tlas,
    instances: &Buffer,
    instance_count: u32,
    scratch: &Buffer,
) -> Option<()> {
    use hal::CommandEncoder as _;

    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
    let hal_tlas = unsafe { tlas.as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
    let hal_instances = unsafe { instances.as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard
    let hal_scratch = unsafe { scratch.as_hal::<A>() }?;

    let entries =
        hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
            buffer: Some(&*hal_instances),
            offset: 0,
            count: instance_count,
        });

    let descriptor = hal::BuildAccelerationStructureDescriptor {
        entries: &entries,
        mode: hal::AccelerationStructureBuildMode::Build,
        flags: TLAS_BUILD_FLAGS,
        source_acceleration_structure: None,
        destination_acceleration_structure: &*hal_tlas,
        scratch_buffer: &*hal_scratch,
        scratch_buffer_offset: 0,
    };

    // Both buffers were already transitioned by the caller, through `wgpu-core`, so the only
    // barriers left are the acceleration structure's own, which have no public spelling. Both
    // mirror what `wgpu-core` places around its own TLAS build.
    //
    // SAFETY: every resource in `descriptor` belongs to backend `A` and to this device, the
    // scratch buffer is sized by `tlas_scratch_size` and used by no other build in this
    // submission, and the encoder is neither ended nor its raw handle destroyed here.
    unsafe {
        encoder.as_hal_mut::<A, _, _>(|encoder| {
            let encoder = encoder?;

            // Write-after-read against earlier frames' traces. This parity's TLAS stays bound as
            // the previous frame's for a second frame after it stops being current, so a submission
            // still tracing it can overlap this build. Having two parities widens that window
            // rather than closing it.
            //
            // This also makes a BLAS compaction copy visible to this build. `Queue::compact_blas`
            // records that copy into wgpu's pending writes, which are submitted before these
            // command buffers, but wgpu can't know that the raw build consumes the COPY_DST as a
            // BUILD_INPUT, and so can't insert the dependency itself. A barrier's first scope
            // covers everything already submitted to the queue, which is what makes the combined
            // transition enough for both hazards.
            encoder.place_acceleration_structure_barrier(hal::AccelerationStructureBarrier {
                usage: hal::StateTransition {
                    from: hal::AccelerationStructureUses::SHADER_INPUT
                        | hal::AccelerationStructureUses::COPY_DST,
                    to: hal::AccelerationStructureUses::BUILD_OUTPUT
                        | hal::AccelerationStructureUses::BUILD_INPUT,
                },
            });

            encoder.build_acceleration_structures(1, [descriptor]);

            // And read-after-write, for the passes later this frame that trace the result
            encoder.place_acceleration_structure_barrier(hal::AccelerationStructureBarrier {
                usage: hal::StateTransition {
                    from: hal::AccelerationStructureUses::BUILD_OUTPUT,
                    to: hal::AccelerationStructureUses::SHADER_INPUT,
                },
            });

            Some(())
        })
    }
}
