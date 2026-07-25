//! Building the TLAS through `wgpu_hal` directly, bypassing `wgpu-core`.
//!
//! `wgpu-core` re-derives an entire TLAS build from its `Vec<Option<TlasInstance>>` on every call:
//! per instance it resolves and clones an `Arc<Blas>` three times, heap-allocates a `Vec<u8>` for
//! the descriptor bytes, and appends to a staging buffer that grows from empty each frame. At
//! 100k instances that dominates the frame regardless of how few instances actually moved.
//!
//! Solari already keeps every instance's transform on the GPU, so instead the descriptors are
//! packed by a compute shader into a buffer we own, and the build is recorded straight into the
//! hal encoder. The CPU cost per frame becomes one dispatch and one build call.
//!
//! # Backends
//!
//! Vulkan's `VkAccelerationStructureInstanceKHR` and DXR's `D3D12_RAYTRACING_INSTANCE_DESC` are
//! byte-identical. Metal's `MTLIndirectAccelerationStructureInstanceDescriptor` is 72 bytes with a
//! column-major matrix and unpacked mask/user id, so it gets a second layout in the shader; see
//! [`InstanceLayout`].
//!
//! Rather than matching on [`Backend`](bevy_render::settings::Backend), every operation tries each
//! compiled-in backend and relies on `as_hal` returning `None` for a resource that doesn't belong
//! to it. That can't disagree with the adapter actually in use.
//!
//! # Barriers
//!
//! Anything recorded here is invisible to `wgpu-core`'s state tracker, so it emits no barriers for
//! it and learns nothing about the states left behind. Rather than hand-rolling those, the buffers
//! are transitioned with `CommandEncoder::transition_resources` from the *other* system — the one
//! that runs the pack pass through the regular API. That keeps the tracker's view accurate, which
//! matters on DX12 where a resource transition carries an explicit before-state, and has the side
//! benefit of holding the buffers alive for the submission. Only the acceleration structure
//! barrier, which has no such public spelling, is placed by hand here.

#![expect(
    unsafe_code,
    reason = "Building the TLAS without wgpu-core's per-instance cost requires wgpu_hal."
)]

use alloc::{vec, vec::Vec};
use bevy_render::{
    render_resource::{Blas, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Tlas},
    renderer::RenderDevice,
};
use bevy_shader::ShaderDefVal;
use core::mem::size_of;
use wgpu::{hal, BufferUses};

/// Memory layout of a TLAS instance descriptor, which differs by backend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InstanceLayout {
    /// `VkAccelerationStructureInstanceKHR`, and identically `D3D12_RAYTRACING_INSTANCE_DESC`:
    /// 64 bytes, row-major 3x4 matrix, custom data and mask bit-packed into one word.
    Packed64,
    /// `MTLIndirectAccelerationStructureInstanceDescriptor`: 72 bytes, column-major 4x3 matrix of
    /// packed float3s, and separate words for options, mask, table offset and user id.
    #[cfg_attr(
        not(target_vendor = "apple"),
        expect(
            dead_code,
            reason = "only constructed by the Metal arm of `instance_layout`"
        )
    )]
    Metal72,
}

impl InstanceLayout {
    /// Size of one descriptor in bytes.
    pub fn size(self) -> u64 {
        match self {
            Self::Packed64 => 64,
            Self::Metal72 => 72,
        }
    }

    /// Shader def selecting this layout in `tlas_instances.wgsl`.
    pub fn shader_defs(self) -> Vec<ShaderDefVal> {
        match self {
            Self::Packed64 => vec![],
            Self::Metal72 => vec!["TLAS_INSTANCE_LAYOUT_METAL".into()],
        }
    }

    /// Bytes of immediate data `tlas_instances.wgsl` takes, which is only what a dead slot points
    /// at — and only where that isn't the constant zero. See [`needs_dummy_blas`].
    pub fn immediate_size(self) -> u32 {
        if needs_dummy_blas(self) {
            size_of::<u64>() as u32
        } else {
            0
        }
    }
}

/// Runs `$body` against each backend with a raw build path until one produces `Some`.
///
/// The backend cfgs mirror `wgpu-hal`'s own (`vulkan` is every non-wasm target, `dx12` is Windows,
/// `metal` is Apple), since those aliases aren't visible outside that crate. `bevy_render` enables
/// all three features unconditionally, so no feature check is needed here.
macro_rules! first_supported_backend {
    (|$api:ident| $body:block) => {{
        let mut result = None;

        #[cfg(not(target_arch = "wasm32"))]
        if result.is_none() {
            type $api = hal::api::Vulkan;
            result = $body;
        }

        #[cfg(target_os = "windows")]
        if result.is_none() {
            type $api = hal::api::Dx12;
            result = $body;
        }

        #[cfg(target_vendor = "apple")]
        if result.is_none() {
            type $api = hal::api::Metal;
            result = $body;
        }

        result
    }};
}

/// The descriptor layout this device's backend expects, or `None` if it has no raw build path.
pub fn instance_layout(render_device: &RenderDevice) -> Option<InstanceLayout> {
    // Deliberately not written with `first_supported_backend!`: the answer depends on which arm
    // matched, not just that one did.
    let device = render_device.wgpu_device();

    #[cfg(not(target_arch = "wasm32"))]
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    if unsafe { device.as_hal::<hal::api::Vulkan>() }.is_some() {
        return Some(InstanceLayout::Packed64);
    }

    #[cfg(target_os = "windows")]
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    if unsafe { device.as_hal::<hal::api::Dx12>() }.is_some() {
        return Some(InstanceLayout::Packed64);
    }

    #[cfg(target_vendor = "apple")]
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    if unsafe { device.as_hal::<hal::api::Metal>() }.is_some() {
        return Some(InstanceLayout::Metal72);
    }

    let _ = device;
    None
}

/// Whether dead instance slots need to point at a real, masked-off acceleration structure.
///
/// Vulkan and DXR both define a zero acceleration structure reference as an inactive instance that
/// the build discards, so a hole costs nothing there. Metal references its structures by
/// [`MTLResourceID`], an opaque handle rather than an address, and documents nothing about a zero
/// one — so holes get a degenerate dummy with a zero mask instead of a value the driver may
/// dereference.
///
/// [`MTLResourceID`]: https://developer.apple.com/documentation/metal/mtlresourceid
pub fn needs_dummy_blas(layout: InstanceLayout) -> bool {
    match layout {
        InstanceLayout::Packed64 => false,
        InstanceLayout::Metal72 => true,
    }
}

/// The device address a TLAS instance descriptor refers to a BLAS by.
///
/// `None` on a backend with no raw build path, which `RaytracingScenePlugin` declines to load on.
pub fn blas_device_address(render_device: &RenderDevice, blas: &mut Blas) -> Option<u64> {
    first_supported_backend!(|A| { blas_device_address_impl::<A>(render_device, blas) })
}

fn blas_device_address_impl<A: hal::Api>(
    render_device: &RenderDevice,
    blas: &mut Blas,
) -> Option<u64> {
    use hal::Device as _;

    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    let hal_blas = unsafe { blas.as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    let hal_device = unsafe { render_device.wgpu_device().as_hal::<A>() }?;

    // SAFETY: `hal_blas` came from `hal_device`, which `as_hal` just confirmed is backend `A`.
    Some(unsafe { hal_device.get_acceleration_structure_device_address(&hal_blas) })
}

/// Scratch space a TLAS build over `instance_count` instances needs.
pub fn tlas_scratch_size(render_device: &RenderDevice, instance_count: u32) -> Option<u64> {
    first_supported_backend!(|A| { tlas_scratch_size_impl::<A>(render_device, instance_count) })
}

fn tlas_scratch_size_impl<A: hal::Api>(
    render_device: &RenderDevice,
    instance_count: u32,
) -> Option<u64> {
    use hal::Device as _;

    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    let hal_device = unsafe { render_device.wgpu_device().as_hal::<A>() }?;

    // Buffers and addresses are documented as ignored for sizing, so the instance buffer doesn't
    // have to exist yet — only the count matters.
    let entries =
        hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
            buffer: None,
            offset: 0,
            count: instance_count,
        });

    // SAFETY: `entries` describes instances with no buffer, which is what the sizing query wants.
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
/// adds `SHADER_DEVICE_ADDRESS` for acceleration-structure usages, and scratch is addressed by
/// pointer. So the buffer is created through hal and handed straight to `wgpu-core`, which then
/// owns its destruction.
///
/// Note that being owned by `wgpu-core` is not by itself enough to make replacing this buffer
/// safe. It defers a free only for resources a submission's tracker references, and a buffer used
/// exclusively through `as_hal` is referenced by nothing. What actually earns the deferral is the
/// caller transitioning it with `CommandEncoder::transition_resources` each frame it is used.
pub fn create_scratch_buffer(render_device: &RenderDevice, size: u64) -> Option<Buffer> {
    first_supported_backend!(|A| { create_scratch_buffer_impl::<A>(render_device, size) })
}

fn create_scratch_buffer_impl<A: hal::Api>(
    render_device: &RenderDevice,
    size: u64,
) -> Option<Buffer> {
    use hal::Device as _;

    let device = render_device.wgpu_device();
    let hal_buffer = {
        // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
        let hal_device = unsafe { device.as_hal::<A>() }?;
        // SAFETY: the descriptor is valid and the buffer is handed straight to `wgpu-core` below,
        // which takes over destroying it.
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
    // size (callers never ask for zero scratch).
    Some(unsafe { device.create_buffer_from_hal::<A>(hal_buffer, &descriptor) }.into())
}

/// Flags the TLAS is created with, which its build has to be given again.
const TLAS_BUILD_FLAGS: wgpu::AccelerationStructureFlags =
    wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;

/// Records a TLAS build reading instance descriptors straight out of `instances`.
///
/// Returns `false` on a backend without a raw path, having recorded nothing. That shouldn't
/// happen: `RaytracingScenePlugin` declines to load on such a backend.
///
/// # Preconditions
///
/// The caller must already have transitioned `instances` to
/// [`BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT`] and `scratch` to
/// [`BufferUses::ACCELERATION_STRUCTURE_SCRATCH`] through
/// `CommandEncoder::transition_resources` on an earlier, regularly-encoded command buffer in the
/// same submission. The scratch transition is not optional even when its state is unchanged:
/// scratch is an exclusive usage, so the redundant-looking transition is what keeps consecutive
/// frames' builds from overlapping in it.
///
/// [`BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT`]: wgpu::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT
/// [`BufferUses::ACCELERATION_STRUCTURE_SCRATCH`]: wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH
///
/// # Encoder exclusivity
///
/// `wgpu-core` panics if an encoder mixes the wgpu and raw encoding APIs, so `encoder` must not
/// have had any wgpu command recorded into it — including timestamp writes. [`RenderContext`]
/// hands each system its own encoder, so a system that calls only this is fine.
/// `mark_acceleration_structures_built`, which this calls to keep the TLAS bindable, is itself a
/// raw-API call and so doesn't conflict.
///
/// [`RenderContext`]: bevy_render::renderer::RenderContext
///
/// # Lifetimes
///
/// Marking the build clears the TLAS's dependency list, which is what `wgpu-core` otherwise uses
/// to keep the BLASes a TLAS points at alive. The caller becomes responsible for retaining them
/// past every submission that might still trace this TLAS.
pub fn build_tlas(
    encoder: &mut CommandEncoder,
    tlas: &mut Tlas,
    instances: &Buffer,
    instance_count: u32,
    scratch: &Buffer,
) -> bool {
    let built = first_supported_backend!(|A| {
        build_tlas_impl::<A>(encoder, tlas, instances, instance_count, scratch)
    })
    .is_some();

    if built {
        // Without this the build is invisible to `wgpu-core` and it rejects the TLAS when bound.
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

    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    let hal_tlas = unsafe { tlas.as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
    let hal_instances = unsafe { instances.as_hal::<A>() }?;
    // SAFETY: the handle is only read from, never destroyed, and is dropped with the guard.
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
    // barrier left is the acceleration structure's own — which has no public spelling.
    //
    // SAFETY: every resource in `descriptor` belongs to backend `A` and to this device, the
    // scratch buffer is sized by `tlas_scratch_size` and used by no other build in this
    // submission, and the encoder is neither ended nor its raw handle destroyed here.
    unsafe {
        encoder.as_hal_mut::<A, _, _>(|encoder| {
            let encoder = encoder?;

            encoder.build_acceleration_structures(1, [descriptor]);

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
