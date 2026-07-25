use super::{blas::BlasManager, extract::StandardMaterialAssets, tlas_build, RaytracingMesh3d};
use alloc::sync::Arc;
use bevy_asset::{load_embedded_asset, AssetId};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::{Entity, EntityHashMap, EntityHashSet},
    lifecycle::RemovedComponents,
    query::{Changed, Or, With},
    resource::Resource,
    system::{Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::Image;
use bevy_math::{ops::cos, Affine3, Affine3Ext, Vec3, Vec4};
use bevy_mesh::Mesh;
use bevy_pbr::{
    DfgLut, ExtractedDirectionalLight, MeshMaterial3d, PreviousGlobalTransform, StandardMaterial,
};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    impl_atomic_pod,
    mesh::allocator::MeshAllocator,
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{default, once};
use bytemuck::{Pod, Zeroable};
use core::{f32::consts::TAU, hash::Hash, mem::size_of, num::NonZeroU32, ops::Deref};
use tracing::{info_span, warn};
use wgpu::{BufferTransition, BufferUses};

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

/// Smallest TLAS instance capacity handed out, and the floor [`tlas_capacity_for`] grows from.
const TLAS_MIN_CAPACITY: u32 = 128;

/// Instance capacity to allocate to hold `instance_count` slots.
///
/// Geometric rather than a fixed step, so that the number of reallocations over a scene's lifetime
/// is logarithmic in its size rather than linear. That matters because a reallocation is not just
/// the allocation: [`RaytracingSceneBindings::reserve_tlas`] invalidates the scene bind group, since
/// the TLAS it binds really did change, and rebuilding that walks both mesh slab binding arrays and
/// every bound texture. Growing 128 at a time meant paying that once per 128 instances that ever
/// appeared — around 780 times per parity while a 100k-instance scene streamed in. At 1.5x it is
/// about 18.
///
/// The cost is up to 50% slack in the descriptor buffer and the TLAS, including `wgpu`'s CPU-side
/// instance mirror. A few MB at scene scale, against work that scaled with instance count.
///
/// A pure function of the count rather than a multiple of the previous capacity, so that the
/// descriptor buffer and both TLASes agree on the capacity for a given slot count without having to
/// coordinate — which is what lets [`RaytracingSceneBindings::reserve_tlas_scratch`] size scratch
/// from one of them and have it cover the other.
fn tlas_capacity_for(instance_count: u32) -> u32 {
    let mut capacity = TLAS_MIN_CAPACITY;
    while capacity < instance_count {
        // 1.5x, rounded up so that it always makes progress, and saturating so that a count near
        // `u32::MAX` terminates rather than wrapping.
        capacity = capacity.saturating_add(capacity.div_ceil(2));
    }
    capacity
}

/// Width of a TLAS instance's custom data in both Vulkan (`instanceCustomIndex`) and DXR
/// (`InstanceID`). `tlas_instances.wgsl` packs instance slots into that field, so they have to fit.
const TLAS_CUSTOM_DATA_BITS: u32 = 24;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    _padding: Vec3,
    reflectance: f32,
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        if triangle_count > u16::MAX as u32 {
            panic!("Too many triangles ({triangle_count}) in an emissive mesh, maximum is 65535.");
        }

        Self {
            kind: triangle_count << 1,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

/// A world-from-local affine transform, stored transposed as three rows.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct GpuTransform([Vec4; 3]);

/// A bare `u32` element. Needed because `AtomicPod` can't be implemented for `u32` itself.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
struct GpuU32(u32);

/// The device address of a slot's acceleration structure.
///
/// `tlas_instances.wgsl` copies this straight into the instance descriptor. It reads the buffer as
/// `array<vec2<u32>>` rather than `array<u64>`, since nothing does arithmetic on an address and
/// declaring it 64-bit would cost a `SHADER_INT64` requirement; the two are the same bytes on any
/// little-endian host.
///
/// Zero means the slot has no acceleration structure — it was never handed out, or its instance
/// isn't currently drawable. Both Vulkan and DXR define a zero reference as an inactive instance
/// that the build discards, so holes in the slot allocation cost nothing and need no dummy BLAS.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
struct GpuBlasRef(u64);

impl GpuBlasRef {
    const NONE: Self = Self(0);

    fn new(address: u64) -> Self {
        Self(address)
    }
}

impl_atomic_pod!(GpuInstanceGeometryIds, GpuInstanceGeometryIdsBlob);
impl_atomic_pod!(GpuMaterial, GpuMaterialBlob);
impl_atomic_pod!(GpuLightSource, GpuLightSourceBlob);
impl_atomic_pod!(GpuDirectionalLight, GpuDirectionalLightBlob);
impl_atomic_pod!(GpuTransform, GpuTransformBlob);
impl_atomic_pod!(GpuU32, GpuU32Blob);
impl_atomic_pod!(GpuBlasRef, GpuBlasRefBlob);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Writes `value` at `index`, growing the buffer first if the index is past the end.
///
/// A write that wouldn't change anything is dropped rather than dirtying the element, so data that
/// gets recomputed every frame without actually moving doesn't get re-uploaded every frame.
fn set_at<T: AtomicPod + PartialEq>(buffer: &mut AtomicSparseBufferVec<T>, index: u32, value: T) {
    if buffer.len() > index {
        if buffer.get(index) == value {
            return;
        }
    } else {
        buffer.grow(index + 1);
    }
    buffer.set(index, value);
}

/// Writes `value` at an index the buffer already covers, skipping writes that change nothing.
///
/// The `&self` counterpart to [`set_at`], for paths that run in parallel and so can't grow the
/// buffer. Growing is the caller's job.
fn set_existing<T: AtomicPod + PartialEq>(buffer: &AtomicSparseBufferVec<T>, index: u32, value: T) {
    debug_assert!(
        index < buffer.len(),
        "buffer was not grown past index {index}"
    );
    if buffer.get(index) != value {
        buffer.set(index, value);
    }
}

fn new_storage_buffer<T: AtomicPod>(label: &'static str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, Arc::from(label))
}

/// Drops `entity` from the reverse index under `key`, discarding the entry once it's empty.
///
/// Without the discard these maps only ever grow: meshes and materials that no longer have any
/// instances would leave an empty set behind forever.
fn unlink<K: Eq + Hash>(map: &mut HashMap<K, EntityHashSet>, key: &K, entity: Entity) {
    let now_empty = map.get_mut(key).is_some_and(|instances| {
        instances.remove(&entity);
        instances.is_empty()
    });
    if now_empty {
        map.remove(key);
    }
}

/// Moves `entity` in a reverse index out of `previous`'s entry and into `key`'s.
fn relink<K: Copy + Eq + Hash>(
    map: &mut HashMap<K, EntityHashSet>,
    entity: Entity,
    previous: Option<K>,
    key: K,
) {
    if previous == Some(key) {
        return;
    }
    if let Some(previous) = previous {
        unlink(map, &previous, entity);
    }
    map.entry(key).or_default().insert(entity);
}

// ---------------------------------------------------------------------------
// Instance descriptor packing
// ---------------------------------------------------------------------------

/// Workgroup size of `tlas_instances.wgsl`. Has to match the shader.
const TLAS_INSTANCE_PACK_WORKGROUP_SIZE: u32 = 64;

/// The compute pipeline that turns per-slot transforms and BLAS addresses into TLAS instance
/// descriptors.
#[derive(Resource)]
pub struct TlasInstancePackPipeline {
    pub layout: BindGroupLayoutDescriptor,
    /// `None` on a backend that builds the TLAS through `wgpu-core`, where this shader is never
    /// dispatched and so isn't worth compiling.
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

// ---------------------------------------------------------------------------
// Index and slot allocation
// ---------------------------------------------------------------------------

/// Hands out indices that stay put for as long as they're held.
///
/// Indices given back by removals get handed out again to later callers, so the index space stays
/// about as dense as the live set.
struct IndexAllocator {
    free: Vec<u32>,
    len: u32,
}

impl IndexAllocator {
    fn new() -> Self {
        Self {
            free: Vec::new(),
            len: 0,
        }
    }

    /// One past the highest index ever allocated.
    fn len(&self) -> u32 {
        self.len
    }

    /// How many more indices can be handed out before running past `capacity`.
    fn vacancies(&self, capacity: u32) -> u32 {
        capacity.saturating_sub(self.len) + self.free.len() as u32
    }

    fn allocate(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            let index = self.len;
            self.len += 1;
            index
        })
    }

    fn release(&mut self, index: u32) {
        self.free.push(index);
    }
}

/// Assigns each key an index that stays put for as long as the key is live.
struct SlotAllocator<K> {
    slots: HashMap<K, u32>,
    indices: IndexAllocator,
}

impl<K: Eq + Hash> SlotAllocator<K> {
    fn new() -> Self {
        Self {
            slots: HashMap::default(),
            indices: IndexAllocator::new(),
        }
    }

    fn get(&self, key: &K) -> Option<u32> {
        self.slots.get(key).copied()
    }

    fn contains(&self, key: &K) -> bool {
        self.slots.contains_key(key)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        self.slots.keys()
    }

    /// How many more distinct keys can be taken on before running past `capacity`.
    fn vacancies(&self, capacity: u32) -> u32 {
        self.indices.vacancies(capacity)
    }

    fn get_or_allocate(&mut self, key: K) -> u32 {
        if let Some(&slot) = self.slots.get(&key) {
            return slot;
        }

        let slot = self.indices.allocate();
        self.slots.insert(key, slot);
        slot
    }

    fn remove(&mut self, key: &K) -> Option<u32> {
        let slot = self.slots.remove(key)?;
        self.indices.release(slot);
        Some(slot)
    }
}

/// One occupied slot of a [`RetainedBindingArray`].
struct BindingSlot<T> {
    item: T,
    /// Live references to this slot. The slot is freed when the last one goes away, so an occupied
    /// slot always has at least one — which is what makes this a `NonZeroU32`.
    references: NonZeroU32,
}

/// A binding array whose indices are stable across frames.
///
/// Slots are reference counted by whatever points at them — materials for textures, instances for
/// mesh slab buffers — and are reused once the last reference goes away. `dirty` records whether
/// the contents changed, which is what forces the bind group to be rebuilt.
struct RetainedBindingArray<K, T> {
    allocator: SlotAllocator<K>,
    slots: Vec<Option<BindingSlot<T>>>,
    dirty: bool,
}

impl<K: Eq + Hash, T> RetainedBindingArray<K, T> {
    fn new() -> Self {
        Self {
            allocator: SlotAllocator::new(),
            slots: Vec::new(),
            dirty: false,
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.allocator.contains(key)
    }

    /// How many more distinct keys can be held before running past `capacity`.
    fn vacancies(&self, capacity: u32) -> u32 {
        self.allocator.vacancies(capacity)
    }

    /// Whether [`Self::acquire`] would be able to hand out a reference to `key`.
    ///
    /// Callers that need more than one slot at once check this for all of them before acquiring
    /// any, so that they never have to hand a slot straight back — see [`Self::acquire`].
    fn has_room(&self, key: &K, capacity: u32) -> bool {
        self.contains(key) || self.vacancies(capacity) > 0
    }

    /// The array's contents in slot order, with `None` for slots that are currently free.
    fn iter(&self) -> impl Iterator<Item = Option<&T>> {
        self.slots
            .iter()
            .map(|slot| slot.as_ref().map(|slot| &slot.item))
    }

    /// Takes a reference to `key`'s slot, allocating and filling it if this is the first one.
    ///
    /// Returns `None` if `key` would need a new slot and every slot below `capacity` is taken.
    /// The bind group layout declares these arrays with a fixed length, so running past it makes
    /// `create_bind_group` fail outright — callers have to drop whatever wanted the slot instead.
    fn acquire(&mut self, key: K, capacity: u32, item: impl FnOnce() -> T) -> Option<u32> {
        if !self.has_room(&key, capacity) {
            return None;
        }

        let slot = self.allocator.get_or_allocate(key);
        let index = slot as usize;

        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || None);
        }

        if let Some(occupied) = self.slots[index].as_mut() {
            occupied.references = occupied.references.saturating_add(1);
        } else {
            self.slots[index] = Some(BindingSlot {
                item: item(),
                references: NonZeroU32::MIN,
            });
            self.dirty = true;
        }

        Some(slot)
    }

    /// Drops a reference to `key`'s slot, freeing it if that was the last one.
    fn release(&mut self, key: &K) {
        let Some(slot) = self.allocator.get(key) else {
            return;
        };
        let index = slot as usize;
        let Some(occupied) = self.slots[index].as_mut() else {
            return;
        };

        if let Some(remaining) = NonZeroU32::new(occupied.references.get() - 1) {
            occupied.references = remaining;
            return;
        }

        self.slots[index] = None;
        self.allocator.remove(key);
        self.dirty = true;
    }

    /// Repoints an already-allocated slot at a new value, leaving its index and refcount alone.
    fn replace(&mut self, key: &K, item: T) {
        if let Some(slot) = self.allocator.get(key)
            && let Some(occupied) = self.slots[slot as usize].as_mut()
        {
            occupied.item = item;
            self.dirty = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse buffers
// ---------------------------------------------------------------------------

/// The number of [`AtomicSparseBufferVec`]s the scene bindings own.
const SPARSE_BUFFER_COUNT: usize = 9;

/// Lets the scene's sparse buffers, which all have different element types, be driven as a group.
///
/// Every one of them is grown, written, uploaded and identity-checked the same way, so they're only
/// ever enumerated once, in [`RaytracingSceneBindings::sparse_buffers`].
trait SceneBuffer {
    /// Grows the buffer to at least `len` elements.
    fn grow_to(&mut self, len: u32);

    /// The id of the backing GPU buffer, if one has been allocated.
    fn buffer_id(&self) -> Option<BufferId>;

    /// Uploads whatever changed, reallocating the GPU buffer first if the data outgrew it.
    fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue);

    /// Finalizes a scheduled sparse upload.
    fn prepare_upload(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        jobs: &mut SparseBufferUpdateJobs,
        bind_groups: &mut SparseBufferUpdateBindGroups,
        pipelines: &SparseBufferUpdatePipelines,
    );
}

impl<T: AtomicPod> SceneBuffer for AtomicSparseBufferVec<T> {
    fn grow_to(&mut self, len: u32) {
        self.grow(len);
    }

    fn buffer_id(&self) -> Option<BufferId> {
        Some(self.buffer()?.id())
    }

    fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.write_buffers(render_device, render_queue);
    }

    fn prepare_upload(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        jobs: &mut SparseBufferUpdateJobs,
        bind_groups: &mut SparseBufferUpdateBindGroups,
        pipelines: &SparseBufferUpdatePipelines,
    ) {
        self.prepare_to_populate_buffers(
            render_device,
            pipeline_cache,
            jobs,
            bind_groups,
            pipelines,
        );
    }
}

// ---------------------------------------------------------------------------
// The scene bindings
// ---------------------------------------------------------------------------

/// The four textures a [`StandardMaterial`] can reference, in the order they appear in
/// [`GpuMaterial`]. `None` means the material doesn't use that texture.
type MaterialTextures = [Option<AssetId<Image>>; 4];

/// Everything the scene tracks per raytracing instance.
///
/// One record rather than a map per field, so that a refresh is a single lookup and there's no way
/// to update some of an instance's bookkeeping and forget the rest.
#[derive(Clone, Copy)]
struct Instance {
    /// Index into every per-instance buffer, and the TLAS custom data the shaders resolve hits
    /// with. Stable for as long as the instance exists.
    slot: u32,
    mesh: AssetId<Mesh>,
    material: AssetId<StandardMaterial>,
    /// Mesh slab buffers this instance is holding binding array references to.
    buffers: Option<(BufferId, BufferId)>,
}

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,

    // Retained binding arrays.
    vertex_buffers: RetainedBindingArray<BufferId, Buffer>,
    index_buffers: RetainedBindingArray<BufferId, Buffer>,
    textures: RetainedBindingArray<AssetId<Image>, (TextureView, Sampler)>,

    // Retained GPU buffers, updated in place. Enumerated in `sparse_buffers`.
    materials: AtomicSparseBufferVec<GpuMaterial>,
    transforms: AtomicSparseBufferVec<GpuTransform>,
    previous_frame_transforms: AtomicSparseBufferVec<GpuTransform>,
    geometry_ids: AtomicSparseBufferVec<GpuInstanceGeometryIds>,
    material_ids: AtomicSparseBufferVec<GpuU32>,
    /// Per-slot BLAS device address, and so also the record of which slots are drawable.
    ///
    /// Read by `tlas_instances.wgsl` rather than by the CPU. Unlike the transforms this only
    /// changes when an instance resolves or its mesh's acceleration structure is replaced, so it
    /// is nearly always empty of updates.
    blas_refs: AtomicSparseBufferVec<GpuBlasRef>,
    light_sources: AtomicSparseBufferVec<GpuLightSource>,
    directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    previous_frame_light_id_translations: AtomicSparseBufferVec<GpuU32>,

    // Material bookkeeping. A material only gets a slot once all of its textures have been
    // uploaded; until then it stays unresolved and its instances stay out of the scene.
    material_slots: SlotAllocator<AssetId<StandardMaterial>>,
    material_textures: HashMap<AssetId<StandardMaterial>, MaterialTextures>,
    emissive_materials: HashSet<AssetId<StandardMaterial>>,
    /// Materials to retry next frame, because a texture they need hadn't been uploaded yet.
    ///
    /// This is polled rather than woken by image asset events: `prepare_assets` can defer an
    /// upload to a later frame when the bytes-per-frame budget runs out, and on the frame the
    /// image finally lands it is no longer reported as added or modified. An event-driven wakeup
    /// would miss it and strand the material — and every instance using it — permanently.
    unresolved_materials: HashSet<AssetId<StandardMaterial>>,
    /// Images that were reported as changed but whose new GPU data hadn't landed yet.
    ///
    /// Polled for the same reason as [`Self::unresolved_materials`], and needed for the same
    /// reason: `prepare_assets` drops the old [`GpuImage`] before it uploads the replacement, so
    /// during a deferral there is nothing to swap in, and by the time there is, the image is no
    /// longer reported as changed. Without the retry, the binding array would keep serving the old
    /// texture view forever. Bounded by the textures actually bound, since ids we hold no slot for
    /// are dropped rather than retried.
    pending_texture_updates: HashSet<AssetId<Image>>,

    // Instance bookkeeping.
    instance_slots: IndexAllocator,
    instances: EntityHashMap<Instance>,
    /// How many slots currently have an acceleration structure, tracked alongside [`Self::blas_refs`]
    /// so that "is there anything to trace" doesn't need a scan.
    live_instance_count: u32,
    /// Instances to re-resolve next frame, because something they depend on wasn't ready.
    ///
    /// An instance whose mesh or material never arrives stays in here and is retried every frame.
    /// That's a handful of hash lookups each, bounded by the number of instances waiting on an
    /// asset, so it isn't worth waking on an event instead — especially as the events that would
    /// do the waking (`BlasManager::changed`, material changes, slab growth) already queue their
    /// own refreshes, which would leave this as a redundant second path.
    pending_refresh: EntityHashSet,
    mesh_instances: HashMap<AssetId<Mesh>, EntityHashSet>,
    material_instances: HashMap<AssetId<StandardMaterial>, EntityHashSet>,

    // Light bookkeeping. The shaders sample `light_sources` using `arrayLength`, so it has to
    // stay dense; removals swap the last light down into the hole.
    light_index: EntityHashMap<u32>,
    light_entities: Vec<Entity>,
    previous_light_index: EntityHashMap<u32>,
    light_index_changed: EntityHashSet,
    /// Translation entries written last frame, so they can be reset to identity.
    nonidentity_translations: Vec<u32>,
    directional_light_slots: SlotAllocator<Entity>,

    // Two TLASes, so one can be bound as the previous frame's while the other is rebuilt. Only one
    // instance buffer is needed for the pair: a built acceleration structure is self contained, so
    // last frame's keeps describing last frame's scene no matter what the buffer says now.
    tlas: [Option<Tlas>; 2],
    tlas_capacity: [u32; 2],
    /// Whether each parity's TLAS has had a build recorded into it since it was allocated.
    ///
    /// `wgpu` rejects a TLAS that is bound without having been built, and the previous-frame entry
    /// is bound for a second frame after it stops being current — so a frame that allocated one
    /// but bailed before building would surface as a submit error a frame later. Binding is gated
    /// on this instead.
    tlas_built: [bool; 2],
    /// Acceleration structures each TLAS points into, held for as long as that TLAS might still be
    /// traced against.
    ///
    /// Building through [`tlas_build`] means `wgpu-core` never learns which BLASes a TLAS depends
    /// on — `mark_acceleration_structures_built` records an empty dependency list — so it won't
    /// keep them alive. A built TLAS holds pointers into that memory, and the off-parity one is
    /// still bound as the previous frame, so both parities' sets have to be retained.
    ///
    /// One entry per distinct mesh rather than per instance.
    tlas_blas_handles: [Vec<Blas>; 2],
    /// [`BlasManager::generation`] each parity's [`Self::tlas_blas_handles`] was built from, so that
    /// the refresh is skipped on the frames the acceleration structures didn't change — which is
    /// nearly all of them. Zero means never built.
    tlas_blas_generation: [u64; 2],
    frame_parity: usize,

    /// Packed TLAS instance descriptors, filled by `tlas_instances.wgsl`.
    ///
    /// GPU only: every field is either already on the GPU (the transform, the BLAS address) or
    /// implied by the slot index, so a CPU mirror would be pure duplication.
    instance_descriptors: Option<Buffer>,
    instance_descriptor_capacity: u32,
    /// Scratch space for the TLAS build, grown to fit the largest build so far.
    tlas_scratch: Option<Buffer>,
    tlas_scratch_capacity: u64,
    /// Descriptor capacity [`Self::tlas_scratch_capacity`] was queried for.
    ///
    /// The query is a driver call, and the answer only depends on the instance count, so this is
    /// what keeps it from being asked again every frame.
    tlas_scratch_sized_for: u32,
    /// Whether the pack pass recorded this frame, and with it the transitions the build needs.
    instances_packed: bool,
    /// Acceleration structures no longer referenced by either TLAS, awaiting release.
    ///
    /// A TLAS built through [`tlas_build`] never registers its BLAS dependencies, so `wgpu-core`
    /// won't hold these for us and dropping one destroys it immediately — while a submission that
    /// traces the TLAS pointing into it may still be running. `retire_raytracing_resources` hands
    /// them to `Queue::on_submitted_work_done` once this frame is submitted, which drops them at
    /// exactly the right moment rather than after a guessed number of frames.
    pending_retire: Vec<Blas>,
    /// Bind group for `tlas_instances.wgsl`, rebuilt whenever one of its three buffers moves.
    instance_pack_bind_group: Option<BindGroup>,
    /// The buffers [`Self::instance_pack_bind_group`] was built against.
    instance_pack_buffer_ids: Option<[BufferId; 3]>,

    // One cached bind group per TLAS parity, since only the two acceleration structure entries
    // differ between them.
    cached_bind_groups: [Option<BindGroup>; 2],
    bind_group_invalid: bool,
    last_buffer_ids: [Option<BufferId>; SPARSE_BUFFER_COUNT],
    last_light_count: u32,
    /// The DFG LUT falls back to a placeholder until it finishes uploading, so the bind group has
    /// to be rebuilt once the real one shows up.
    last_dfg_ids: Option<(TextureViewId, SamplerId)>,
    /// Bound into binding array slots that are currently free.
    dummy_buffer: Buffer,
}

impl FromWorld for RaytracingSceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Binding arrays are dense slices, so freed slots still need something valid bound into
        // them. A few elements' worth of zeroes covers the shader's runtime-sized arrays.
        let dummy_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_dummy_binding_array_buffer"),
            size: 256,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            bind_group: None,
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT),
                        storage_buffer_read_only_sized(false, None),
                        acceleration_structure(),
                        acceleration_structure(),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                    ),
                ),
            ),

            vertex_buffers: RetainedBindingArray::new(),
            index_buffers: RetainedBindingArray::new(),
            textures: RetainedBindingArray::new(),

            materials: new_storage_buffer("solari_materials"),
            transforms: new_storage_buffer("solari_transforms"),
            previous_frame_transforms: new_storage_buffer("solari_previous_frame_transforms"),
            geometry_ids: new_storage_buffer("solari_geometry_ids"),
            material_ids: new_storage_buffer("solari_material_ids"),
            blas_refs: new_storage_buffer("solari_blas_refs"),
            light_sources: new_storage_buffer("solari_light_sources"),
            directional_lights: new_storage_buffer("solari_directional_lights"),
            previous_frame_light_id_translations: new_storage_buffer(
                "solari_previous_frame_light_id_translations",
            ),

            material_slots: SlotAllocator::new(),
            material_textures: HashMap::default(),
            emissive_materials: HashSet::default(),
            unresolved_materials: HashSet::default(),
            pending_texture_updates: HashSet::default(),

            instance_slots: IndexAllocator::new(),
            instances: EntityHashMap::default(),
            live_instance_count: 0,
            pending_refresh: EntityHashSet::default(),
            mesh_instances: HashMap::default(),
            material_instances: HashMap::default(),

            light_index: EntityHashMap::default(),
            light_entities: Vec::new(),
            previous_light_index: EntityHashMap::default(),
            light_index_changed: EntityHashSet::default(),
            nonidentity_translations: Vec::new(),
            directional_light_slots: SlotAllocator::new(),

            tlas: [None, None],
            tlas_capacity: [0, 0],
            tlas_built: [false, false],
            tlas_blas_handles: [Vec::new(), Vec::new()],
            tlas_blas_generation: [0, 0],
            frame_parity: 0,

            instance_descriptors: None,
            instance_descriptor_capacity: 0,
            tlas_scratch: None,
            tlas_scratch_capacity: 0,
            tlas_scratch_sized_for: 0,
            instances_packed: false,
            pending_retire: Vec::new(),
            instance_pack_bind_group: None,
            instance_pack_buffer_ids: None,

            cached_bind_groups: [None, None],
            bind_group_invalid: true,
            last_buffer_ids: [None; SPARSE_BUFFER_COUNT],
            last_light_count: 0,
            last_dfg_ids: None,
            dummy_buffer,
        }
    }
}

impl RaytracingSceneBindings {
    fn sparse_buffers(&self) -> [&dyn SceneBuffer; SPARSE_BUFFER_COUNT] {
        [
            &self.materials,
            &self.transforms,
            &self.previous_frame_transforms,
            &self.geometry_ids,
            &self.material_ids,
            &self.blas_refs,
            &self.light_sources,
            &self.directional_lights,
            &self.previous_frame_light_id_translations,
        ]
    }

    fn sparse_buffers_mut(&mut self) -> [&mut dyn SceneBuffer; SPARSE_BUFFER_COUNT] {
        [
            &mut self.materials,
            &mut self.transforms,
            &mut self.previous_frame_transforms,
            &mut self.geometry_ids,
            &mut self.material_ids,
            &mut self.blas_refs,
            &mut self.light_sources,
            &mut self.directional_lights,
            &mut self.previous_frame_light_id_translations,
        ]
    }

    fn write_sparse_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        let _span = info_span!("write_buffers").entered();

        for buffer in self.sparse_buffers_mut() {
            // Every buffer needs at least one element to be bindable, even if the scene doesn't
            // use it.
            buffer.grow_to(1);
            buffer.write(render_device, render_queue);
        }
    }
}

// ---------------------------------------------------------------------------
// Material updates
// ---------------------------------------------------------------------------

impl RaytracingSceneBindings {
    fn update_materials(
        &mut self,
        material_assets: &StandardMaterialAssets,
        texture_assets: &RenderAssets<GpuImage>,
    ) {
        let _span = info_span!("update_materials").entered();

        for material_id in &material_assets.removed {
            self.remove_material(*material_id);
        }
        // Deliberately unordered with respect to the removals above: `update_material` re-checks
        // whether the material is still in `material_assets`, so it corrects itself whichever
        // order the two events arrived in.
        for material_id in &material_assets.changed {
            self.update_material(*material_id, material_assets, texture_assets);
        }
    }

    /// Resolves a material's textures and writes it into its slot.
    ///
    /// If any of its textures haven't been uploaded yet the material stays unresolved: it holds no
    /// slot, and instances using it stay out of the scene until the texture arrives.
    fn update_material(
        &mut self,
        material_id: AssetId<StandardMaterial>,
        material_assets: &StandardMaterialAssets,
        texture_assets: &RenderAssets<GpuImage>,
    ) {
        let Some(material) = material_assets.get(&material_id) else {
            self.remove_material(material_id);
            return;
        };

        let was_resolved = self.material_slots.contains(&material_id);

        let handles = [
            &material.normal_map_texture,
            &material.base_color_texture,
            &material.emissive_texture,
            &material.metallic_roughness_texture,
        ];

        // Resolve everything up front, so that a missing texture leaves our state untouched.
        let mut textures: MaterialTextures = [None; 4];
        for (slot, handle) in textures.iter_mut().zip(handles) {
            let Some(handle) = handle else { continue };
            let image_id = handle.id();

            if texture_assets.get(image_id).is_none() {
                self.defer_material(material_id);
                return;
            }

            *slot = Some(image_id);
        }

        // Check for room before taking any slots. Acquiring some and then handing them straight
        // back would dirty the binding array — and so rebuild the whole bind group — on every
        // frame for as long as the scene stayed over the limit.
        if self.new_texture_count(&textures) > self.textures.vacancies(MAX_TEXTURE_COUNT.get()) {
            // There isn't room while this material is still holding the slots it took last time.
            // Some of those are about to be given up anyway, so drop them and look again —
            // otherwise a material could never swap one texture for another once the array filled
            // up, and would stay deferred forever. This gives up the slots shared with the new set
            // too, so it costs a bind group rebuild, but it only happens at the limit.
            self.release_material_textures(material_id);
            self.material_textures.remove(&material_id);

            if self.new_texture_count(&textures) > self.textures.vacancies(MAX_TEXTURE_COUNT.get())
            {
                once!(warn!(
                    "Solari scene needs more than {} textures. Materials past that limit will not \
                     be rendered.",
                    MAX_TEXTURE_COUNT.get()
                ));
                self.defer_material(material_id);
                return;
            }
        }

        // Acquire before releasing, so that textures shared between the old and new state aren't
        // freed and immediately reallocated. The over-limit path above is the one exception: it has
        // to release first to have any hope of fitting, and has already done so.
        let mut texture_ids = [TEXTURE_MAP_NONE; 4];
        for (texture_id, image_id) in texture_ids.iter_mut().zip(textures) {
            let Some(image_id) = image_id else { continue };
            let image = texture_assets.get(image_id).unwrap();
            // Can't fail: the check above accounted for every slot this material still needs. If
            // it somehow did, the material draws without that texture rather than not at all.
            if let Some(slot) = self
                .textures
                .acquire(image_id, MAX_TEXTURE_COUNT.get(), || {
                    (image.texture_view.clone(), image.sampler.clone())
                })
            {
                *texture_id = slot;
            }
        }

        self.release_material_textures(material_id);
        self.material_textures.insert(material_id, textures);
        self.unresolved_materials.remove(&material_id);

        let slot = self.material_slots.get_or_allocate(material_id);
        let emissive = material.emissive.to_vec3();
        let is_emissive = emissive != Vec3::ZERO;

        set_at(
            &mut self.materials,
            slot,
            GpuMaterial {
                normal_map_texture_id: texture_ids[0],
                base_color_texture_id: texture_ids[1],
                emissive_texture_id: texture_ids[2],
                metallic_roughness_texture_id: texture_ids[3],

                base_color: LinearRgba::from(material.base_color).to_vec3(),
                perceptual_roughness: material.perceptual_roughness,
                emissive,
                metallic: material.metallic,
                reflectance: material.reflectance,
                _padding: Vec3::ZERO,
            },
        );

        // A material's slot never moves, and its instances don't carry any of its data, so the
        // only material edits that reach instances are the ones that change whether the material
        // resolves at all or whether it emits light.
        let was_emissive = if is_emissive {
            !self.emissive_materials.insert(material_id)
        } else {
            self.emissive_materials.remove(&material_id)
        };
        if !was_resolved || was_emissive != is_emissive {
            self.invalidate_material_instances(material_id);
        }
    }

    /// How many slots the texture binding array would have to hand out to cover `textures`.
    ///
    /// Duplicates within `textures` and images already in the array cost nothing, since they share
    /// a slot with what's already there.
    fn new_texture_count(&self, textures: &MaterialTextures) -> u32 {
        let mut count = 0;
        for (index, image_id) in textures.iter().enumerate() {
            let Some(image_id) = image_id else { continue };
            let counted_already = textures[..index].contains(&Some(*image_id));
            if !counted_already && !self.textures.contains(image_id) {
                count += 1;
            }
        }
        count
    }

    /// Drops the material's GPU state and queues it to be retried on a later frame.
    fn defer_material(&mut self, material_id: AssetId<StandardMaterial>) {
        // `remove_material` clears the retry entry, so insert after it rather than before.
        self.remove_material(material_id);
        self.unresolved_materials.insert(material_id);
    }

    fn remove_material(&mut self, material_id: AssetId<StandardMaterial>) {
        // Unconditional: a material can be waiting on a texture without ever having held a slot,
        // and it still has to stop being retried once it's gone.
        self.unresolved_materials.remove(&material_id);

        if self.material_slots.remove(&material_id).is_none() {
            return;
        }

        self.release_material_textures(material_id);
        self.material_textures.remove(&material_id);
        self.emissive_materials.remove(&material_id);

        self.invalidate_material_instances(material_id);
    }

    fn release_material_textures(&mut self, material_id: AssetId<StandardMaterial>) {
        let Some(textures) = self.material_textures.get(&material_id).copied() else {
            return;
        };
        for image_id in textures.into_iter().flatten() {
            self.textures.release(&image_id);
        }
    }

    /// Queues every instance using `material_id` to be re-resolved.
    fn invalidate_material_instances(&mut self, material_id: AssetId<StandardMaterial>) {
        let Self {
            material_instances,
            pending_refresh,
            ..
        } = self;

        if let Some(instances) = material_instances.get(&material_id) {
            pending_refresh.extend(instances.iter().copied());
        }
    }

    /// Swaps in the GPU data of images that finished uploading, and retries whatever was waiting on
    /// one.
    fn update_textures(
        &mut self,
        extracted_images: &ExtractedAssets<GpuImage>,
        texture_assets: &RenderAssets<GpuImage>,
        material_assets: &StandardMaterialAssets,
    ) {
        let _span = info_span!("update_textures").entered();

        // `extract_render_asset` records every id it extracts in `added`, modifications included,
        // so that set alone covers everything whose GPU data could have changed.
        let mut pending = core::mem::take(&mut self.pending_texture_updates);
        pending.extend(extracted_images.added.iter().copied());

        for image_id in pending {
            // Nothing bound for this image, so there's nothing to swap. A material that needs it
            // acquires it fresh once it lands, and skipping it here is also what keeps this set
            // from growing without bound when an image is dropped mid-retry.
            if !self.textures.contains(&image_id) {
                continue;
            }

            match texture_assets.get(image_id) {
                Some(image) => self.textures.replace(
                    &image_id,
                    (image.texture_view.clone(), image.sampler.clone()),
                ),
                None => {
                    self.pending_texture_updates.insert(image_id);
                }
            }
        }

        // Retry materials that were waiting on a texture. Normally empty; during load it drains as
        // images arrive.
        for material_id in core::mem::take(&mut self.unresolved_materials) {
            self.update_material(material_id, material_assets, texture_assets);
        }
    }
}

// ---------------------------------------------------------------------------
// Light source bookkeeping
// ---------------------------------------------------------------------------

impl RaytracingSceneBindings {
    fn update_lights(&mut self, directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>) {
        // There are few enough directional lights to just walk them every frame.
        let _span = info_span!("update_lights").entered();

        let mut live_directional_lights = EntityHashSet::default();
        for (entity, directional_light) in directional_lights {
            live_directional_lights.insert(entity);

            let slot = self.directional_light_slots.get_or_allocate(entity);
            set_at(
                &mut self.directional_lights,
                slot,
                GpuDirectionalLight::new(directional_light),
            );
            self.add_light(entity, GpuLightSource::new_directional_light(slot));
        }

        let stale: Vec<Entity> = self
            .directional_light_slots
            .keys()
            .copied()
            .filter(|entity| !live_directional_lights.contains(entity))
            .collect();
        for entity in stale {
            self.directional_light_slots.remove(&entity);
            self.remove_light(entity);
        }

        self.write_light_id_translations();

        if self.light_entities.len() > u16::MAX as usize {
            panic!("Too many light sources in the scene, maximum is 65535.");
        }
    }

    fn add_light(&mut self, entity: Entity, source: GpuLightSource) {
        let index = match self.light_index.get(&entity) {
            Some(&index) => index,
            None => {
                let index = self.light_entities.len() as u32;
                self.light_entities.push(entity);
                self.light_index.insert(entity, index);
                self.light_index_changed.insert(entity);
                index
            }
        };
        set_at(&mut self.light_sources, index, source);
    }

    /// Removes a light, moving the last one down into the hole to keep the array dense.
    fn remove_light(&mut self, entity: Entity) {
        let Some(index) = self.light_index.remove(&entity) else {
            return;
        };
        self.light_index_changed.insert(entity);

        let last = self.light_entities.len() as u32 - 1;
        self.light_entities.swap_remove(index as usize);

        if index != last {
            let moved = self.light_entities[index as usize];
            self.light_index.insert(moved, index);
            self.light_index_changed.insert(moved);

            let source = self.light_sources.get(last);
            set_at(&mut self.light_sources, index, source);
        }
    }

    /// Restores the translation entries written last frame back to identity.
    fn reset_light_id_translations(&mut self) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            set_at(
                &mut self.previous_frame_light_id_translations,
                index,
                GpuU32(index),
            );
        }
    }

    /// Records where each light that moved or disappeared this frame ended up, so that reservoirs
    /// still carrying last frame's light ids can be remapped.
    fn write_light_id_translations(&mut self) {
        let changed: Vec<Entity> = self.light_index_changed.drain().collect();

        for entity in &changed {
            // Lights that first appeared this frame have no previous id to translate from.
            let Some(&previous) = self.previous_light_index.get(entity) else {
                continue;
            };
            let current = self
                .light_index
                .get(entity)
                .copied()
                .unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);

            if current != previous {
                set_at(
                    &mut self.previous_frame_light_id_translations,
                    previous,
                    GpuU32(current),
                );
                self.nonidentity_translations.push(previous);
            }
        }

        for entity in changed {
            match self.light_index.get(&entity) {
                Some(&index) => self.previous_light_index.insert(entity, index),
                None => self.previous_light_index.remove(&entity),
            };
        }

        // Every index the shader might read has to be backed by a real element.
        let light_count = self.light_entities.len() as u32;
        let translations = &mut self.previous_frame_light_id_translations;
        if translations.len() < light_count {
            let start = translations.len();
            translations.grow(light_count);
            for index in start..light_count {
                translations.set(index, GpuU32(index));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Instance updates
// ---------------------------------------------------------------------------

type InstanceQueryData<'w> = (
    &'w RaytracingMesh3d,
    &'w MeshMaterial3d<StandardMaterial>,
    &'w GlobalTransform,
    &'w PreviousGlobalTransform,
);

/// Instances whose mesh or material was swapped out.
type ChangedInstanceFilter = (
    With<RaytracingMesh3d>,
    Or<(
        Changed<RaytracingMesh3d>,
        Changed<MeshMaterial3d<StandardMaterial>>,
    )>,
);

impl RaytracingSceneBindings {
    /// Grows everything indexed by instance slot to cover `slot`.
    ///
    /// Called for every slot as it's handed out, so that afterwards the per-instance paths only
    /// ever write into elements that already exist. That's what lets them take `&self` and run in
    /// parallel — growing a buffer needs `&mut`, and a slot can first be touched from the parallel
    /// pass, so the growth can't wait until then.
    fn reserve_slot(&mut self, slot: u32) {
        let len = slot + 1;

        self.transforms.grow(len);
        self.previous_frame_transforms.grow(len);
        self.blas_refs.grow(len);
    }

    fn remove_instances(&mut self, removed: impl IntoIterator<Item = Entity>) {
        let _span = info_span!("remove_instances").entered();

        for entity in removed {
            self.remove_instance(entity);
        }
    }

    fn refresh_instances(
        &mut self,
        instances: &Query<InstanceQueryData>,
        changed_instances: &Query<Entity, ChangedInstanceFilter>,
        blas_manager: &BlasManager,
        mesh_allocator: &MeshAllocator,
    ) {
        let _span = info_span!("refresh_instances").entered();

        let mut refresh = core::mem::take(&mut self.pending_refresh);
        refresh.extend(changed_instances.iter());

        // A mesh invalidates every instance using it in two cases: its BLAS was rebuilt, and its
        // data moved to a different slab buffer. The second is reported separately because growing
        // a slab replaces the buffer for every mesh resident in it, so a mesh can be invalidated
        // without changing in any way of its own — nothing else names those meshes. (`Res` change
        // detection is no use for either: `allocate_and_free_meshes` takes `ResMut<MeshAllocator>`
        // unconditionally, so it would report a change every single frame.)
        let moved_meshes = mesh_allocator.meshes_with_changed_buffers();
        for mesh_id in blas_manager.changed().iter().copied().chain(moved_meshes) {
            if let Some(mesh_instances) = self.mesh_instances.get(&mesh_id) {
                refresh.extend(mesh_instances.iter().copied());
            }
        }

        for entity in refresh {
            let Ok((mesh, material, transform, previous_frame_transform)) = instances.get(entity)
            else {
                // The entity is gone, or is no longer a complete instance. Either way it can't go
                // on holding a slot, a TLAS entry and binding array references.
                self.remove_instance(entity);
                continue;
            };
            self.refresh_instance(
                entity,
                mesh,
                material,
                transform,
                previous_frame_transform,
                blas_manager,
                mesh_allocator,
            );
        }
    }

    /// Re-resolves an instance and rewrites all of its per-instance data.
    ///
    /// If its mesh, BLAS or material isn't ready, the instance is dropped from the TLAS and queued
    /// for another attempt next frame.
    fn refresh_instance(
        &mut self,
        entity: Entity,
        mesh: &RaytracingMesh3d,
        material: &MeshMaterial3d<StandardMaterial>,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
        blas_manager: &BlasManager,
        mesh_allocator: &MeshAllocator,
    ) {
        let mesh_id = mesh.id();
        let material_id = material.id();

        let previous = self.instances.get(&entity).copied();

        // Keep the reverse indices current, so mesh and material changes can find their instances
        // without scanning the whole scene.
        relink(
            &mut self.mesh_instances,
            entity,
            previous.map(|instance| instance.mesh),
            mesh_id,
        );
        relink(
            &mut self.material_instances,
            entity,
            previous.map(|instance| instance.material),
            material_id,
        );

        let slot = match previous {
            Some(previous) => previous.slot,
            None => self.instance_slots.allocate(),
        };
        // Unconditional, rather than only for freshly allocated slots: an instance that fails to
        // resolve still owns its slot, and the parallel move pass can reach it.
        self.reserve_slot(slot);

        // Seeded when the slot is handed out rather than when the instance first resolves. A slot
        // waiting on a mesh or a material still owns its transform, and the extract schedule only
        // writes one when `GlobalTransform` changes — so an instance that resolves on a later
        // frame and then never moves would otherwise trace against the zeroes `reserve_slot` grew
        // into.
        //
        // Only for a new slot: after this the extract schedule owns the transforms, and the render
        // world components read here are a snapshot from when the instance was spawned. Writing
        // them back on a later refresh (a mesh finally loading, say) would undo every move the
        // instance has made since.
        if previous.is_none() {
            self.write_transforms(slot, transform, previous_frame_transform);
        }

        let mut instance = Instance {
            slot,
            mesh: mesh_id,
            material: material_id,
            buffers: previous.and_then(|instance| instance.buffers),
        };

        let resolved = self.resolve_instance(entity, &mut instance, blas_manager, mesh_allocator);

        // Written back on both paths, so that a failed attempt keeps its slot and its reverse
        // index links and can simply be retried.
        self.instances.insert(entity, instance);
        if !resolved {
            self.pending_refresh.insert(entity);
        }
    }

    /// Resolves everything an instance's slot needs and writes it out, reporting whether the
    /// instance ended up drawable.
    fn resolve_instance(
        &mut self,
        entity: Entity,
        instance: &mut Instance,
        blas_manager: &BlasManager,
        mesh_allocator: &MeshAllocator,
    ) -> bool {
        let slot = instance.slot;

        let (Some(vertex_slice), Some(index_slice), Some(material_slot)) = (
            mesh_allocator.mesh_vertex_slice(&instance.mesh),
            mesh_allocator.mesh_index_slice(&instance.mesh),
            self.material_slots.get(&instance.material),
        ) else {
            self.deactivate_instance(entity, instance);
            return false;
        };
        let Some(blas_address) = blas_manager.address(&instance.mesh) else {
            self.deactivate_instance(entity, instance);
            return false;
        };

        let vertex_buffer_key = vertex_slice.buffer.id();
        let index_buffer_key = index_slice.buffer.id();
        let capacity = MAX_MESH_SLAB_COUNT.get();

        // Check both arrays for room before taking either slot. Acquiring one and then handing it
        // straight back would dirty the binding array — and so rebuild the whole bind group — every
        // frame, since an instance that fails here is queued for another attempt next frame.
        if !self.vertex_buffers.has_room(&vertex_buffer_key, capacity)
            || !self.index_buffers.has_room(&index_buffer_key, capacity)
        {
            once!(warn!(
                "Solari scene needs more than {} mesh slabs. Instances past that limit will \
                 not be rendered.",
                MAX_MESH_SLAB_COUNT.get()
            ));
            self.deactivate_instance(entity, instance);
            return false;
        }

        // Take the new slab references before dropping the old ones, so that a mesh which didn't
        // change slabs doesn't have its slot freed and immediately reallocated — that would dirty
        // the binding array and rebuild the whole bind group for nothing.
        //
        // Neither acquire can fail: the check above confirmed both arrays have room.
        let previous_buffers = instance.buffers.take();
        let vertex_buffer_id = self
            .vertex_buffers
            .acquire(vertex_buffer_key, capacity, || vertex_slice.buffer.clone())
            .expect("vertex slab binding array had room but handed out no slot");
        let index_buffer_id = self
            .index_buffers
            .acquire(index_buffer_key, capacity, || index_slice.buffer.clone())
            .expect("index slab binding array had room but handed out no slot");
        instance.buffers = Some((vertex_buffer_key, index_buffer_key));
        self.release_buffers(previous_buffers);

        let triangle_count = (index_slice.range.len() / 3) as u32;

        set_at(
            &mut self.geometry_ids,
            slot,
            GpuInstanceGeometryIds {
                vertex_buffer_id,
                vertex_buffer_offset: vertex_slice.range.start,
                index_buffer_id,
                index_buffer_offset: index_slice.range.start,
                triangle_count,
            },
        );
        set_at(&mut self.material_ids, slot, GpuU32(material_slot));

        // The transforms are `refresh_instance`'s job: they belong to the slot rather than to the
        // instance being drawable, and are seeded when the slot is handed out.
        self.set_live(slot, blas_address);

        if self.emissive_materials.contains(&instance.material) {
            self.add_light(
                entity,
                GpuLightSource::new_emissive_mesh_light(slot, triangle_count),
            );
        } else {
            self.remove_light(entity);
        }

        true
    }

    /// Records that an instance moved, writing its transforms into the per-slot buffers.
    ///
    /// Driven straight from the extract schedule rather than by way of render world components:
    /// a moving instance costs nothing else per frame, so copying its transform into the render
    /// world first only to read it back out in the same frame is pure overhead.
    ///
    /// Every write lands in an element that already exists, on a slot no other instance shares,
    /// so this takes `&self` and the caller can run it in parallel.
    pub fn move_instance(
        &self,
        entity: Entity,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        let Some(slot) = self.instances.get(&entity).map(|instance| instance.slot) else {
            // Spawned this frame and not given a slot yet. `refresh_instance` seeds the buffers
            // when it allocates one, later this same frame.
            return;
        };

        // Nothing else to do: `tlas_instances.wgsl` reads the transform straight out of the buffer
        // when it packs this slot's descriptor, so there is no TLAS-side copy to invalidate.
        self.write_transforms(slot, transform, previous_frame_transform);
    }

    /// Writes an instance's transforms into the per-slot buffers.
    ///
    /// Takes `&self` so it can be called from a parallel pass. `reserve_slot` has to
    /// have grown the buffers past `slot` first.
    fn write_transforms(
        &self,
        slot: u32,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        set_existing(
            &self.transforms,
            slot,
            GpuTransform(Affine3::from(transform.affine()).to_transpose()),
        );
        set_existing(
            &self.previous_frame_transforms,
            slot,
            GpuTransform(Affine3::from(previous_frame_transform.0).to_transpose()),
        );
    }

    /// Marks a slot as drawable by pointing it at its mesh's acceleration structure.
    fn set_live(&mut self, slot: u32, address: u64) {
        if self.set_blas_ref(slot, GpuBlasRef::new(address)) {
            self.live_instance_count += 1;
        }
    }

    /// Drops a slot out of the TLAS, if it was in it.
    fn clear_live(&mut self, slot: u32) {
        if self.set_blas_ref(slot, GpuBlasRef::NONE) {
            self.live_instance_count -= 1;
        }
    }

    /// Writes a slot's acceleration structure reference, reporting whether that changed whether the
    /// slot is drawable at all.
    ///
    /// The buffer doubles as the liveness record, so there's no second structure to keep in step.
    /// Note this can't use [`set_at`]: an unchanged write still has to be skipped for the upload's
    /// sake, but the caller's live count depends on the *previous* value either way.
    fn set_blas_ref(&mut self, slot: u32, reference: GpuBlasRef) -> bool {
        self.blas_refs.grow(slot + 1);

        let previous = self.blas_refs.get(slot);
        if previous == reference {
            return false;
        }
        self.blas_refs.set(slot, reference);

        (previous == GpuBlasRef::NONE) != (reference == GpuBlasRef::NONE)
    }

    /// Drops an instance out of the TLAS without giving up its slot.
    fn deactivate_instance(&mut self, entity: Entity, instance: &mut Instance) {
        self.clear_live(instance.slot);
        self.remove_light(entity);
        // An instance that isn't drawing shouldn't pin a binding array slot for a slab it may no
        // longer even be allocated in. A later successful refresh takes fresh references.
        self.release_buffers(instance.buffers.take());
    }

    fn release_buffers(&mut self, buffers: Option<(BufferId, BufferId)>) {
        if let Some((vertex_key, index_key)) = buffers {
            self.vertex_buffers.release(&vertex_key);
            self.index_buffers.release(&index_key);
        }
    }

    fn remove_instance(&mut self, entity: Entity) {
        let Some(instance) = self.instances.remove(&entity) else {
            return;
        };

        // Drop out of the TLAS before giving the slot back, so the slot can't be handed to another
        // instance while this one still has a live entry.
        self.clear_live(instance.slot);
        self.instance_slots.release(instance.slot);
        self.pending_refresh.remove(&entity);
        self.remove_light(entity);
        self.release_buffers(instance.buffers);

        unlink(&mut self.mesh_instances, &instance.mesh, entity);
        unlink(&mut self.material_instances, &instance.material, entity);
    }
}

// ---------------------------------------------------------------------------
// TLAS
// ---------------------------------------------------------------------------

impl RaytracingSceneBindings {
    /// Moves to the next TLAS parity and brings it up to date with this frame's changes.
    ///
    /// The two acceleration structures alternate: this frame's is rebuilt, and last frame's stays
    /// intact so the shaders can trace against it.
    ///
    /// `build_ready` reports whether this frame will be able to record a build. When it is false
    /// nothing happens at all — not even the parity flip.
    ///
    /// Skipping the build but still advancing would strand an acceleration structure that never
    /// gets built, and a TLAS is bound for two frames, so it would resurface as the previous frame
    /// long after the cause went away. `wgpu` rejects a TLAS that was used without being built, so
    /// this is a hard error rather than a visual glitch.
    fn advance_tlas(
        &mut self,
        render_device: &RenderDevice,
        blas_manager: &BlasManager,
        build_ready: bool,
    ) {
        let _span = info_span!("advance_tlas").entered();

        // Nothing to trace and nothing to build. Worth returning before allocating rather than
        // after: `reserve_tlas` never allocates fewer than `TLAS_MIN_CAPACITY` instances, so it
        // would otherwise hand out a TLAS for an empty scene that the pack pass then declines to
        // fill, leaving an allocated-but-unbuilt structure to resurface as a later frame's
        // previous-frame entry.
        if !build_ready || self.instance_slots.len() == 0 {
            return;
        }

        // The custom data a hit is resolved through is 24 bits wide, and `tlas_instances.wgsl`
        // masks rather than checks, so a slot past that would silently alias another.
        debug_assert!(
            self.instance_slots.len() < 1 << TLAS_CUSTOM_DATA_BITS,
            "instance slot count {} does not fit in a TLAS instance's custom data",
            self.instance_slots.len()
        );

        // Everything the build reads is secured before the parity flip commits, for the same
        // reason as `build_ready` above: an allocation failure here would otherwise leave a TLAS
        // that never gets built.
        self.reserve_instance_descriptors(render_device);
        self.reserve_tlas_scratch(render_device);
        if self.instance_descriptors.is_none() || self.tlas_scratch.is_none() {
            return;
        }

        self.frame_parity ^= 1;
        let parity = self.frame_parity;

        self.reserve_tlas(parity, render_device);

        // Refreshed rather than appended to, so a BLAS that no scene instance references any more
        // is eventually released. Bounded by the number of distinct meshes, so it's cheaper than
        // tracking exactly which of them this parity ended up pointing at — but it is still an
        // atomic refcount bump per mesh here and another when the retired copy is dropped, so it's
        // gated on the set having actually changed rather than paid on every frame.
        //
        // Retired rather than dropped: the TLAS this parity held was traced by submissions that
        // may still be in flight, and since the build never told `wgpu-core` about these
        // dependencies, nothing else is keeping them alive.
        let generation = blas_manager.generation();
        if self.tlas_blas_generation[parity] != generation {
            let stale = core::mem::take(&mut self.tlas_blas_handles[parity]);
            self.pending_retire.extend(stale);
            self.tlas_blas_handles[parity].extend(blas_manager.handles().cloned());
            self.tlas_blas_generation[parity] = generation;
        }
    }

    /// Makes sure the instance descriptor buffer covers every slot, reallocating it if not.
    ///
    /// Sized off the slot allocator rather than the live count: descriptors are indexed by slot, so
    /// holes take up room too. They cost nothing at build time, since a hole's zero acceleration
    /// structure address is what both Vulkan and DXR treat as an inactive instance.
    fn reserve_instance_descriptors(&mut self, render_device: &RenderDevice) {
        let needed = self.instance_slots.len();
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

        // The pack shader writes every slot every frame, so a fresh buffer needs no seeding — but
        // its bind group is now stale.
        self.instance_pack_bind_group = None;
    }

    /// Makes sure the build scratch buffer is big enough for this frame's instance count.
    ///
    /// Has to run after [`Self::reserve_instance_descriptors`], which is what sets the instance
    /// count being sized for.
    fn reserve_tlas_scratch(&mut self, render_device: &RenderDevice) {
        let capacity = self.instance_descriptor_capacity;
        if self.tlas_scratch.is_some() && capacity <= self.tlas_scratch_sized_for {
            return;
        }

        let Some(instances) = self.instance_descriptors.as_ref() else {
            return;
        };
        let Some(needed) = tlas_build::tlas_scratch_size(render_device, instances, capacity) else {
            return;
        };
        self.tlas_scratch_sized_for = capacity;

        if self.tlas_scratch.is_some() && needed <= self.tlas_scratch_capacity {
            return;
        }

        // Dropping the outgrown buffer here is safe: `pack_raytracing_tlas_instances` transitions
        // it through `wgpu-core` every frame it is used, so a submission still reading it holds a
        // reference and the free is deferred until that submission retires. Without that
        // transition nothing would track it and this would be a use-after-free.
        self.tlas_scratch = tlas_build::create_scratch_buffer(render_device, needed);
        self.tlas_scratch_capacity = if self.tlas_scratch.is_some() {
            needed
        } else {
            0
        };
    }

    /// Makes sure `tlas[parity]` can hold every instance slot, reallocating just that one if not.
    ///
    /// The other TLAS is deliberately left alone. It only has to keep holding last frame's
    /// contents, which by definition fit in whatever size it already is, so growing the pair in
    /// lockstep would discard usable previous-frame data and pay for a second full build.
    fn reserve_tlas(&mut self, parity: usize, render_device: &RenderDevice) {
        let needed = self.instance_slots.len();
        if self.tlas[parity].is_some() && needed <= self.tlas_capacity[parity] {
            return;
        }

        let capacity = tlas_capacity_for(needed);

        self.tlas[parity] = Some(
            render_device
                .wgpu_device()
                .create_tlas(&CreateTlasDescriptor {
                    label: Some("tlas"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                    max_instances: capacity,
                }),
        );
        self.tlas_capacity[parity] = capacity;
        // A fresh acceleration structure has nothing in it until this frame's build runs.
        self.tlas_built[parity] = false;
        self.bind_group_invalid = true;
    }

    /// Rebuilds the pack shader's bind group if any of the three buffers it reads has moved.
    ///
    /// Sparse buffers reallocate whenever they outgrow their allocation, and the descriptor buffer
    /// whenever the slot count does, so the ids are compared rather than assumed stable.
    fn update_instance_pack_bind_group(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        pipeline: &TlasInstancePackPipeline,
    ) {
        let (Some(transforms), Some(blas_refs), Some(instances)) = (
            self.transforms.buffer(),
            self.blas_refs.buffer(),
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

// ---------------------------------------------------------------------------
// Bind group
// ---------------------------------------------------------------------------

/// Collects a binding array of slab buffers, standing `dummy` in for the free slots.
fn buffer_bindings<'a>(
    buffers: &'a RetainedBindingArray<BufferId, Buffer>,
    dummy: &'a Buffer,
) -> Vec<BufferBinding<'a>> {
    buffers
        .iter()
        .map(|buffer| buffer.unwrap_or(dummy).as_entire_buffer_binding())
        .collect()
}

impl RaytracingSceneBindings {
    /// The ids of the buffers behind the sparse vectors, in `sparse_buffers` order.
    fn buffer_ids(&self) -> [Option<BufferId>; SPARSE_BUFFER_COUNT] {
        let buffers = self.sparse_buffers();
        core::array::from_fn(|index| buffers[index].buffer_id())
    }

    /// Returns true if anything the bind group captures has changed since it was last built.
    fn take_bind_group_invalidation(
        &mut self,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> bool {
        let mut invalid = self.bind_group_invalid;
        self.bind_group_invalid = false;

        for dirty in [
            &mut self.vertex_buffers.dirty,
            &mut self.index_buffers.dirty,
            &mut self.textures.dirty,
        ] {
            invalid |= core::mem::replace(dirty, false);
        }

        // Any of the sparse buffers may have reallocated as it grew.
        let buffer_ids = self.buffer_ids();
        if self.last_buffer_ids != buffer_ids {
            self.last_buffer_ids = buffer_ids;
            invalid = true;
        }

        // `light_sources` is bound with an explicit size, because the shaders derive the light
        // count from `arrayLength` and the buffer is allocated with slack.
        let light_count = self.light_entities.len() as u32;
        if self.last_light_count != light_count {
            self.last_light_count = light_count;
            invalid = true;
        }

        // The DFG LUT stands in with the fallback image until it has been uploaded. Nothing else
        // here would notice the swap, so a bind group built during that window would keep the
        // placeholder bound for the rest of the run.
        let dfg_ids = Some((dfg_view.id(), dfg_sampler.id()));
        if self.last_dfg_ids != dfg_ids {
            self.last_dfg_ids = dfg_ids;
            invalid = true;
        }

        invalid
    }

    /// Builds the bind group for one TLAS parity.
    ///
    /// Only the requested parity is built, because the other one may not be bindable yet: on the
    /// very first frame the previous-frame TLAS hasn't been allocated at all.
    fn create_bind_group(
        &self,
        parity: usize,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        fallback_texture: &FallbackImage,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> BindGroup {
        let _span = info_span!("create_bind_group").entered();

        let vertex_buffers = buffer_bindings(&self.vertex_buffers, &self.dummy_buffer);
        let index_buffers = buffer_bindings(&self.index_buffers, &self.dummy_buffer);

        let (mut textures, mut samplers): (Vec<_>, Vec<_>) = self
            .textures
            .iter()
            .map(|texture| match texture {
                Some((view, sampler)) => (view.deref(), sampler.deref()),
                None => (
                    fallback_texture.d2.texture_view.deref(),
                    fallback_texture.d2.sampler.deref(),
                ),
            })
            .unzip();
        if textures.is_empty() {
            textures.push(fallback_texture.d2.texture_view.deref());
            samplers.push(fallback_texture.d2.sampler.deref());
        }

        let light_sources = BufferBinding {
            buffer: self.light_sources.buffer().unwrap(),
            offset: 0,
            size: BufferSize::new(
                self.light_entities.len() as u64 * size_of::<GpuLightSource>() as u64,
            ),
        };

        let materials = self.materials.buffer().unwrap().as_entire_buffer_binding();
        let transforms = self.transforms.buffer().unwrap().as_entire_buffer_binding();
        let previous_frame_transforms = self
            .previous_frame_transforms
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let geometry_ids = self
            .geometry_ids
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let material_ids = self
            .material_ids
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let directional_lights = self
            .directional_lights
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();
        let translations = self
            .previous_frame_light_id_translations
            .buffer()
            .unwrap()
            .as_entire_buffer_binding();

        // Checked by the caller, which declines to build a bind group at all when this parity has
        // no TLAS — `advance_tlas` won't allocate one it can't also build.
        let current = self.tlas[parity].as_ref().unwrap();
        // The other one is only rebuilt on the frames it's current, so it still holds last frame's
        // contents — unless it has never been built, either because this is the very first frame
        // or because the frame that allocated it bailed before building. There's nothing valid to
        // trace as the previous frame in either case, so bind the current TLAS in its place. See
        // `cache_bind_group` for why that isn't cached.
        let previous = self.tlas[parity ^ 1]
            .as_ref()
            .filter(|_| self.tlas_built[parity ^ 1])
            .unwrap_or(current);

        render_device.create_bind_group(
            "raytracing_scene_bind_group",
            layout,
            &BindGroupEntries::sequential((
                vertex_buffers.as_slice(),
                index_buffers.as_slice(),
                textures.as_slice(),
                samplers.as_slice(),
                materials,
                current.as_binding(),
                previous.as_binding(),
                transforms,
                previous_frame_transforms,
                geometry_ids,
                material_ids,
                light_sources,
                directional_lights,
                translations,
                dfg_view,
                dfg_sampler,
            )),
        )
    }

    /// Returns this parity's bind group, building it if the cache is cold.
    ///
    /// The result is only cached once a real previous-frame TLAS exists — which means built, not
    /// merely allocated, exactly as [`Self::create_bind_group`] decides it. Until then the
    /// previous-frame entry aliases the current TLAS, and caching that would leave the alias in
    /// place every time this parity came around again.
    fn cache_bind_group(
        &mut self,
        parity: usize,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        fallback_texture: &FallbackImage,
        dfg_view: &TextureView,
        dfg_sampler: &Sampler,
    ) -> BindGroup {
        if let Some(bind_group) = &self.cached_bind_groups[parity] {
            return bind_group.clone();
        }

        let bind_group = self.create_bind_group(
            parity,
            render_device,
            layout,
            fallback_texture,
            dfg_view,
            dfg_sampler,
        );

        // `tlas_built` rather than `tlas.is_some()`: an allocated but never built parity is what
        // `create_bind_group` aliases past, so caching on the weaker condition would pin the alias
        // for as long as this parity's TLAS went un-reallocated. (Built implies allocated, so this
        // covers both.)
        if self.tlas_built[parity ^ 1] {
            self.cached_bind_groups[parity] = Some(bind_group.clone());
        }

        bind_group
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Applies this frame's scene changes to the retained buffers, binding arrays and TLAS.
pub fn prepare_raytracing_scene_resources(
    instances: Query<InstanceQueryData>,
    changed_instances: Query<Entity, ChangedInstanceFilter>,
    mut removed_instances: RemovedComponents<RaytracingMesh3d>,
    directional_lights: Query<(Entity, &ExtractedDirectionalLight)>,
    mesh_allocator: Res<MeshAllocator>,
    blas_manager: Res<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    extracted_images: Res<ExtractedAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    instance_pack_pipeline: Res<TlasInstancePackPipeline>,
    mut bindings: ResMut<RaytracingSceneBindings>,
) {
    let bindings = &mut *bindings;

    bindings.reset_light_id_translations();

    bindings.update_materials(&material_assets, &texture_assets);
    bindings.update_textures(&extracted_images, &texture_assets, &material_assets);

    bindings.remove_instances(removed_instances.read());
    bindings.refresh_instances(
        &instances,
        &changed_instances,
        &blas_manager,
        &mesh_allocator,
    );

    bindings.update_lights(&directional_lights);

    bindings.write_sparse_buffers(&render_device, &render_queue);

    // The raw path can't build until the pack shader exists to fill the descriptors, and the
    // pipeline cache takes a few frames to get there from a cold start.
    let build_ready = instance_pack_pipeline
        .id
        .and_then(|id| pipeline_cache.get_compute_pipeline(id))
        .is_some();
    bindings.advance_tlas(&render_device, &blas_manager, build_ready);
}

/// Packs this frame's TLAS instance descriptors on the GPU.
///
/// Only does anything on the raw build path; the fallback hands `wgpu-core` a CPU-side instance
/// array instead. Has to run after [`SparseBufferSystems::Update`], whose dispatches are what put
/// this frame's transforms and acceleration structure addresses on the GPU, and before
/// [`build_raytracing_tlas`] consumes the result.
pub fn pack_raytracing_tlas_instances(
    mut bindings: ResMut<RaytracingSceneBindings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<TlasInstancePackPipeline>,
    mut render_context: RenderContext,
) {
    let bindings = &mut *bindings;
    bindings.instances_packed = false;

    // `advance_tlas` only hands out a TLAS for a frame it can also build, so without one there is
    // nothing to pack for. This isn't redundant with the checks below: the pipeline cache is
    // processed between `advance_tlas` and here, so it can become ready on the very frame
    // `advance_tlas` gave up on it — and packing then would rebuild the TLAS that is already bound
    // as this frame's, leaving the previous-frame entry a frame staler than it should be.
    if bindings.tlas[bindings.frame_parity].is_none() {
        return;
    }

    let (Some(bind_group), Some(compute_pipeline), Some(instances), Some(scratch)) = (
        bindings.instance_pack_bind_group.as_ref(),
        pipeline
            .id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id)),
        bindings.instance_descriptors.as_ref(),
        bindings.tlas_scratch.as_ref(),
    ) else {
        return;
    };

    let slot_count = bindings.instance_slots.len();
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

    // Hand both buffers to the build in the states it needs, from here rather than from the raw
    // encoder, so `wgpu-core` emits the barriers from a previous state it actually knows and keeps
    // the buffers alive for the submission. See `tlas_build`'s module docs.
    //
    // The scratch transition looks redundant — its state never changes — but scratch is an
    // exclusive usage, so the transition is emitted anyway, and it is what stops this frame's
    // build from overlapping the last one inside a buffer they share.
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

    bindings.instances_packed = true;
}

/// Records this frame's TLAS build into the render graph's command encoder.
///
/// This runs as part of the graph rather than submitting its own command buffer, so the build
/// rides along with everything else the frame submits.
pub fn build_raytracing_tlas(
    mut bindings: ResMut<RaytracingSceneBindings>,
    mut render_context: RenderContext,
) {
    let bindings = &mut *bindings;
    let parity = bindings.frame_parity;

    let Some(tlas) = bindings.tlas[parity].as_mut() else {
        return;
    };

    // Gated on the pack pass having actually recorded. If the pipeline is still compiling there
    // are no descriptors to build from, and — just as importantly — the transitions the build
    // relies on didn't happen either.
    //
    // Reaching any of these with a TLAS allocated means `advance_tlas` handed out one it couldn't
    // build, which `wgpu` reports much later and far less helpfully as "Tlas is used before it is
    // built".
    let (true, Some(instances), Some(scratch)) = (
        bindings.instances_packed,
        bindings.instance_descriptors.as_ref(),
        bindings.tlas_scratch.as_ref(),
    ) else {
        once!(warn!(
            "TLAS allocated but not built: packed={}, descriptors={}, scratch={}",
            bindings.instances_packed,
            bindings.instance_descriptors.is_some(),
            bindings.tlas_scratch.is_some(),
        ));
        return;
    };

    // `wgpu-core` panics if one encoder mixes the wgpu and raw encoding APIs, and a timestamp
    // write is a wgpu call — so the build gets an encoder of its own and is handed over as a
    // finished command buffer. `add_command_buffer` flushes the render context's encoder first,
    // which keeps the ordering the span needs: start timestamp, then the build, then the end
    // timestamp.
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
        bindings.instance_slots.len(),
        scratch,
    );
    render_context.add_command_buffer(command_encoder.finish());

    time_span.end(render_context.command_encoder());
    if built {
        bindings.tlas_built[parity] = true;
    } else {
        once!(warn!(
            "TLAS build recorded nothing; the backend probe and the build disagree about hal \
             access."
        ));
    }
}

/// Releases acceleration structures that no TLAS points at any more.
///
/// Has to run after this frame's submission, so that the work handed to
/// [`Queue::on_submitted_work_done`] covers every submission that could still be tracing a TLAS
/// which references them. The callback fires when that work completes, which is the exact moment
/// they stop being reachable by the GPU.
///
/// [`Queue::on_submitted_work_done`]: bevy_render::renderer::RenderQueue::on_submitted_work_done
pub fn retire_raytracing_resources(
    mut bindings: ResMut<RaytracingSceneBindings>,
    render_queue: Res<RenderQueue>,
) {
    if bindings.pending_retire.is_empty() {
        return;
    }

    let retired = core::mem::take(&mut bindings.pending_retire);
    render_queue.on_submitted_work_done(move || drop(retired));
}

/// Finalizes the sparse buffer uploads and selects this frame's bind group.
pub fn prepare_raytracing_scene_bind_group(
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    sparse_buffer_update_pipelines: Res<SparseBufferUpdatePipelines>,
    instance_pack_pipeline: Res<TlasInstancePackPipeline>,
    mut sparse_buffer_update_jobs: ResMut<SparseBufferUpdateJobs>,
    mut sparse_buffer_update_bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    mut bindings: ResMut<RaytracingSceneBindings>,
) {
    let bindings = &mut *bindings;

    {
        let _span = info_span!("prepare_sparse_uploads").entered();

        for buffer in bindings.sparse_buffers_mut() {
            buffer.prepare_upload(
                &render_device,
                &pipeline_cache,
                &mut sparse_buffer_update_jobs,
                &mut sparse_buffer_update_bind_groups,
                &sparse_buffer_update_pipelines,
            );
        }
    }

    bindings.update_instance_pack_bind_group(
        &render_device,
        &pipeline_cache,
        &instance_pack_pipeline,
    );

    bindings.bind_group = None;

    // Solari has nothing to trace against without both geometry and a light to sample.
    if bindings.live_instance_count == 0 || bindings.light_entities.is_empty() {
        return;
    }

    // `advance_tlas` declines to allocate one until it can also build it, so a missing TLAS means
    // this frame has nothing valid to trace. Handing out a bind group anyway would bind an unbuilt
    // acceleration structure, which `wgpu` rejects at submit.
    if bindings.tlas[bindings.frame_parity].is_none() {
        return;
    }

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|image| (&image.texture_view, &image.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    if bindings.take_bind_group_invalidation(dfg_view, dfg_sampler) {
        bindings.cached_bind_groups = [None, None];
    }

    let parity = bindings.frame_parity;
    let layout = pipeline_cache.get_bind_group_layout(&bindings.bind_group_layout);
    bindings.bind_group = Some(bindings.cache_bind_group(
        parity,
        &render_device,
        &layout,
        &fallback_texture,
        dfg_view,
        dfg_sampler,
    ));
}
