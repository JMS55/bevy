use alloc::collections::VecDeque;
use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::HashMap;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::{
        allocator::{MeshAllocator, MeshBufferSlice},
        RenderMesh,
    },
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};

/// After compacting this many vertices worth of meshes per frame, no further BLAS will be compacted.
/// Lower this number to distribute the work across more frames.
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

#[derive(Resource)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
    changed: Vec<AssetId<Mesh>>,
    /// Bumped every time the set of acceleration structures held here changes. See
    /// [`Self::generation`].
    generation: u64,
}

impl FromWorld for BlasManager {
    fn from_world(_world: &mut World) -> Self {
        Self {
            blas: HashMap::default(),
            compaction_queue: VecDeque::new(),
            changed: Vec::new(),
            // Starts at one so that zero can mean "never seen a generation" to a consumer
            // comparing against this.
            generation: 1,
        }
    }
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
    }

    /// A counter bumped every time the set of acceleration structures held here changes — a
    /// creation, a replacement or a removal.
    ///
    /// Anything mirroring the whole set rather than tracking individual meshes compares this
    /// against the value its mirror was built from, so that it does nothing at all on the frames
    /// nothing moved — which is nearly all of them. [`Self::handles`] is the mirror that matters:
    /// rebuilding it costs an atomic refcount bump per distinct mesh, and another when the old copy
    /// is dropped, to arrive at the list it already had.
    ///
    /// Never zero, so zero is usable as "no mirror built yet".
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Records a mesh's newly built or newly compacted acceleration structure.
    ///
    /// Goes through here rather than touching the map directly so that [`Self::changed`] and
    /// [`Self::generation`] can't be left behind.
    fn insert(&mut self, mesh: AssetId<Mesh>, blas: Blas) {
        self.blas.insert(mesh, blas);
        self.changed.push(mesh);
        self.generation += 1;
    }

    /// Drops a mesh's acceleration structure. See [`Self::insert`].
    fn remove(&mut self, mesh: AssetId<Mesh>) {
        self.blas.remove(&mesh);
        self.changed.push(mesh);
        self.generation += 1;
    }

    /// The device address to point a TLAS instance descriptor at.
    ///
    /// `None` on a backend where `wgpu` can't hand one out, which is every backend Solari declines
    /// to load on anyway.
    pub fn address(&self, mesh: &AssetId<Mesh>) -> Option<u64> {
        self.blas.get(mesh)?.handle()
    }

    /// Every acceleration structure currently held, so a caller can retain them.
    ///
    /// A TLAS built through [`crate::scene::tlas_build`] is invisible to `wgpu-core`, which therefore
    /// stops tracking which BLASes it points into. Since a built TLAS holds pointers to that
    /// memory, whoever builds one has to keep these alive for as long as it might still be traced
    /// against.
    pub fn handles(&self) -> impl Iterator<Item = &Blas> {
        self.blas.values()
    }

    /// Meshes whose [`Blas`] was created, replaced, or removed this frame.
    ///
    /// Compaction swaps out the [`Blas`] object entirely, so anything holding a reference to one
    /// (such as a TLAS instance) has to be rebuilt when its mesh appears here.
    ///
    /// A mesh that was modified appears twice, once for the removal and once for the rebuild.
    /// Consumers are expected to be idempotent rather than pay for deduplication: this is empty
    /// on the overwhelming majority of frames, and short whenever it isn't.
    pub fn changed(&self) -> &[AssetId<Mesh>] {
        &self.changed
    }
}

/// Builds the degenerate acceleration structure dead instance slots point at, on the backends that
/// need one.
///
/// A single triangle at the origin with all coordinates zero: it has no area, and every instance
/// referencing it is written with a zero mask, so it can never be hit. It exists purely so a hole
/// in the slot allocation refers to something the driver is willing to dereference.
///
/// Built here, at render startup, rather than from a system: it is needed before the first
/// instance resolves and never again, so a system would spend every later frame re-deciding that
pub fn prepare_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
) {
    blas_manager.changed.clear();

    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.remove(*asset_id);
    }

    if extracted_meshes.extracted.is_empty() {
        return;
    }

    // Create new BLAS for added or changed meshes
    let blas_resources = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
        .map(|(asset_id, _)| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();

            let (blas, blas_size) =
                allocate_blas(&vertex_slice, &index_slice, asset_id, &render_device);

            blas_manager.insert(*asset_id, blas);
            blas_manager
                .compaction_queue
                .push_back((*asset_id, blas_size.vertex_count, false));

            (*asset_id, vertex_slice, index_slice, blas_size)
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, vertex_slice, index_slice, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertex_slice.buffer,
                first_vertex: vertex_slice.range.start,
                vertex_stride: 48,
                index_buffer: Some(index_slice.buffer),
                first_index: Some(index_slice.range.start),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &blas_manager.blas[asset_id],
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blas_build_command_encoder"),
    });
    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(&mut command_encoder, "blas_build"));
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    if let Some(time_span) = time_span {
        time_span.end(&mut command_encoder);
    }
    render_queue.submit([command_encoder.finish()]);
}

pub fn compact_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
    let queue_size = blas_manager.compaction_queue.len();
    let mut meshes_processed = 0;
    let mut vertices_compacted = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_compacted < MAX_COMPACTION_VERTICES_PER_FRAME
        && meshes_processed < queue_size
    {
        meshes_processed += 1;

        let (mesh, vertex_count, compaction_started) =
            blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&mesh) else {
            continue;
        };

        if !compaction_started {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            // Compaction moves the acceleration structure, so the old address is dead. Reporting
            // the mesh through `changed` is what makes every instance using it pick up the new
            // one; see `RaytracingSceneBindings::refresh_instances`.
            let compacted_blas = render_queue.compact_blas(blas);

            blas_manager.insert(mesh, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((mesh, vertex_count, true));
    }
}

fn allocate_blas(
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    asset_id: &AssetId<Mesh>,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_slice.range.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    // TODO: Switching to refit (`ALLOW_UPDATE` + `AccelerationStructureUpdateMode::PreferUpdate`)
    // has two consequences that aren't local to this function:
    //
    // 1. A refit mutates the BLAS in place, whereas a rebuild allocates a fresh one. Solari keeps
    //    two TLASes and traces the frame-old one as the previous frame; that one points at the
    //    BLAS being mutated. Per the DXR spec, "if a bottom-level acceleration structure at a given
    //    address is pointed to by top-level acceleration structures ever changes, those top-level
    //    acceleration structures are stale and must either be rebuilt or updated before they are
    //    valid to use again." So refitting a BLAS invalidates the previous-frame TLAS, and either
    //    both TLASes have to be rebuilt that frame or the refit target has to be double buffered.
    // 2. Primitives can't change activity across an update: a triangle hidden by a NaN vertex at
    //    build time can never be un-hidden, and an active one can never be hidden. Don't introduce
    //    NaN-hiding inside a refittable BLAS.
    //
    // Compaction is also mutually exclusive with updates in practice, so `compact_raytracing_blas`
    // would need to skip refittable meshes.
    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&asset_id.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .eq([
            (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
            (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
            (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
            (Mesh::ATTRIBUTE_TANGENT.id, Mesh::ATTRIBUTE_TANGENT.format),
        ]);
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
    mesh.enable_raytracing && triangle_list && vertex_attributes && indexed_32
}
