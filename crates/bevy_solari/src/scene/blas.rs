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

/// TLAS builds a retired [`Blas`] waits out before it can be dropped.
///
/// Solari keeps two TLASes and traces the frame-old one as the previous frame, so a retired
/// structure stays pointed at until both have been rebuilt without it. A build only happens on a
/// frame where the binder flipped parity — an empty scene or an unready pack pipeline stops the
/// build as well as the flip — so builds strictly alternate parities and two of them is exactly
/// "both TLASes have been rebuilt", however many frames that takes.
const TLAS_BUILDS_BEFORE_DELETION: usize = 2;

#[derive(Resource)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
    changed: Vec<AssetId<Mesh>>,
    /// Retired since the last TLAS build.
    dying: Vec<Blas>,
    /// One batch of retirements per TLAS build still to be waited out, oldest at the front.
    deletion_queue: VecDeque<Vec<Blas>>,
    /// Retirements that have waited out their builds and now only await the GPU.
    deletable: Vec<Blas>,
}

impl FromWorld for BlasManager {
    fn from_world(_world: &mut World) -> Self {
        Self {
            blas: HashMap::default(),
            compaction_queue: VecDeque::new(),
            changed: Vec::new(),
            dying: Vec::new(),
            deletion_queue: VecDeque::new(),
            deletable: Vec::new(),
        }
    }
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
    }

    /// Records a mesh's newly built or newly compacted acceleration structure.
    ///
    /// Goes through here rather than touching the map directly so that [`Self::changed`] and the
    /// deletion queue can't be left behind. Compaction hands over a replacement for a mesh that
    /// already had one, and the structure it displaces is exactly as live as a removed one.
    fn insert(&mut self, mesh: AssetId<Mesh>, blas: Blas) {
        if let Some(displaced) = self.blas.insert(mesh, blas) {
            self.dying.push(displaced);
        }
        self.changed.push(mesh);
    }

    /// Retires a mesh's acceleration structure, if it had one. See [`Self::insert`].
    ///
    /// Called for every mesh removed or modified this frame, most of which never had one — a mesh
    /// that isn't raytracing compatible, or that no raytraced instance uses.
    ///
    /// [`Self::changed`] is reported either way. It is a push onto a short list, and its consumers
    /// are idempotent by contract.
    fn remove(&mut self, mesh: AssetId<Mesh>) {
        self.changed.push(mesh);
        if let Some(dying) = self.blas.remove(&mesh) {
            self.dying.push(dying);
        }
    }

    /// Advances the deletion queue by one TLAS build, which the binder calls when it records one.
    ///
    /// A retired structure cannot be dropped when it leaves the map: a TLAS built earlier still
    /// holds pointers into its memory, and stays that way until it is built again. So retirements
    /// wait here for [`TLAS_BUILDS_BEFORE_DELETION`] builds rather than for a number of frames —
    /// a scene that stops drawing and later resumes traces its previous-frame TLAS from before the
    /// pause, and a frame count would already have dropped what that TLAS points at.
    ///
    /// Costs a push of an empty `Vec` on a frame where nothing was retired, and is otherwise
    /// proportional to the number of meshes that actually changed — never to the size of the scene.
    pub fn note_tlas_build(&mut self) {
        self.deletion_queue
            .push_back(core::mem::take(&mut self.dying));
        if self.deletion_queue.len() > TLAS_BUILDS_BEFORE_DELETION {
            let expired = self.deletion_queue.pop_front().unwrap_or_default();
            self.deletable.extend(expired);
        }
    }

    /// The device address to point a TLAS instance descriptor at.
    ///
    /// `None` on a backend where `wgpu` can't hand one out, which is every backend Solari declines
    /// to load on anyway.
    pub fn address(&self, mesh: &AssetId<Mesh>) -> Option<u64> {
        self.blas.get(mesh)?.handle()
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

/// Builds an acceleration structure for every mesh that arrived or changed this frame.
///
/// A modified mesh has its old structure dropped first rather than reused: an acceleration
/// structure is built from the geometry, so it says nothing about the new data. Both that and the
/// rebuild are reported through [`BlasManager::changed`], which is what makes the instances using
/// the mesh re-resolve and pick up the new address.
///
/// The builds go out as their own submission rather than riding the frame's, because the TLAS
/// build later in the frame reads the structures this produces.
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

/// Frees the acceleration structures that have waited out their TLAS builds.
///
/// Runs at the end of the render graph so that the fence it registers covers this frame's tracing
/// as well: waiting out the builds establishes that nothing will point at these again, and the
/// fence establishes that nothing still in flight is reading them.
pub fn delete_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
    if blas_manager.deletable.is_empty() {
        return;
    }

    let deletable = core::mem::take(&mut blas_manager.deletable);
    render_queue.on_submitted_work_done(move || drop(deletable));
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
