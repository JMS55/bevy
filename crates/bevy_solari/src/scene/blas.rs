use super::tlas_build;
use alloc::collections::VecDeque;
use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
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

/// A mesh's acceleration structure, and the address TLAS instance descriptors refer to it by.
struct ManagedBlas {
    blas: Blas,
    /// Device address of `blas`, or `None` on a backend without a raw TLAS build path, where
    /// descriptors are `wgpu-core`'s job and no address is ever needed.
    ///
    /// Resolved once here rather than per instance: an address is a property of the acceleration
    /// structure, and a scene has far fewer meshes than instances.
    address: Option<u64>,
}

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, ManagedBlas>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
    changed: Vec<AssetId<Mesh>>,
    /// Degenerate acceleration structure for dead instance slots, on backends that need one.
    ///
    /// Built once, never compacted or freed, so its address is stable for the whole run. See
    /// [`tlas_build::needs_dummy_blas`].
    dummy: Option<ManagedBlas>,
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh).map(|managed| &managed.blas)
    }

    /// The device address to point a TLAS instance descriptor at.
    pub fn address(&self, mesh: &AssetId<Mesh>) -> Option<u64> {
        self.blas.get(mesh).and_then(|managed| managed.address)
    }

    /// What a dead instance slot should point at, or zero where a null reference is legal.
    pub fn dead_blas_address(&self) -> u64 {
        self.dummy
            .as_ref()
            .and_then(|dummy| dummy.address)
            .unwrap_or(0)
    }

    /// Every acceleration structure currently held, so a caller can retain them.
    ///
    /// A TLAS built through [`crate::scene::tlas_build`] is invisible to `wgpu-core`, which therefore
    /// stops tracking which BLASes it points into. Since a built TLAS holds pointers to that
    /// memory, whoever builds one has to keep these alive for as long as it might still be traced
    /// against.
    pub fn handles(&self) -> impl Iterator<Item = &Blas> {
        self.blas
            .values()
            .chain(self.dummy.as_ref())
            .map(|managed| &managed.blas)
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

/// Builds the degenerate acceleration structure dead instance slots point at.
///
/// Only on backends that need one, and only once. A single triangle at the origin with all
/// coordinates zero: it has no area, and every instance referencing it is written with a zero
/// mask, so it can never be hit. It exists purely so a hole in the slot allocation refers to
/// something the driver is willing to dereference.
pub fn prepare_dummy_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if blas_manager.dummy.is_some() {
        return;
    }
    let Some(layout) = tlas_build::instance_layout(&render_device) else {
        return;
    };
    if !tlas_build::needs_dummy_blas(layout) {
        return;
    }

    // One zeroed triangle. The vertex stride matches what `allocate_blas` uses for real meshes, so
    // the geometry description stays the same shape.
    let vertices = render_device.create_buffer(&BufferDescriptor {
        label: Some("solari_dummy_blas_vertices"),
        size: 48 * 3,
        usage: BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let indices = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("solari_dummy_blas_indices"),
        contents: bytemuck::cast_slice(&[0u32, 0, 0]),
        usage: BufferUsages::BLAS_INPUT,
    });

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: 3,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(3),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let mut blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some("solari_dummy_blas"),
            // No `ALLOW_COMPACTION`: this is never compacted, so its address stays put and the
            // pack shader's uniform never has to be rewritten.
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );
    let address = tlas_build::blas_device_address(&render_device, &mut blas);

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("dummy_blas_build_command_encoder"),
    });
    command_encoder.build_acceleration_structures(
        &[BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: 48,
                index_buffer: Some(&indices),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        &[],
    );
    render_queue.submit([command_encoder.finish()]);

    blas_manager.dummy = Some(ManagedBlas { blas, address });
}

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
        blas_manager.blas.remove(asset_id);
        blas_manager.changed.push(*asset_id);
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

            let (mut blas, blas_size) =
                allocate_blas(&vertex_slice, &index_slice, asset_id, &render_device);
            let address = tlas_build::blas_device_address(&render_device, &mut blas);

            blas_manager
                .blas
                .insert(*asset_id, ManagedBlas { blas, address });
            blas_manager.changed.push(*asset_id);
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
                blas: &blas_manager.blas[asset_id].blas,
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
    render_device: Res<RenderDevice>,
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
            let mut compacted_blas = render_queue.compact_blas(blas);
            // Compaction moves the acceleration structure, so the old address is dead. Reporting
            // the mesh through `changed` is what makes every instance using it pick up the new
            // one; see `RaytracingSceneBindings::refresh_instances`.
            let address = tlas_build::blas_device_address(&render_device, &mut compacted_blas);

            blas_manager.blas.insert(
                mesh,
                ManagedBlas {
                    blas: compacted_blas,
                    address,
                },
            );
            blas_manager.changed.push(mesh);

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
