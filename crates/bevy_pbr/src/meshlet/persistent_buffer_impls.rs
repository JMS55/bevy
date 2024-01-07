use super::{persistent_buffer::PersistentGpuBufferable, Meshlet, MeshletBoundingSphere};
use std::sync::Arc;

impl PersistentGpuBufferable for Arc<[u8]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> u64 {
        self.len() as u64
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(self);
    }
}

impl PersistentGpuBufferable for Arc<[u32]> {
    type Metadata = u64;

    fn size_in_bytes(&self) -> u64 {
        self.len() as u64 * 4
    }

    fn write_bytes_le(&self, offset: Self::Metadata, buffer: &mut Vec<u8>) {
        let offset = offset as u32 / 48;

        for index in self.iter() {
            let bytes = (*index + offset).to_le_bytes();
            buffer.extend_from_slice(&bytes);
        }
    }
}

impl PersistentGpuBufferable for Arc<[Meshlet]> {
    type Metadata = (u64, u64);

    fn size_in_bytes(&self) -> u64 {
        self.len() as u64 * 12
    }

    fn write_bytes_le(&self, (vertex_offset, index_offset): Self::Metadata, buffer: &mut Vec<u8>) {
        let vertex_offset = vertex_offset as u32 / 4;
        let index_offset = index_offset as u32;

        for meshlet in self.iter() {
            let bytes = bytemuck::cast::<_, [u8; 12]>(Meshlet {
                start_vertex_id: meshlet.start_vertex_id + vertex_offset,
                start_index_id: meshlet.start_index_id + index_offset,
                triangle_count: meshlet.triangle_count,
            });
            buffer.extend_from_slice(&bytes);
        }
    }
}

impl PersistentGpuBufferable for Arc<[MeshletBoundingSphere]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> u64 {
        self.len() as u64 * 16
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(bytemuck::cast_slice(self));
    }
}
