use super::{
    buffers::{new_storage_buffer, set_at, GpuU32},
    slots::SlotAllocator,
};
use bevy_color::ColorToComponents;
use bevy_ecs::{
    entity::{Entity, EntityHashSet},
    system::Query,
};
use bevy_math::{ops::cos, Vec3};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::render_resource::AtomicSparseBufferVec;
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};
use core::{f32::consts::TAU, hash::Hash};
use tracing::info_span;

const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLightSource {
    kind: u32,
    id: u32,
}

/// Stable identity for one source in the dense light array.
///
/// An entity can contribute both kinds at once, so the entity alone is not enough to identify a
/// source.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum LightSourceId {
    EmissiveMesh(Entity),
    Directional(Entity),
}

#[derive(Default)]
pub struct DenseLightIndex {
    indices: HashMap<LightSourceId, u32>,
    ids: Vec<LightSourceId>,
    changed: HashSet<LightSourceId>,
}

impl DenseLightIndex {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn get(&self, id: &LightSourceId) -> Option<u32> {
        self.indices.get(id).copied()
    }

    fn insert(&mut self, id: LightSourceId) -> u32 {
        if let Some(&index) = self.indices.get(&id) {
            return index;
        }

        let index = self.ids.len() as u32;
        self.ids.push(id);
        self.indices.insert(id, index);
        self.changed.insert(id);
        index
    }

    /// Removes `id` and reports both its old index and the old final index.
    ///
    /// When these differ, the caller must move the final GPU light source into the hole.
    fn remove(&mut self, id: LightSourceId) -> Option<(u32, u32)> {
        let index = self.indices.remove(&id)?;
        self.changed.insert(id);

        let last = self.ids.len() as u32 - 1;
        self.ids.swap_remove(index as usize);

        if index != last {
            let moved = self.ids[index as usize];
            self.indices.insert(moved, index);
            self.changed.insert(moved);
        }

        Some((index, last))
    }
}

impl GpuLightSource {
    pub fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
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
pub struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl_atomic_pod!(GpuLightSource, GpuLightSourceBlob);
impl_atomic_pod!(GpuDirectionalLight, GpuDirectionalLightBlob);

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

/// Dense light slots and the incremental previous-frame id translation state.
pub struct LightState {
    /// Kept dense because shaders derive the light count with `arrayLength`.
    pub sources: AtomicSparseBufferVec<GpuLightSource>,
    pub directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    pub previous_frame_id_translations: AtomicSparseBufferVec<GpuU32>,
    pub index: DenseLightIndex,
    previous_index: HashMap<LightSourceId, u32>,
    nonidentity_translations: Vec<u32>,
    directional_slots: SlotAllocator<Entity>,
}

impl LightState {
    pub fn new() -> Self {
        Self {
            sources: new_storage_buffer("solari_light_sources"),
            directional_lights: new_storage_buffer("solari_directional_lights"),
            previous_frame_id_translations: new_storage_buffer(
                "solari_previous_frame_light_id_translations",
            ),
            index: DenseLightIndex::default(),
            previous_index: HashMap::default(),
            nonidentity_translations: Vec::new(),
            directional_slots: SlotAllocator::new(),
        }
    }

    pub fn update(&mut self, directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>) {
        // There are few enough directional lights to just walk them every frame
        let _span = info_span!("update_lights").entered();

        let mut live_directional_lights = EntityHashSet::default();
        for (entity, directional_light) in directional_lights {
            live_directional_lights.insert(entity);

            let slot = self.directional_slots.get_or_allocate(entity);
            set_at(
                &mut self.directional_lights,
                slot,
                GpuDirectionalLight::new(directional_light),
            );
            self.add_light(
                LightSourceId::Directional(entity),
                GpuLightSource::new_directional_light(slot),
            );
        }

        let stale: Vec<Entity> = self
            .directional_slots
            .keys()
            .copied()
            .filter(|entity| !live_directional_lights.contains(entity))
            .collect();
        for entity in stale {
            self.directional_slots.remove(&entity);
            self.remove_light(LightSourceId::Directional(entity));
        }

        self.write_light_id_translations();

        if self.index.len() > u16::MAX as usize {
            panic!("Too many light sources in the scene, maximum is 65535.");
        }
    }

    pub fn add_light(&mut self, id: LightSourceId, source: GpuLightSource) {
        let index = self.index.insert(id);
        set_at(&mut self.sources, index, source);
    }

    /// Removes a light, moving the last one down into the hole to keep the array dense.
    pub fn remove_light(&mut self, id: LightSourceId) {
        let Some((index, last)) = self.index.remove(id) else {
            return;
        };

        if index != last {
            let source = self.sources.get(last);
            set_at(&mut self.sources, index, source);
        }
    }

    /// Restores the translation entries written last frame back to identity.
    pub fn reset_id_translations(&mut self) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            set_at(
                &mut self.previous_frame_id_translations,
                index,
                GpuU32(index),
            );
        }
    }

    /// Records where each light that moved or disappeared this frame ended up, so that reservoirs
    /// still carrying last frame's light ids can be remapped.
    fn write_light_id_translations(&mut self) {
        let changed: Vec<LightSourceId> = self.index.changed.drain().collect();

        for id in &changed {
            // Lights that first appeared this frame have no previous id to translate from
            let Some(&previous) = self.previous_index.get(id) else {
                continue;
            };
            let current = self.index.get(id).unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);

            if current != previous {
                set_at(
                    &mut self.previous_frame_id_translations,
                    previous,
                    GpuU32(current),
                );
                self.nonidentity_translations.push(previous);
            }
        }

        for id in changed {
            match self.index.get(&id) {
                Some(index) => self.previous_index.insert(id, index),
                None => self.previous_index.remove(&id),
            };
        }

        // Every index the shader might read has to be backed by a real element
        let light_count = self.index.len() as u32;
        let translations = &mut self.previous_frame_id_translations;
        if translations.len() < light_count {
            let start = translations.len();
            translations.grow(light_count);
            for index in start..light_count {
                translations.set(index, GpuU32(index));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseLightIndex, LightSourceId};
    use bevy_ecs::entity::Entity;

    #[test]
    fn light_index_keeps_sources_on_the_same_entity_independent() {
        let entity = Entity::PLACEHOLDER;
        let emissive = LightSourceId::EmissiveMesh(entity);
        let directional = LightSourceId::Directional(entity);
        let mut lights = DenseLightIndex::default();

        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.insert(directional), 1);
        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.len(), 2);

        assert_eq!(lights.remove(emissive), Some((0, 1)));
        assert_eq!(lights.get(&emissive), None);
        assert_eq!(lights.get(&directional), Some(0));
        assert_eq!(lights.len(), 1);

        assert_eq!(lights.remove(directional), Some((0, 0)));
        assert!(lights.is_empty());
    }
}
