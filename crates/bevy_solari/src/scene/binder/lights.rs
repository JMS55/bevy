use super::allocator::SlotAllocator;
use super::instances::InstanceState;
use bevy_color::{ColorToComponents, LinearRgba, Luminance};
use bevy_ecs::{
    entity::{Entity, EntityHashSet},
    system::Query,
};
use bevy_math::{ops::cos, Vec3};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::render_resource::{AtomicSparseBufferVec, Buffer, BufferUsages, RawBufferVec};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicBool, Ordering};
use core::{f32::consts::TAU, hash::Hash};
use tracing::info_span;

pub const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

fn luminance(color: Vec3) -> f32 {
    LinearRgba::from_vec3(color).luminance()
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLightSource {
    kind: u32,
    id: u32,
}

/// Stable identity for one source in the light array.
///
/// An entity can contribute both kinds at once, so the entity alone is not enough to identify a
/// source.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum LightSourceId {
    EmissiveMesh(Entity),
    Directional(Entity),
}

#[derive(Default)]
pub struct LightIndex {
    indices: HashMap<LightSourceId, u32>,
    ids: Vec<LightSourceId>,
    changed: HashSet<LightSourceId>,
}

impl LightIndex {
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
    /// Only the ids tracked here are swapped down. When the two indices differ, the caller has to
    /// mirror that swap in `sources`, copying the element at the old final index into the hole.
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

/// How a light's power weight is derived.
#[derive(Clone, Copy, PartialEq)]
enum LightWeightSource {
    /// A weight that stays put for as long as the light is registered.
    Fixed(f32),
    /// An emissive mesh light, whose area depends on the instance's current transform.
    EmissiveMesh {
        instance_slot: u32,
        /// Luminance of the material's emissive color.
        luminance: f32,
        /// Half extents of the mesh's AABB, in mesh space.
        local_half_extents: Vec3,
    },
}

impl Default for LightWeightSource {
    fn default() -> Self {
        Self::Fixed(0.0)
    }
}

impl LightWeightSource {
    fn evaluate(&self, instances: &InstanceState) -> f32 {
        match *self {
            Self::Fixed(weight) => weight,
            Self::EmissiveMesh {
                instance_slot,
                luminance,
                local_half_extents,
            } => luminance * instances.world_aabb_surface_area(instance_slot, local_half_extents),
        }
    }
}

/// Per-light power weights and the discrete distribution the sampler draws lights from.
///
/// The weights are re-evaluated every frame, since an emissive mesh light's area moves with its
/// instance. The CDF is an inclusive prefix sum, so any weight change invalidates every entry
/// after it: it is rebuilt and reuploaded whole, but only on frames where a weight actually moved.
pub struct LightWeights {
    weights: RawBufferVec<f32>,
    cdf: RawBufferVec<f32>,
    /// How to recompute each light's weight, indexed by light id.
    sources: Vec<LightWeightSource>,
    needs_upload: bool,
}

impl LightWeights {
    fn new() -> Self {
        let mut weights = RawBufferVec::new(BufferUsages::STORAGE);
        weights.set_label(Some("solari_light_weights"));
        let mut cdf = RawBufferVec::new(BufferUsages::STORAGE);
        cdf.set_label(Some("solari_light_cdf"));

        Self {
            weights,
            cdf,
            sources: Vec::new(),
            needs_upload: false,
        }
    }

    pub fn weights_buffer(&self) -> Option<&Buffer> {
        self.weights.buffer()
    }

    pub fn cdf_buffer(&self) -> Option<&Buffer> {
        self.cdf.buffer()
    }

    fn set(&mut self, index: u32, source: LightWeightSource) {
        let index = index as usize;
        if index >= self.sources.len() {
            self.sources.resize(index + 1, LightWeightSource::default());
        }
        self.sources[index] = source;
    }

    /// Mirrors the swap-down that keeps the light array gap-free.
    fn remove(&mut self, index: u32, last: u32) {
        if index != last {
            self.sources[index as usize] = self.sources[last as usize];
        }
        self.sources.truncate(last as usize);
    }

    /// Recomputes every weight and, if any of them moved, rebuilds the CDF.
    fn refresh(&mut self, instances: &InstanceState) {
        let mut changed = self.weights.len() != self.sources.len();
        self.weights.values_mut().resize(self.sources.len(), 0.0);
        for (weight, source) in self.weights.values_mut().iter_mut().zip(&self.sources) {
            let new_weight = source.evaluate(instances);
            changed |= *weight != new_weight;
            *weight = new_weight;
        }

        if !changed {
            return;
        }

        let cdf = self.cdf.values_mut();
        cdf.clear();
        let mut running = 0.0;
        for weight in self.weights.values() {
            running += weight;
            cdf.push(running);
        }
        self.needs_upload = true;
    }

    fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        // The bind group needs a buffer to point at even before the first light shows up
        self.weights
            .reserve(self.weights.len().max(1), render_device);
        self.cdf.reserve(self.cdf.len().max(1), render_device);

        if !core::mem::take(&mut self.needs_upload) {
            return;
        }
        self.weights.write_buffer(render_device, render_queue);
        self.cdf.write_buffer(render_device, render_queue);
    }
}

/// Light slots and the incremental previous-frame id translation state.
pub struct LightState {
    /// Kept gap-free because shaders derive the light count with `arrayLength`.
    pub sources: AtomicSparseBufferVec<GpuLightSource>,
    pub directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    pub previous_frame_id_translations: AtomicSparseBufferVec<u32>,
    pub weights: LightWeights,
    pub index: LightIndex,
    /// Light ids as of the last frame whose translation table the lighting shader actually read.
    previous_index: HashMap<LightSourceId, u32>,
    nonidentity_translations: Vec<u32>,
    directional_slots: SlotAllocator<Entity>,
    /// Set by the lighting node once it has recorded work reading the translation table.
    translations_consumed: AtomicBool,
}

impl LightState {
    pub fn new() -> Self {
        Self {
            sources: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_light_sources".into(),
            ),
            directional_lights: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_directional_lights".into(),
            ),
            previous_frame_id_translations: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_previous_frame_light_id_translations".into(),
            ),
            weights: LightWeights::new(),
            index: LightIndex::default(),
            previous_index: HashMap::default(),
            nonidentity_translations: Vec::new(),
            directional_slots: SlotAllocator::new(),
            translations_consumed: AtomicBool::new(false),
        }
    }

    pub fn update(&mut self, directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>) {
        // There are few enough directional lights to just walk them every frame
        let _span = info_span!("update_lights").entered();

        let mut live_directional_lights = EntityHashSet::default();
        for (entity, directional_light) in directional_lights {
            live_directional_lights.insert(entity);

            let slot = self.directional_slots.get_or_allocate(entity);
            let gpu_light = GpuDirectionalLight::new(directional_light);
            self.directional_lights.grow_and_set(slot, gpu_light);
            // Luminance times the solid angle of the sun disk, an estimate of radiant power
            let weight = luminance(gpu_light.luminance) * gpu_light.inverse_pdf;
            self.add_light(
                LightSourceId::Directional(entity),
                GpuLightSource::new_directional_light(slot),
                LightWeightSource::Fixed(weight),
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

    /// Registers or updates a light, returning its index in the light array.
    fn add_light(
        &mut self,
        id: LightSourceId,
        source: GpuLightSource,
        weight: LightWeightSource,
    ) -> u32 {
        let index = self.index.insert(id);
        self.sources.grow_and_set(index, source);
        self.weights.set(index, weight);
        index
    }

    /// Removes a light, moving the last one down into the hole so the array stays gap-free.
    pub fn remove_light(&mut self, id: LightSourceId) {
        let Some((index, last)) = self.index.remove(id) else {
            return;
        };

        if index != last {
            let source = self.sources.get(last);
            self.sources.grow_and_set(index, source);
        }
        self.weights.remove(index, last);
    }

    /// Registers an emissive mesh light, weighted by emissive luminance times world-space area.
    pub fn set_emissive_mesh_light(
        &mut self,
        entity: Entity,
        instance_slot: u32,
        triangle_count: u32,
        emissive: Vec3,
        local_half_extents: Option<Vec3>,
    ) -> u32 {
        let luminance = luminance(emissive);
        let weight = match local_half_extents {
            Some(local_half_extents) => LightWeightSource::EmissiveMesh {
                instance_slot,
                luminance,
                local_half_extents,
            },
            // No bounds to estimate an area from, so weight by luminance alone
            None => LightWeightSource::Fixed(luminance),
        };

        self.add_light(
            LightSourceId::EmissiveMesh(entity),
            GpuLightSource::new_emissive_mesh_light(instance_slot, triangle_count),
            weight,
        )
    }

    /// Lights whose index may have moved since instances last recorded it.
    pub fn changed_ids(&self) -> impl Iterator<Item = LightSourceId> + '_ {
        self.index.changed.iter().copied()
    }

    /// Current index of a light, or [`LIGHT_NOT_PRESENT_THIS_FRAME`] if it is gone.
    pub fn light_id(&self, id: &LightSourceId) -> u32 {
        self.index.get(id).unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME)
    }

    /// Recomputes the power weights and, if any changed, the distribution the sampler draws from.
    pub fn refresh_weights(&mut self, instances: &InstanceState) {
        let _span = info_span!("refresh_light_weights").entered();
        self.weights.refresh(instances);
    }

    pub fn write_weight_buffers(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        self.weights.write_buffers(render_device, render_queue);
    }

    /// Rolls the translation table over for a new frame.
    ///
    /// `previous_index` and `changed` only advance once the shader has read the table. The
    /// lighting node bails out while its pipelines compile, and the reservoirs keep the older ids
    /// across such a gap, so the next table has to translate from those instead. `has_consumers`
    /// is false when no view runs Solari lighting, where deferring forever would grow both
    /// without bound.
    pub fn begin_frame(&mut self, has_consumers: bool) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            self.previous_frame_id_translations
                .grow_and_set(index, index);
        }

        if !has_consumers || self.translations_consumed.swap(false, Ordering::Relaxed) {
            for id in core::mem::take(&mut self.index.changed) {
                match self.index.get(&id) {
                    Some(index) => self.previous_index.insert(id, index),
                    None => self.previous_index.remove(&id),
                };
            }
        }
    }

    /// Records that the lighting shader read this frame's translation table.
    pub fn note_translations_consumed(&self) {
        self.translations_consumed.store(true, Ordering::Relaxed);
    }

    /// Records where each light that moved or disappeared this frame ended up, so that reservoirs
    /// still carrying last frame's light ids can be remapped.
    fn write_light_id_translations(&mut self) {
        for id in &self.index.changed {
            // Lights that first appeared since the last read table have no previous id
            let Some(&previous) = self.previous_index.get(id) else {
                continue;
            };
            let current = self.index.get(id).unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);

            if current != previous {
                self.previous_frame_id_translations
                    .grow_and_set(previous, current);
                self.nonidentity_translations.push(previous);
            }
        }

        // Every index the shader might read has to be backed by a real element
        let light_count = self.index.len() as u32;
        let translations = &mut self.previous_frame_id_translations;
        if translations.len() < light_count {
            let start = translations.len();
            translations.grow(light_count);
            for index in start..light_count {
                translations.set(index, index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LightIndex, LightSourceId};
    use bevy_ecs::entity::Entity;

    #[test]
    fn light_index_keeps_sources_on_the_same_entity_independent() {
        let entity = Entity::PLACEHOLDER;
        let emissive = LightSourceId::EmissiveMesh(entity);
        let directional = LightSourceId::Directional(entity);
        let mut lights = LightIndex::default();

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
