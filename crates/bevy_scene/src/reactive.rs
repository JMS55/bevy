//! Reactive scene support for BSN.
//!
//! This module provides React-like reactivity for BSN scenes. A [`SceneFactory`] component stores
//! a scene factory function that is re-invoked every frame. The resulting scene is diffed against
//! the existing entity tree and minimal updates are applied.
//!
//! # Usage
//!
//! ```ignore
//! fn counter_ui(ctx: &SceneContext) -> impl Scene {
//!     let count = ctx.use_state_or(Counter(0)).0;
//!     bsn! {
//!         Text({format!("Count: {count}")})
//!         on(|_: On<Pointer<Press>>, mut q: Query<&mut Counter>| {
//!             q.single_mut().0 += 1;
//!         })
//!     }
//! }
//!
//! world.spawn_reactive_scene(counter_ui);
//! ```

extern crate alloc;

use crate::{Scene, ScenePatch};
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use bevy_asset::{AssetServer, Assets};
use bevy_ecs::{
    component::Component, entity::Entity, prelude::*, resource::Resource, world::World,
};
use core::cell::RefCell;
use tracing::error;

/// A component that stores a scene factory function for reactive re-rendering.
///
/// Entities with this component are automatically re-rendered every frame by the
/// [`reapply_reactive_scenes`] system. The factory function is called to produce a new [`Scene`],
/// which is then diffed against the existing entity state and applied with minimal changes.
///
/// Use [`WorldSceneExt::spawn_reactive_scene`] or [`CommandsSceneExt::spawn_reactive_scene`] to
/// create entities with this component.
///
/// [`WorldSceneExt::spawn_reactive_scene`]: crate::WorldSceneExt::spawn_reactive_scene
/// [`CommandsSceneExt::spawn_reactive_scene`]: crate::CommandsSceneExt::spawn_reactive_scene
#[derive(Component)]
pub struct SceneFactory {
    factory: Arc<dyn Fn(&SceneContext) -> Box<dyn Scene> + Send + Sync>,
}

impl SceneFactory {
    /// Creates a new [`SceneFactory`] from the given factory function.
    ///
    /// The factory function returns `Box<dyn Scene>` to avoid higher-ranked lifetime issues
    /// with generic return types.
    pub fn new(factory: impl Fn(&SceneContext) -> Box<dyn Scene> + Send + Sync + 'static) -> Self {
        Self {
            factory: Arc::new(factory),
        }
    }
}

/// Marker component indicating that an entity's [`SceneFactory`] has completed its initial render.
/// After the first render, subsequent frames use `reapply` (diff-aware) instead of `apply`.
#[derive(Component)]
pub struct SceneFactoryInitialized;

/// Context provided to [`SceneFactory`] functions during re-rendering.
///
/// Provides read-only access to the [`World`] and the owning [`Entity`], along with helper methods
/// like [`use_state_or`] for reading component state with automatic initialization.
///
/// [`use_state_or`]: SceneContext::use_state_or
pub struct SceneContext<'a> {
    world: &'a World,
    entity: Entity,
    pending_inits: RefCell<Vec<Box<dyn FnOnce(&mut EntityWorldMut) + Send>>>,
}

impl<'a> SceneContext<'a> {
    /// Creates a new [`SceneContext`].
    pub fn new(world: &'a World, entity: Entity) -> Self {
        Self {
            world,
            entity,
            pending_inits: RefCell::new(Vec::new()),
        }
    }

    /// Reads component state of type `T` from the entity. If the component does not exist,
    /// returns `initial` and queues initialization so the component will be inserted on the entity
    /// after the factory function returns.
    ///
    /// This is analogous to React's `useState` hook: it provides a default value on first render
    /// and reads the persisted state on subsequent renders.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn my_ui(ctx: &SceneContext) -> impl Scene {
    ///     let count = ctx.use_state_or(Counter(0)).0;
    ///     bsn! { Text({format!("Count: {count}")}) }
    /// }
    /// ```
    pub fn use_state_or<T: Component + Clone>(&self, initial: T) -> T {
        if let Some(value) = self.world.get::<T>(self.entity) {
            value.clone()
        } else {
            let clone = initial.clone();
            self.pending_inits
                .borrow_mut()
                .push(Box::new(move |entity| {
                    if !entity.contains::<T>() {
                        entity.insert(initial);
                    }
                }));
            clone
        }
    }

    /// Access a global [`Resource`].
    pub fn resource<R: Resource>(&self) -> &R {
        self.world.resource::<R>()
    }

    /// The [`Entity`] this scene factory lives on.
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Direct read-only [`World`] access for advanced queries.
    pub fn world(&self) -> &'a World {
        self.world
    }

    /// Drain the pending state initializations. Called by the system after the factory returns.
    pub(crate) fn take_pending_inits(&self) -> Vec<Box<dyn FnOnce(&mut EntityWorldMut) + Send>> {
        core::mem::take(&mut *self.pending_inits.borrow_mut())
    }
}

/// Exclusive system that re-renders all entities with a [`SceneFactory`] component every frame.
///
/// For entities that have not yet been initialized ([`SceneFactoryInitialized`] is absent),
/// this performs the initial `apply`. For already-initialized entities, it performs a diff-aware
/// `reapply` that skips unchanged components and reconciles children.
pub fn reapply_reactive_scenes(world: &mut World) {
    // Collect all SceneFactory entities
    let entities: Vec<(Entity, Arc<dyn Fn(&SceneContext) -> Box<dyn Scene> + Send + Sync>, bool)> = {
        let mut query = world.query::<(Entity, &SceneFactory, Option<&SceneFactoryInitialized>)>();
        query
            .iter(world)
            .map(|(entity, sf, init)| (entity, sf.factory.clone(), init.is_some()))
            .collect()
    };

    for (entity, factory, is_initialized) in entities {
        // Create context and call factory (reborrow &mut World as &World)
        let (scene, pending_inits) = {
            let ctx = SceneContext::new(&*world, entity);
            let scene = (factory)(&ctx);
            let pending_inits = ctx.take_pending_inits();
            (scene, pending_inits)
        };

        // Flush pending state initializations
        if !pending_inits.is_empty() {
            if let Ok(mut entity_mut) = world.get_entity_mut(entity) {
                for init in pending_inits {
                    init(&mut entity_mut);
                }
            }
        }

        // Resolve the scene
        let assets = world.resource::<AssetServer>();
        let mut patch = ScenePatch::load(assets, scene);
        let resolve_result = patch.resolve(assets, world.resource::<Assets<ScenePatch>>());
        match resolve_result {
            Ok(()) => {}
            Err(err) => {
                error!("Failed to resolve reactive scene for entity {entity}: {err}");
                continue;
            }
        }

        let resolved = match &patch.resolved {
            Some(resolved) => resolved.clone(),
            None => continue,
        };

        // Apply or reapply
        if let Ok(mut entity_mut) = world.get_entity_mut(entity) {
            let result = if is_initialized {
                resolved.reapply(&mut entity_mut)
            } else {
                entity_mut.insert(SceneFactoryInitialized);
                resolved.apply(&mut entity_mut)
            };

            if let Err(err) = result {
                error!("Failed to apply reactive scene for entity {entity}: {err}");
            }
        }
    }
}
