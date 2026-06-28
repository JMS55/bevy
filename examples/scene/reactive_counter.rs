//! A prototype of **react-like reactive scenes** authored directly in `bsn!`.
//!
//! A reactive scene's render function is a genuine Bevy system whose parameters are real
//! `SystemParam`s and which returns a `bsn!` scene — no `view()` / `reactive_node()` wrappers:
//!
//! ```ignore
//! #[reactive_scene]
//! fn panel(mut hooks: ReactiveHooks, score: Res<Score>, healths: Query<&Health>) -> impl Scene { .. }
//! ```
//!
//! The runtime resolves the returned scene and reconciles it against the live entities.
//!
//! ## What's implemented
//! - **Per-node reconcile** with **required-component-aware** component removal.
//! - **Reactive reads**: `Res<T>` and `Query<&C>` parameters re-render when their data changes.
//! - **Keyed children** (`ReactiveKey(n)`): reused by identity across reorders.
//! - **Nested props** propagate to reused reactive children on a parent re-render.
//! - **Unmount cleanups**: despawning a reactive subtree runs its effects' [`Cleanup`]s.
//! - **Observer reconciliation**: observers are matched positionally across renders — unchanged
//!   ones are kept, only newly-added observers are attached and only removed ones are despawned
//!   (no churn for a stable observer set).
//! - **`set_if_neq` value commits** — re-applying an unchanged component does not bump its change
//!   tick. This is done in `bevy_scene` via `feature(specialization)` (it commits `PartialEq` +
//!   mutable components with `set_if_neq`, everything else with a plain insert). `min_specialization`
//!   can't specialize on `PartialEq`, so full `specialization` is required.
//! - **One cached system per reactive-scene function** (`register_system_cached`) shared across all
//!   instances — no per-instance/per-render registration.
//! - Hooks: `use_state` / `use_memo` / `use_effect`.
//!
//! Requires a **nightly** toolchain (for `feature(specialization)` in `bevy_scene`).

use bevy::{
    ecs::{
        change_detection::Tick,
        component::ComponentId,
        observer::ObservedBy,
        system::{IntoSystem, SystemId, SystemParam},
        template::{SceneEntityReferences, Template, TemplateContext},
    },
    prelude::*,
    scene::{ResolveContext, ResolveSceneError, ResolvedScene, ResolvedSceneRoot, ScenePatch},
};
use reactive_scene_macro::reactive_scene;
use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    sync::{Arc, Mutex},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(Score(0))
        .insert_resource(Accent(Color::srgb(0.4, 0.7, 1.0)))
        .add_systems(Startup, setup)
        .add_systems(Update, render_dirty_instances)
        .run();
}

#[derive(Resource, Default, Clone)]
struct Score(i32);

#[derive(Resource, Clone)]
struct Accent(Color);

#[derive(Component, Default, Clone)]
struct Health(u32);

/// A "prop" passed to a nested reactive scene as a component (read via `Query`).
#[derive(Component, Default, Clone)]
struct TallyLabel(&'static str);

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn(Health(40));
    commands.spawn(Health(60));
    commands.spawn_scene(app_root());
}

// ----------------------------------------------------------------------------
// The demo UI — pure `bsn!`, each render fn is a real system.
// ----------------------------------------------------------------------------

#[reactive_scene]
fn app_root(mut hooks: ReactiveHooks, score: Res<Score>, accent: Res<Accent>) -> impl Scene {
    let (count, set_count) = hooks.use_state(|| 0i32);
    let doubled = hooks.use_memo(count, move || count * 2);
    hooks.use_effect(count, move |_world| {
        info!("local count is now {count}");
        Cleanup::none()
    });

    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            row_gap: px(12),
        }
        Children [
            (
                Text({ format!("local: {count}   (x2 = {doubled})") })
                TextFont { font_size: px(32.0) }
                TextColor({ accent.0 })
            ),
            (
                Text({ format!("shared score: {}", score.0) })
                TextFont { font_size: px(32.0) }
            ),
            (
                button("local +1")
                on(move |_: On<Pointer<Press>>, mut commands: Commands| {
                    set_count.update(&mut commands, |c| *c += 1);
                })
            ),
            (
                button("score +1")
                on(|_: On<Pointer<Press>>, mut score: ResMut<Score>| {
                    score.0 += 1;
                })
            ),
            ( {health_readout()} ),
            ( {tally()} TallyLabel("A") ReactiveKey(1) ),
            ( {tally()} TallyLabel("B") ReactiveKey(2) ),
        ]
    }
}

#[reactive_scene]
fn tally(mut hooks: ReactiveHooks, labels: Query<&TallyLabel>) -> impl Scene {
    let label = labels.get(hooks.entity()).map(|l| l.0).unwrap_or("?");
    let (n, set_n) = hooks.use_state(|| 0i32);
    bsn! {
        Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(10),
            align_items: AlignItems::Center,
        }
        Children [
            (
                Text({ format!("{label}: {n}") })
                TextFont { font_size: px(24.0) }
            ),
            (
                button("+1")
                on(move |_: On<Pointer<Press>>, mut commands: Commands| {
                    set_n.update(&mut commands, |x| *x += 1);
                })
            ),
        ]
    }
}

#[reactive_scene]
fn health_readout(healths: Query<&Health>) -> impl Scene {
    let total: u32 = healths.iter().map(|h| h.0).sum();
    bsn! {
        Text({ format!("total health: {total}") })
        TextFont { font_size: px(24.0) }
    }
}

fn button(label: &str) -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(220),
            height: px(55),
            border: px(3),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
        }
        BorderColor::from(Color::BLACK)
        BackgroundColor(Color::srgb(0.15, 0.15, 0.15))
        Children [(
            Text(label)
            TextColor(Color::srgb(0.9, 0.9, 0.9))
        )]
    }
}

// ----------------------------------------------------------------------------
// Hooks (a SystemParam) + per-instance state.
// ----------------------------------------------------------------------------

#[derive(Resource)]
struct CurrentRender {
    entity: Entity,
    state: ReactiveState,
    cursor: usize,
    pending: Vec<PendingEffect>,
}

/// The hooks handle, usable inside any render system as a normal `SystemParam`.
#[derive(SystemParam)]
pub struct ReactiveHooks<'w> {
    current: ResMut<'w, CurrentRender>,
}

impl ReactiveHooks<'_> {
    /// The entity this reactive instance renders onto.
    pub fn entity(&self) -> Entity {
        self.current.entity
    }

    /// React's `useState`: a snapshot value for this render, plus a `Copy` [`Setter`].
    pub fn use_state<T: Clone + PartialEq + Send + Sync + 'static>(
        &mut self,
        init: impl FnOnce() -> T,
    ) -> (T, Setter<T>) {
        let cr = &mut *self.current;
        let slot = cr.cursor;
        cr.cursor += 1;
        if slot == cr.state.slots.len() {
            cr.state.slots.push(Slot::State(Box::new(init())));
        }
        let value = match &cr.state.slots[slot] {
            Slot::State(b) => b
                .downcast_ref::<T>()
                .expect("use_state slot type mismatch")
                .clone(),
            _ => panic!("hook order changed at slot {slot} (expected state)"),
        };
        (
            value,
            Setter {
                entity: cr.entity,
                slot,
                _marker: PhantomData,
            },
        )
    }

    /// React's `useMemo`: recompute only when `deps` change.
    pub fn use_memo<D, T>(&mut self, deps: D, compute: impl FnOnce() -> T) -> T
    where
        D: PartialEq + Send + Sync + 'static,
        T: Clone + Send + Sync + 'static,
    {
        let cr = &mut *self.current;
        let slot = cr.cursor;
        cr.cursor += 1;
        if slot == cr.state.slots.len() {
            let value = compute();
            cr.state.slots.push(Slot::Memo {
                deps: Box::new(deps),
                value: Box::new(value.clone()),
            });
            return value;
        }
        match &mut cr.state.slots[slot] {
            Slot::Memo {
                deps: stored,
                value,
            } => {
                if stored.eq_any(&deps) {
                    value
                        .downcast_ref::<T>()
                        .expect("use_memo slot type mismatch")
                        .clone()
                } else {
                    let v = compute();
                    *stored = Box::new(deps);
                    *value = Box::new(v.clone());
                    v
                }
            }
            _ => panic!("hook order changed at slot {slot} (expected memo)"),
        }
    }

    /// React's `useEffect`: runs after commit, only when `deps` change; cleanup before re-run
    /// and on unmount.
    pub fn use_effect<D>(
        &mut self,
        deps: D,
        effect: impl FnOnce(&mut World) -> Cleanup + Send + Sync + 'static,
    ) where
        D: PartialEq + Send + Sync + 'static,
    {
        let cr = &mut *self.current;
        let slot = cr.cursor;
        cr.cursor += 1;
        let run = if slot == cr.state.slots.len() {
            cr.state.slots.push(Slot::Effect {
                deps: Box::new(deps),
                cleanup: Cleanup::none(),
            });
            true
        } else {
            match &mut cr.state.slots[slot] {
                Slot::Effect { deps: stored, .. } => {
                    if stored.eq_any(&deps) {
                        false
                    } else {
                        *stored = Box::new(deps);
                        true
                    }
                }
                _ => panic!("hook order changed at slot {slot} (expected effect)"),
            }
        };
        if run {
            cr.pending.push(PendingEffect {
                slot,
                run: Box::new(effect),
            });
        }
    }
}

#[derive(Component, Default)]
struct ReactiveState {
    slots: Vec<Slot>,
}

enum Slot {
    State(Box<dyn Any + Send + Sync>),
    Memo {
        deps: Box<dyn Deps>,
        value: Box<dyn Any + Send + Sync>,
    },
    Effect {
        deps: Box<dyn Deps>,
        cleanup: Cleanup,
    },
}

/// A teardown callback returned by an effect.
pub struct Cleanup(Option<Box<dyn FnOnce(&mut World) + Send + Sync>>);

impl Cleanup {
    /// No teardown.
    pub fn none() -> Self {
        Cleanup(None)
    }
    /// A teardown closure run before the effect re-runs, or when the instance unmounts.
    pub fn new(f: impl FnOnce(&mut World) + Send + Sync + 'static) -> Self {
        Cleanup(Some(Box::new(f)))
    }
    fn run(self, world: &mut World) {
        if let Some(f) = self.0 {
            f(world);
        }
    }
}

trait Deps: Send + Sync + 'static {
    fn eq_any(&self, other: &dyn Any) -> bool;
}
impl<T: PartialEq + Send + Sync + 'static> Deps for T {
    fn eq_any(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<T>().is_some_and(|o| self == o)
    }
}

struct PendingEffect {
    slot: usize,
    run: Box<dyn FnOnce(&mut World) -> Cleanup + Send + Sync>,
}

/// A `Copy` token that writes back to a `use_state` slot, like React's setter.
pub struct Setter<T> {
    entity: Entity,
    slot: usize,
    _marker: PhantomData<fn(T)>,
}

impl<T> Clone for Setter<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Setter<T> {}

impl<T: PartialEq + Clone + Send + Sync + 'static> Setter<T> {
    /// Read-modify-write the state; if it changed, mark the instance for re-render.
    pub fn update(self, commands: &mut Commands, f: impl FnOnce(&mut T) + Send + 'static) {
        let entity = self.entity;
        let slot = self.slot;
        commands.queue(move |world: &mut World| {
            let changed = world
                .get_mut::<ReactiveState>(entity)
                .map(|mut state| match &mut state.slots[slot] {
                    Slot::State(b) => {
                        let value = b.downcast_mut::<T>().expect("setter slot type mismatch");
                        let old = value.clone();
                        f(value);
                        *value != old
                    }
                    _ => false,
                })
                .unwrap_or(false);
            if changed && let Ok(mut entity_mut) = world.get_entity_mut(entity) {
                entity_mut.insert(RenderDirty);
            }
        });
    }
}

// ----------------------------------------------------------------------------
// Reactive dependencies (change-tick subscriptions).
// ----------------------------------------------------------------------------

/// "Did my dependency change since the instance's last render?" Stored per reactive scene.
type DepChecker = Box<dyn Fn(&mut World, Tick) -> bool + Send + Sync>;
type Deps_ = Arc<Vec<DepChecker>>;

/// Re-render when resource `T` changes. Emitted by the macro for each `Res<T>` parameter.
pub fn res_dep<T: Resource>() -> DepChecker {
    Box::new(|world: &mut World, since: Tick| {
        let now = world.change_tick();
        world
            .get_resource_ref::<T>()
            .is_some_and(|r| r.last_changed().is_newer_than(since, now))
    })
}

/// Re-render when any `C` changes. Emitted by the macro for each `Query<&C>` parameter.
pub fn query_dep<C: Component>() -> DepChecker {
    Box::new(|world: &mut World, since: Tick| {
        let now = world.change_tick();
        let mut query = world.query::<Ref<C>>();
        query
            .iter(world)
            .any(|c| c.last_changed().is_newer_than(since, now))
    })
}

// ----------------------------------------------------------------------------
// Registering & spawning a reactive scene.
// ----------------------------------------------------------------------------

/// Lazily registers a reactive scene's render system, cached by the system's function type so all
/// instances of a scene share a single registration, and yields its [`SystemId`].
type RegisterFn = Arc<dyn Fn(&mut World) -> SystemId<(), Box<dyn Scene>> + Send + Sync>;

/// A [`Scene`] that turns the entity it is applied to into a reactive instance. Produced by
/// `#[reactive_scene]` constructors.
pub struct ReactiveScene {
    register: RegisterFn,
    deps: Deps_,
}

/// Builds a [`ReactiveScene`] from a render system + its reactive dependency checkers.
pub fn reactive_scene_system<M, S>(system: S, deps: Vec<DepChecker>) -> ReactiveScene
where
    S: IntoSystem<(), Box<dyn Scene>, M> + Copy + Send + Sync + 'static,
{
    ReactiveScene {
        register: Arc::new(move |world: &mut World| world.register_system_cached(system)),
        deps: Arc::new(deps),
    }
}

impl Scene for ReactiveScene {
    fn resolve(
        self,
        _context: &mut ResolveContext,
        scene: &mut ResolvedScene,
    ) -> Result<(), ResolveSceneError> {
        scene.push_template(ReactiveSeed {
            register: self.register,
            deps: self.deps,
        });
        Ok(())
    }
}

/// Carries a reactive scene's registration + deps into a resolved scene as a component, so the
/// runtime can recognize "this node is a reactive instance" and seed it.
#[derive(Component)]
struct ReactiveMarker {
    register: RegisterFn,
    deps: Deps_,
}

struct ReactiveSeed {
    register: RegisterFn,
    deps: Deps_,
}

impl Template for ReactiveSeed {
    type Output = ReactiveMarker;

    fn build_template(&self, _context: &mut TemplateContext) -> Result<Self::Output> {
        Ok(ReactiveMarker {
            register: self.register.clone(),
            deps: self.deps.clone(),
        })
    }

    fn clone_template(&self) -> Self {
        ReactiveSeed {
            register: self.register.clone(),
            deps: self.deps.clone(),
        }
    }
}

// ----------------------------------------------------------------------------
// Per-instance runtime components.
// ----------------------------------------------------------------------------

#[derive(Component)]
struct ReactiveRender {
    system: SystemId<(), Box<dyn Scene>>,
    deps: Deps_,
}

#[derive(Component)]
struct RenderDirty;

/// The observer entities this node attached on its last render, for reconciliation.
#[derive(Component, Default)]
struct ReactiveObservers(Vec<Entity>);

#[derive(Component, Default)]
struct ManagedComponents(HashSet<ComponentId>);

/// The world change-tick at which this instance last rendered (for dependency change detection).
#[derive(Component)]
struct LastRender(Tick);

/// A stable identity for keyed child reconciliation. Authored as `ReactiveKey(n)` in `bsn!`.
#[derive(Component, Default, Clone)]
struct ReactiveKey(u64);

// ----------------------------------------------------------------------------
// The render + reconcile runner.
// ----------------------------------------------------------------------------

fn render_dirty_instances(world: &mut World) {
    for _ in 0..32 {
        seed_reactive_markers(world);
        mark_dependency_dirty(world);

        let mut query = world.query_filtered::<Entity, With<RenderDirty>>();
        let dirty: Vec<Entity> = query.iter(world).collect();
        if dirty.is_empty() {
            break;
        }
        for entity in dirty {
            world.entity_mut(entity).remove::<RenderDirty>();
            render_instance(world, entity);
        }
    }
}

/// Turn freshly-applied [`ReactiveMarker`]s into live reactive instances.
fn seed_reactive_markers(world: &mut World) {
    let mut query =
        world.query_filtered::<Entity, (With<ReactiveMarker>, Without<ReactiveRender>)>();
    let pending: Vec<Entity> = query.iter(world).collect();
    for entity in pending {
        let Some((register, deps)) = world
            .get::<ReactiveMarker>(entity)
            .map(|m| (m.register.clone(), m.deps.clone()))
        else {
            continue;
        };
        let system = register(world);
        let mut entity_mut = world.entity_mut(entity);
        entity_mut.remove::<ReactiveMarker>();
        entity_mut.insert((
            ReactiveRender { system, deps },
            ReactiveState::default(),
            RenderDirty,
        ));
    }
}

/// Dirty any instance whose reactive `Res`/`Query` reads changed since it last rendered.
fn mark_dependency_dirty(world: &mut World) {
    let mut query = world.query_filtered::<Entity, (With<ReactiveRender>, With<LastRender>)>();
    let instances: Vec<Entity> = query.iter(world).collect();
    let mut to_dirty = Vec::new();
    for entity in instances {
        let Some(deps) = world.get::<ReactiveRender>(entity).map(|r| r.deps.clone()) else {
            continue;
        };
        let Some(last) = world.get::<LastRender>(entity).map(|l| l.0) else {
            continue;
        };
        if deps.iter().any(|check| check(world, last)) {
            to_dirty.push(entity);
        }
    }
    for entity in to_dirty {
        world.entity_mut(entity).insert(RenderDirty);
    }
}

fn render_instance(world: &mut World, entity: Entity) {
    let id = match world.get::<ReactiveRender>(entity) {
        Some(render) => render.system,
        None => return,
    };

    let state = std::mem::take(&mut *world.get_mut::<ReactiveState>(entity).unwrap());
    world.insert_resource(CurrentRender {
        entity,
        state,
        cursor: 0,
        pending: Vec::new(),
    });

    let scene = match world.run_system(id) {
        Ok(scene) => scene,
        Err(err) => {
            error!("reactive render failed for {entity}: {err}");
            world.remove_resource::<CurrentRender>();
            return;
        }
    };

    let CurrentRender { state, pending, .. } = world.remove_resource::<CurrentRender>().unwrap();
    *world.get_mut::<ReactiveState>(entity).unwrap() = state;

    let resolved = {
        let asset_server = world.resource::<AssetServer>().clone();
        let patches = world.resource::<Assets<ScenePatch>>();
        match ResolvedSceneRoot::resolve(scene, &asset_server, patches) {
            Ok(root) => root,
            Err(err) => {
                error!("reactive resolve failed for {entity}: {err}");
                return;
            }
        }
    };
    reconcile_node(world, entity, &resolved.scene);

    // Run effects after commit.
    for effect in pending {
        let old = world
            .get_mut::<ReactiveState>(entity)
            .and_then(|mut s| match s.slots.get_mut(effect.slot) {
                Some(Slot::Effect { cleanup, .. }) => {
                    Some(core::mem::replace(cleanup, Cleanup::none()))
                }
                _ => None,
            });
        if let Some(cleanup) = old {
            cleanup.run(world);
        }
        let new_cleanup = (effect.run)(world);
        if let Some(mut s) = world.get_mut::<ReactiveState>(entity) {
            if let Some(Slot::Effect { cleanup, .. }) = s.slots.get_mut(effect.slot) {
                *cleanup = new_cleanup;
            }
        }
    }

    let tick = world.change_tick();
    world.entity_mut(entity).insert(LastRender(tick));
}

/// Reconcile one node of a resolved `bsn!` tree onto `entity`.
fn reconcile_node(world: &mut World, entity: Entity, resolved: &ResolvedScene) {
    // 1. Apply this node's component templates. `set_if_neq` (no tick bump for unchanged values)
    //    is applied automatically in `bevy_scene` via specialization.
    {
        let mut refs = SceneEntityReferences::default();
        let mut entity_mut = world.entity_mut(entity);
        let mut ctx = TemplateContext::new(&mut entity_mut, &mut refs);
        if let Err(err) = resolved.apply_component_templates(&mut ctx) {
            error!("reactive component apply failed for {entity}: {err}");
        }
    }

    // 2. Reactive child? (A `{scene()}` include added a `ReactiveMarker`.)
    if world.get::<ReactiveMarker>(entity).is_some() {
        if world.get::<ReactiveRender>(entity).is_some() {
            // Reused reactive instance: props were just re-applied; drop the redundant marker
            // and re-render with the new props (state preserved).
            let mut entity_mut = world.entity_mut(entity);
            entity_mut.remove::<ReactiveMarker>();
            entity_mut.insert(RenderDirty);
        }
        // On first mount, leave the marker — `seed_reactive_markers` converts it next pass.
        return;
    }

    // 3. Per-node component removal (required-component-aware).
    let this: HashSet<ComponentId> = resolved
        .component_type_ids()
        .iter()
        .filter_map(|type_id| world.components().get_id(*type_id))
        .collect();
    let previous = world
        .get::<ManagedComponents>(entity)
        .map(|m| m.0.clone())
        .unwrap_or_default();
    let mut required: HashSet<ComponentId> = HashSet::new();
    for component_id in &this {
        if let Some(info) = world.components().get_info(*component_id) {
            required.extend(info.required_components().iter_ids());
        }
    }
    for component_id in previous.difference(&this) {
        if !required.contains(component_id) {
            world.entity_mut(entity).remove_by_id(*component_id);
        }
    }
    world.entity_mut(entity).insert(ManagedComponents(this));

    // 4. Observers: keep the ones unchanged across renders, attach only newly-added observers,
    //    and despawn only removed ones. A stable observer set touches nothing (no churn).
    //    Identity is positional: the i-th `on(...)` keeps its observer entity across renders, so a
    //    handler's captured values are fixed at first attach — handlers should read live state.
    {
        let previous: Vec<Entity> = world
            .get::<ReactiveObservers>(entity)
            .map(|o| o.0.clone())
            .unwrap_or_default();
        let count = resolved.bundle_template_count();
        let keep = previous.len().min(count);

        // Despawn observers that are no longer rendered.
        for &observer in &previous[keep..] {
            if let Ok(observer_mut) = world.get_entity_mut(observer) {
                observer_mut.despawn();
            }
        }

        // Attach only the newly-added observers (indices `keep..count`); keep the rest untouched.
        let mut observers: Vec<Entity> = previous[..keep].to_vec();
        for index in keep..count {
            let before: HashSet<Entity> = world
                .get::<ObservedBy>(entity)
                .map(|o| o.get().iter().copied().collect())
                .unwrap_or_default();
            {
                let mut refs = SceneEntityReferences::default();
                let mut entity_mut = world.entity_mut(entity);
                let mut ctx = TemplateContext::new(&mut entity_mut, &mut refs);
                if let Err(err) = resolved.apply_bundle_template(index, &mut ctx) {
                    error!("reactive observer apply failed for {entity}: {err}");
                }
            }
            if let Some(observed) = world.get::<ObservedBy>(entity) {
                observers.extend(observed.get().iter().copied().filter(|e| !before.contains(e)));
            }
        }
        world.entity_mut(entity).insert(ReactiveObservers(observers));
    }

    // 5. Children.
    reconcile_children(world, entity, resolved);
}

fn reconcile_children(world: &mut World, parent: Entity, resolved: &ResolvedScene) {
    let child_scenes = resolved.related_scenes_for::<ChildOf>().unwrap_or(&[]);

    let old: Vec<Entity> = world
        .get::<Children>(parent)
        .map(|c| c.iter().collect())
        .unwrap_or_default();
    let mut by_key: HashMap<u64, Entity> = HashMap::new();
    let mut unkeyed: VecDeque<Entity> = VecDeque::new();
    for child in old {
        match world.get::<ReactiveKey>(child) {
            Some(key) => {
                by_key.insert(key.0, child);
            }
            None => unkeyed.push_back(child),
        }
    }

    let mut new_order = Vec::with_capacity(child_scenes.len());
    for child_scene in child_scenes {
        let entity = match extract_key(child_scene) {
            Some(key) => by_key
                .remove(&key)
                .unwrap_or_else(|| world.spawn_empty().id()),
            None => unkeyed
                .pop_front()
                .unwrap_or_else(|| world.spawn_empty().id()),
        };
        reconcile_node(world, entity, child_scene);
        new_order.push(entity);
    }

    for (_, entity) in by_key {
        run_unmount_cleanups(world, entity);
        world.entity_mut(entity).despawn();
    }
    for entity in unkeyed {
        run_unmount_cleanups(world, entity);
        world.entity_mut(entity).despawn();
    }

    if new_order.is_empty() {
        world.entity_mut(parent).remove::<Children>();
    } else {
        world.entity_mut(parent).replace_children(&new_order);
    }
}

/// Read a child scene's `ReactiveKey` value (if any) directly from its resolved component
/// template — no entity is needed. A `ReactiveKey(n)` resolves to a canonical template whose
/// concrete type is `ReactiveKey`, so we can downcast and read it. Returns `None` if unkeyed.
fn extract_key(scene: &ResolvedScene) -> Option<u64> {
    scene
        .get_direct_erased_template(std::any::TypeId::of::<ReactiveKey>())?
        .as_any()
        .downcast_ref::<ReactiveKey>()
        .map(|key| key.0)
}

/// Run the effect cleanups for an entity and its descendants before it is despawned.
fn run_unmount_cleanups(world: &mut World, entity: Entity) {
    let children: Vec<Entity> = world
        .get::<Children>(entity)
        .map(|c| c.iter().collect())
        .unwrap_or_default();
    for child in children {
        run_unmount_cleanups(world, child);
    }
    let cleanups: Vec<Cleanup> = world
        .get_mut::<ReactiveState>(entity)
        .map(|mut state| {
            state
                .slots
                .iter_mut()
                .filter_map(|slot| match slot {
                    Slot::Effect { cleanup, .. } => {
                        Some(core::mem::replace(cleanup, Cleanup::none()))
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    for cleanup in cleanups {
        cleanup.run(world);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::AssetPlugin, scene::ScenePlugin};
    use std::sync::atomic::{AtomicI32, AtomicU32, Ordering::SeqCst};

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), ScenePlugin))
            .add_systems(Update, render_dirty_instances);
        app
    }

    fn fire_i32(app: &mut App, setter: Setter<i32>, delta: i32) {
        let world = app.world_mut();
        let mut commands = world.commands();
        setter.update(&mut commands, move |x| *x += delta);
        drop(commands);
        world.flush();
    }

    fn fire_bool(app: &mut App, setter: Setter<bool>, value: bool) {
        let world = app.world_mut();
        let mut commands = world.commands();
        setter.update(&mut commands, move |b| *b = value);
        drop(commands);
        world.flush();
    }

    // --- nested state preserved across a parent re-render ------------------

    static T1_VALUE: AtomicI32 = AtomicI32::new(-1);
    static T1_SETTER: Mutex<Option<Setter<i32>>> = Mutex::new(None);
    static T1_ENTITY: Mutex<Option<Entity>> = Mutex::new(None);

    #[reactive_scene]
    fn t1_child(mut hooks: ReactiveHooks) -> impl Scene {
        let (n, set) = hooks.use_state(|| 0i32);
        T1_VALUE.store(n, SeqCst);
        *T1_SETTER.lock().unwrap() = Some(set);
        *T1_ENTITY.lock().unwrap() = Some(hooks.entity());
        bsn! { Name({ format!("child:{n}") }) }
    }

    #[reactive_scene]
    fn t1_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (_t, _s) = hooks.use_state(|| 0i32);
        bsn! { Node Children [ ( {t1_child()} ) ] }
    }

    #[test]
    fn nested_state_preserved_across_parent_rerender() {
        let mut app = test_app();
        let root = app.world_mut().spawn_scene(t1_root()).unwrap().id();
        app.update();
        assert_eq!(T1_VALUE.load(SeqCst), 0);
        let original = T1_ENTITY.lock().unwrap().unwrap();

        let setter = T1_SETTER.lock().unwrap().unwrap();
        fire_i32(&mut app, setter, 1);
        app.update();
        assert_eq!(T1_VALUE.load(SeqCst), 1);

        app.world_mut().entity_mut(root).insert(RenderDirty);
        app.update();
        assert_eq!(T1_VALUE.load(SeqCst), 1, "nested state survives parent re-render");
        assert_eq!(T1_ENTITY.lock().unwrap().unwrap(), original, "child entity reused");
    }

    // --- per-node component removal ----------------------------------------

    #[derive(Component, Default, Clone)]
    struct Flag;

    static T_REMOVE_SETTER: Mutex<Option<Setter<bool>>> = Mutex::new(None);

    #[reactive_scene]
    fn removal_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (flagged, set) = hooks.use_state(|| true);
        *T_REMOVE_SETTER.lock().unwrap() = Some(set);
        let flag = flagged.then(|| bsn! { Flag });
        bsn! { Node Children [ ( {flag} ) ] }
    }

    #[test]
    fn per_node_component_removal() {
        let mut app = test_app();
        app.world_mut().spawn_scene(removal_root()).unwrap();
        app.update();

        let child = {
            let world = app.world_mut();
            let mut q = world.query_filtered::<Entity, With<Flag>>();
            q.iter(world).next()
        };
        assert!(child.is_some(), "Flag present while flagged == true");
        let child = child.unwrap();

        let setter = T_REMOVE_SETTER.lock().unwrap().unwrap();
        fire_bool(&mut app, setter, false);
        app.update();
        assert!(
            app.world().get::<Flag>(child).is_none(),
            "Flag removed when no longer rendered"
        );
    }

    // --- required-component-aware removal ----------------------------------

    #[derive(Component, Default, Clone)]
    struct Req;

    #[derive(Component, Default, Clone)]
    #[require(Req)]
    struct Holder;

    static T_REQ_SETTER: Mutex<Option<Setter<bool>>> = Mutex::new(None);

    #[reactive_scene]
    fn req_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (explicit, set) = hooks.use_state(|| true);
        *T_REQ_SETTER.lock().unwrap() = Some(set);
        // (`{req}` must precede `Holder` so it isn't parsed as `Holder { req }` struct fields.)
        let req = explicit.then(|| bsn! { Req });
        bsn! { Node Children [ ( {req} Holder ) ] }
    }

    #[test]
    fn required_component_not_removed() {
        let mut app = test_app();
        app.world_mut().spawn_scene(req_root()).unwrap();
        app.update();

        let child = {
            let world = app.world_mut();
            let mut q = world.query_filtered::<Entity, With<Holder>>();
            q.iter(world).next().unwrap()
        };
        assert!(app.world().get::<Req>(child).is_some());

        let setter = T_REQ_SETTER.lock().unwrap().unwrap();
        fire_bool(&mut app, setter, false);
        app.update();
        assert!(
            app.world().get::<Req>(child).is_some(),
            "required component kept despite being dropped from the scene"
        );
    }

    // --- nested props propagate to a reused child --------------------------

    static T_PROP_SEEN: Mutex<Option<&'static str>> = Mutex::new(None);
    static T_PROP_SETTER: Mutex<Option<Setter<u32>>> = Mutex::new(None);

    #[reactive_scene]
    fn prop_child(labels: Query<&TallyLabel>, hooks: ReactiveHooks) -> impl Scene {
        let label = labels.get(hooks.entity()).map(|l| l.0).unwrap_or("?");
        *T_PROP_SEEN.lock().unwrap() = Some(label);
        bsn! { Name({ format!("label:{label}") }) }
    }

    #[reactive_scene]
    fn prop_parent(mut hooks: ReactiveHooks) -> impl Scene {
        let (tick, set) = hooks.use_state(|| 0u32);
        *T_PROP_SETTER.lock().unwrap() = Some(set);
        let label = if tick == 0 { "A" } else { "B" };
        bsn! { Node Children [ ( {prop_child()} TallyLabel({ label }) ) ] }
    }

    #[test]
    fn nested_props_propagate_on_parent_rerender() {
        let mut app = test_app();
        app.world_mut().spawn_scene(prop_parent()).unwrap();
        app.update();
        assert_eq!(*T_PROP_SEEN.lock().unwrap(), Some("A"));

        let setter = T_PROP_SETTER.lock().unwrap().unwrap();
        {
            let world = app.world_mut();
            let mut commands = world.commands();
            setter.update(&mut commands, |t| *t = 1);
            drop(commands);
            world.flush();
        }
        app.update();
        assert_eq!(
            *T_PROP_SEEN.lock().unwrap(),
            Some("B"),
            "nested child re-rendered with the updated prop"
        );
    }

    // --- keyed children: state follows the key across a reorder ------------

    static KEYED_A_VALUE: AtomicI32 = AtomicI32::new(-1);
    static KEYED_A_SETTER: Mutex<Option<Setter<i32>>> = Mutex::new(None);
    static KEYED_A_ENTITY: Mutex<Option<Entity>> = Mutex::new(None);
    static KEYED_LIST_SETTER: Mutex<Option<Setter<bool>>> = Mutex::new(None);

    #[reactive_scene]
    fn keyed_item(labels: Query<&TallyLabel>, mut hooks: ReactiveHooks) -> impl Scene {
        let label = labels.get(hooks.entity()).map(|l| l.0).unwrap_or("?");
        let (n, set) = hooks.use_state(|| 0i32);
        if label == "A" {
            KEYED_A_VALUE.store(n, SeqCst);
            *KEYED_A_SETTER.lock().unwrap() = Some(set);
            *KEYED_A_ENTITY.lock().unwrap() = Some(hooks.entity());
        }
        bsn! { Name({ format!("{label}:{n}") }) }
    }

    #[reactive_scene]
    fn keyed_list(mut hooks: ReactiveHooks) -> impl Scene {
        let (reversed, set) = hooks.use_state(|| false);
        *KEYED_LIST_SETTER.lock().unwrap() = Some(set);
        let a: Box<dyn Scene> = Box::new(bsn! { {keyed_item()} TallyLabel("A") ReactiveKey(1) });
        let b: Box<dyn Scene> = Box::new(bsn! { {keyed_item()} TallyLabel("B") ReactiveKey(2) });
        let items: Vec<Box<dyn Scene>> = if reversed { vec![b, a] } else { vec![a, b] };
        bsn! { Node Children [ {items} ] }
    }

    #[test]
    fn keyed_children_reuse_across_reorder() {
        let mut app = test_app();
        app.world_mut().spawn_scene(keyed_list()).unwrap();
        app.update();
        assert_eq!(KEYED_A_VALUE.load(SeqCst), 0);
        let a_entity = KEYED_A_ENTITY.lock().unwrap().unwrap();

        let setter = KEYED_A_SETTER.lock().unwrap().unwrap();
        fire_i32(&mut app, setter, 1);
        app.update();
        assert_eq!(KEYED_A_VALUE.load(SeqCst), 1);

        let list_setter = KEYED_LIST_SETTER.lock().unwrap().unwrap();
        fire_bool(&mut app, list_setter, true);
        app.update();
        assert_eq!(KEYED_A_VALUE.load(SeqCst), 1, "keyed child kept its state across reorder");
        assert_eq!(
            KEYED_A_ENTITY.lock().unwrap().unwrap(),
            a_entity,
            "keyed child reused its entity (not matched positionally)"
        );
    }

    // --- unmount runs effect cleanups --------------------------------------

    static UNMOUNT_CLEANED: AtomicU32 = AtomicU32::new(0);
    static UNMOUNT_SETTER: Mutex<Option<Setter<bool>>> = Mutex::new(None);

    #[reactive_scene]
    fn unmount_child(mut hooks: ReactiveHooks) -> impl Scene {
        hooks.use_effect((), |_world| {
            Cleanup::new(|_world| {
                UNMOUNT_CLEANED.fetch_add(1, SeqCst);
            })
        });
        bsn! { Name("unmount-child") }
    }

    #[reactive_scene]
    fn unmount_parent(mut hooks: ReactiveHooks) -> impl Scene {
        let (show, set) = hooks.use_state(|| true);
        *UNMOUNT_SETTER.lock().unwrap() = Some(set);
        let children: Vec<Box<dyn Scene>> = if show {
            vec![Box::new(bsn! { {unmount_child()} })]
        } else {
            vec![]
        };
        bsn! { Node Children [ {children} ] }
    }

    #[test]
    fn unmount_runs_effect_cleanup() {
        let mut app = test_app();
        app.world_mut().spawn_scene(unmount_parent()).unwrap();
        app.update();
        assert_eq!(UNMOUNT_CLEANED.load(SeqCst), 0);

        let setter = UNMOUNT_SETTER.lock().unwrap().unwrap();
        fire_bool(&mut app, setter, false);
        app.update();
        assert_eq!(UNMOUNT_CLEANED.load(SeqCst), 1, "unmount ran the effect cleanup");
    }

    // --- observers ---------------------------------------------------------

    #[derive(EntityEvent)]
    struct Ping(Entity);

    #[derive(Resource, Default)]
    struct Hits(u32);

    #[derive(Component, Default, Clone)]
    struct ButtonMarker;

    #[reactive_scene]
    fn t2_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (_t, _s) = hooks.use_state(|| 0i32);
        bsn! {
            Node
            Children [
                ( ButtonMarker on(|_: On<Ping>, mut hits: ResMut<Hits>| hits.0 += 1) )
            ]
        }
    }

    #[test]
    fn observer_fires_once_across_rerenders() {
        let mut app = test_app();
        app.insert_resource(Hits(0));
        let root = app.world_mut().spawn_scene(t2_root()).unwrap().id();
        app.update();

        for _ in 0..3 {
            app.world_mut().entity_mut(root).insert(RenderDirty);
            app.update();
        }

        let child = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<Entity, With<ButtonMarker>>();
            query.iter(world).next().unwrap()
        };
        app.world_mut().trigger(Ping(child));
        assert_eq!(app.world().resource::<Hits>().0, 1, "observer fires exactly once");
    }

    // #3: a conditional observer is removed when the node stops rendering it.
    static OBS_SHOW_SETTER: Mutex<Option<Setter<bool>>> = Mutex::new(None);

    #[reactive_scene]
    fn conditional_observer_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (show, set) = hooks.use_state(|| true);
        *OBS_SHOW_SETTER.lock().unwrap() = Some(set);
        let handler = show.then(|| bsn! { on(|_: On<Ping>, mut hits: ResMut<Hits>| hits.0 += 1) });
        bsn! { Node Children [ ( {handler} ButtonMarker ) ] }
    }

    #[test]
    fn observer_removed_when_no_longer_rendered() {
        let mut app = test_app();
        app.insert_resource(Hits(0));
        app.world_mut().spawn_scene(conditional_observer_root()).unwrap();
        app.update();

        let child = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<Entity, With<ButtonMarker>>();
            query.iter(world).next().unwrap()
        };
        app.world_mut().trigger(Ping(child));
        assert_eq!(app.world().resource::<Hits>().0, 1, "observer fires while present");

        // Stop rendering the observer -> it must be removed and no longer fire.
        let setter = OBS_SHOW_SETTER.lock().unwrap().unwrap();
        fire_bool(&mut app, setter, false);
        app.update();
        app.world_mut().flush();
        app.world_mut().trigger(Ping(child));
        assert_eq!(
            app.world().resource::<Hits>().0,
            1,
            "removed observer does not fire again"
        );
    }

    // --- effect runs on dep change with cleanup ----------------------------

    static T3_RUNS: AtomicU32 = AtomicU32::new(0);
    static T3_CLEANS: AtomicU32 = AtomicU32::new(0);
    static T3_SETTER: Mutex<Option<Setter<i32>>> = Mutex::new(None);

    #[reactive_scene]
    fn t3_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (dep, set) = hooks.use_state(|| 0i32);
        *T3_SETTER.lock().unwrap() = Some(set);
        hooks.use_effect(dep, |_world| {
            T3_RUNS.fetch_add(1, SeqCst);
            Cleanup::new(|_world| {
                T3_CLEANS.fetch_add(1, SeqCst);
            })
        });
        bsn! { Node }
    }

    #[test]
    fn effect_runs_on_dep_change_with_cleanup() {
        let mut app = test_app();
        let root = app.world_mut().spawn_scene(t3_root()).unwrap().id();
        app.update();
        assert_eq!(T3_RUNS.load(SeqCst), 1);
        assert_eq!(T3_CLEANS.load(SeqCst), 0);

        let setter = T3_SETTER.lock().unwrap().unwrap();
        fire_i32(&mut app, setter, 1);
        app.update();
        assert_eq!(T3_RUNS.load(SeqCst), 2);
        assert_eq!(T3_CLEANS.load(SeqCst), 1);

        app.world_mut().entity_mut(root).insert(RenderDirty);
        app.update();
        assert_eq!(T3_RUNS.load(SeqCst), 2, "effect skipped when deps unchanged");
    }

    // --- reactive Res read re-renders --------------------------------------

    static T4_RENDERS: AtomicU32 = AtomicU32::new(0);
    static T4_SEEN: AtomicI32 = AtomicI32::new(-1);

    #[reactive_scene]
    fn t4_root(score: Res<Score>) -> impl Scene {
        T4_RENDERS.fetch_add(1, SeqCst);
        T4_SEEN.store(score.0, SeqCst);
        bsn! { Node }
    }

    #[test]
    fn reactive_res_read_rerenders() {
        let mut app = test_app();
        app.insert_resource(Score(0));
        app.world_mut().spawn_scene(t4_root()).unwrap();
        app.update();
        assert_eq!(T4_RENDERS.load(SeqCst), 1);
        assert_eq!(T4_SEEN.load(SeqCst), 0);

        app.world_mut().resource_mut::<Score>().0 = 5;
        app.update();
        assert_eq!(T4_RENDERS.load(SeqCst), 2);
        assert_eq!(T4_SEEN.load(SeqCst), 5);

        app.update();
        assert_eq!(T4_RENDERS.load(SeqCst), 2, "no re-render when Score unchanged");
    }

    // --- reactive Query read re-renders on component change ----------------

    static T6_TOTAL: AtomicU32 = AtomicU32::new(0);
    static T6_RENDERS: AtomicU32 = AtomicU32::new(0);

    #[reactive_scene]
    fn t6_root(healths: Query<&Health>) -> impl Scene {
        T6_RENDERS.fetch_add(1, SeqCst);
        T6_TOTAL.store(healths.iter().map(|h| h.0).sum(), SeqCst);
        bsn! { Node }
    }

    #[test]
    fn reactive_query_rerenders_on_component_change() {
        let mut app = test_app();
        let h = app.world_mut().spawn(Health(40)).id();
        app.world_mut().spawn(Health(60));
        app.world_mut().spawn_scene(t6_root()).unwrap();
        app.update();
        assert_eq!(T6_TOTAL.load(SeqCst), 100);
        assert_eq!(T6_RENDERS.load(SeqCst), 1);

        app.world_mut().get_mut::<Health>(h).unwrap().0 = 90;
        app.update();
        assert_eq!(T6_TOTAL.load(SeqCst), 150);
        assert_eq!(T6_RENDERS.load(SeqCst), 2);
    }

    // --- set_if_neq: unchanged re-applied components don't bump change ticks ---

    #[derive(Component, PartialEq, Clone, Default)]
    struct Probe(i32);

    static PROBE_SETTER: Mutex<Option<Setter<i32>>> = Mutex::new(None);

    #[reactive_scene]
    fn probe_root(mut hooks: ReactiveHooks) -> impl Scene {
        let (n, set) = hooks.use_state(|| 0i32);
        *PROBE_SETTER.lock().unwrap() = Some(set);
        bsn! { Probe({ n }) }
    }

    #[test]
    fn set_if_neq_skips_unchanged_writes() {
        let mut app = test_app();
        let root = app.world_mut().spawn_scene(probe_root()).unwrap().id();
        app.update();
        let tick0 = app
            .world()
            .entity(root)
            .get_change_ticks::<Probe>()
            .unwrap()
            .changed;

        // Re-render with the SAME value -> set_if_neq skips the write (tick unchanged).
        app.world_mut().entity_mut(root).insert(RenderDirty);
        app.update();
        let tick1 = app
            .world()
            .entity(root)
            .get_change_ticks::<Probe>()
            .unwrap()
            .changed;
        assert_eq!(tick0.get(), tick1.get(), "unchanged value must not bump the change tick");

        // Change the value -> the write happens (tick advances).
        let setter = PROBE_SETTER.lock().unwrap().unwrap();
        fire_i32(&mut app, setter, 1);
        app.update();
        let tick2 = app
            .world()
            .entity(root)
            .get_change_ticks::<Probe>()
            .unwrap()
            .changed;
        assert_ne!(tick1.get(), tick2.get(), "changed value must bump the change tick");
    }
}
