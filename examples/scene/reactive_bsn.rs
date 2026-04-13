//! This example demonstrates reactive BSN scenes that re-render every frame,
//! similar to React components. A counter UI updates automatically when its
//! state changes.

use bevy::{ecs::template::template, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

#[derive(Component, Default, Clone, PartialEq)]
struct Counter(usize);

fn setup(world: &mut World) {
    world.spawn(Camera2d);
    world.spawn_reactive_scene(counter_ui);
}

fn counter_ui(ctx: &SceneContext) -> Box<dyn Scene> {
    let count = ctx.use_state_or(Counter(0)).0;

    Box::new(bsn! {
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(10.),
        }
        Children [
            (
                #CounterLabel
                Node {
                    padding: UiRect::all(Val::Px(10.)),
                }
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2))
                Children [(
                    Text({format!("Count: {count}")})
                    template(|context| {
                        Ok(TextFont {
                            font: context
                                .resource::<AssetServer>()
                                .load("fonts/FiraSans-Bold.ttf").into(),
                            font_size: FontSize::Px(40.0),
                            ..default()
                        })
                    })
                    TextColor(Color::WHITE)
                )]
            ),
            (
                #ButtonRow
                Node {
                    column_gap: Val::Px(10.),
                }
                Children [
                    (
                        counter_button("+")
                        on(|_event: On<Pointer<Press>>, mut query: Query<&mut Counter>| {
                            for mut counter in &mut query {
                                counter.0 += 1;
                            }
                        })
                    ),
                    (
                        counter_button("-")
                        on(|_event: On<Pointer<Press>>, mut query: Query<&mut Counter>| {
                            for mut counter in &mut query {
                                counter.0 = counter.0.saturating_sub(1);
                            }
                        })
                    ),
                ]
            ),
        ]
    })
}

fn counter_button(label: &'static str) -> impl Scene {
    bsn! {
        Button
        Node {
            width: Val::Px(100.0),
            height: Val::Px(50.0),
            border: UiRect::all(Val::Px(3.0)),
            border_radius: BorderRadius::MAX,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
        }
        BorderColor::from(Color::BLACK)
        BackgroundColor(Color::srgb(0.15, 0.15, 0.15))
        Children [(
            Text(label)
            template(|context| {
                Ok(TextFont {
                    font: context
                        .resource::<AssetServer>()
                        .load("fonts/FiraSans-Bold.ttf").into(),
                    font_size: FontSize::Px(24.0),
                    ..default()
                })
            })
            TextColor(Color::srgb(0.9, 0.9, 0.9))
        )]
    }
}
