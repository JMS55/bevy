pub mod extract_component;
pub mod extract_instances;
mod extract_param;
pub mod extract_resource;

pub use extract_param::Extract;

use bevy_ecs::schedule::ScheduleLabel;

/// Schedule which extract data from the main world and inserts it into the render world.
///
/// This step should be kept as short as possible to increase the "pipelining potential" for
/// running the next frame while rendering the current frame.
///
/// This schedule is run on the main world, but its buffers are not applied
/// until it is returned to the render world.
#[derive(ScheduleLabel, PartialEq, Eq, Debug, Clone, Hash)]
pub struct ExtractSchedule;
