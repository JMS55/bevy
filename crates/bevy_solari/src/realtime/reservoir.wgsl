// https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf

#define_import_path bevy_solari::reservoir

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::rand_f
#import bevy_solari::sampling::{LightSample, calculate_light_contribution}
