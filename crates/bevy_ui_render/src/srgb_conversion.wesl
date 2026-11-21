// Converts a camera's main texture to sRGB-encoded values before the UI pass,
// and back afterwards, so the hardware blends UI in sRGB space.
//
// The main texture stores linear RGB, or Oklab when the `OKLAB` shader def is set.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::color_operations::{linear_to_srgb, srgb_to_linear};
@if(OKLAB)
import bevy_render::color_operations::{linear_rgb_to_oklab, oklab_to_linear_rgb};

@group(0) @binding(0) var source: texture_2d<f32>;

@fragment
fn to_srgb(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureLoad(source, vec2<i32>(in.position.xy), 0);
    var rgb = color.rgb;
@if(OKLAB)
    rgb = oklab_to_linear_rgb(rgb);
    return vec4(linear_to_srgb(rgb), color.a);
}

@fragment
fn from_srgb(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureLoad(source, vec2<i32>(in.position.xy), 0);
    var rgb = srgb_to_linear(color.rgb);
@if(OKLAB)
    rgb = linear_rgb_to_oklab(rgb);
    return vec4(rgb, color.a);
}
