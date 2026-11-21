---
title: "UI blends in sRGB space, and `UiMaterial` shaders must encode their output"
pull_requests: [TODO]
---

UI now alpha blends in sRGB space on every camera, as web browsers and most image editors do. Previously it blended in linear space, so semi-transparent UI looked lighter than the same colors on the web: black at 50% alpha over white was a light gray instead of a mid gray. Text antialiasing changes the same way: dark text on a light background looks bolder, and light text on a dark background looks thinner.

To get this, UI shaders output sRGB-encoded colors, and UI renders to a target that stores those values without converting them. Custom `UiMaterial` fragment shaders must now pass their final linear color through `encode_output`. Without it, their colors come out too dark.

```wesl
import bevy_ui_render::ui::encode_output;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let color = ...;
    return encode_output(color);
}
```

UI pipelines no longer render to the camera's `ExtractedView::target_format`. Custom pipelines that draw into the `TransparentUi` phase should use the format from the camera's new `UiRenderTarget` component, and encode their output with `encode_output` too.

On cameras whose main texture has no sRGB format, such as cameras with `Hdr`, the UI pass now adds a fullscreen pass on frames that draw UI, to convert the main texture to sRGB first. Upscaling then decodes it. When another camera renders to the same main texture afterwards, a second pass converts it back instead.
