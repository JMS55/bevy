@group(1) @binding(23) var<storage, read_write> foo: vec3<f32>;
@group(1) @binding(24) var<storage, read_write> bar: vec3<f32>;

@compute @workgroup_size(1, 1, 1)
fn combine() {
    bar = mix(bar, foo, 0.1);
}
