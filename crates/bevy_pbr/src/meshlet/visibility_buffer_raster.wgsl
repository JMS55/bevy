#import bevy_pbr::{
    meshlet_bindings::{
        meshlet_thread_meshlet_ids,
        meshlets,
        meshlet_vertex_ids,
        meshlet_vertex_data,
        meshlet_thread_instance_ids,
        meshlet_instance_uniforms,
        meshlet_instance_material_ids,
        view,
        get_meshlet_index,
        unpack_meshlet_vertex,
    },
    mesh_functions::mesh_position_local_to_world,
}
#import bevy_render::maths::affine3_to_square

/// Vertex/fragment shader for rasterizing meshlets into a visibility buffer.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
    @location(0) @interpolate(flat) visibility: u32,
    @location(1) @interpolate(flat) material_depth: u32,
#endif
#ifdef DEPTH_CLAMP_ORTHO
    @location(0) unclamped_clip_depth: f32,
#endif
}

#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
struct FragmentOutput {
    @location(0) visibility: vec4<u32>,
    @location(1) material_depth: vec4<u32>,
}
#endif

@vertex
fn vertex(@builtin(vertex_index) cull_output: u32) -> VertexOutput {
    let thread_id = cull_output >> 8u;
    let meshlet_id = meshlet_thread_meshlet_ids[thread_id];
    let meshlet = meshlets[meshlet_id];
    let index_id = extractBits(cull_output, 0u, 8u);
    let index = get_meshlet_index(meshlet.start_index_id + index_id);
    let vertex_id = meshlet_vertex_ids[meshlet.start_vertex_id + index];
    let vertex = unpack_meshlet_vertex(meshlet_vertex_data[vertex_id]);
    let instance_id = meshlet_thread_instance_ids[thread_id];
    let instance_uniform = meshlet_instance_uniforms[instance_id];

    let model = affine3_to_square(instance_uniform.model);
    let world_position = mesh_position_local_to_world(model, vec4(vertex.position, 1.0));
    var clip_position = view.view_proj * vec4(world_position.xyz, 1.0);
#ifdef DEPTH_CLAMP_ORTHO
    let unclamped_clip_depth = clip_position.z;
    clip_position.z = min(clip_position.z, 1.0);
#endif

    return VertexOutput(
        clip_position,
#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
        (thread_id << 7u) | (index_id / 3u),
        meshlet_instance_material_ids[instance_id],
#endif
#ifdef DEPTH_CLAMP_ORTHO
        unclamped_clip_depth,
#endif
    );
}

#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
@fragment
fn fragment(vertex_output: VertexOutput) -> FragmentOutput {
    return FragmentOutput(
        vec4(vertex_output.visibility, 0u, 0u, 0u),
        vec4(vertex_output.material_depth, 0u, 0u, 0u),
    );
}
#endif

#ifdef DEPTH_CLAMP_ORTHO
@fragment
fn fragment(vertex_output: VertexOutput) -> @builtin(frag_depth) f32 {
    return vertex_output.unclamped_clip_depth;
}
#endif
