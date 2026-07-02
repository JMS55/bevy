// Line-by-line GLSL translation of bevy_solari's restir.wgsl and the functions it
// transitively reaches. Compiled to SPIR-V with function/source debug info (glslang -gVS)
// and loaded via wgpu SPIR-V passthrough for source-level debugging in Nsight Graphics.
//
// This is the shared translation unit. It is #included by restir_initial.comp,
// restir_temporal.comp and restir_spatial_and_shade.comp *after* their #version /
// #extension / (optional) #define DLSS_RR_GUIDE_BUFFERS lines.
//
// Bindings mirror the two (three, with DLSS) bind group layouts built in realtime/node.rs
// and raytracing_scene_bindings.wgsl. Struct layouts match the WGSL/Rust definitions:
// Reservoir = 48B, ResolvedLightSamplePacked = 24B, LightSample = 8B (see prepare.rs).

// ============================================================================
// Constants
// ============================================================================
const float PI = 3.141592653589793;
const float PI_2 = 6.283185307179586;
const float HALF_PI = 1.57079632679;

const float RAY_T_MIN = 0.001;
const float RAY_T_MAX = 100000.0;
const uint RAY_NO_CULL = 0xFFu;

const float MIRROR_ROUGHNESS_THRESHOLD = 0.001;

const uint NULL_LIGHT_ID = 0xFFFFFFFFu;
const uint LIGHT_NOT_PRESENT_THIS_FRAME = 0xFFFFFFFFu;
const uint TEXTURE_MAP_NONE = 0xFFFFFFFFu;

const uint LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0u;
const uint LIGHT_SOURCE_KIND_DIRECTIONAL = 1u;

const float U12MAXF = 4095.0;

// World cache (WORLD_CACHE_SIZE baked in: WORLD_CACHE_SIZE = 2^20, see prepare.rs)
const uint WORLD_CACHE_SIZE = 1048576u;
const uint WORLD_CACHE_CELL_LIFETIME = 10u;
const uint WORLD_CACHE_MAX_SEARCH_STEPS = 3u;
const uint WORLD_CACHE_EMPTY_CELL = 0u;

// restir.wgsl module constants
const float SPATIAL_REUSE_RADIUS_PIXELS = 30.0;
const float SPECULAR_DOMINANCE_SKIP_RESAMPLING_THRESHOLD = 0.2;

// initial_path.wgsl module constants
const float RECONNECTION_FOOTPRINT_KAPPA = 0.02;
const float RECONNECTION_ROUGHNESS_MIN = 0.3;
const float RECONNECTION_RELAX_DISTANCE = 1.0;

// Ray query committed-intersection kinds. gl_RayQueryCommittedIntersectionNoneEXT == 0,
// matching WGSL RAY_QUERY_INTERSECTION_NONE.
#define RAY_QUERY_INTERSECTION_NONE gl_RayQueryCommittedIntersectionNoneEXT

// ============================================================================
// saturate helpers (WGSL builtin, not present in GLSL)
// ============================================================================
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3 saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }

// ============================================================================
// Structs
// ============================================================================
struct ColorGrading {
    mat3 balance;
    vec3 saturation;
    vec3 contrast;
    vec3 gamma;
    vec3 gain;
    vec3 lift;
    vec2 midtone_range;
    float exposure;
    float hue;
    float post_saturation;
};

struct View {
    mat4 clip_from_world;
    mat4 unjittered_clip_from_world;
    mat4 world_from_clip;
    mat4 world_from_view;
    mat4 view_from_world;
    mat4 clip_from_view;
    mat4 view_from_clip;
    vec3 world_position;
    float exposure;
    vec4 viewport;
    vec4 main_pass_viewport;
    vec4 frustum[6];
    vec3 lod_view_world_position;
    ColorGrading color_grading;
    float mip_bias;
    uint frame_count;
};

struct PreviousViewUniforms {
    mat4 view_from_world;
    mat4 clip_from_world;
    mat4 unjittered_clip_from_world;
    mat4 clip_from_view;
    mat4 world_from_clip;
    mat4 view_from_clip;
};

struct SolariLightingSettings {
    float confidence_weight_cap;
    uint primary_di_samples;
    uint secondary_di_samples;
    uint max_bounces;
    float world_cache_max_temporal_samples;
    uint world_cache_direct_light_sample_count;
    float world_cache_max_gi_ray_distance;
    uint world_cache_cell_updates_soft_target;
    float world_cache_position_base_cell_size;
    float world_cache_position_lod_scale;
    uint frame_rng;
    uint reset;
};

struct LightSample {
    uint light_id;
    uint seed;
};

// 24 bytes, scalar-packed (std430). Matches prepare::RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE.
struct ResolvedLightSamplePacked {
    float world_position_x;
    float world_position_y;
    float world_position_z;
    uint world_normal;
    uint radiance;
    float inverse_pdf;
};

// 48 bytes (std430). Matches prepare::RESERVOIR_STRUCT_SIZE.
struct Reservoir {
    vec3 sample_point_world_position;
    float unbiased_contribution_weight;
    vec3 radiance;
    float confidence_weight;
    vec2 sample_point_world_normal;
    LightSample light_sample;
};

struct WorldCacheGeometryData {
    vec3 world_position;
    uint padding_a;
    vec3 world_normal;
    uint padding_b;
};

struct InstanceGeometryIds {
    uint vertex_buffer_id;
    uint vertex_buffer_offset;
    uint index_buffer_id;
    uint index_buffer_offset;
    uint triangle_count;
};

struct PackedVertex {
    vec4 a;
    vec4 b;
    vec4 tangent;
};

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
};

struct Material {
    uint normal_map_texture_id;
    uint base_color_texture_id;
    uint emissive_texture_id;
    uint metallic_roughness_texture_id;
    vec3 base_color;
    float perceptual_roughness;
    vec3 emissive;
    float metallic;
    vec3 _padding;
    float reflectance;
};

struct LightSource {
    uint kind;
    uint id;
};

struct DirectionalLight {
    vec3 direction_to_light;
    float cos_theta_max;
    vec3 luminance;
    float inverse_pdf;
};

struct ResolvedMaterial {
    vec3 base_color;
    vec3 emissive;
    float reflectance;
    float perceptual_roughness;
    float roughness;
    float metallic;
};

struct ResolvedRayHitFull {
    vec3 world_position;
    vec3 previous_frame_world_position;
    vec3 world_normal;
    vec3 geometric_world_normal;
    vec4 world_tangent;
    vec2 uv;
    float triangle_area;
    uint triangle_count;
    ResolvedMaterial material;
};

// GLSL rayQueryEXT is opaque and can't be returned, so trace_ray populates this value type
// (mirroring WGSL's RayIntersection value semantics). instance_id == WGSL instance_index.
struct RayIntersection {
    uint kind;
    float t;
    uint instance_id;
    uint primitive_index;
    vec2 barycentrics;
};

struct ResolvedLightSample {
    vec4 world_position;
    vec3 world_normal;
    vec3 radiance;
    float inverse_pdf;
};

struct LightContribution {
    vec3 radiance;
    float inverse_pdf;
    float inverse_solid_angle_pdf;
    vec3 wi;
    bool brdf_rays_can_hit;
};

struct ResolvedGPixel {
    vec3 world_position;
    vec3 world_normal;
    ResolvedMaterial material;
};

struct EvaluateAndSampleBrdfResult {
    vec3 wi;
    vec3 throughput;
    float pdf;
    bool diffuse_selected;
};

struct LobeReflectances {
    vec3 specular;
    vec3 diffuse;
};

struct DiSample {
    float unbiased_contribution_weight;
    LightSample light_sample;
    vec3 wi;
    vec3 brdf_radiance;
    float inverse_solid_angle_pdf;
    bool brdf_rays_can_hit;
};

struct PathState {
    vec3 ray_origin;
    vec3 normal;
    vec3 wo;
    ResolvedMaterial material;
    vec3 throughput_past_x1;
    vec3 x2_position;
    vec3 x2_normal;
    bool x2_reusable;
    vec3 x1_brdf;
};

struct InitialSamplingResult {
    Reservoir reservoir;
    vec3 non_resampled_radiance;
};

struct NeighborInfo {
    Reservoir reservoir;
    vec3 world_position;
    vec3 world_normal;
    ResolvedMaterial material;
};

struct ReservoirMergeResult {
    Reservoir merged_reservoir;
    vec3 selected_sample_brdf_radiance;
};

struct ReservoirContribution {
    vec3 brdf_radiance;
    float target_function;
    vec4 sample_world_position;
};

// ============================================================================
// Bindings — set 0: raytracing scene bindings
// ============================================================================
layout(std430, set = 0, binding = 0) readonly buffer VertexBuffer { PackedVertex vertices[]; } vertex_buffers[];
layout(std430, set = 0, binding = 1) readonly buffer IndexBuffer { uint indices[]; } index_buffers[];
layout(set = 0, binding = 2) uniform texture2D textures[];
layout(set = 0, binding = 3) uniform sampler samplers[];
layout(std430, set = 0, binding = 4) readonly buffer MaterialsBuffer { Material materials[]; };
layout(set = 0, binding = 5) uniform accelerationStructureEXT tlas;
layout(std430, set = 0, binding = 6) readonly buffer TransformsBuffer { mat4 transforms[]; };
layout(std430, set = 0, binding = 7) readonly buffer PrevTransformsBuffer { mat4 previous_frame_transforms[]; };
layout(std430, set = 0, binding = 8) readonly buffer GeometryIdsBuffer { InstanceGeometryIds geometry_ids[]; };
layout(std430, set = 0, binding = 9) readonly buffer MaterialIdsBuffer { uint material_ids[]; };
layout(std430, set = 0, binding = 10) readonly buffer LightSourcesBuffer { LightSource light_sources[]; };
layout(std430, set = 0, binding = 11) readonly buffer DirectionalLightsBuffer { DirectionalLight directional_lights[]; };
layout(std430, set = 0, binding = 12) readonly buffer PrevLightIdTranslationsBuffer { uint previous_frame_light_id_translations[]; };
layout(set = 0, binding = 13) uniform texture2D brdf_dfg_lut;
layout(set = 0, binding = 14) uniform sampler brdf_dfg_lut_sampler;

// ============================================================================
// Bindings — set 1: realtime bindings (only the bindings restir actually uses)
// ============================================================================
layout(set = 1, binding = 0, rgba16f) uniform image2D view_output;
layout(std430, set = 1, binding = 1) buffer LightTileSamplesBuffer { LightSample light_tile_samples[]; };
layout(std430, set = 1, binding = 2) buffer LightTileResolvedSamplesBuffer { ResolvedLightSamplePacked light_tile_resolved_samples[]; };
layout(std430, set = 1, binding = 3) buffer ReservoirsABuffer { Reservoir reservoirs_a[]; };
layout(std430, set = 1, binding = 4) buffer ReservoirsBBuffer { Reservoir reservoirs_b[]; };
layout(set = 1, binding = 5) uniform utexture2D gbuffer;
layout(set = 1, binding = 6) uniform texture2D depth_buffer;
layout(set = 1, binding = 7) uniform texture2D motion_vectors;
layout(set = 1, binding = 8) uniform utexture2D previous_gbuffer;
layout(set = 1, binding = 9) uniform texture2D previous_depth_buffer;
layout(std140, set = 1, binding = 10) uniform ViewBuffer { View view; };
layout(std140, set = 1, binding = 11) uniform PreviousViewBuffer { PreviousViewUniforms previous_view; };
layout(std430, set = 1, binding = 12) buffer WorldCacheChecksumsBuffer { uint world_cache_checksums[]; };
layout(std430, set = 1, binding = 13) buffer WorldCacheLifeBuffer { uint world_cache_life[]; };
layout(std430, set = 1, binding = 14) buffer WorldCacheRadianceBuffer { vec4 world_cache_radiance[]; };
layout(std430, set = 1, binding = 15) buffer WorldCacheGeometryDataBuffer { WorldCacheGeometryData world_cache_geometry_data[]; };
layout(std140, set = 1, binding = 22) uniform ConstantsBuffer { SolariLightingSettings constants; };

#ifdef DLSS_RR_GUIDE_BUFFERS
// ============================================================================
// Bindings — set 2: DLSS Ray Reconstruction guide buffers (initial-with-PSR only)
// ============================================================================
layout(set = 2, binding = 0, rgba8) uniform writeonly image2D diffuse_albedo;
layout(set = 2, binding = 1, rgba8) uniform writeonly image2D specular_albedo;
layout(set = 2, binding = 2, rgba16f) uniform writeonly image2D normal_roughness;
layout(set = 2, binding = 3, rg16f) uniform writeonly image2D specular_motion_vectors;
#endif

// ============================================================================
// Function prototypes (GLSL requires declaration-before-use)
// ============================================================================
float luminance(vec3 v);
uint rand_u(inout uint state);
float rand_f(inout uint state);
vec2 rand_vec2f(inout uint state);
uint rand_range_u(uint n, inout uint state);
vec2 sample_disk(float disk_radius, inout uint rng);
vec3 sample_cosine_hemisphere(vec3 normal, inout uint rng);
float copysign(float a, float b);
mat3 orthonormalize(vec3 z_basis);
vec3 octahedral_decode_signed(vec2 v);
vec3 octahedral_decode(vec2 v);
vec2 octahedral_encode(vec3 v);
vec3 rgb9e5_to_vec3_(uint v);
vec2 unpack_24bit_normal(uint packed);
float depth_ndc_to_view_z(float ndc_depth, mat4 clip_from_view, mat4 view_from_clip);
float D_GGX(float roughness, float NdotH);
float V_SmithGGXCorrelated(float roughness, float NdotV, float NdotL);
vec3 specular_multiscatter(float D, float V, vec3 F, vec3 F0, vec2 F_ab, float specular_intensity);
vec3 calculate_F0_dielectric(vec3 reflectance);
vec3 calculate_F0(vec3 base_color, float metallic, vec3 reflectance);
vec3 calculate_diffuse_color(vec3 base_color, float metallic, float specular_transmission, float diffuse_transmission);
mat3 calculate_tbn_mikktspace(vec3 world_normal, vec4 world_tangent);

Vertex unpack_vertex(PackedVertex packed);
vec3 sample_texture(uint id, vec2 uv);
ResolvedMaterial resolve_material(Material material, vec2 uv);
RayIntersection trace_ray(vec3 ray_origin, vec3 ray_direction, float ray_t_min, float ray_t_max, uint ray_flag);
ResolvedRayHitFull resolve_ray_hit_full(RayIntersection ray_hit);
Vertex[3] load_vertices(InstanceGeometryIds instance_geometry_ids, uint triangle_id);
vec3[3] transform_positions(mat4 transform, Vertex vertices[3]);
ResolvedRayHitFull resolve_triangle_data_full(uint instance_id, uint triangle_id, vec3 barycentrics);

float power_heuristic(float f, float g);
float balance_heuristic(float f, float g);
vec3 sample_ggx_vndf(vec3 wi_tangent, float roughness, inout uint rng);
bool ggx_vndf_sample_invalid(vec3 ray_tangent);
float ggx_vndf_pdf(vec3 wi_tangent, vec3 wo_tangent, float roughness);
ResolvedLightSample resolve_light_sample(LightSample light_sample, LightSource light_source);
LightContribution calculate_resolved_light_contribution(ResolvedLightSample resolved_light_sample, vec3 ray_origin, vec3 origin_world_normal);
float trace_light_visibility(vec3 ray_origin, vec4 light_sample_world_position);
vec3 triangle_barycentrics(uint seed);
ResolvedLightSample unpack_resolved_light_sample(ResolvedLightSamplePacked packed, float exposure);

LobeReflectances lobe_reflectances(vec3 F0_metal, vec3 F0_dielectric, ResolvedMaterial material, vec2 F_ab);
EvaluateAndSampleBrdfResult evaluate_and_sample_brdf(vec3 wo, vec3 world_normal, ResolvedMaterial material, vec2 F_ab, inout uint rng);
vec3 evaluate_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab);
vec3 evaluate_diffuse_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab);
vec3 evaluate_specular_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab);
float brdf_pdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab);
vec3 fresnel(vec3 f0, float LdotH);
vec2 F_AB(float perceptual_roughness, float NdotV);

ResolvedGPixel gpixel_resolve(uvec4 gpixel, float depth, uvec2 pixel_id, vec2 view_size, mat4 world_from_clip);
vec3 reconstruct_world_position(uvec2 pixel_id, float depth, vec2 view_size, mat4 world_from_clip);
bool pixel_dissimilar(float depth, vec3 world_position, vec3 other_world_position, vec3 normal, vec3 other_normal, View v);
uvec2 permute_pixel(uvec2 pixel_id, uint frame_index, vec2 view_size);

vec3 query_world_cache(vec3 world_position_in, vec3 world_normal, vec3 view_position, float ray_t, uint cell_lifetime, inout uint rng);
float get_cell_size(vec3 world_position, vec3 view_position, float ray_t, inout uint rng);
vec3 quantize_position(vec3 world_position, float quantization_factor);
vec3 quantize_normal(vec3 world_normal);
uint compute_key(uvec3 world_position, uvec3 world_normal);
uint compute_checksum(uvec3 world_position, uvec3 world_normal);
uint pcg_hash(uint input_val);
uint iqint_hash(uint input_val);
uint wrap_key(uint key);

Reservoir empty_reservoir();
ResolvedMaterial empty_material();

InitialSamplingResult generate_initial_reservoir(vec3 world_position, vec3 world_normal, ResolvedMaterial material, uvec2 workgroup_id, uvec2 pixel_id, inout uint rng);
void generate_nee_candidate(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, vec2 F_ab, float p_nee, uint di_samples, uvec2 workgroup_id, uint bounce, inout uint rng);
DiSample sample_light_ris(vec3 ray_origin, vec3 normal, vec3 wo, ResolvedMaterial material, vec2 F_ab, uint di_samples, uvec2 workgroup_id, uint bounce, inout uint rng);
void generate_emissive_candidate(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, ResolvedRayHitFull ray_hit, vec3 wi, float p_brdf, float ray_t, float p_nee, uint di_samples, uint bounce, inout uint rng);
bool terminate_into_cache(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, ResolvedRayHitFull ray_hit, float ray_t, uint bounce, inout uint rng);
bool reconnection_reusable(float ray_t, float p_brdf, vec3 wi, bool diffuse_selected, ResolvedRayHitFull ray_hit, vec3 world_position, float x1_perceptual_roughness, float primary_NdotV);

#ifdef DLSS_RR_GUIDE_BUFFERS
mat3 reflection_matrix(vec3 plane_normal);
void replace_primary_surface(uvec2 pixel_id, ResolvedRayHitFull ray_hit, mat3 mirror_rotations, vec3 primary_surface_world_position);
vec2 calculate_motion_vector(vec3 world_position, vec3 previous_world_position);
vec3 env_brdf_approx2(vec3 specular_color, float alpha, vec3 N, vec3 V);
#endif

NeighborInfo load_temporal_reservoir(uvec2 pixel_id, float depth, vec3 world_position, vec3 world_normal);
NeighborInfo load_spatial_reservoir(uvec2 pixel_id, float depth, vec3 world_position, vec3 world_normal, inout uint rng);
uvec2 get_neighbor_pixel_id(uvec2 center_pixel_id, float search_radius, inout uint rng);
float jacobian(vec3 new_world_position, vec3 original_world_position, vec3 sample_point_world_position, vec3 sample_point_world_normal);
ReservoirMergeResult merge_reservoirs(Reservoir canonical_reservoir, vec3 canonical_world_position, vec3 canonical_world_normal, ResolvedMaterial canonical_material, Reservoir other_reservoir, vec3 other_world_position, vec3 other_world_normal, ResolvedMaterial other_material, vec3 other_view_position, bool is_spatial, inout uint rng);
ReservoirContribution reservoir_contribution(Reservoir reservoir, ResolvedLightSample resolved, vec3 world_position, vec3 world_normal, vec3 wo, ResolvedMaterial material, vec2 F_ab);

// ============================================================================
// Leaf utilities (bevy_pbr::utils, bevy_render::maths, bevy_render::utils,
// bevy_core_pipeline::tonemapping, bevy_pbr::{rgb9e5,pbr_deferred_types,lighting,pbr_functions})
// ============================================================================
float luminance(vec3 v) {
    return dot(v, vec3(0.2126, 0.7152, 0.0722));
}

uint rand_u(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_f(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * uintBitsToFloat(0x2f800004u);
}

vec2 rand_vec2f(inout uint state) {
    return vec2(rand_f(state), rand_f(state));
}

uint rand_range_u(uint n, inout uint state) {
    return rand_u(state) % n;
}

vec2 sample_disk(float disk_radius, inout uint rng) {
    vec2 ab = 2.0 * rand_vec2f(rng) - 1.0;
    float a = ab.x;
    float b = ab.y;
    if (b == 0.0) { b = 1.0; }

    float phi;
    float r;
    if (a * a > b * b) {
        r = disk_radius * a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = disk_radius * b;
        phi = (PI / 2.0) - (PI / 4.0) * (a / b);
    }

    float x = r * cos(phi);
    float y = r * sin(phi);
    return vec2(x, y);
}

vec3 sample_cosine_hemisphere(vec3 normal, inout uint rng) {
    float cos_theta = 1.0 - 2.0 * rand_f(rng);
    float phi = PI_2 * rand_f(rng);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    vec3 direction = normal + vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    float len_sq = dot(direction, direction);
    if (len_sq < 1e-8) { return normal; }
    return direction * inversesqrt(len_sq);
}

float copysign(float a, float b) {
    return uintBitsToFloat((floatBitsToUint(a) & 0x7FFFFFFFu) | (floatBitsToUint(b) & 0x80000000u));
}

mat3 orthonormalize(vec3 z_basis) {
    float sign = copysign(1.0, z_basis.z);
    float a = -1.0 / (sign + z_basis.z);
    float b = z_basis.x * z_basis.y * a;
    vec3 x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    vec3 y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3(x_basis, y_basis, z_basis);
}

vec3 octahedral_decode_signed(vec2 v) {
    vec3 n = vec3(v.xy, 1.0 - abs(v.x) - abs(v.y));
    float t = saturate(-n.z);
    vec2 w = mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.0)));
    n = vec3(n.xy + w, n.z);
    return normalize(n);
}

vec3 octahedral_decode(vec2 v) {
    vec2 f = v * 2.0 - 1.0;
    return octahedral_decode_signed(f);
}

vec2 octahedral_encode(vec3 v) {
    vec3 n = v / (abs(v.x) + abs(v.y) + abs(v.z));
    vec2 octahedral_wrap = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThan(n.xy, vec2(0.0)));
    vec2 n_xy = (n.z >= 0.0) ? n.xy : octahedral_wrap;
    return n_xy * 0.5 + 0.5;
}

vec3 rgb9e5_to_vec3_(uint v) {
    int exponent = int(bitfieldExtract(v, 27, 5)) - 15 - 9;
    float scale = exp2(float(exponent));
    return vec3(
        float(bitfieldExtract(v, 0, 9)),
        float(bitfieldExtract(v, 9, 9)),
        float(bitfieldExtract(v, 18, 9))
    ) * scale;
}

vec2 unpack_24bit_normal(uint packed) {
    uint unorm1 = packed & 0xFFFu;
    uint unorm2 = (packed >> 12u) & 0xFFFu;
    return vec2(float(unorm1) / U12MAXF, float(unorm2) / U12MAXF);
}

float depth_ndc_to_view_z(float ndc_depth, mat4 clip_from_view, mat4 view_from_clip) {
    vec4 view_pos = view_from_clip * vec4(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
}

float D_GGX(float roughness, float NdotH) {
    float oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    float a = NdotH * roughness;
    float k = roughness / (oneMinusNdotHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

float V_SmithGGXCorrelated(float roughness, float NdotV, float NdotL) {
    float a2 = roughness * roughness;
    float lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    float lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    float v = 0.5 / (lambdaV + lambdaL);
    return v;
}

vec3 specular_multiscatter(float D, float V, vec3 F, vec3 F0, vec2 F_ab, float specular_intensity) {
    vec3 Fr = (specular_intensity * D * V) * F;
    Fr *= 1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0);
    return Fr;
}

vec3 calculate_F0_dielectric(vec3 reflectance) {
    return 0.16 * reflectance * reflectance;
}

vec3 calculate_F0(vec3 base_color, float metallic, vec3 reflectance) {
    return mix(calculate_F0_dielectric(reflectance), base_color, metallic);
}

vec3 calculate_diffuse_color(vec3 base_color, float metallic, float specular_transmission, float diffuse_transmission) {
    return base_color * (1.0 - metallic) * (1.0 - specular_transmission) * (1.0 - diffuse_transmission);
}

mat3 calculate_tbn_mikktspace(vec3 world_normal, vec4 world_tangent) {
    vec3 N = world_normal;
    vec3 T = world_tangent.xyz;
    vec3 B = world_tangent.w * cross(N, T);
    return mat3(T, B, N);
}

// ============================================================================
// Scene bindings (raytracing_scene_bindings.wgsl)
// ============================================================================
Vertex unpack_vertex(PackedVertex packed) {
    Vertex vertex;
    vertex.position = packed.a.xyz;
    vertex.normal = vec3(packed.a.w, packed.b.xy);
    vertex.uv = packed.b.zw;
    vertex.tangent = packed.tangent;
    return vertex;
}

vec3 sample_texture(uint id, vec2 uv) {
    return textureLod(sampler2D(textures[nonuniformEXT(id)], samplers[nonuniformEXT(id)]), uv, 0.0).rgb;
}

RayIntersection trace_ray(vec3 ray_origin, vec3 ray_direction, float ray_t_min, float ray_t_max, uint ray_flag) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, ray_flag, RAY_NO_CULL, ray_origin, ray_t_min, ray_direction, ray_t_max);
    while (rayQueryProceedEXT(rq)) {}

    RayIntersection hit;
    hit.kind = rayQueryGetIntersectionTypeEXT(rq, true);
    hit.t = 0.0;
    hit.instance_id = 0u;
    hit.primitive_index = 0u;
    hit.barycentrics = vec2(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        hit.t = rayQueryGetIntersectionTEXT(rq, true);
        hit.instance_id = rayQueryGetIntersectionInstanceIdEXT(rq, true);
        hit.primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        hit.barycentrics = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    }
    return hit;
}

ResolvedMaterial resolve_material(Material material, vec2 uv) {
    ResolvedMaterial m;

    m.base_color = material.base_color.rgb;
    if (material.base_color_texture_id != TEXTURE_MAP_NONE) {
        m.base_color *= sample_texture(material.base_color_texture_id, uv);
    }

    m.emissive = material.emissive.rgb;
    if (material.emissive_texture_id != TEXTURE_MAP_NONE) {
        m.emissive *= sample_texture(material.emissive_texture_id, uv);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if (material.metallic_roughness_texture_id != TEXTURE_MAP_NONE) {
        vec3 metallic_roughness = sample_texture(material.metallic_roughness_texture_id, uv);
        m.perceptual_roughness *= metallic_roughness.g;
        m.metallic *= metallic_roughness.b;
    }

    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    return m;
}

ResolvedRayHitFull resolve_ray_hit_full(RayIntersection ray_hit) {
    vec3 barycentrics = vec3(1.0 - ray_hit.barycentrics.x - ray_hit.barycentrics.y, ray_hit.barycentrics);
    return resolve_triangle_data_full(ray_hit.instance_id, ray_hit.primitive_index, barycentrics);
}

Vertex[3] load_vertices(InstanceGeometryIds instance_geometry_ids, uint triangle_id) {
    uvec3 indices_i = (triangle_id * 3u) + uvec3(0u, 1u, 2u) + instance_geometry_ids.index_buffer_offset;
    uint ib = nonuniformEXT(instance_geometry_ids.index_buffer_id);
    uvec3 indices = uvec3(
        index_buffers[ib].indices[indices_i.x],
        index_buffers[ib].indices[indices_i.y],
        index_buffers[ib].indices[indices_i.z]
    ) + instance_geometry_ids.vertex_buffer_offset;

    uint vb = nonuniformEXT(instance_geometry_ids.vertex_buffer_id);
    return Vertex[3](
        unpack_vertex(vertex_buffers[vb].vertices[indices.x]),
        unpack_vertex(vertex_buffers[vb].vertices[indices.y]),
        unpack_vertex(vertex_buffers[vb].vertices[indices.z])
    );
}

vec3[3] transform_positions(mat4 transform, Vertex vertices[3]) {
    return vec3[3](
        (transform * vec4(vertices[0].position, 1.0)).xyz,
        (transform * vec4(vertices[1].position, 1.0)).xyz,
        (transform * vec4(vertices[2].position, 1.0)).xyz
    );
}

ResolvedRayHitFull resolve_triangle_data_full(uint instance_id, uint triangle_id, vec3 barycentrics) {
    uint material_id = material_ids[instance_id];
    Material material = materials[material_id];

    mat4 transform = transforms[instance_id];
    mat4 previous_frame_transform = previous_frame_transforms[instance_id];

    InstanceGeometryIds instance_geometry_ids = geometry_ids[instance_id];
    Vertex vertices[3] = load_vertices(instance_geometry_ids, triangle_id);

    vec3 world_vertices[3] = transform_positions(transform, vertices);
    vec3 world_position = mat3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    vec3 previous_frame_world_vertices[3] = transform_positions(previous_frame_transform, vertices);
    vec3 previous_frame_world_position = mat3(previous_frame_world_vertices[0], previous_frame_world_vertices[1], previous_frame_world_vertices[2]) * barycentrics;

    vec2 uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;

    vec3 local_tangent = mat3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    vec4 world_tangent = vec4(
        normalize(mat3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_tangent),
        vertices[0].tangent.w
    );

    vec3 local_normal = mat3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    vec3 world_normal = normalize(mat3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_normal);
    vec3 geometric_world_normal = world_normal;
    if (material.normal_map_texture_id != TEXTURE_MAP_NONE) {
        mat3 TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        vec3 T = TBN[0];
        vec3 B = TBN[1];
        vec3 N = TBN[2];
        vec3 Nt = sample_texture(material.normal_map_texture_id, uv);
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    vec3 triangle_edge0 = world_vertices[0] - world_vertices[1];
    vec3 triangle_edge1 = world_vertices[0] - world_vertices[2];
    float triangle_area = length(cross(triangle_edge0, triangle_edge1)) / 2.0;

    ResolvedMaterial resolved_material = resolve_material(material, uv);

    ResolvedRayHitFull hit;
    hit.world_position = world_position;
    hit.previous_frame_world_position = previous_frame_world_position;
    hit.world_normal = world_normal;
    hit.geometric_world_normal = geometric_world_normal;
    hit.world_tangent = world_tangent;
    hit.uv = uv;
    hit.triangle_area = triangle_area;
    hit.triangle_count = instance_geometry_ids.triangle_count;
    hit.material = resolved_material;
    return hit;
}

// ============================================================================
// Sampling (sampling.wgsl) + presample unpack (presample_light_tiles.wgsl)
// ============================================================================
float balance_heuristic(float f, float g) {
    if (f == 0.0) {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}

float power_heuristic(float f, float g) {
    return balance_heuristic(f * f, g * g);
}

vec3 sample_ggx_vndf(vec3 wi_tangent, float roughness, inout uint rng) {
    if (roughness <= MIRROR_ROUGHNESS_THRESHOLD) {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    vec3 i = wi_tangent;
    vec2 rand = rand_vec2f(rng);
    vec3 i_std = normalize(vec3(i.xy * roughness, i.z));
    float phi = PI_2 * rand.x;
    float a = roughness;
    float s = 1.0 + length(vec2(i.xy));
    float a2 = a * a;
    float s2 = s * s;
    float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    float b = (i.z > 0.0) ? k * i_std.z : i_std.z;
    float z = fma(1.0 - rand.y, 1.0 + b, -b);
    float sin_theta = sqrt(saturate(1.0 - z * z));
    vec3 o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    vec3 m_std = i_std + o_std;
    vec3 m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

bool ggx_vndf_sample_invalid(vec3 ray_tangent) {
    return !(ray_tangent.z > 0.0);
}

float ggx_vndf_pdf(vec3 wi_tangent, vec3 wo_tangent, float roughness) {
    if (roughness <= MIRROR_ROUGHNESS_THRESHOLD) {
        vec3 mirror_wo = vec3(-wi_tangent.xy, wi_tangent.z);
        if (all(lessThan(abs(mirror_wo - wo_tangent), vec3(0.0001)))) {
            return uintBitsToFloat(0x7F800000u);
        } else {
            return 0.0;
        }
    }

    vec3 i = wi_tangent;
    vec3 o = wo_tangent;
    vec3 m = normalize(i + o);
    float ndf = D_GGX(roughness, saturate(m.z));
    vec2 ai = roughness * i.xy;
    float len2 = dot(ai, ai);
    float t = sqrt(len2 + i.z * i.z);
    float pdf;
    if (i.z >= 0.0) {
        float a = roughness;
        float s = 1.0 + length(i.xy);
        float a2 = a * a;
        float s2 = s * s;
        float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        pdf = ndf / (2.0 * (k * i.z + t));
    } else {
        pdf = ndf * (t - i.z) / (2.0 * len2);
    }

    return isnan(pdf) ? 0.0 : pdf;
}

vec3 triangle_barycentrics(uint seed) {
    uint rng = seed;
    vec2 barycentrics = rand_vec2f(rng);
    if (barycentrics.x + barycentrics.y > 1.0) { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}

ResolvedLightSample resolve_light_sample(LightSample light_sample, LightSource light_source) {
    if (light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL) {
        DirectionalLight directional_light = directional_lights[light_source.id];

        // NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS is not defined for restir: use soft shadows.
        uint rng = light_sample.seed;
        vec2 random = rand_vec2f(rng);
        float cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        float phi = random.y * PI_2;
        float x = cos(phi) * sin_theta;
        float y = sin(phi) * sin_theta;
        vec3 direction_to_light = vec3(x, y, cos_theta);

        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;

        ResolvedLightSample rls;
        rls.world_position = vec4(direction_to_light, 0.0);
        rls.world_normal = -direction_to_light;
        rls.radiance = directional_light.luminance;
        rls.inverse_pdf = directional_light.inverse_pdf;
        return rls;
    } else {
        uint triangle_count = light_source.kind >> 1u;
        uint triangle_id = light_sample.light_id & 0xFFFFu;
        vec3 barycentrics = triangle_barycentrics(light_sample.seed);
        ResolvedRayHitFull triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        ResolvedLightSample rls;
        rls.world_position = vec4(triangle_data.world_position, 1.0);
        rls.world_normal = triangle_data.world_normal;
        rls.radiance = triangle_data.material.emissive.rgb;
        rls.inverse_pdf = float(triangle_count) * triangle_data.triangle_area;
        return rls;
    }
}

LightContribution calculate_resolved_light_contribution(ResolvedLightSample resolved_light_sample, vec3 ray_origin, vec3 origin_world_normal) {
    vec3 ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    float light_distance = length(ray);
    vec3 wi = ray / light_distance;

    float cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    float light_distance_squared = light_distance * light_distance;
    float denominator = cos_theta_light / light_distance_squared;

    vec3 radiance = resolved_light_sample.radiance * denominator;
    float inverse_solid_angle_pdf = resolved_light_sample.inverse_pdf * denominator;

    LightContribution lc;
    lc.radiance = radiance;
    lc.inverse_pdf = resolved_light_sample.inverse_pdf;
    lc.inverse_solid_angle_pdf = inverse_solid_angle_pdf;
    lc.wi = wi;
    lc.brdf_rays_can_hit = resolved_light_sample.world_position.w == 1.0;
    return lc;
}

float trace_light_visibility(vec3 ray_origin, vec4 light_sample_world_position) {
    vec3 ray_direction = light_sample_world_position.xyz;
    float ray_t_max = RAY_T_MAX;

    if (light_sample_world_position.w == 1.0) {
        vec3 ray = ray_direction - ray_origin;
        float dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }

    if (ray_t_max < RAY_T_MIN) { return 0.0; }

    RayIntersection ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, gl_RayFlagsTerminateOnFirstHitEXT);
    return float(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

ResolvedLightSample unpack_resolved_light_sample(ResolvedLightSamplePacked packed, float exposure) {
    ResolvedLightSample rls;
    rls.world_position = vec4(packed.world_position_x, packed.world_position_y, packed.world_position_z, (packed.inverse_pdf < 0.0) ? 0.0 : 1.0);
    rls.world_normal = octahedral_decode(unpackUnorm2x16(packed.world_normal));
    rls.radiance = (exp2(rgb9e5_to_vec3_(packed.radiance)) - 1.0) / exposure;
    rls.inverse_pdf = abs(packed.inverse_pdf);
    return rls;
}

// ============================================================================
// BRDF (brdf.wgsl)
// ============================================================================
LobeReflectances lobe_reflectances(vec3 F0_metal, vec3 F0_dielectric, ResolvedMaterial material, vec2 F_ab) {
    float multiscattering_factor = 1.0 / (F_ab.x + F_ab.y) - 1.0;
    vec3 rho_specular_metallic = (F0_metal * F_ab.x + F_ab.y) * (1.0 + F0_metal * multiscattering_factor);
    vec3 rho_specular_dielectric = (F0_dielectric * F_ab.x + F_ab.y) * (1.0 + F0_dielectric * multiscattering_factor);
    LobeReflectances lr;
    lr.specular = mix(rho_specular_dielectric, rho_specular_metallic, material.metallic);
    lr.diffuse = (1.0 - material.metallic) * (1.0 - rho_specular_dielectric) * material.base_color;
    return lr;
}

EvaluateAndSampleBrdfResult evaluate_and_sample_brdf(vec3 wo, vec3 world_normal, ResolvedMaterial material, vec2 F_ab, inout uint rng) {
    float NdotV = dot(world_normal, wo);
    if (NdotV < 0.0001) { return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0, false); }
    vec3 F0_metal = material.base_color;
    vec3 F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    LobeReflectances rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    float specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    float diffuse_weight = 1.0 - specular_weight;

    mat3 TBN = orthonormalize(world_normal);
    vec3 T = TBN[0];
    vec3 B = TBN[1];
    vec3 N = TBN[2];

    vec3 wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));

    vec3 wi;
    vec3 wi_tangent;
    bool diffuse_selected = rand_f(rng) < diffuse_weight;
    if (diffuse_selected) {
        wi = sample_cosine_hemisphere(world_normal, rng);
        wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));
    } else {
        wi_tangent = sample_ggx_vndf(wo_tangent, material.roughness, rng);
        if (ggx_vndf_sample_invalid(wi_tangent)) {
            return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0, false);
        }
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;

        if (material.roughness <= MIRROR_ROUGHNESS_THRESHOLD) {
            return EvaluateAndSampleBrdfResult(
                wi,
                evaluate_specular_brdf(wo, wi, world_normal, material, F_ab) / specular_weight,
                uintBitsToFloat(0x7F800000u),
                false
            );
        }
    }

    float diffuse_pdf = wi_tangent.z / PI;
    float specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    float pdf = (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
    vec3 throughput = evaluate_brdf(wo, wi, world_normal, material, F_ab) / pdf;
    return EvaluateAndSampleBrdfResult(wi, throughput, pdf, diffuse_selected);
}

vec3 evaluate_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab) {
    return max(evaluate_diffuse_brdf(wo, wi, world_normal, material, F_ab) + evaluate_specular_brdf(wo, wi, world_normal, material, F_ab), vec3(0.0));
}

vec3 evaluate_diffuse_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab) {
    float NdotL = dot(world_normal, wi);
    float NdotV = dot(world_normal, wo);
    if (NdotL < 0.0001 || NdotV < 0.0001) { return vec3(0.0); }
    vec3 F0_metal = material.base_color;
    vec3 F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    LobeReflectances rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    return rho.diffuse / PI * NdotL;
}

vec3 evaluate_specular_brdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab) {
    vec3 H = normalize(wi + wo);
    float NdotL = dot(world_normal, wi);
    float NdotH = dot(world_normal, H);
    float LdotH = dot(wi, H);
    float NdotV = dot(world_normal, wo);
    if (NdotL < 0.0001 || NdotH < 0.0001 || LdotH < 0.0001 || NdotV < 0.0001) { return vec3(0.0); }

    vec3 F0_metal = material.base_color;
    vec3 F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));

    if (material.roughness <= MIRROR_ROUGHNESS_THRESHOLD) {
        if (abs(NdotH - 1.0) < 0.0001) {
            vec3 F_metal = fresnel(F0_metal, LdotH);
            vec3 F_dielectric = fresnel(F0_dielectric, LdotH);
            return mix(F_dielectric, F_metal, material.metallic);
        } else {
            return vec3(0.0);
        }
    }

    float D = D_GGX(material.roughness, NdotH);
    float Vs = V_SmithGGXCorrelated(material.roughness, NdotV, NdotL);
    vec3 F_metal = fresnel(F0_metal, LdotH);
    vec3 F_dielectric = fresnel(F0_dielectric, LdotH);
    return mix(specular_multiscatter(D, Vs, F_dielectric, F0_dielectric, F_ab, 1.0),
               specular_multiscatter(D, Vs, F_metal, F0_metal, F_ab, 1.0),
               material.metallic) * NdotL;
}

float brdf_pdf(vec3 wo, vec3 wi, vec3 world_normal, ResolvedMaterial material, vec2 F_ab) {
    float NdotV = max(dot(world_normal, wo), 0.0001);
    vec3 F0_metal = material.base_color;
    vec3 F0_dielectric = calculate_F0_dielectric(vec3(material.reflectance));
    LobeReflectances rho = lobe_reflectances(F0_metal, F0_dielectric, material, F_ab);
    float specular_weight = luminance(rho.specular) / luminance(rho.specular + rho.diffuse);
    float diffuse_weight = 1.0 - specular_weight;

    mat3 TBN = orthonormalize(world_normal);
    vec3 T = TBN[0];
    vec3 B = TBN[1];
    vec3 N = TBN[2];

    vec3 wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
    vec3 wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));

    float diffuse_pdf = wi_tangent.z / PI;
    float specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
    return (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
}

vec3 fresnel(vec3 f0, float LdotH) {
    return f0 + (1.0 - f0) * pow(1.0 - LdotH, 5.0);
}

vec2 F_AB(float perceptual_roughness, float NdotV) {
    return textureLod(sampler2D(brdf_dfg_lut, brdf_dfg_lut_sampler), vec2(NdotV, perceptual_roughness), 0.0).rg;
}

// ============================================================================
// G-buffer utils (gbuffer_utils.wgsl)
// ============================================================================
vec3 reconstruct_world_position(uvec2 pixel_id, float depth, vec2 view_size, mat4 world_from_clip) {
    vec2 uv = (vec2(pixel_id) + 0.5) / view_size;
    vec2 xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    vec4 world_pos = world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

ResolvedGPixel gpixel_resolve(uvec4 gpixel, float depth, uvec2 pixel_id, vec2 view_size, mat4 world_from_clip) {
    vec3 world_position = reconstruct_world_position(pixel_id, depth, view_size, world_from_clip);
    vec3 world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    vec4 base_rough = unpackUnorm4x8(gpixel.r);
    vec3 base_color = pow(base_rough.rgb, vec3(2.2));
    float perceptual_roughness = base_rough.a;
    float roughness = perceptual_roughness * perceptual_roughness;
    vec4 props = unpackUnorm4x8(gpixel.b);
    float reflectance = props.r;
    float metallic = props.g;
    vec3 emissive = rgb9e5_to_vec3_(gpixel.g);
    ResolvedMaterial material = ResolvedMaterial(base_color, emissive, reflectance, perceptual_roughness, roughness, metallic);

    return ResolvedGPixel(world_position, world_normal, material);
}

bool pixel_dissimilar(float depth, vec3 world_position, vec3 other_world_position, vec3 normal, vec3 other_normal, View v) {
    float tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    float view_z = -depth_ndc_to_view_z(depth, v.clip_from_view, v.view_from_clip);

    return tangent_plane_distance / view_z > 0.003 || dot(normal, other_normal) < 0.0;
}

uvec2 permute_pixel(uvec2 pixel_id, uint frame_index, vec2 view_size) {
    uint r = frame_index;
    uvec2 offset = uvec2(r & 3u, (r >> 2u) & 3u);
    uvec2 shifted_pixel_id = pixel_id + offset;
    shifted_pixel_id ^= uvec2(3u);
    shifted_pixel_id -= offset;
    return min(shifted_pixel_id, uvec2(view_size - 1.0));
}

// ============================================================================
// World cache (world_cache_query.wgsl)
// ============================================================================
uint pcg_hash(uint input_val) {
    uint state = input_val * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint iqint_hash(uint input_val) {
    uint n = (input_val << 13u) ^ input_val;
    return n * (n * n * 15731u + 789221u) + 1376312589u;
}

uint wrap_key(uint key) {
    return key & (WORLD_CACHE_SIZE - 1u);
}

uint compute_key(uvec3 world_position, uvec3 world_normal) {
    uint key = pcg_hash(world_position.x);
    key = pcg_hash(key + world_position.y);
    key = pcg_hash(key + world_position.z);
    key = pcg_hash(key + world_normal.x);
    key = pcg_hash(key + world_normal.y);
    key = pcg_hash(key + world_normal.z);
    return wrap_key(key);
}

uint compute_checksum(uvec3 world_position, uvec3 world_normal) {
    uint key = iqint_hash(world_position.x);
    key = iqint_hash(key + world_position.y);
    key = iqint_hash(key + world_position.z);
    key = iqint_hash(key + world_normal.x);
    key = iqint_hash(key + world_normal.y);
    key = iqint_hash(key + world_normal.z);
    return max(key, 1u);
}

vec3 quantize_position(vec3 world_position, float quantization_factor) {
    return floor(world_position / quantization_factor + 0.0001);
}

vec3 quantize_normal(vec3 world_normal) {
    return floor(world_normal + 0.0001);
}

float get_cell_size(vec3 world_position, vec3 view_position, float ray_t, inout uint rng) {
    float camera_distance = distance(view_position, world_position) / constants.world_cache_position_lod_scale;
    float lod_f = log2(1.0 + camera_distance);
    float lod_fract = fract(lod_f);
    float lod = floor(lod_f) + ((rand_f(rng) < lod_fract * lod_fract * lod_fract) ? 1.0 : 0.0);
    float cell_size = constants.world_cache_position_base_cell_size * exp2(lod);

    if (ray_t < cell_size) {
        float shrunk_lod = max(floor(log2(ray_t / constants.world_cache_position_base_cell_size)), 0.0);
        cell_size = constants.world_cache_position_base_cell_size * exp2(shrunk_lod);
    }

    return cell_size;
}

vec3 query_world_cache(vec3 world_position_in, vec3 world_normal, vec3 view_position, float ray_t, uint cell_lifetime, inout uint rng) {
    vec3 world_position = world_position_in;
    float cell_size = get_cell_size(world_position, view_position, ray_t, rng);

    // NO_JITTER_WORLD_CACHE is not defined for restir: jitter the query point.
    mat3 TBN = orthonormalize(world_normal);
    vec2 offset = (rand_vec2f(rng) * 2.0 - 1.0) * cell_size * 0.5;
    world_position += offset.x * TBN[0] + offset.y * TBN[1];
    cell_size = get_cell_size(world_position, view_position, ray_t, rng);

    uvec3 world_position_quantized = floatBitsToUint(quantize_position(world_position, cell_size));
    uvec3 world_normal_quantized = floatBitsToUint(quantize_normal(world_normal));
    uint key = compute_key(world_position_quantized, world_normal_quantized);
    uint checksum = compute_checksum(world_position_quantized, world_normal_quantized);

    for (uint i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        uint existing_checksum = atomicCompSwap(world_cache_checksums[key], WORLD_CACHE_EMPTY_CELL, checksum);
        bool exchanged = existing_checksum == WORLD_CACHE_EMPTY_CELL;

        // Cell already exists or is empty - reset lifetime.
        // WORLD_CACHE_QUERY_ATOMIC_MAX_LIFETIME is not defined for restir: atomic store.
        if (existing_checksum == checksum || existing_checksum == WORLD_CACHE_EMPTY_CELL) {
            atomicExchange(world_cache_life[key], cell_lifetime);
        }

        if (existing_checksum == checksum) {
            return world_cache_radiance[key].rgb;
        } else if (existing_checksum == WORLD_CACHE_EMPTY_CELL && exchanged) {
            world_cache_geometry_data[key].world_position = world_position;
            world_cache_geometry_data[key].world_normal = world_normal;
            return vec3(0.0);
        } else {
            key += 1u;
        }
    }

    return vec3(0.0);
}

// ============================================================================
// Reservoir / material helpers (realtime_bindings.wgsl + restir.wgsl)
// ============================================================================
Reservoir empty_reservoir() {
    Reservoir r;
    r.sample_point_world_position = vec3(0.0);
    r.unbiased_contribution_weight = 0.0;
    r.radiance = vec3(0.0);
    r.confidence_weight = 0.0;
    r.sample_point_world_normal = vec2(0.0);
    r.light_sample = LightSample(NULL_LIGHT_ID, 0u);
    return r;
}

ResolvedMaterial empty_material() {
    return ResolvedMaterial(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
}

// ============================================================================
// Initial path tracing (initial_path.wgsl)
// ============================================================================
#ifdef DLSS_RR_GUIDE_BUFFERS
mat3 reflection_matrix(vec3 plane_normal) {
    mat3 n_nt = mat3(
        plane_normal * plane_normal.x,
        plane_normal * plane_normal.y,
        plane_normal * plane_normal.z
    );
    mat3 identity_matrix = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    return identity_matrix - n_nt * 2.0;
}

vec2 calculate_motion_vector(vec3 world_position, vec3 previous_world_position) {
    vec4 clip_position_t = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    vec2 clip_position = clip_position_t.xy / clip_position_t.w;
    vec4 previous_clip_position_t = previous_view.unjittered_clip_from_world * vec4(previous_world_position, 1.0);
    vec2 previous_clip_position = previous_clip_position_t.xy / previous_clip_position_t.w;
    return (clip_position - previous_clip_position) * vec2(0.5, -0.5);
}

vec3 env_brdf_approx2(vec3 specular_color, float alpha, vec3 N, vec3 V) {
    float NoV = abs(dot(N, V));

    vec4 X;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    vec4 Y;
    Y.x = 1.0;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    mat2 M1 = mat2(0.99044, 1.29678, -1.28514, -0.755907);
    mat3 M2 = mat3(1.0, 20.3225, 121.563, 2.92338, -27.0302, 626.13, 59.4188, 222.592, 316.627);
    mat2 M3 = mat2(0.0365463, 9.0632, 3.32707, -9.04756);
    mat3 M4 = mat3(1.0, 9.04401, 5.56589, 3.59685, -16.3174, 19.7886, -1.36772, 9.22949, -20.2123);

    float bias = dot(M1 * X.xy, Y.xy) / dot(M2 * X.xyw, Y.xyw);
    float scale = dot(M3 * X.xy, Y.xy) / dot(M4 * X.xzw, Y.xyw);

    bias *= saturate(specular_color.g * 50.0);

    return fma(specular_color, vec3(max(0.0, scale)), vec3(max(0.0, bias)));
}

void replace_primary_surface(uvec2 pixel_id, ResolvedRayHitFull ray_hit, mat3 mirror_rotations, vec3 primary_surface_world_position) {
    vec3 virtual_position = (mirror_rotations * (ray_hit.world_position - primary_surface_world_position)) + primary_surface_world_position;
    vec3 virtual_previous_frame_position = (mirror_rotations * (ray_hit.previous_frame_world_position - primary_surface_world_position)) + primary_surface_world_position;
    vec2 specular_motion_vector = calculate_motion_vector(virtual_position, virtual_previous_frame_position);

    vec3 F0 = calculate_F0(ray_hit.material.base_color, ray_hit.material.metallic, vec3(ray_hit.material.reflectance));
    vec3 wo = normalize(view.world_position - virtual_position);
    vec3 virtual_normal = normalize(mirror_rotations * ray_hit.world_normal);

    imageStore(specular_motion_vectors, ivec2(pixel_id), vec4(specular_motion_vector, vec2(0.0)));
    imageStore(diffuse_albedo, ivec2(pixel_id), vec4(calculate_diffuse_color(ray_hit.material.base_color, ray_hit.material.metallic, 0.0, 0.0), 0.0));
    imageStore(specular_albedo, ivec2(pixel_id), vec4(env_brdf_approx2(F0, ray_hit.material.roughness, virtual_normal, wo), 0.0));
    imageStore(normal_roughness, ivec2(pixel_id), vec4(virtual_normal, ray_hit.material.perceptual_roughness));
}
#endif

DiSample sample_light_ris(vec3 ray_origin, vec3 normal, vec3 wo, ResolvedMaterial material, vec2 F_ab, uint di_samples, uvec2 workgroup_id, uint bounce, inout uint rng) {
    uint workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y + bounce;
    uint light_tile_start = rand_range_u(128u, workgroup_rng) * 1024u;

    float weight_sum = 0.0;
    float selected_target_function = 0.0;
    LightSample selected_light_sample = LightSample(NULL_LIGHT_ID, 0u);
    vec4 selected_world_position = vec4(0.0);
    vec3 selected_wi = vec3(0.0);
    vec3 selected_brdf_radiance = vec3(0.0);
    float selected_inverse_solid_angle_pdf = 0.0;
    bool selected_brdf_rays_can_hit = false;
    float mis_weight = 1.0 / float(di_samples);
    for (uint i = 0u; i < di_samples; i++) {
        uint tile_sample = light_tile_start + rand_range_u(1024u, rng);
        ResolvedLightSample resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        LightContribution light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, normal);
        vec3 brdf_current = evaluate_brdf(wo, light_contribution.wi, normal, material, F_ab);
        vec3 brdf_radiance = brdf_current * light_contribution.radiance;

        float target_function = luminance(brdf_radiance);
        float resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
            selected_target_function = target_function;
            selected_light_sample = light_tile_samples[tile_sample];
            selected_world_position = resolved_light_sample.world_position;
            selected_wi = light_contribution.wi;
            selected_inverse_solid_angle_pdf = light_contribution.inverse_solid_angle_pdf;
            selected_brdf_rays_can_hit = light_contribution.brdf_rays_can_hit;
            selected_brdf_radiance = brdf_radiance;
        }
    }

    float unbiased_contribution_weight = 0.0;
    if (selected_target_function > 0.0) {
        unbiased_contribution_weight = weight_sum / selected_target_function;
        unbiased_contribution_weight *= trace_light_visibility(ray_origin, selected_world_position);
    }

    return DiSample(unbiased_contribution_weight, selected_light_sample, selected_wi, selected_brdf_radiance, selected_inverse_solid_angle_pdf, selected_brdf_rays_can_hit);
}

void generate_nee_candidate(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, vec2 F_ab, float p_nee, uint di_samples, uvec2 workgroup_id, uint bounce, inout uint rng) {
    if (rand_f(rng) >= p_nee) { return; }

    DiSample di = sample_light_ris(path.ray_origin, path.normal, path.wo, path.material, F_ab, di_samples, workgroup_id, bounce, rng);
    float di_target_function = luminance(di.brdf_radiance);
    if (di_target_function <= 0.0) { return; }

    float nee_mis_weight = 1.0;
    if (di.brdf_rays_can_hit && di.inverse_solid_angle_pdf > 0.0) {
        float p_nee_strategy = float(di_samples) * (1.0 / di.inverse_solid_angle_pdf) * p_nee;
        float p_brdf_at_nee = brdf_pdf(path.wo, di.wi, path.normal, path.material, F_ab);
        nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
    }

    if (bounce == 0u) {
        float target_function = di_target_function * nee_mis_weight;
        float resampling_weight = target_function * di.unbiased_contribution_weight / p_nee;

        weight_sum += resampling_weight;
        if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
            reservoir.light_sample = di.light_sample;
            selected_target_function = target_function;
        }
    } else {
        vec3 L_at_reconnection = path.throughput_past_x1 * di.brdf_radiance * di.unbiased_contribution_weight * nee_mis_weight / p_nee;
        if (!path.x2_reusable) {
            non_resampled_radiance += path.x1_brdf * L_at_reconnection;
        } else {
            float target_function = luminance(path.x1_brdf * L_at_reconnection);
            float resampling_weight = target_function;

            weight_sum += resampling_weight;
            if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
                reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
                reservoir.sample_point_world_position = path.x2_position;
                reservoir.sample_point_world_normal = octahedral_encode(path.x2_normal);
                reservoir.radiance = L_at_reconnection;
                selected_target_function = target_function;
            }
        }
    }
}

void generate_emissive_candidate(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, ResolvedRayHitFull ray_hit, vec3 wi, float p_brdf, float ray_t, float p_nee, uint di_samples, uint bounce, inout uint rng) {
    float NdotV_hit = max(dot(ray_hit.world_normal, -wi), 0.0001);
    uint light_count = light_sources.length();
    float area_pdf = 1.0 / (float(light_count) * float(ray_hit.triangle_count) * ray_hit.triangle_area);
    float p_light = area_pdf * ray_t * ray_t / NdotV_hit;
    float emissive_mis_weight = power_heuristic(p_brdf, p_light * p_nee * float(di_samples));

    if (!path.x2_reusable) {
        non_resampled_radiance += path.x1_brdf * path.throughput_past_x1 * ray_hit.material.emissive * emissive_mis_weight;
        return;
    }

    if (bounce == 0u) {
        float target_function = luminance(path.x1_brdf * ray_hit.material.emissive) * emissive_mis_weight;
        float resampling_weight = luminance(path.x1_brdf * path.throughput_past_x1 * ray_hit.material.emissive) * emissive_mis_weight;

        weight_sum += resampling_weight;
        if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
            reservoir.light_sample = LightSample(NULL_LIGHT_ID, floatBitsToUint(area_pdf));
            reservoir.sample_point_world_position = path.x2_position;
            reservoir.sample_point_world_normal = octahedral_encode(path.x2_normal);
            reservoir.radiance = ray_hit.material.emissive;
            selected_target_function = target_function;
        }
    } else {
        vec3 emissive_L_at_reconnection = path.throughput_past_x1 * ray_hit.material.emissive * emissive_mis_weight;
        float target_function = luminance(path.x1_brdf * emissive_L_at_reconnection);
        float resampling_weight = target_function;

        weight_sum += resampling_weight;
        if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
            reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
            reservoir.sample_point_world_position = path.x2_position;
            reservoir.sample_point_world_normal = octahedral_encode(path.x2_normal);
            reservoir.radiance = emissive_L_at_reconnection;
            selected_target_function = target_function;
        }
    }
}

bool terminate_into_cache(inout Reservoir reservoir, inout float weight_sum, inout float selected_target_function, inout vec3 non_resampled_radiance, PathState path, ResolvedRayHitFull ray_hit, float ray_t, uint bounce, inout uint rng) {
    float p_term = mix(1.0, path.material.perceptual_roughness, path.material.metallic);
    bool stochastic_terminate = rand_f(rng) < p_term;
    bool forced_terminate = bounce == constants.max_bounces - 1u;
    if (!(stochastic_terminate || forced_terminate)) { return false; }

    uint rng_copy = rng;
    float world_cache_cell_size = get_cell_size(ray_hit.world_position, view.world_position, ray_t, rng_copy);
    if (ray_t <= sqrt(3.0) * world_cache_cell_size) { return false; }

    vec3 cached_radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, view.world_position, ray_t, WORLD_CACHE_CELL_LIFETIME, rng);

    vec3 cache_outgoing = (ray_hit.material.base_color / PI) * cached_radiance;
    vec3 cache_L_at_reconnection = path.throughput_past_x1 * cache_outgoing;
    if (!path.x2_reusable) {
        non_resampled_radiance += path.x1_brdf * cache_L_at_reconnection;
        return true;
    }

    float target_function = luminance(path.x1_brdf * cache_L_at_reconnection);
    float resampling_weight = target_function;
    weight_sum += resampling_weight;
    if (weight_sum > 0.0 && rand_f(rng) * weight_sum < resampling_weight) {
        reservoir.light_sample = LightSample(NULL_LIGHT_ID, 0u);
        reservoir.sample_point_world_position = path.x2_position;
        reservoir.sample_point_world_normal = octahedral_encode(path.x2_normal);
        reservoir.radiance = cache_L_at_reconnection;
        selected_target_function = target_function;
    }

    return true;
}

bool reconnection_reusable(float ray_t, float p_brdf, vec3 wi, bool diffuse_selected, ResolvedRayHitFull ray_hit, vec3 world_position, float x1_perceptual_roughness, float primary_NdotV) {
    float cos_x2 = max(dot(ray_hit.world_normal, -wi), 0.0001);
    float ray_footprint = (ray_t * ray_t) / (p_brdf * cos_x2);
    float primary_dist = length(view.world_position - world_position);
    float primary_footprint = 4.0 * PI * primary_dist * primary_dist / primary_NdotV;
    bool footprint_ok = ray_footprint >= (RECONNECTION_FOOTPRINT_KAPPA / 100.0) * primary_footprint;

    bool x1_lobe_ok = diffuse_selected || x1_perceptual_roughness >= RECONNECTION_ROUGHNESS_MIN;

    bool x2_is_light = any(greaterThan(ray_hit.material.emissive, vec3(0.0)));
    float x2_roughness = mix(1.0, ray_hit.material.perceptual_roughness, ray_hit.material.metallic);
    float x2_roughness_floor = RECONNECTION_ROUGHNESS_MIN * saturate(RECONNECTION_RELAX_DISTANCE / ray_t);
    bool x2_end_ok = x2_is_light || x2_roughness >= x2_roughness_floor;

    return footprint_ok && x1_lobe_ok && x2_end_ok;
}

InitialSamplingResult generate_initial_reservoir(vec3 world_position, vec3 world_normal, ResolvedMaterial material, uvec2 workgroup_id, uvec2 pixel_id, inout uint rng) {
    Reservoir reservoir = empty_reservoir();
    reservoir.confidence_weight = 1.0;

    vec3 non_resampled_radiance = vec3(0.0);
    float weight_sum = 0.0;
    float selected_target_function = 0.0;

#ifdef DLSS_RR_GUIDE_BUFFERS
    mat3 mirror_rotations = reflection_matrix(world_normal);
    bool psr_finished = material.roughness > MIRROR_ROUGHNESS_THRESHOLD || material.metallic <= 0.9999;
#endif

    vec3 wo = normalize(view.world_position - world_position);
    float primary_NdotV = max(dot(world_normal, wo), 0.0001);
    vec2 primary_F_ab = F_AB(material.perceptual_roughness, primary_NdotV);

    PathState path;
    path.ray_origin = world_position + (world_normal * RAY_T_MIN);
    path.normal = world_normal;
    path.wo = wo;
    path.material = material;
    path.throughput_past_x1 = vec3(1.0);
    path.x2_position = vec3(0.0);
    path.x2_normal = vec3(0.0);
    path.x2_reusable = false;
    path.x1_brdf = vec3(0.0);

    for (uint bounce = 0u; bounce < constants.max_bounces; bounce++) {
        float NdotV = max(dot(path.normal, path.wo), 0.0001);
        vec2 F_ab = F_AB(path.material.perceptual_roughness, NdotV);

        float p_nee = mix(1.0, path.material.perceptual_roughness, path.material.metallic);
        uint di_samples = (bounce == 0u) ? constants.primary_di_samples : constants.secondary_di_samples;
        generate_nee_candidate(reservoir, weight_sum, selected_target_function, non_resampled_radiance,
            path, F_ab, p_nee, di_samples, workgroup_id, bounce, rng);

        EvaluateAndSampleBrdfResult next_bounce = evaluate_and_sample_brdf(path.wo, path.normal, path.material, F_ab, rng);
        if (next_bounce.pdf == 0.0) { break; }
        RayIntersection ray = trace_ray(path.ray_origin, next_bounce.wi, RAY_T_MIN, RAY_T_MAX, gl_RayFlagsNoneEXT);
        if (ray.kind == RAY_QUERY_INTERSECTION_NONE) { break; }
        ResolvedRayHitFull ray_hit = resolve_ray_hit_full(ray);
        float p_brdf = next_bounce.pdf;

#ifdef DLSS_RR_GUIDE_BUFFERS
        if (!psr_finished) {
            if (!isinf(p_brdf)) {
                psr_finished = true;
            } else if (ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999) {
                mirror_rotations = mirror_rotations * reflection_matrix(ray_hit.world_normal);
            } else {
                psr_finished = true;
                replace_primary_surface(pixel_id, ray_hit, mirror_rotations, world_position);
            }
        }
#endif

        if (bounce == 0u) {
            path.x2_position = ray_hit.world_position;
            path.x2_normal = ray_hit.world_normal;

            path.x1_brdf = evaluate_brdf(wo, next_bounce.wi, world_normal, material, primary_F_ab);

            path.x2_reusable = reconnection_reusable(ray.t, p_brdf, next_bounce.wi, next_bounce.diffuse_selected, ray_hit, world_position, material.perceptual_roughness, primary_NdotV);

            path.throughput_past_x1 *= next_bounce.throughput / max(path.x1_brdf, vec3(0.0001));
        } else {
            path.throughput_past_x1 *= next_bounce.throughput;
        }

        if (any(greaterThan(ray_hit.material.emissive, vec3(0.0)))) {
            generate_emissive_candidate(reservoir, weight_sum, selected_target_function, non_resampled_radiance,
                path, ray_hit, next_bounce.wi, p_brdf, ray.t, p_nee, di_samples, bounce, rng);
        }

        if (terminate_into_cache(reservoir, weight_sum, selected_target_function, non_resampled_radiance, path, ray_hit, ray.t, bounce, rng)) {
            break;
        }

        path.ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
        path.normal = ray_hit.world_normal;
        path.wo = -next_bounce.wi;
        path.material = ray_hit.material;

        vec3 full_throughput = path.throughput_past_x1 * max(path.x1_brdf, vec3(0.0001));
        float rr = saturate(luminance(full_throughput));
        if (rand_f(rng) >= rr) { break; }
        path.throughput_past_x1 /= rr;
    }

    if (selected_target_function > 0.0) {
        reservoir.unbiased_contribution_weight = weight_sum / selected_target_function;
    }

    return InitialSamplingResult(reservoir, non_resampled_radiance);
}

// ============================================================================
// ReSTIR reuse (restir.wgsl) — used by temporal & spatial_and_shade
// ============================================================================
uvec2 get_neighbor_pixel_id(uvec2 center_pixel_id, float search_radius, inout uint rng) {
    vec2 spatial_id = vec2(center_pixel_id) + sample_disk(search_radius, rng);
    spatial_id = clamp(spatial_id, vec2(0.0), view.main_pass_viewport.zw - 1.0);
    return uvec2(spatial_id);
}

float jacobian(vec3 new_world_position, vec3 original_world_position, vec3 sample_point_world_position, vec3 sample_point_world_normal) {
    vec3 r = new_world_position - sample_point_world_position;
    vec3 q = original_world_position - sample_point_world_position;
    float rl = length(r);
    float ql = length(q);
    float phi_r = saturate(dot(r / rl, sample_point_world_normal));
    float phi_q = saturate(dot(q / ql, sample_point_world_normal));
    float jacobian_val = (phi_r * ql * ql) / (phi_q * rl * rl);
    return (isinf(jacobian_val) || isnan(jacobian_val)) ? 0.0 : jacobian_val;
}

ReservoirContribution reservoir_contribution(Reservoir reservoir, ResolvedLightSample resolved, vec3 world_position, vec3 world_normal, vec3 wo, ResolvedMaterial material, vec2 F_ab) {
    if (reservoir.light_sample.light_id != NULL_LIGHT_ID) {
        LightContribution light_contribution = calculate_resolved_light_contribution(resolved, world_position, world_normal);

        float nee_mis_weight = 1.0;
        if (light_contribution.brdf_rays_can_hit && light_contribution.inverse_solid_angle_pdf > 0.0) {
            uint light_count = light_sources.length();
            float inverse_solid_angle_pdf = light_contribution.inverse_solid_angle_pdf * float(light_count);
            float p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
            float p_nee_strategy = float(constants.primary_di_samples) * (1.0 / inverse_solid_angle_pdf) * p_nee;
            float p_brdf_at_nee = brdf_pdf(wo, light_contribution.wi, world_normal, material, F_ab);
            nee_mis_weight = power_heuristic(p_nee_strategy, p_brdf_at_nee);
        }

        vec3 brdf_radiance = light_contribution.radiance * evaluate_brdf(wo, light_contribution.wi, world_normal, material, F_ab) * nee_mis_weight;
        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), resolved.world_position);
    } else if (any(notEqual(reservoir.radiance, vec3(0.0)))) {
        vec3 delta = reservoir.sample_point_world_position - (world_position + world_normal * RAY_T_MIN);
        float sample_distance = length(delta);
        vec3 wi = delta / sample_distance;
        vec3 brdf_radiance = reservoir.radiance * evaluate_brdf(wo, wi, world_normal, material, F_ab);

        if (reservoir.light_sample.seed != 0u) {
            float area_pdf = uintBitsToFloat(reservoir.light_sample.seed);
            vec3 light_normal = octahedral_decode(reservoir.sample_point_world_normal);
            float cos_theta_light = max(dot(-wi, light_normal), 0.0001);
            float p_light = area_pdf * sample_distance * sample_distance / cos_theta_light;
            float p_nee = mix(1.0, material.perceptual_roughness, material.metallic);
            float p_brdf = brdf_pdf(wo, wi, world_normal, material, F_ab);
            brdf_radiance *= power_heuristic(p_brdf, p_light * p_nee * float(constants.primary_di_samples));
        }

        return ReservoirContribution(brdf_radiance, luminance(brdf_radiance), vec4(reservoir.sample_point_world_position, 1.0));
    } else {
        return ReservoirContribution(vec3(0.0), 0.0, vec4(reservoir.sample_point_world_position, 1.0));
    }
}

NeighborInfo load_temporal_reservoir(uvec2 pixel_id, float depth, vec3 world_position, vec3 world_normal) {
    if (bool(constants.reset)) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    vec2 motion_vector = texelFetch(motion_vectors, ivec2(pixel_id), 0).xy;
    vec2 temporal_pixel_id_float = round(vec2(pixel_id) - (motion_vector * view.main_pass_viewport.zw));

    uvec2 point_temporal_pixel_id = pixel_id;
    if (all(greaterThanEqual(temporal_pixel_id_float, vec2(0.0))) && all(lessThan(temporal_pixel_id_float, view.main_pass_viewport.zw))) {
        point_temporal_pixel_id = uvec2(temporal_pixel_id_float);
    }

    uint permute_rng = constants.frame_rng;
    uvec2 permuted_temporal_pixel_id = permute_pixel(point_temporal_pixel_id, rand_u(permute_rng), view.main_pass_viewport.zw);

    float temporal_depth = texelFetch(previous_depth_buffer, ivec2(permuted_temporal_pixel_id), 0).x;
    ResolvedGPixel temporal_surface = gpixel_resolve(texelFetch(previous_gbuffer, ivec2(permuted_temporal_pixel_id), 0), temporal_depth, permuted_temporal_pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
    if (pixel_dissimilar(depth, world_position, temporal_surface.world_position, world_normal, temporal_surface.world_normal, view)) {
        return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
    }

    uint temporal_pixel_index = permuted_temporal_pixel_id.x + permuted_temporal_pixel_id.y * uint(view.main_pass_viewport.z);
    NeighborInfo temporal = NeighborInfo(reservoirs_a[temporal_pixel_index], temporal_surface.world_position, temporal_surface.world_normal, temporal_surface.material);

    if (temporal.reservoir.light_sample.light_id != NULL_LIGHT_ID) {
        uint previous_light_id = temporal.reservoir.light_sample.light_id >> 16u;
        uint triangle_id = temporal.reservoir.light_sample.light_id & 0xFFFFu;
        uint light_id = previous_frame_light_id_translations[previous_light_id];
        if (light_id == LIGHT_NOT_PRESENT_THIS_FRAME) {
            return NeighborInfo(empty_reservoir(), vec3(0.0), vec3(0.0), empty_material());
        }
        temporal.reservoir.light_sample.light_id = (light_id << 16u) | triangle_id;
    }

    temporal.reservoir.confidence_weight = min(temporal.reservoir.confidence_weight, constants.confidence_weight_cap);

    return temporal;
}

NeighborInfo load_spatial_reservoir(uvec2 pixel_id, float depth, vec3 world_position, vec3 world_normal, inout uint rng) {
    for (uint i = 0u; i < 5u; i++) {
        uvec2 spatial_pixel_id = get_neighbor_pixel_id(pixel_id, SPATIAL_REUSE_RADIUS_PIXELS, rng);

        if (all(equal(spatial_pixel_id, pixel_id))) {
            continue;
        }

        float spatial_depth = texelFetch(depth_buffer, ivec2(spatial_pixel_id), 0).x;
        ResolvedGPixel spatial_surface = gpixel_resolve(texelFetch(gbuffer, ivec2(spatial_pixel_id), 0), spatial_depth, spatial_pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
        if (pixel_dissimilar(depth, world_position, spatial_surface.world_position, world_normal, spatial_surface.world_normal, view)) {
            continue;
        }

        uint spatial_pixel_index = spatial_pixel_id.x + spatial_pixel_id.y * uint(view.main_pass_viewport.z);
        return NeighborInfo(reservoirs_b[spatial_pixel_index], spatial_surface.world_position, spatial_surface.world_normal, spatial_surface.material);
    }

    return NeighborInfo(empty_reservoir(), world_position, world_normal, empty_material());
}

// The two cross-domain MIS visibility rays in merge_reservoirs are split into dedicated
// functions (each with its own ray-query loop) purely so Nsight attributes their traversal
// cost to distinct source lines instead of collapsing both onto the shared trace_ray. They
// are otherwise identical to trace_light_visibility.
float trace_visibility_other_at_canonical(vec3 ray_origin, vec4 light_sample_world_position) {
    vec3 ray_direction = light_sample_world_position.xyz;
    float ray_t_max = RAY_T_MAX;
    if (light_sample_world_position.w == 1.0) {
        vec3 ray = ray_direction - ray_origin;
        float dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }
    if (ray_t_max < RAY_T_MIN) { return 0.0; }

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, RAY_NO_CULL, ray_origin, RAY_T_MIN, ray_direction, ray_t_max);
    while (rayQueryProceedEXT(rq)) {}
    return float(rayQueryGetIntersectionTypeEXT(rq, true) == RAY_QUERY_INTERSECTION_NONE);
}

float trace_visibility_canonical_at_other(vec3 ray_origin, vec4 light_sample_world_position) {
    vec3 ray_direction = light_sample_world_position.xyz;
    float ray_t_max = RAY_T_MAX;
    if (light_sample_world_position.w == 1.0) {
        vec3 ray = ray_direction - ray_origin;
        float dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }
    if (ray_t_max < RAY_T_MIN) { return 0.0; }

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, RAY_NO_CULL, ray_origin, RAY_T_MIN, ray_direction, ray_t_max);
    while (rayQueryProceedEXT(rq)) {}
    return float(rayQueryGetIntersectionTypeEXT(rq, true) == RAY_QUERY_INTERSECTION_NONE);
}

ReservoirMergeResult merge_reservoirs(Reservoir canonical_reservoir, vec3 canonical_world_position, vec3 canonical_world_normal, ResolvedMaterial canonical_material, Reservoir other_reservoir, vec3 other_world_position, vec3 other_world_normal, ResolvedMaterial other_material, vec3 other_view_position, bool is_spatial, inout uint rng) {
    ResolvedLightSample canonical_resolved;
    if (canonical_reservoir.light_sample.light_id != NULL_LIGHT_ID) {
        canonical_resolved = resolve_light_sample(canonical_reservoir.light_sample, light_sources[canonical_reservoir.light_sample.light_id >> 16u]);
    }

    vec3 canonical_wo = normalize(view.world_position - canonical_world_position);
    float canonical_NdotV = max(dot(canonical_world_normal, canonical_wo), 0.0001);
    vec2 canonical_F_ab = F_AB(canonical_material.perceptual_roughness, canonical_NdotV);
    ReservoirContribution canonical_sample_at_canonical = reservoir_contribution(canonical_reservoir, canonical_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);

    if (other_reservoir.confidence_weight == 0.0) {
        return ReservoirMergeResult(canonical_reservoir, canonical_sample_at_canonical.brdf_radiance);
    }

    ResolvedLightSample other_resolved;
    if (other_reservoir.light_sample.light_id != NULL_LIGHT_ID) {
        other_resolved = resolve_light_sample(other_reservoir.light_sample, light_sources[other_reservoir.light_sample.light_id >> 16u]);
    }
    vec3 other_wo = normalize(other_view_position - other_world_position);
    float other_NdotV = max(dot(other_world_normal, other_wo), 0.0001);
    vec2 other_F_ab = F_AB(other_material.perceptual_roughness, other_NdotV);

    ReservoirContribution other_sample_at_canonical = reservoir_contribution(other_reservoir, other_resolved, canonical_world_position, canonical_world_normal, canonical_wo, canonical_material, canonical_F_ab);
    ReservoirContribution canonical_sample_at_other = reservoir_contribution(canonical_reservoir, canonical_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);
    ReservoirContribution other_sample_at_other = reservoir_contribution(other_reservoir, other_resolved, other_world_position, other_world_normal, other_wo, other_material, other_F_ab);

    float other_sample_at_canonical_jacobian = 1.0;
    if (other_reservoir.light_sample.light_id == NULL_LIGHT_ID) {
        other_sample_at_canonical_jacobian = jacobian(
            canonical_world_position,
            other_world_position,
            other_reservoir.sample_point_world_position,
            octahedral_decode(other_reservoir.sample_point_world_normal)
        );
    }
    float canonical_sample_at_other_jacobian = 1.0;
    if (canonical_reservoir.light_sample.light_id == NULL_LIGHT_ID) {
        canonical_sample_at_other_jacobian = jacobian(
            other_world_position,
            canonical_world_position,
            canonical_reservoir.sample_point_world_position,
            octahedral_decode(canonical_reservoir.sample_point_world_normal)
        );
    }

    if (other_sample_at_canonical_jacobian < 0.125 || other_sample_at_canonical_jacobian > 8.0) {
        other_sample_at_canonical_jacobian = 0.0;
    }
    if (canonical_sample_at_other_jacobian < 0.125 || canonical_sample_at_other_jacobian > 8.0) {
        canonical_sample_at_other_jacobian = 0.0;
    }

    if (other_sample_at_canonical.target_function > 0.0 && other_sample_at_canonical_jacobian > 0.0) {
        float visibility = trace_visibility_other_at_canonical(canonical_world_position + canonical_world_normal * RAY_T_MIN, other_sample_at_canonical.sample_world_position);
        other_sample_at_canonical.target_function *= visibility;
    }
    if (canonical_sample_at_other.target_function > 0.0 && canonical_sample_at_other_jacobian > 0.0) {
        float visibility = trace_visibility_canonical_at_other(other_world_position + other_world_normal * RAY_T_MIN, canonical_sample_at_other.sample_world_position);
        canonical_sample_at_other.target_function *= visibility;
    }

    float total_confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    float defensive_t_c = float(is_spatial) * ((total_confidence_weight > 0.0) ? canonical_reservoir.confidence_weight / total_confidence_weight : 1.0);

    float canonical_balance_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_sample_at_canonical.target_function,
        other_reservoir.confidence_weight * canonical_sample_at_other.target_function * canonical_sample_at_other_jacobian
    );
    float canonical_sample_mis_weight = mix(canonical_balance_mis_weight, 1.0, defensive_t_c);
    float canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_sample_at_canonical.target_function * canonical_reservoir.unbiased_contribution_weight;

    float other_balance_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_sample_at_other.target_function,
        canonical_reservoir.confidence_weight * other_sample_at_canonical.target_function * other_sample_at_canonical_jacobian
    );
    float other_sample_mis_weight = mix(other_balance_mis_weight, 0.0, defensive_t_c);
    float other_sample_resampling_weight = other_sample_mis_weight * other_sample_at_canonical.target_function * other_reservoir.unbiased_contribution_weight * other_sample_at_canonical_jacobian;

    Reservoir combined_reservoir = empty_reservoir();
    combined_reservoir.confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    float weight_sum = canonical_sample_resampling_weight + other_sample_resampling_weight;

    if (weight_sum > 0.0 && rand_f(rng) * weight_sum < other_sample_resampling_weight) {
        combined_reservoir.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = other_reservoir.radiance;
        combined_reservoir.light_sample = other_reservoir.light_sample;

        float inverse_target_function = (other_sample_at_canonical.target_function > 0.0) ? 1.0 / other_sample_at_canonical.target_function : 0.0;
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, other_sample_at_canonical.brdf_radiance);
    } else {
        combined_reservoir.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined_reservoir.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined_reservoir.radiance = canonical_reservoir.radiance;
        combined_reservoir.light_sample = canonical_reservoir.light_sample;

        float inverse_target_function = (canonical_sample_at_canonical.target_function > 0.0) ? 1.0 / canonical_sample_at_canonical.target_function : 0.0;
        combined_reservoir.unbiased_contribution_weight = weight_sum * inverse_target_function;

        return ReservoirMergeResult(combined_reservoir, canonical_sample_at_canonical.brdf_radiance);
    }
}
