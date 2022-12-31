#define_import_path bevy_pbr::environment_map

@group(3) @binding(0)
var environment_map_diffuse: texture_cube<f32>;
@group(3) @binding(1)
var environment_map_specular: texture_cube<f32>;
@group(3) @binding(2)
var environment_map_sampler: sampler;

let ENVIRONMENT_MAP_SPECULAR_SMALLEST_MIP_LEVEL = 10.0;

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn environment_map_light(
    perceptual_roughness: f32, 
    roughness: f32, 
    diffuse_color: vec3<f32>, 
    NdotV: f32, 
    f_ab: vec2<f32>, 
    N: vec3<f32>, 
    R: vec3<f32>, 
    F0: vec3<f32>,
) -> EnvironmentMapLight {

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    let irradiance = textureSample(environment_map_diffuse, environment_map_sampler, N).rgb;
    let radiance = textureSampleLevel(environment_map_specular, environment_map_sampler, R, perceptual_roughness * ENVIRONMENT_MAP_SPECULAR_SMALLEST_MIP_LEVEL).rgb;

    // Multiscattering approximation: https://www.jcgt.org/published/0008/01/03/paper.pdf
    // Useful reference: https://bruop.github.io/ibl
    let Fr = max(vec3(1.0 - roughness), F0) - F0;
    let kS = F0 + Fr * pow(1.0 - NdotV, 5.0);
    let FssEss = kS * f_ab.x + f_ab.y;
    let Ess = f_ab.x + f_ab.y;
    let Ems = 1.0 - Ess;
    let Favg = F0 + (1.0 - F0) / 21.0;
    let Fms = FssEss * Favg / (1.0 - Ems * Favg);
    let Edss = 1.0 - (FssEss + Fms * Ems);
    let kD = diffuse_color * Edss;

    var out: EnvironmentMapLight;
    out.diffuse = (Fms * Ems + kD) * irradiance;
    out.specular = FssEss * radiance;
    return out;
}
