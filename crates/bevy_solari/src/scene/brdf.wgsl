#define_import_path bevy_solari::brdf

#import bevy_pbr::lighting::{F_AB, D_GGX, V_SmithGGXCorrelated, fresnel, specular_multiscatter}
#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_render::maths::PI
#import bevy_solari::scene_bindings::{ResolvedMaterial, MIRROR_ROUGHNESS_THRESHOLD}

fn evaluate_diffuse_brdf(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial) -> vec3<f32> {
    let diffuse_color = calculate_diffuse_color(material.base_color, material.metallic, 0.0, 0.0) / PI;

    let LdotH = saturate(dot(wi, world_normal));
    let F0 = calculate_F0(material.base_color, material.metallic, material.reflectance);
    let F = fresnel(F0, LdotH);

    return diffuse_color * saturate(dot(world_normal, wi)) * (1.0 - F);
}

fn evaluate_specular_brdf(wo: vec3<f32>, wi: vec3<f32>, world_normal: vec3<f32>, material: ResolvedMaterial) -> vec3<f32> {
    let H = normalize(wi + wo);
    let NdotL = saturate(dot(world_normal, wi));
    let NdotH = saturate(dot(world_normal, H));
    let LdotH = saturate(dot(wi, world_normal));
    let NdotV = max(dot(world_normal, wo), 0.0001);

    let F0 = calculate_F0(material.base_color, material.metallic, material.reflectance);
    let F = fresnel(F0, LdotH);

    if material.roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        return F;
    }

    let D = D_GGX(material.roughness, NdotH);
    let Vs = V_SmithGGXCorrelated(material.roughness, NdotV, NdotL);
    let F_ab = F_AB(material.perceptual_roughness, NdotV);
    return specular_multiscatter(D, Vs, F, F0, F_ab, 1.0) * NdotL;
}
