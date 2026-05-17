#import bevy_solari::scene_bindings::{materials, simplified_materials, textures, TEXTURE_MAP_NONE}

@compute @workgroup_size(64, 1, 1)
fn simplify_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let material_id = global_id.x;
    if material_id >= arrayLength(&materials) { return; }

    let material = materials[material_id];
    var simplified_material = simplified_materials[material_id];

    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        simplified_material.base_color *= load_last_mip_texel(material.base_color_texture_id);
    }

    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        simplified_material.emissive *= load_last_mip_texel(material.emissive_texture_id);
    }

    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = load_last_mip_texel(material.metallic_roughness_texture_id);
        simplified_material.perceptual_roughness *= metallic_roughness.g;
        simplified_material.roughness = simplified_material.perceptual_roughness * simplified_material.perceptual_roughness;
        simplified_material.metallic *= metallic_roughness.b;
    }

    simplified_materials[material_id] = simplified_material;
}

fn load_last_mip_texel(texture_id: u32) -> vec3<f32> {
    let last_mip = textureNumLevels(textures[texture_id]) - 1u;
    return textureLoad(textures[texture_id], vec2<i32>(0, 0), i32(last_mip)).rgb;
}
