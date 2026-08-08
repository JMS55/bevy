enable wgpu_ray_query;
#define_import_path bevy_solari::resolve_dlss_rr_textures

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0, calculate_F0_dielectric}
#import bevy_render::utils::octahedral_decode
#import bevy_solari::brdf::{F_AB, lobe_reflectances}
#import bevy_solari::gbuffer_utils::{gpixel_resolve, ResolvedGPixel}
#import bevy_solari::realtime_bindings::{gbuffer, depth_buffer, motion_vectors, view, previous_view, constants, view_output, diffuse_albedo, specular_albedo, normal_roughness, specular_motion_vectors, dlss_rr_depth, dlss_rr_motion_vectors}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, MIRROR_ROUGHNESS_THRESHOLD, RAY_T_MIN, RAY_T_MAX, ResolvedMaterial, ResolvedRayHitFull}

@compute @workgroup_size(8, 8, 1)
fn resolve_dlss_rr_textures(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    if any(pixel_id >= vec2u(view.main_pass_viewport.zw)) { return; }

    let depth = textureLoad(depth_buffer, pixel_id, 0);
    let surface_motion_vector = textureLoad(motion_vectors, pixel_id, 0).xy;

    // Defaults describe this pixel's own surface, which is the right answer for everything that
    // isn't a mirror. `follow_mirror_chain` overwrites all of them when it replaces the surface.
    textureStore(specular_motion_vectors, pixel_id, vec4(surface_motion_vector, vec2(0.0)));
    textureStore(dlss_rr_motion_vectors, pixel_id, vec4(surface_motion_vector, vec2(0.0)));
    textureStore(dlss_rr_depth, pixel_id, vec4(depth));

    if depth == 0.0 {
        psr_debug_write(pixel_id, vec3(0.0));
        textureStore(diffuse_albedo, pixel_id, vec4(0.0));
        textureStore(specular_albedo, pixel_id, vec4(0.5));
        textureStore(normal_roughness, pixel_id, vec4(0.0, 0.0, 1.0, 0.0));
        return;
    }

    let surface = gpixel_resolve(textureLoad(gbuffer, pixel_id, 0), depth, pixel_id, view.main_pass_viewport.zw, view.world_from_clip);

    let wo = normalize(view.world_position - surface.world_position);
    let NdotV = max(dot(surface.world_normal, wo), 0.0001);
    let split = delta_split(surface.material, NdotV);

    // Anything with a delta lobe gets the chain followed. A near-pure mirror has its surface replaced
    // outright; a merely polished one keeps its own depth and motion and only mixes the reflected
    // surface into the guide buffers, because the pixel really is mostly still itself.
    if split.fraction > 0.0
        && follow_mirror_chain(pixel_id, surface, split, wo)
    {
        return;
    }
    if split.fraction == 0.0 {
        psr_debug_write(pixel_id, PSR_DEBUG_NOT_DELTA);
    }

    let F0 = calculate_F0(surface.material.base_color, surface.material.metallic, vec3(surface.material.reflectance));

    textureStore(diffuse_albedo, pixel_id, vec4(calculate_diffuse_color(surface.material.base_color, surface.material.metallic, 0.0, 0.0), 0.0));
    textureStore(specular_albedo, pixel_id, vec4(env_brdf_approx2(F0, surface.material.roughness, surface.world_normal, wo), 0.0));
    textureStore(normal_roughness, pixel_id, vec4(surface.world_normal, surface.material.perceptual_roughness));
}

// False-colour classification, written when `psr_debug_overlay` is set. ReSTIR skips its own writes
// to `view_output` in that case, so these survive to the screen.
//
// The distinction that matters is dark grey versus yellow or red: a surface that was never eligible
// looks the same in the final image as one that was eligible and refused, but the causes are
// unrelated and so are the fixes.
const PSR_DEBUG_NOT_DELTA = vec3(0.04, 0.04, 0.05);
const PSR_DEBUG_REPLACED = vec3(0.0, 0.65, 0.1);
const PSR_DEBUG_BLENDED = vec3(0.1, 0.3, 0.9);
const PSR_DEBUG_TOO_CURVED = vec3(0.85, 0.65, 0.0);
const PSR_DEBUG_CHAIN_FAILED = vec3(0.85, 0.08, 0.04);
const PSR_DEBUG_ENVIRONMENT = vec3(0.55, 0.15, 0.75);

fn psr_debug_write(pixel_id: vec2<u32>, color: vec3<f32>) {
    if constants.psr_debug_overlay != 0u {
        textureStore(view_output, pixel_id, vec4(color, 1.0));
    }
}

// Roughness, as alpha, below which a surface gets a virtual specular motion vector at all. About 0.25
// perceptual.
//
// Deliberately much looser than `MIRROR_ROUGHNESS_THRESHOLD`, and separate from it. Using the strict
// delta threshold here put a cliff in the middle of the roughness range: a floor whose roughness
// texture straddled it had neighbouring pixels alternately given a virtual and a surface motion
// vector, so the denoiser saw an incoherent motion field and smeared. The cliff, not either value,
// was the defect.
//
// Two separate questions were being answered with one number. Swapping a surface out demands its lobe
// really be delta, because a virtual position is meaningless otherwise. Describing how its reflection
// *moves* does not: it changes no light transport, and the mirror direction stands in for the lobe
// centre well past the delta range — NRD's dominant-direction fit is identically the mirror direction
// below 0.279 perceptual.
const SPECULAR_GUIDE_ROUGHNESS_THRESHOLD = 0.0625;

// A surface is replaced outright only when nearly all of its reflectance leaves through the delta
// lobe. Below that it still contributes, but as a blend rather than a takeover.
const FULL_REPLACEMENT_FRACTION = 0.9;

struct DeltaSplit {
    // Colored reflectance of the delta lobe. This is literally what multiplies whatever the chain
    // finds, so it doubles as the per-bounce chain throughput.
    specular: vec3<f32>,
    diffuse: vec3<f32>,
    // Specular share of the total, in 0..1.
    fraction: f32,
}

// Splits a surface's reflectance into the part that goes through the delta lobe and the part that
// does not. Replaces the old `metallic > 0.9999` test: what matters is not whether a surface is a
// metal but whether its mirror lobe carries the energy, which is a question a dielectric can also
// answer yes to.
fn delta_split(material: ResolvedMaterial, NdotV: f32) -> DeltaSplit {
    if material.roughness > SPECULAR_GUIDE_ROUGHNESS_THRESHOLD {
        return DeltaSplit(vec3(0.0), vec3(0.0), 0.0);
    }
    if constants.psr_dielectric == 0u {
        // Legacy behaviour, kept for A/B: metals only, and all-or-nothing when they qualify. A unit
        // specular reflectance makes the chain tint an identity, so this really is the old path.
        if material.metallic <= 0.9999 {
            return DeltaSplit(vec3(0.0), vec3(0.0), 0.0);
        }
        return DeltaSplit(vec3(1.0), vec3(0.0), 1.0);
    }
    let F_ab = F_AB(material.perceptual_roughness, NdotV);
    let rho = lobe_reflectances(
        material.base_color,
        calculate_F0_dielectric(vec3(material.reflectance)),
        material,
        F_ab,
    );
    let specular_luminance = luminance(rho.specular);
    let fraction = specular_luminance / max(specular_luminance + luminance(rho.diffuse), 0.0001);
    return DeltaSplit(rho.specular, rho.diffuse, fraction);
}

// Whether this surface should be replaced outright rather than blended.
//
// Evaluated at normal incidence on purpose. Specular reflectance climbs toward 1 at grazing angles,
// so a view-dependent test would flip a polished floor between replaced and not as the camera moved
// — reintroducing exactly the frame-to-frame guide instability this pass exists to remove. At normal
// incidence the answer is a property of the material and cannot change under motion.
fn is_delta_mirror(material: ResolvedMaterial) -> bool {
    // The strict threshold, not the loose guide one: replacement moves the surface, which is only
    // meaningful when the lobe is genuinely a delta.
    return material.roughness <= MIRROR_ROUGHNESS_THRESHOLD
        && delta_split(material, 1.0).fraction >= FULL_REPLACEMENT_FRACTION;
}

// How far a reflector's virtual image may shift between neighbouring pixels, as a fraction of the
// virtual distance itself, before the surface is treated as too curved to replace.
//
// Being a ratio is the point. An angle-per-pixel threshold silently changes meaning with render
// resolution, which is untenable when DLSS renders below output resolution and the quality mode
// would then decide which surfaces get replaced.
const MAX_VIRTUAL_DEPTH_JITTER = 0.002;

// Even a perfectly flat surface measures a small nonzero normal change, because G-buffer normals are
// 24-bit octahedral and neighbouring texels round differently. Subtract that floor: without it a flat
// mirror reflecting something far away fails the test purely on encoding noise, since the jitter term
// scales with reflection length.
const NORMAL_QUANTIZATION_FLOOR = 0.002;

// Distance, in metres, standing in for "infinitely far" when a reflection ray finds nothing.
//
// Large enough that camera translation moves the image by well under a pixel at any sane scene scale,
// so what remains is the rotation term — which is the whole content of a distant reflection's motion.
// Not larger, because the virtual position still goes through a `clip_from_world` multiply, and
// pushing it out to 1e30 spends float precision to say the same thing.
const ENVIRONMENT_VIRTUAL_DISTANCE = 10000.0;


fn load_gbuffer_normal(pixel_id: vec2<u32>) -> vec3<f32> {
    return octahedral_decode(unpack_24bit_normal(textureLoad(gbuffer, pixel_id, 0).a));
}

// Angle, in radians, that the shading normal turns over one pixel — a screen-space stand-in for
// curvature. Noise floor removed, so a flat surface reads as exactly zero.
//
// Two alternatives were tried and both were worse. A geometric normal reconstructed from depth ignores
// normal maps, but is flat across a triangle and jumps at the seams, so a smooth-shaded low-poly sphere
// came out as replaced facets with refused creases between them — reflections follow the interpolated
// shading normal, so that is what has to be measured. Widening the baseline and normalising by it does
// suppress normal-map pollution, but widens the refused border around every mirror by the same amount,
// which costs more at silhouettes than the pollution costs on flat panels.
//
// So: one pixel, shading normals, and accept that surface detail reads as some curvature.
//
// Also fires at object boundaries, where the neighbour sits on unrelated geometry. That costs a
// one-pixel border around every mirror and errs toward not replacing, the safe way to be wrong.
fn reflector_curvature(pixel_id: vec2<u32>, normal: vec3<f32>) -> f32 {
    let max_pixel_id = vec2<u32>(view.main_pass_viewport.zw) - vec2(1u);
    let right = load_gbuffer_normal(min(pixel_id + vec2(1u, 0u), max_pixel_id));
    let down = load_gbuffer_normal(min(pixel_id + vec2(0u, 1u), max_pixel_id));

    // For unit vectors |a - b| is the angle between them to first order, and |a - b|^2 = 2(1 - a.b).
    let min_cos = min(dot(normal, right), dot(normal, down));
    let angle = sqrt(max(0.0, 2.0 * (1.0 - min_cos)));
    return max(0.0, angle - NORMAL_QUANTIZATION_FLOOR);
}

// Walk the specular chain from the primary surface. Unlike the path tracer, this takes the mirror
// direction directly instead of sampling the BRDF, so the result is identical every frame for a
// static camera - which is the whole reason this lives here and not inside the ReSTIR loop.
//
// A glossy surface is not a mirror and never walks a chain: it takes one ray along its lobe centre
// and stops, whatever it lands on. The lobe centre is a fair stand-in for a slightly rough lobe, but
// each further bounce blurs it again, so a chain of them describes nothing in particular — and the
// result is only ever used to say how far away the reflection is.
//
// Returns true if the guide buffers were written and the caller should stop. The one case that writes
// something and still returns false is a glossy reflection that found nothing: it has a specular motion
// vector to record, but no reflected surface, so its own albedo and normal are left to the caller.
fn follow_mirror_chain(pixel_id: vec2<u32>, surface: ResolvedGPixel, primary_split: DeltaSplit, primary_wo: vec3<f32>) -> bool {
    let primary_position = surface.world_position;
    let primary_normal = surface.world_normal;
    // Decided once, from the surface alone. Everything that separates the glossy path from the mirror
    // path keys off this, and re-deriving it per bounce invites the two halves to disagree.
    let primary_is_mirror = is_delta_mirror(surface.material);
    // Keyed on roughness alone, deliberately — not on `!primary_is_mirror`. A black polished panel
    // fails the mirror test on its *energy* split while still having a razor-thin lobe; it is a
    // smooth dielectric, not a glossy surface, and it wants the chain and the blend it already gets.
    // What makes a surface glossy is that its lobe has measurable width.
    let glossy = constants.psr_glossy != 0u && surface.material.roughness > MIRROR_ROUGHNESS_THRESHOLD;

    let camera_to_primary = primary_position - view.world_position;
    let primary_distance = length(camera_to_primary);
    let primary_direction = camera_to_primary / primary_distance;

    let curvature = reflector_curvature(pixel_id, primary_normal);

    var mirror_rotations = reflection_matrix(primary_normal);
    var ray_origin = primary_position + (primary_normal * RAY_T_MIN);
    var normal = primary_normal;
    var wo = -primary_direction;
    var path_length = primary_distance;
    // Product of each bounce's specular reflectance. This is what the reflected surface's colour gets
    // multiplied by before it reaches the eye, so it is both the chain tint and, for a dielectric,
    // the reason a reflection contributes only a few percent of the pixel.
    var chain_throughput = primary_split.specular;

    // TODO: This wants an independent cap rather than borrowing the lighting bounce count, but
    // matching it keeps this pass byte-identical to the previous in-ReSTIR implementation.
    for (var step = 0u; step < constants.max_bounces; step++) {
        let wi = reflect(-wo, normal);
        let ray = trace_ray(ray_origin, wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            // Nothing out there, so the reflection is the environment and sits effectively at
            // infinity. That is a real answer, not a failure: a distant image does not parallax, so it
            // shifts only with camera rotation, and saying so is much better than the surface motion
            // vector this pixel would otherwise keep — which claims the reflection is painted on and
            // slides with the floor.
            //
            // Only for glossy. Replacing a surface outright needs a virtual *depth* too, and a depth
            // at infinity handed to the denoiser next to a neighbour's real one is exactly the
            // shattered depth field the curvature gate exists to avoid.
            if glossy {
                let far_position = view.world_position + (primary_direction * ENVIRONMENT_VIRTUAL_DISTANCE);
                textureStore(specular_motion_vectors, pixel_id, vec4(calculate_motion_vector(far_position, far_position), vec2(0.0)));
                psr_debug_write(pixel_id, PSR_DEBUG_ENVIRONMENT);
                return false;
            }
            psr_debug_write(pixel_id, PSR_DEBUG_CHAIN_FAILED);
            return false;
        }
        let ray_hit = resolve_ray_hit_full(ray);
        path_length += ray.t;

        // `glossy` terminates unconditionally: one ray, then describe whatever it found, even another
        // mirror. Following a mirror on from a glossy surface would be describing a reflection of a
        // reflection with a lobe that was already too wide to be a direction.
        if glossy || !is_delta_mirror(ray_hit.material) {
            // A normal turning by some angle turns the reflected ray by twice that, and the virtual
            // image sits at the reflection distance, so neighbouring pixels' virtual depths differ
            // by roughly 2 * curvature * reflection_length. Once that is a meaningful fraction of
            // the depth itself, the denoiser reads a shattered surface and blurs instead of
            // reprojecting — worse than never having replaced the surface at all.
            //
            // Only knowable here, after the chain has run: the same reflector is fine looking at
            // something close and unusable looking at something far, and curvature alone cannot
            // tell those apart. Containment only; correcting the distance for curvature is the
            // actual fix, and would keep these pixels rather than discarding them.
            //
            // Glossy pixels are exempt, because the gate is about *depth*. They write no virtual
            // depth — only a specular motion vector, which the denoiser applies to a signal that is
            // already blurry and which has no neighbour-to-neighbour continuity requirement. Refusing
            // them cost every curved polished surface its motion vector to protect a buffer those
            // pixels never touch.
            let reflection_length = path_length - primary_distance;
            if constants.psr_skip_curved_reflectors != 0u
                && !glossy
                && 2.0 * curvature * reflection_length > MAX_VIRTUAL_DEPTH_JITTER * path_length
            {
                psr_debug_write(pixel_id, PSR_DEBUG_TOO_CURVED);
                return false;
            }

            // Unfolding a specular path about a chain of planes straightens it into a single
            // straight ray from the camera, so the virtual image sits along the primary direction at
            // the total path length. Exact for planar mirrors at any chain length, and only a scalar
            // to carry - where reflecting the hit position about the first mirror's plane is exact
            // for one mirror and drifts for two, because later mirrors reflect about their own plane.
            //
            // Curvature is deliberately ignored: a convex reflector's image really sits nearer than
            // this. NVIDIA's own DLSS-RR sample does the same, and NRD notes it costs reprojection
            // artefacts on curved surfaces rather than breaking outright.
            var virtual_position = view.world_position + (primary_direction * path_length);
            if constants.psr_unfold_along_camera_ray == 0u {
                // Legacy construction, kept for A/B: fold the whole chain about the first mirror.
                virtual_position = (mirror_rotations * (ray_hit.world_position - primary_position)) + primary_position;
            }
            replace_primary_surface(pixel_id, surface, primary_split, primary_wo, ray_hit, mirror_rotations, virtual_position, chain_throughput, primary_is_mirror);
            return true;
        }

        // Still in the mirror chain, so fold this mirror in and keep going. Accumulating by
        // right-multiplication gives M1 * M2 * ... * Mk, which is the order that unfolds the path
        // correctly when applied to the final hit. Reversing it looks fine on parallel mirrors and
        // is badly wrong on skew ones.
        mirror_rotations = mirror_rotations * reflection_matrix(ray_hit.world_normal);
        ray_origin = ray_hit.world_position + (ray_hit.geometric_world_normal * RAY_T_MIN);
        wo = -wi;
        normal = ray_hit.world_normal;
        chain_throughput *= delta_split(ray_hit.material, max(dot(normal, wo), 0.0001)).specular;
    }

    // Ran out of bounces still inside the mirror chain.
    psr_debug_write(pixel_id, PSR_DEBUG_CHAIN_FAILED);
    return false;
}

// https://en.wikipedia.org/wiki/Householder_transformation
fn reflection_matrix(plane_normal: vec3<f32>) -> mat3x3<f32> {
    // N times Nᵀ
    let n_nt = mat3x3<f32>(
        plane_normal * plane_normal.x,
        plane_normal * plane_normal.y,
        plane_normal * plane_normal.z,
    );
    let identity_matrix = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    return identity_matrix - n_nt * 2.0;
}

// Describe the chain's terminating surface, reflected into the mirror's virtual space, to the guide
// buffers. For a near-pure mirror that means replacing this pixel's surface outright; for a merely
// polished one it means mixing the reflection into what is already there.
// https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/#primary_surface_replacement
fn replace_primary_surface(
    pixel_id: vec2<u32>,
    surface: ResolvedGPixel,
    primary_split: DeltaSplit,
    primary_wo: vec3<f32>,
    ray_hit: ResolvedRayHitFull,
    mirror_rotations: mat3x3<f32>,
    virtual_position: vec3<f32>,
    chain_throughput: vec3<f32>,
    // Whether this surface is being replaced outright, or merely having its reflection's motion
    // described. A glossy surface gets the latter: its own depth, motion and normal are still the
    // truth about it, and only the specular motion vector is new.
    replace_fully: bool,
) {
    // The position comes from path length, so only the object's frame-to-frame *motion* needs the
    // chain applied to it. Mirroring the delta is what makes a moving reflection track properly;
    // note this assumes the mirrors themselves are static, since a moving mirror would drag the
    // whole virtual world with it.
    let world_motion = ray_hit.previous_frame_world_position - ray_hit.world_position;
    let virtual_previous_position = virtual_position + (mirror_rotations * world_motion);
    let specular_motion_vector = calculate_motion_vector(virtual_position, virtual_previous_position);

    let wo = normalize(view.world_position - virtual_position);
    let virtual_normal = normalize(mirror_rotations * ray_hit.world_normal);

    // Depth of the reflection rather than of the mirror, so the denoiser reprojects it at its true
    // optical distance. This has to move together with the motion vectors below: the integration
    // guide's one statement about depth is that it must be the same data the motion vectors describe.
    // Jittered `clip_from_world` to match the prepass depth we copy for every other pixel, while the
    // motion vectors use the unjittered matrices, also matching the prepass convention.
    if replace_fully && constants.psr_virtual_depth != 0u {
        let virtual_clip_position = view.clip_from_world * vec4(virtual_position, 1.0);
        textureStore(dlss_rr_depth, pixel_id, vec4(virtual_clip_position.z / virtual_clip_position.w));
        textureStore(dlss_rr_motion_vectors, pixel_id, vec4(specular_motion_vector, vec2(0.0)));
    }

    textureStore(specular_motion_vectors, pixel_id, vec4(specular_motion_vector, vec2(0.0)));

    // The albedo guides are documented as this pixel's reflectance, and that is exactly what this
    // is: the chain's tint times what the chain found, plus whatever the surface reflects on its own
    // account. No threshold and no lerp — a metal mirror contributes all of the first term and none
    // of the second, a polished floor mostly the reverse, and everything between falls out.
    //
    // The tint is what makes a gold mirror's reflection read as gold rather than as the wall behind
    // it, and for a dielectric it is also why the reflection is only a few percent of the pixel.
    // NVIDIA's DLSS-RR sample applies the same product to both albedo guides.
    let reflected_diffuse = calculate_diffuse_color(ray_hit.material.base_color, ray_hit.material.metallic, 0.0, 0.0);
    let reflected_F0 = calculate_F0(ray_hit.material.base_color, ray_hit.material.metallic, vec3(ray_hit.material.reflectance));
    let reflected_specular = env_brdf_approx2(reflected_F0, ray_hit.material.roughness, virtual_normal, wo);

    let own_diffuse = calculate_diffuse_color(surface.material.base_color, surface.material.metallic, 0.0, 0.0);
    let own_F0 = calculate_F0(surface.material.base_color, surface.material.metallic, vec3(surface.material.reflectance));
    let own_specular = env_brdf_approx2(own_F0, surface.material.roughness, surface.world_normal, primary_wo);

    var tint = chain_throughput;
    if constants.psr_tint_albedo == 0u { tint = vec3(1.0); }

    textureStore(diffuse_albedo, pixel_id, vec4(tint * reflected_diffuse + own_diffuse, 0.0));
    // The surface's own specular reflectance is already the chain's first factor, so adding it back
    // would count it twice — only the reflected surface's own specular term is new here.
    textureStore(specular_albedo, pixel_id, vec4(mix(own_specular, tint * reflected_specular, primary_split.fraction), 0.0));

    // Normals cannot be summed the way reflectance can, and averaging a floor's normal with a
    // reflected object's produces one that describes neither — RTXPT blends theirs across planes and
    // it is the one part of their approach worth not copying. Hand over the reflected surface only
    // when it genuinely dominates, and otherwise leave this pixel's own normal alone.
    if replace_fully {
        textureStore(normal_roughness, pixel_id, vec4(virtual_normal, ray_hit.material.perceptual_roughness));
        psr_debug_write(pixel_id, PSR_DEBUG_REPLACED);
    } else {
        textureStore(normal_roughness, pixel_id, vec4(surface.world_normal, surface.material.perceptual_roughness));
        psr_debug_write(pixel_id, PSR_DEBUG_BLENDED);
    }
}

fn calculate_motion_vector(world_position: vec3<f32>, previous_world_position: vec3<f32>) -> vec2<f32> {
    let clip_position_t = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    let clip_position = clip_position_t.xy / clip_position_t.w;
    let previous_clip_position_t = previous_view.unjittered_clip_from_world * vec4(previous_world_position, 1.0);
    let previous_clip_position = previous_clip_position_t.xy / previous_clip_position_t.w;
    // Motion vectors are UV-space offsets in [-1, 1], from one corner to the diagonally-opposite one.
    // A clip-space diagonal difference is in [-2, 2], so scale by 0.5, and flip y since V goes down
    // where clip-space y goes up.
    return (clip_position - previous_clip_position) * vec2(0.5, -0.5);
}

fn env_brdf_approx2(specular_color: vec3<f32>, alpha: f32, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    let NoV = abs(dot(N, V));

    var X: vec4<f32>;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    var Y: vec4<f32>;
    Y.x = 1.0;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    let M1 = mat2x2<f32>(0.99044, 1.29678, -1.28514, -0.755907);
    let M2 = mat3x3<f32>(1.0, 20.3225, 121.563, 2.92338, -27.0302, 626.13, 59.4188, 222.592, 316.627);
    let M3 = mat2x2<f32>(0.0365463, 9.0632, 3.32707, -9.04756);
    let M4 = mat3x3<f32>(1.0, 9.04401, 5.56589, 3.59685, -16.3174, 19.7886, -1.36772, 9.22949, -20.2123);

    var bias = dot(M1 * X.xy, Y.xy) / dot(M2 * X.xyw, Y.xyw);
    let scale = dot(M3 * X.xy, Y.xy) / dot(M4 * X.xzw, Y.xyw);

    bias *= saturate(specular_color.g * 50.0);

    return fma(specular_color, vec3(max(0.0, scale)), vec3(max(0.0, bias)));
}
