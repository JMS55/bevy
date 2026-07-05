"""Re-export an ORCA Zero-Day FBX as a single self-contained glTF binary (.glb) for
the `zero_day` Bevy example.

Zero-Day (NVIDIA ORCA) ships each measure (`MEASURE_ONE`, `MEASURE_SEVEN`, ...) as an
`.fbx` plus a sibling `tex/` folder of `.dds` textures. This script converts whichever
`.fbx` you pass; the example's README lists the per-measure output names. Bevy can't load
FBX, and -- more importantly -- Blender's FBX
importer mis-reads this Octane-exported asset's material conventions (it routes the
ORM map into `KHR_materials_specular`, turns the BaseColor opacity into alpha blend,
and drops most emissive maps). So instead of trusting the imported material graph,
this rebuilds every material from the naming convention documented in the download's
README:

    <name>_BaseColor.dds  RGB = base color            (sRGB)
    <name>_Specular.dds   R = occlusion, G = roughness, B = metallic (Non-Color, ORM)
    <name>_Normal.dds     DirectX normal map          (Non-Color)
    <name>_Emissive.dds   RGB = emissive color        (sRGB)

Roughness (G) + metallic (B) from the shared `_Specular` image pack into a single
glTF `metallicRoughnessTexture`. Normals are left DirectX-convention and green-flipped
in the Bevy example (`flip_normal_map_y`), matching how the `bistro` example handles
Sponza. Occlusion (the R channel) is left out -- it needs the fragile glTF-settings
node group and contributes little.

Meshes the FBX marks hidden (which the film never renders, but which the glTF exporter
would happily export as visible, solid geometry) are deleted -- see the comment at the
deletion site.

Usage:
    blender --background --python convert.py -- <input.fbx> <output.glb>
"""

import glob
import os
import sys

import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if len(argv) != 2:
    raise SystemExit(
        "usage: blender --background --python convert.py -- <input.fbx> <output.glb>"
    )
src, dst = argv
# Absolute so Blender can still resolve the textures when it re-reads them at export time
# (linking the BaseColor alpha makes the exporter re-open each .dds; a relative path that
# worked at load time fails there because the export CWD/base differs).
texdir = os.path.abspath(os.path.join(os.path.dirname(src), "tex"))

# Start from an empty scene and import the FBX (for geometry, material assignment,
# and names). We rebuild the materials below, so we only need Blender's import for
# the mesh <-> material mapping.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, use_image_search=True)

# The FBX statically hides many meshes the film never renders (~1,700 in Measure
# Seven): proxy tubes with placeholder materials that the flythrough camera spends the
# whole take *inside*, ON/OFF light-state variants, duplicated machinery. Octane and
# Blender skip hidden objects at render time, but the glTF exporter exports them as
# ordinary visible geometry and Bevy has no per-object visibility to recover the
# distinction -- rasterized or ray-traced (Solari treats every BLAS triangle as an
# opaque occluder), they read as solid walls wrapped around the camera and phantom
# emissive lights. Delete them. Geometry is stripped first so a hidden mesh that still
# parents other objects can keep its node -- its children inherit its (possibly
# animated) transform -- while rendering nothing; the rest delete leaf-first so no
# child is ever orphaned into a different world transform.
hidden = [
    obj
    for obj in bpy.context.scene.objects
    if obj.type == "MESH"
    and (obj.hide_render or obj.hide_viewport or not obj.visible_get())
]
for obj in hidden:
    obj.data = bpy.data.meshes.new(obj.name + "_hidden")
removed = 0
remaining = set(hidden)
while True:
    deletable = [obj for obj in remaining if not obj.children]
    if not deletable:
        break
    for obj in deletable:
        remaining.discard(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        removed += 1
print(
    "ZERO_DAY_HIDDEN removed=%d kept_as_empty_nodes=%d" % (removed, len(remaining))
)

# Index the texture set by base name: {base_lower: {channel_lower: path}}.
tex = {}
for path in glob.glob(os.path.join(texdir, "*.dds")):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" not in stem:
        continue
    base, chan = stem.rsplit("_", 1)
    tex.setdefault(base.lower(), {})[chan.lower()] = path


def base_for_material(mat):
    """The texture base name for a material, from its ORCA material NAME.

    The name is the authored identity the download's README keys `tex/` by, and the mapping
    is deterministic, so match on it alone -- don't guess from whatever BaseColor image
    Blender's FBX import happens to link, which can point at a different set and drop the
    material's emissive (the scene's only light). A name that doesn't resolve is reported in
    `skipped` so the mismatch surfaces instead of being papered over with a wrong guess."""
    nm = mat.name.lower()
    # The full name, then the name with any trailing "_suffix" (e.g. Blender's "_c4d")
    # stripped; dict.fromkeys keeps order and drops the duplicate when they are equal.
    for cand in dict.fromkeys((nm, nm.rsplit("_", 1)[0])):
        if cand in tex:
            return cand
    return None


def load_image(path, non_color):
    img = bpy.data.images.load(os.path.abspath(path), check_existing=True)
    img.filepath = os.path.abspath(path)  # keep it absolute for the export-time re-read
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


_cutout_cache = {}


def has_cutout_texels(img, cutoff=0.5):
    """True if any texel's alpha falls below the mask cutoff.

    Only the ~30 decal/label materials actually use their BaseColor alpha; every other
    texture's alpha plane is solid 1.0. Linking the alpha indiscriminately would export
    ~90 fully-opaque materials as alpha-masked, which costs the raster prepass early-z
    for discards that can never happen -- so a material only becomes a cutout when its
    texture really cuts something out."""
    key = img.filepath
    if key not in _cutout_cache:
        result = False
        if img.channels == 4 and img.size[0] > 0 and img.size[1] > 0:
            pixels = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
            img.pixels.foreach_get(pixels)
            result = bool((pixels[3 :: img.channels] < cutoff).any())
        _cutout_cache[key] = result
    return _cutout_cache[key]


def rebuild(mat, base):
    channels = tex[base]
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    basecolor_node = None
    if "basecolor" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["basecolor"], non_color=False)
        nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])
        basecolor_node = n

    if "specular" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["specular"], non_color=True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(n.outputs["Color"], sep.inputs["Color"])
        # ORM: G -> roughness, B -> metallic (shared image packs into one glTF map).
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    if "normal" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["normal"], non_color=True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if "emissive" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["emissive"], non_color=False)
        nt.links.new(n.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    # The BaseColor alpha is an opacity map: the decal/label materials are see-through
    # outside their glyphs. Preserve that as an alpha CLIP (cutout), not BLEND: Bevy
    # renders blended materials in a forward pass that Solari's deferred-G-buffer
    # primary visibility can't see, while a hard cutout stays in the G-buffer and just
    # discards the see-through texels. Materials whose alpha plane is solid 1.0 (nearly
    # all of them) stay opaque -- a mask that never discards would only cost perf.
    if basecolor_node is not None and has_cutout_texels(basecolor_node.image):
        nt.links.new(basecolor_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = 0.5
        except (AttributeError, TypeError):
            pass


rebuilt = 0
skipped = []
for mat in bpy.data.materials:
    base = base_for_material(mat)
    if base is None:
        skipped.append(mat.name)
        continue
    rebuild(mat, base)
    rebuilt += 1
print("ZERO_DAY_MATERIALS rebuilt=%d skipped=%d" % (rebuilt, len(skipped)))
if skipped:
    print("ZERO_DAY_SKIPPED", skipped[:20])

# Report the world-space mesh bounds so the example's scale/camera can be tuned.
lo = mathutils.Vector((1e30, 1e30, 1e30))
hi = -lo
for obj in bpy.context.scene.objects:
    if obj.type != "MESH":
        continue
    for corner in obj.bound_box:
        v = obj.matrix_world @ mathutils.Vector(corner)
        lo = mathutils.Vector((min(lo.x, v.x), min(lo.y, v.y), min(lo.z, v.z)))
        hi = mathutils.Vector((max(hi.x, v.x), max(hi.y, v.y), max(hi.z, v.z)))
print(
    "ZERO_DAY_BOUNDS min=(%.2f, %.2f, %.2f) max=(%.2f, %.2f, %.2f)"
    % (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
)

# The actions run past the scene's playback range (e.g. Measure One's camera runs to
# frame 412, Measure Seven's to 441, while the scene ends at 250), so extend the range to
# cover every action. Otherwise the baked animation -- and the flythrough -- is cut short.
max_frame = 1
for obj in bpy.context.scene.objects:
    ad = obj.animation_data
    if ad and ad.action:
        max_frame = max(max_frame, int(round(ad.action.frame_range[1])))
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = max_frame
print("ZERO_DAY_FRAME_RANGE 1..%d" % max_frame)

# Export a single self-contained .glb. glTF is Y-up; the scene has no real lights, but it
# carries the film's animated camera (named per measure -- `DynamicCamera2`,
# `DynamicCamera`, ...) and ~550-640 animated objects. `export_animation_mode="SCENE"`
# bakes them into ONE clip over the scene frame range, so every object stays on the film's
# shared timeline -- exporting per-action instead gives each object its own duration, and
# looping them makes the short ones race.
bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_yup=True,
    export_cameras=True,
    export_lights=False,
    export_animations=True,
    export_animation_mode="SCENE",
    export_apply=True,
)
print("ZERO_DAY_EXPORT_DONE", dst)


def patch_alpha_modes_to_mask(path, cutoff=0.5):
    """Rewrite every ``alphaMode: BLEND`` material in the exported .glb to ``MASK``.

    Blender's glTF exporter emits ``BLEND`` for any material with a linked opacity, regardless
    of blend mode. ``BLEND`` is wrong for the Bevy Solari example: blended surfaces render in a
    forward pass and never populate the deferred G-buffer Solari resolves primary visibility
    from, so they'd vanish from the trace. ``MASK`` (alpha cutout) keeps opaque texels in the
    G-buffer and only discards the see-through ones -- the see-through shells and decal
    backgrounds read through, everything else stays solid. Patches the JSON chunk in place;
    the binary chunk is untouched."""
    import json
    import struct

    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    json_len, json_type = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20 : 20 + json_len])
    bin_chunk = data[20 + json_len :]  # BIN chunk header + payload, unchanged

    patched = 0
    for material in gltf.get("materials", []):
        if material.get("alphaMode") == "BLEND":
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = cutoff
            patched += 1

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)  # glTF pads the JSON chunk with spaces
    total = 12 + 8 + len(new_json) + len(bin_chunk)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", magic, version, total))
        f.write(struct.pack("<II", len(new_json), json_type))
        f.write(new_json)
        f.write(bin_chunk)
    print("ZERO_DAY_ALPHA_MASK patched=%d materials" % patched)


patch_alpha_modes_to_mask(dst)
