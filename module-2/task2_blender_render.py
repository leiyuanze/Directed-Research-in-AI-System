"""Headless Blender renders for the Task 2 report (optional Blender step).

Run with:
    blender --background --factory-startup --python task2_blender_render.py -- manifest.json

The manifest is JSON: {"jobs": [{"ply": "...", "out": "...", "title_color": null, "azim": 35, "elev": 18}, ...]}
Each job is rendered into its own scene with an auto-framed camera, a sun plus
an area fill light, and vertex colours when the imported mesh has them.
"""

import json
import math
import sys

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_ply(path):
    try:
        bpy.ops.wm.ply_import(filepath=path)
    except Exception:
        bpy.ops.import_mesh.ply(filepath=path)
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"no mesh imported from {path}")
    return objs[0]


def setup_material(obj, job):
    mat = bpy.data.materials.new("render_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.8
    attrs = [a.name for a in obj.data.color_attributes] if hasattr(obj.data, "color_attributes") else []
    print("colour attributes:", attrs)
    fixed = job.get("base_color")
    if attrs and not fixed:
        node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        node.layer_name = attrs[0]
        mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    elif fixed:
        bsdf.inputs["Base Color"].default_value = tuple(fixed) + (1.0,)
    else:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.75, 1.0)
    obj.data.materials.append(mat)


def frame_camera(obj, job):
    coords = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(coords, Vector()) / 8.0
    extent = [max(c[i] for c in coords) - min(c[i] for c in coords) for i in range(3)]
    size = max(extent) or 1.0

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 55.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    az = math.radians(job.get("azim", 35.0))
    el = math.radians(job.get("elev", 18.0))
    dist = size * job.get("dist_factor", 1.9)
    cam.location = center + Vector((
        dist * math.cos(el) * math.sin(az),
        -dist * math.cos(el) * math.cos(az),
        dist * math.sin(el),
    ))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(35))

    fill_data = bpy.data.lights.new("fill", "AREA")
    fill_data.energy = 260.0
    fill_data.size = max(size, 0.3) * 2.0
    fill = bpy.data.objects.new("fill", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = center + Vector((-dist, dist * 0.8, dist * 0.6))
    fill.rotation_euler = (center - fill.location).to_track_quat("-Z", "Y").to_euler()


def setup_world():
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.92, 0.92, 0.95, 1.0)
    bg.inputs[1].default_value = 1.0
    bpy.context.scene.world = world


def render(job):
    clear_scene()
    obj = import_ply(job["ply"])
    for poly in obj.data.polygons:
        poly.use_smooth = True
    setup_material(obj, job)
    frame_camera(obj, job)
    setup_world()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = job.get("res", 1000)
    scene.render.resolution_y = job.get("res", 1000)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = job["out"]
    bpy.ops.render.render(write_still=True)
    print("rendered:", job["out"])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    manifest = json.loads(open(argv[0], encoding="utf-8").read())
    for job in manifest["jobs"]:
        render(job)


if __name__ == "__main__":
    main()