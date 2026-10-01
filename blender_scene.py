"""
Animated 3D warehouse twin in Blender.

Headless build:
    blender -b --python blender_scene.py -- trajectories.json out.blend

Inside Blender:
    Open a .blend built by this script, click "Allow Execution" if Blender asks,
    then press N in the 3D viewport -> "Warehouse" tab. Set the number of forklifts
    and the comparison, then click "Run simulation & rebuild".
    (Or: Scripting tab -> open this file -> Run Script, which adds the same panel.)

Vehicles
--------
If models/forklift.obj exists next to this script, every vehicle is that forklift,
with a coloured floor halo: red = human driver, blue = autonomous (plus a beacon).
Tweak FORKLIFT_* below if a different model faces the wrong way or is the wrong size.

Swap in your own models
-----------------------
Template collections (hidden, under "Templates"): Tmpl_human, Tmpl_amr,
Tmpl_coordinated, Tmpl_rack (one rack bay), Tmpl_pack (a loading-dock door).
Every vehicle/rack is an *instance* of those, so replacing a template's contents
updates them all. Model convention: metres, origin on the floor at its centre,
facing +X. Rebuilding keeps your templates.
"""
import importlib
import json
import math
import os
import sys

import bmesh
import bpy

JSON_PATH = r"trajectories.json"
CELL = 2.0                           # metres per grid cell
RACK_H = 3.2
WALL_H = 2.4
FRAMES_PER_STEP = 3
GAP_CELLS = 8                        # space between warehouses

FORKLIFT_OBJ = "models/forklift.obj"
FORKLIFT_AXES = ("Y", "Z")           # OBJ import (forward_axis, up_axis)
FORKLIFT_YAW = 90                    # degrees to turn the model so its forks face +X
FORKLIFT_LENGTH = 1.8                # metres, longest horizontal side after scaling
METRES_PER_ORDER = 0.12              # length of the green orders-completed bar

WALL, FREE, PACK, BARRIER = 0, 1, 2, 3      # must match sim.py
FLEET_COLORS = {"human": (0.85, 0.25, 0.2), "amr": (0.95, 0.65, 0.2), "coordinated": (0.15, 0.45, 0.85)}

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()


# ----------------------------------------------------------------------------- helpers
def material(name, color, emission=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = emission
    m.diffuse_color = (*color, 1)
    return m


def boxes_mesh(name, boxes, mat):
    """boxes: list of (cx, cy, cz, sx, sy, sz) in metres -> one mesh object."""
    from mathutils import Matrix
    bm = bmesh.new()
    for cx, cy, cz, sx, sy, sz in boxes:
        mtx = Matrix.Translation((cx, cy, cz)) @ Matrix.Diagonal((sx, sy, sz, 1))
        bmesh.ops.create_cube(bm, size=1.0, matrix=mtx)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return bpy.data.objects.new(name, me)


def cylinder_obj(name, r, h, z, mat):
    from mathutils import Matrix
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=r, radius2=r, depth=h,
                          matrix=Matrix.Translation((0, 0, z)))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return bpy.data.objects.new(name, me)


def text_obj(name, body, loc, size, col, align="CENTER", mat=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.align_x = align
    cu.size = size
    if mat:
        cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    col.objects.link(o)
    return o


def ensure_collection(name, parent):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    return col


def find_layer_collection(lc, name):
    if lc.collection.name == name:
        return lc
    for ch in lc.children:
        r = find_layer_collection(ch, name)
        if r:
            return r
    return None


# ----------------------------------------------------------------------------- forklift
def forklift_mesh(path):
    """Import the OBJ once, merge its parts, and normalise it: origin on the floor
    at its centre, forks facing +X, FORKLIFT_LENGTH long. Cached as 'forklift_mesh'."""
    me = bpy.data.meshes.get("forklift_mesh")
    if me:
        return me
    import numpy as np
    from mathutils import Matrix
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis=FORKLIFT_AXES[0], up_axis=FORKLIFT_AXES[1])
    parts = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0)
    centre = Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    scale = FORKLIFT_LENGTH / max(hi[0] - lo[0], hi[1] - lo[1])
    me.transform(Matrix.Scale(scale, 4) @ Matrix.Rotation(math.radians(FORKLIFT_YAW), 4, "Z") @ centre)
    me.name = "forklift_mesh"
    bpy.data.objects.remove(obj, do_unlink=True)
    return me


# ----------------------------------------------------------------------------- templates
def build_templates(forklift_path=None):
    root = bpy.context.scene.collection
    tmpl = ensure_collection("Templates", root)

    def make(name, objs_fn):
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
            tmpl.children.link(col)
        if len(col.all_objects) == 0:           # only create placeholders if empty
            for o in objs_fn():
                col.objects.link(o)
        return col

    def forklift(kind):
        def fn():
            fk = bpy.data.objects.new(f"forklift_{kind}", forklift_mesh(forklift_path))
            halo = material(f"mat_halo_{kind}", FLEET_COLORS[kind], emission=1.5)
            objs = [fk, cylinder_obj(f"halo_{kind}", 1.0, 0.02, 0.01, halo)]
            if kind == "coordinated":
                beacon = material("mat_beacon", (0.3, 0.8, 1.0), emission=6.0)
                objs.append(cylinder_obj("beacon", 0.12, 0.2, 2.3, beacon))
            return objs
        return fn

    def placeholder(kind):
        def fn():
            m = material(f"mat_{kind}", FLEET_COLORS[kind])
            return [boxes_mesh(f"{kind}_body", [(0, 0, 0.3, 1.6, 1.0, 0.5), (0.6, 0, 0.7, 0.3, 0.8, 0.4)], m)]
        return fn

    def rack():
        steel = material("mat_rack", (0.2, 0.35, 0.6))
        wood = material("mat_pallet", (0.6, 0.45, 0.25))
        s = CELL
        posts = [(x, y, RACK_H / 2, 0.08, 0.08, RACK_H)
                 for x in (-s / 2 + 0.05, s / 2 - 0.05) for y in (-s / 2 + 0.05, s / 2 - 0.05)]
        shelves = [(0, 0, z, s, s, 0.05) for z in (0.15, 1.15, 2.15, RACK_H)]
        boxes = [(0, 0, z + 0.35, s * 0.8, s * 0.8, 0.6) for z in (0.2, 1.2, 2.2)]
        return [boxes_mesh("rack_frame", posts + shelves, steel), boxes_mesh("rack_boxes", boxes, wood)]

    def dock():
        frame = material("mat_dock", (0.95, 0.75, 0.1))
        door = material("mat_dock_door", (0.35, 0.37, 0.4))
        return [boxes_mesh("dock_frame", [(0, -CELL * 0.45, 1.5, 0.3, 0.12, 3.0),
                                          (0, CELL * 0.45, 1.5, 0.3, 0.12, 3.0),
                                          (0, 0, 3.0, 0.3, CELL, 0.15)], frame),
                boxes_mesh("dock_door", [(0.05, 0, 1.5, 0.12, CELL * 0.85, 2.9)], door)]

    use_fk = forklift_path and os.path.exists(forklift_path)
    cols = {
        "human": make("Tmpl_human", forklift("human") if use_fk else placeholder("human")),
        "amr": make("Tmpl_amr", forklift("amr") if use_fk else placeholder("amr")),
        "coordinated": make("Tmpl_coordinated", forklift("coordinated") if use_fk else placeholder("coordinated")),
        "rack": make("Tmpl_rack", rack),
        "pack": make("Tmpl_pack", dock),
    }
    lc = find_layer_collection(bpy.context.view_layer.layer_collection, "Templates")
    if lc:
        lc.exclude = True                       # templates don't render at the origin
    return cols


def instance(name, col, loc, parent_col):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = col
    e.location = loc
    parent_col.objects.link(e)
    return e


# ----------------------------------------------------------------------------- counters
def counter_node_group():
    """Geometry Nodes group that turns an animated number into 3D text, so the
    counters update every frame without any Python running during playback."""
    ng = bpy.data.node_groups.get("TwinCounter")
    if ng:
        return ng
    ng = bpy.data.node_groups.new("TwinCounter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Value", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket("Size", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket("Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    v2s = n.new("FunctionNodeValueToString")
    s2c = n.new("GeometryNodeStringToCurves")
    real = n.new("GeometryNodeRealizeInstances")
    fill = n.new("GeometryNodeFillCurve")
    setm = n.new("GeometryNodeSetMaterial")
    L = ng.links.new
    L(gi.outputs["Value"], v2s.inputs["Value"])
    L(v2s.outputs["String"], s2c.inputs["String"])
    L(gi.outputs["Size"], s2c.inputs["Size"])
    L(s2c.outputs["Curve Instances"], real.inputs["Geometry"])
    L(real.outputs["Geometry"], fill.inputs["Curve"])
    L(fill.outputs["Mesh"], setm.inputs["Geometry"])
    L(gi.outputs["Material"], setm.inputs["Material"])
    L(setm.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def counter_obj(name, loc, size, mat, frames, values, col):
    ng = counter_node_group()
    o = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    o.location = loc
    col.objects.link(o)
    mod = o.modifiers.new("Counter", "NODES")
    mod.node_group = ng
    ids = {it.name: it.identifier for it in ng.interface.items_tree
           if getattr(it, "in_out", None) == "INPUT" and it.name != "Geometry"}
    mod[ids["Size"]] = size
    mod[ids["Material"]] = mat
    path = f'modifiers["Counter"]["{ids["Value"]}"]'
    mod[ids["Value"]] = float(values[0])
    o.keyframe_insert(data_path=path, frame=frames[0])
    for fc in fcurves_of(o):
        if fc.data_path == path:
            set_keys(fc, frames, [float(v) for v in values], interp="CONSTANT")
    return o


# ----------------------------------------------------------------------------- animation
def fcurves_of(obj):
    """Works with both legacy and slotted (Blender 4.4+/5.x) actions."""
    ad = obj.animation_data
    act = ad.action
    if hasattr(act, "fcurves") and len(getattr(act, "fcurves", [])):
        return list(act.fcurves)
    from bpy_extras import anim_utils
    cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
    return list(cb.fcurves)


def set_keys(fc, frames, values, interp="LINEAR"):
    """Bulk-write keyframes, dropping ones where the value holds steady."""
    keep_f, keep_v = [], []
    n = len(values)
    for i in range(n):
        if 0 < i < n - 1 and values[i] == values[i - 1] and values[i] == values[i + 1]:
            continue
        keep_f.append(frames[i])
        keep_v.append(values[i])
    kp = fc.keyframe_points
    while len(kp):
        kp.remove(kp[0], fast=True)
    kp.add(len(keep_f))
    kp.foreach_set("co", [c for pair in zip(keep_f, keep_v) for c in pair])
    for k in kp:
        k.interpolation = interp
    fc.update()


def animate(obj, frames, xs, ys, rots):
    obj.keyframe_insert("location", frame=frames[0])
    obj.keyframe_insert("rotation_euler", index=2, frame=frames[0])
    for fc in fcurves_of(obj):
        if fc.data_path == "location":
            vals = (xs, ys, [obj.location.z] * len(xs))[fc.array_index]
            set_keys(fc, frames, vals)
        elif fc.data_path == "rotation_euler":
            set_keys(fc, frames, rots)


# ----------------------------------------------------------------------------- build
def clear_warehouse():
    old = bpy.data.collections.get("Warehouse")
    if not old:
        return
    for o in list(old.all_objects):
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and data.users == 0 and data.name != "forklift_mesh":
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
    for c in list(old.children_recursive):
        bpy.data.collections.remove(c)
    bpy.data.collections.remove(old)
    for a in list(bpy.data.actions):
        if a.users == 0:
            bpy.data.actions.remove(a)


def build(data, forklift_path=None):
    scene = bpy.context.scene
    if not bpy.data.filepath:                  # fresh file: clear Blender's default objects
        for n in ("Cube", "Light", "Camera"):
            o = bpy.data.objects.get(n)
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
    clear_warehouse()
    wh = bpy.data.collections.new("Warehouse")
    scene.collection.children.link(wh)
    tmpl = build_templates(forklift_path)

    floor_mat = material("mat_floor", (0.55, 0.55, 0.52))
    hall_mat = material("mat_hall_floor", (0.45, 0.47, 0.5))
    lane_mat = material("mat_shortcut", (0.6, 0.2, 0.85), emission=1.5)
    wall_mat = material("mat_wall", (0.78, 0.74, 0.68))
    dock_mark = material("mat_dock_mark", (0.95, 0.75, 0.1), emission=0.8)
    bar_mat = material("mat_bar", (0.2, 0.8, 0.3), emission=0.5)
    text_mat = material("mat_text", (0.95, 0.95, 0.95), emission=1.0)
    good_mat = material("mat_text_good", (0.3, 0.95, 0.4), emission=2.0)
    warn_mat = material("mat_text_warn", (1.0, 0.45, 0.25), emission=2.0)

    steps = data["steps"]
    n_agents = len(data["scenes"][0]["kinds"])
    frames = [1 + i * FRAMES_PER_STEP for i in range(steps)]
    extents = []
    for idx, sc in enumerate(data["scenes"]):
        grid = sc["grid"]
        H, W = len(grid), len(grid[0])
        x_off = sc.get("col", idx) * (W + GAP_CELLS) * CELL
        y_off = -sc.get("row", 0) * (H + GAP_CELLS + 6) * CELL
        col = bpy.data.collections.new(sc["label"])
        wh.children.link(col)

        def xy(cell_or_rc):
            r, c = divmod(cell_or_rc, W) if isinstance(cell_or_rc, int) else cell_or_rc
            return x_off + c * CELL, y_off + (H - 1 - r) * CELL

        # floor (hallway = columns left of the first building-wall cell)
        hall_w = next((c for c in range(W) if any(grid[r][c] == BARRIER for r in range(H))), 0)
        cx, cy = x_off + (W - 1) * CELL / 2, y_off + (H - 1) * CELL / 2
        col.objects.link(boxes_mesh(f"floor_{idx}", [(cx, cy, -0.05, W * CELL, H * CELL, 0.1)], floor_mat))
        if hall_w:
            hx = x_off + (hall_w - 1) * CELL / 2
            col.objects.link(boxes_mesh(f"hall_floor_{idx}", [(hx, cy, -0.04, hall_w * CELL, H * CELL, 0.1)], hall_mat))

        # racks, building walls, loading docks
        walls = []
        dock_marks = []
        for r in range(H):
            for c in range(W):
                x, y = xy((r, c))
                v = grid[r][c]
                if v == WALL:
                    instance(f"rack_{idx}_{r}_{c}", tmpl["rack"], (x, y, 0), col)
                elif v == BARRIER:
                    walls.append((x, y, WALL_H / 2, CELL * 0.3, CELL, WALL_H))
                elif v == PACK:
                    instance(f"dock_{idx}_{r}_{c}", tmpl["pack"], (x - CELL * 0.65, y, 0), col)
                    dock_marks.append((x, y, 0.006, CELL * 0.9, CELL * 0.9, 0.01))
        # outer wall behind the docks
        walls.append((x_off - CELL * 0.8, cy, WALL_H / 2, 0.3, H * CELL, WALL_H))
        col.objects.link(boxes_mesh(f"walls_{idx}", walls, wall_mat))
        if dock_marks:
            col.objects.link(boxes_mesh(f"dock_marks_{idx}", dock_marks, dock_mark))

        if sc["shortcut"]:
            lane = [(*xy(i), 0.007, CELL, CELL, 0.01) for i in sc["shortcut_cells"]]
            col.objects.link(boxes_mesh(f"shortcut_{idx}", lane, lane_mat))

        # title
        text_obj(f"title_{idx}", sc["label"], (cx, y_off + H * CELL + 0.5, 0.05), 2.2, col, mat=text_mat)

        # vehicles
        pos = sc["positions"]
        for i, kind in enumerate(sc["kinds"]):
            xs, ys, rots = [], [], []
            heading = None
            for s in range(steps):
                x, y = xy(pos[s][i])
                if s > 0 and (x, y) != (xs[-1], ys[-1]):
                    target = math.atan2(y - ys[-1], x - xs[-1])
                    if heading is None:
                        heading = target
                        rots = [target] * len(rots)
                    heading += (target - heading + math.pi) % (2 * math.pi) - math.pi
                xs.append(x)
                ys.append(y)
                rots.append(heading if heading is not None else 0.0)
            e = instance(f"{kind}_{idx}_{i:03d}", tmpl[kind], (xs[0], ys[0], 0), col)
            animate(e, frames, xs, ys, rots)

        # scoreboard in front of the warehouse
        base_y = y_off - 2.2 * CELL
        bar = boxes_mesh(f"orders_bar_{idx}", [(0.5, 0, 0.25, 1, 1, 0.5)], bar_mat)
        bar.location = (x_off - CELL / 2, base_y, 0)
        bar.scale = (0.001, 1.2, 1)
        col.objects.link(bar)
        bar.keyframe_insert("scale", index=0, frame=1)
        orders = sc["orders"]
        sub = list(range(0, steps, 5)) + [steps - 1]
        for fc in fcurves_of(bar):
            set_keys(fc, [frames[s] for s in sub], [max(0.001, orders[s] * METRES_PER_ORDER) for s in sub])

        lx = x_off - CELL / 2
        text_obj(f"lbl_orders_{idx}", "Orders completed", (lx, base_y - 3.2, 0.05), 1.6, col, "LEFT", text_mat)
        counter_obj(f"num_orders_{idx}", (lx + 17, base_y - 3.2, 0.05), 2.2, good_mat, frames, orders, col)
        text_obj(f"lbl_wait_{idx}", "Stuck in traffic now", (lx, base_y - 6.2, 0.05), 1.6, col, "LEFT", text_mat)
        waiting = sc.get("waiting", [0] * steps)
        counter_obj(f"num_wait_{idx}", (lx + 17, base_y - 6.2, 0.05), 2.2, warn_mat, frames, waiting, col)
        text_obj(f"lbl_fleet_{idx}", f"of {n_agents} forklifts", (lx + 21, base_y - 6.2, 0.05), 1.6, col, "LEFT", text_mat)

        extents.append((x_off - 3 * CELL, x_off + W * CELL, base_y - 8, y_off + (H + 2) * CELL))

    # camera, light, world, render settings
    x0, x1 = min(e[0] for e in extents), max(e[1] for e in extents)
    y0, y1 = min(e[2] for e in extents), max(e[3] for e in extents)
    total_w, total_h = x1 - x0, y1 - y0
    center = ((x0 + x1) / 2, (y0 + y1) / 2, 0)
    target = bpy.data.objects.new("CameraTarget", None)
    target.location = center
    wh.objects.link(target)
    cam = bpy.data.objects.get("TwinCamera")
    if cam is None:
        cam = bpy.data.objects.new("TwinCamera", bpy.data.cameras.new("TwinCamera"))
    # orthographic, slightly tilted: every panel the same size, counters readable
    tilt = math.radians(20)                     # angle from straight down
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(total_w, total_h * math.cos(tilt) * 16 / 9) * 1.04
    cam.data.clip_end = 2000
    dist = 300
    cam.location = (center[0], center[1] - dist * math.sin(tilt), dist * math.cos(tilt))
    cam.constraints.clear()
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    wh.objects.link(cam)
    scene.camera = cam

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(35), math.radians(10), math.radians(30))
    wh.objects.link(sun)

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if scene.world.node_tree is None:
        scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
        bg.inputs["Strength"].default_value = 0.8

    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            pass
    scene.frame_start = 1
    scene.frame_end = frames[-1]
    scene.render.fps = 30
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene["twin_dir"] = HERE
    embed_ui_loader()


def find_forklift():
    p = os.path.join(HERE, FORKLIFT_OBJ)
    return p if os.path.exists(p) else None


# ----------------------------------------------------------------------------- Blender UI
COMPARE_ITEMS = [
    ("fleets", "Humans vs autonomous", "Human drivers vs central planner, same layout"),
    ("shortcut", "Shortcut closed vs open", "One fleet, with and without the shortcut"),
    ("all", "All four (2x2)", "Humans/autonomous x shortcut closed/open"),
]
FLEET_ITEMS = [("human", "Human drivers", ""), ("coordinated", "Autonomous", ""), ("amr", "Naive robots", "")]


class TWIN_OT_rebuild(bpy.types.Operator):
    """Re-run the traffic simulation with these settings and rebuild the 3D scene"""
    bl_idname = "twin.rebuild"
    bl_label = "Run simulation & rebuild"

    def execute(self, context):
        s = context.scene
        d = s.get("twin_dir", HERE)
        if d not in sys.path:
            sys.path.insert(0, d)
        import sim
        import export_blender
        importlib.reload(sim)
        importlib.reload(export_blender)
        wm = context.window_manager
        wm.progress_begin(0, 1)
        try:
            data = export_blender.make_data(
                compare=s.twin_compare, fleet=s.twin_fleet,
                shortcut="open" if s.twin_shortcut else "closed",
                agents=s.twin_agents, shortcut_w=s.twin_shortcut_w, seed=s.twin_seed,
                steps=s.twin_steps, log=lambda m: self.report({"INFO"}, m))
            build(data, find_forklift())
        finally:
            wm.progress_end()
        s.frame_set(1)
        self.report({"INFO"}, f"Rebuilt with {s.twin_agents} forklifts")
        return {"FINISHED"}


class TWIN_PT_panel(bpy.types.Panel):
    bl_label = "Warehouse Twin"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Warehouse"

    def draw(self, context):
        s = context.scene
        col = self.layout.column()
        col.prop(s, "twin_agents")
        col.prop(s, "twin_compare", text="")
        if s.twin_compare == "fleets":
            col.prop(s, "twin_shortcut")
        elif s.twin_compare == "shortcut":
            col.prop(s, "twin_fleet", text="Fleet")
        col.prop(s, "twin_shortcut_w")
        col.prop(s, "twin_steps")
        col.prop(s, "twin_seed")
        col.separator()
        col.operator("twin.rebuild", icon="PLAY")
        col.label(text="Takes a few seconds; bigger fleets take longer.")


PROPS = {
    "twin_agents": bpy.props.IntProperty(name="Forklifts per warehouse", default=50, min=1, max=200),
    "twin_compare": bpy.props.EnumProperty(name="Compare", items=COMPARE_ITEMS, default="all"),
    "twin_fleet": bpy.props.EnumProperty(name="Fleet", items=FLEET_ITEMS, default="human"),
    "twin_shortcut": bpy.props.BoolProperty(name="Shortcut open", default=True),
    "twin_shortcut_w": bpy.props.IntProperty(name="Shortcut width (cells)", default=1, min=1, max=2),
    "twin_steps": bpy.props.IntProperty(name="Steps to record", default=500, min=50, max=3000),
    "twin_seed": bpy.props.IntProperty(name="Random seed", default=0, min=0),
}


def register():
    for cls in (TWIN_OT_rebuild, TWIN_PT_panel):
        try:
            bpy.utils.unregister_class(getattr(bpy.types, cls.__name__))
        except (AttributeError, RuntimeError):
            pass
        bpy.utils.register_class(cls)
    for k, p in PROPS.items():
        setattr(bpy.types.Scene, k, p)


def embed_ui_loader():
    """Store a tiny registered script in the .blend so the panel reappears when the
    file is opened (Blender asks you to 'Allow Execution' the first time)."""
    name = "warehouse_twin_ui.py"
    t = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    t.clear()
    t.write(
        "import bpy, sys\n"
        "d = bpy.data.scenes[0].get('twin_dir', '')\n"
        "if d and d not in sys.path:\n"
        "    sys.path.insert(0, d)\n"
        "try:\n"
        "    import blender_scene\n"
        "    blender_scene.register()\n"
        "except Exception as e:\n"
        "    print('Warehouse Twin panel not loaded:', e)\n")
    t.use_module = True


def main():
    register()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:                                   # run from Blender's text editor
        if bpy.data.collections.get("Warehouse") is None:
            bpy.ops.twin.rebuild()
        return
    path = argv[0]
    with open(path) as fh:
        data = json.load(fh)
    fk = find_forklift()
    print("forklift model:", fk or "not found, using placeholders")
    build(data, fk)
    s = bpy.context.scene
    s.twin_agents = data.get("agents", len(data["scenes"][0]["kinds"]))
    s.twin_compare = data.get("compare", "all")
    if len(argv) > 1:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(argv[1]))
        print("saved", argv[1])


if __name__ == "__main__":
    main()
