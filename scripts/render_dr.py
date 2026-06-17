"""
Domain Randomization Renderer for Sim2Real Jaco Robot Picking.

Single unified script for all DR rendering modes:
- grid: Render a NxN grid of single-frame DR variations (random frames)
- animation: Render full animation sequences with DR
- single: Render a single DR frame

Randomization axes:
1. HDRI environment lighting (from env_map_hdri dataset)
2. Robot material randomization per-link (textures + colors)
3. Table/floor/backdrop texture randomization + table scale
4. Camera pose perturbation (position, rotation, focal length)
5. Flying distractors (primitives + USDC objects from meta_assets_2k)
6. Additional area/point lights with random color/intensity
7. Object material randomization
8. HDRI-only background mode (35% — no floor/backdrop)
9. Post-processing (exposure, color management)

Usage:
    # 3x3 grid of random frames
    blender --background <file.blend> --python render_dr.py -- \\
        --mode grid --grid_size 3 --output_dir output/grid

    # Full animation with DR
    blender --background <file.blend> --python render_dr.py -- \\
        --mode animation --n_variations 10 --output_dir output/anim

    # Single frame
    blender --background <file.blend> --python render_dr.py -- \\
        --mode single --frame 30 --output_dir output/single
"""

import bpy
import bmesh
import sys
import os
import random
import math
import argparse
from pathlib import Path
from mathutils import Vector, Euler, Color
import numpy as np


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Domain Randomization Renderer")
    parser.add_argument("--mode", type=str, default="grid",
                        choices=["grid", "animation", "single"],
                        help="Render mode: grid (NxN), animation (full seq), single (one frame)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_variations", type=int, default=10,
                        help="Number of DR variations (animation mode)")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Grid dimension NxN (grid mode)")
    parser.add_argument("--frame", type=int, default=-1,
                        help="Frame to render (-1 = random)")
    parser.add_argument("--hdri_dir", type=str, default=None)
    parser.add_argument("--texture_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8,
                        help="Render samples")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Render every N frames (animation mode)")
    parser.add_argument("--n_distractors_min", type=int, default=5)
    parser.add_argument("--n_distractors_max", type=int, default=15)
    
    return parser.parse_args(argv)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rand_color():
    return (random.random(), random.random(), random.random(), 1.0)


def rand_range(lo, hi):
    return random.uniform(lo, hi)


def get_project_dir():
    """Resolve project directory robustly."""
    _this_file = Path(__file__) if '__file__' in dir() and not str(__file__).startswith('<') else None
    if _this_file and _this_file.exists():
        return _this_file.parent.parent
    for candidate in [Path("/Users/jtremblay/code/blender2dr"), Path.cwd(), Path.cwd().parent]:
        if (candidate / "assets" / "distractors").exists():
            return candidate
    return Path("/Users/jtremblay/code/blender2dr")


# ============================================================
# MESH SMOOTHING
# ============================================================

def apply_mesh_smoothing():
    """Explicitly set FLAT shading on robot — keep the blocky sim look."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'robot' in obj.name:
            for poly in obj.data.polygons:
                poly.use_smooth = False


# ============================================================
# MATERIAL RANDOMIZATION
# ============================================================

def create_random_material(name_prefix="dr_mat", base_color=None, texture_path=None):
    """Create a new material with randomized PBR properties."""
    mat = bpy.data.materials.new(name=f"{name_prefix}_{random.randint(0, 99999)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if texture_path and os.path.exists(texture_path):
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 0)
        tex_node.image = bpy.data.images.load(texture_path)
        
        tex_coord = nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)
        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-600, 0)
        mapping.inputs['Scale'].default_value = (rand_range(0.5, 5.0),) * 3
        mapping.inputs['Rotation'].default_value = (0, 0, rand_range(0, 2 * math.pi))
        
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        if base_color:
            bsdf.inputs['Base Color'].default_value = base_color
        else:
            bsdf.inputs['Base Color'].default_value = rand_color()
    
    # Keep visible: no mirrors, no glass (DR reference tip 4)
    bsdf.inputs['Metallic'].default_value = rand_range(0.0, 0.7)
    bsdf.inputs['Roughness'].default_value = rand_range(0.3, 1.0)
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = rand_range(0.0, 0.8)
    bsdf.inputs['Alpha'].default_value = 1.0
    if 'Transmission Weight' in bsdf.inputs:
        bsdf.inputs['Transmission Weight'].default_value = 0.0
    
    return mat


def randomize_robot_material(texture_files):
    """Randomize robot material PER LINK — 70% textured, 30% solid color."""
    robot_objects = [obj for obj in bpy.data.objects if 'robot' in obj.name and obj.type == 'MESH']
    for obj in robot_objects:
        if texture_files and random.random() < 0.7:
            tex = random.choice(texture_files)
            mat = create_random_material(f"robot_{obj.name}", texture_path=str(tex))
        else:
            mat = create_random_material(f"robot_{obj.name}")
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def randomize_table_geometry():
    """Randomly scale the table in X and Y."""
    table_obj = bpy.data.objects.get('shape_35_table')
    sx = rand_range(0.7, 1.8)
    sy = rand_range(0.7, 1.8)
    if table_obj:
        table_obj.scale.x = sx
        table_obj.scale.y = sy
    for obj in bpy.data.objects:
        if 'leg' in obj.name and obj.type == 'MESH':
            obj.scale.x = sx
            obj.scale.y = sy


def randomize_table_material(texture_files):
    """Randomize table, floor, and backdrop — 80% textured."""
    for group_name, group_filter in [
        ("table", lambda o: 'table' in o.name or 'leg' in o.name),
        ("floor", lambda o: 'floor' in o.name),
        ("backdrop", lambda o: 'backdrop' in o.name),
    ]:
        objs = [o for o in bpy.data.objects if o.type == 'MESH' and group_filter(o)]
        if texture_files and random.random() < 0.8:
            tex = random.choice(texture_files)
            mat = create_random_material(f"{group_name}_dr", texture_path=str(tex))
        else:
            mat = create_random_material(f"{group_name}_dr")
        for obj in objs:
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def randomize_object_material(texture_files):
    """Randomize the hope_object material — 80% textured."""
    obj = bpy.data.objects.get('hope_object')
    if not obj:
        return
    if texture_files and random.random() < 0.8:
        tex = random.choice(texture_files)
        mat = create_random_material("object_dr", texture_path=str(tex))
    else:
        mat = create_random_material("object_dr")
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ============================================================
# LIGHTING RANDOMIZATION
# ============================================================

def setup_hdri_lighting(hdri_dir):
    """Set up HDRI environment lighting with random rotation and intensity."""
    hdri_files = list(Path(hdri_dir).glob("*.hdr")) + list(Path(hdri_dir).glob("*.exr"))
    if not hdri_files:
        return False
    
    hdri_path = str(random.choice(hdri_files))
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.location = (-600, 0)
    env_tex.image = bpy.data.images.load(hdri_path)
    
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-800, 0)
    mapping.inputs['Rotation'].default_value = (
        rand_range(-0.1, 0.1), rand_range(-0.1, 0.1), rand_range(0, 2 * math.pi))
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-1000, 0)
    
    background = nodes.new('ShaderNodeBackground')
    background.location = (-200, 0)
    background.inputs['Strength'].default_value = rand_range(0.3, 3.0)
    
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (0, 0)
    
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])
    return True


def add_random_lights():
    """Add random additional lights."""
    for obj in list(bpy.data.objects):
        if obj.name.startswith("DR_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    for i in range(random.randint(1, 4)):
        light_type = random.choice(['POINT', 'AREA', 'SPOT'])
        light_data = bpy.data.lights.new(name=f"DR_Light_{i}", type=light_type)
        light_obj = bpy.data.objects.new(name=f"DR_Light_{i}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        
        light_obj.location = (rand_range(-2, 2), rand_range(-2, 2), rand_range(0.5, 3))
        light_data.color = (rand_range(0.7, 1), rand_range(0.7, 1), rand_range(0.7, 1))
        light_data.energy = rand_range(10, 500)
        
        if light_type == 'AREA':
            light_data.size = rand_range(0.5, 3.0)
        elif light_type == 'SPOT':
            light_data.spot_size = rand_range(0.3, 1.5)
            light_obj.rotation_euler = Euler((rand_range(0.5, 1.5), rand_range(-0.5, 0.5), rand_range(0, 2*math.pi)))


# ============================================================
# CAMERA RANDOMIZATION
# ============================================================

def randomize_camera():
    """Randomize camera pose — critical for sim2real (MolmoBot)."""
    cam = bpy.data.objects.get('Camera')
    if not cam:
        return
    orig_loc = cam.location.copy()
    orig_rot = cam.rotation_euler.copy()
    
    cam.location.x = orig_loc.x + rand_range(-0.3, 0.3)
    cam.location.y = orig_loc.y + rand_range(-0.3, 0.3)
    cam.location.z = orig_loc.z + rand_range(-0.2, 0.2)
    cam.rotation_euler.x = orig_rot.x + rand_range(-0.08, 0.08)
    cam.rotation_euler.y = orig_rot.y + rand_range(-0.08, 0.08)
    cam.rotation_euler.z = orig_rot.z + rand_range(-0.08, 0.08)
    cam.data.lens = rand_range(25, 60)
    
    if random.random() > 0.7:
        cam.data.dof.use_dof = True
        cam.data.dof.aperture_fstop = rand_range(1.4, 8.0)
        cam.data.dof.focus_distance = rand_range(1.0, 3.0)
    else:
        cam.data.dof.use_dof = False


# ============================================================
# FLYING DISTRACTORS
# ============================================================

def add_flying_distractors(n_min, n_max, texture_files):
    """Add flying distractors: mix of primitives + USDC objects from meta_assets_2k.
    USDC objects keep their original materials. All distractors are animated."""
    
    # Remove existing distractors
    for obj in list(bpy.data.objects):
        if obj.name.startswith("DR_Distractor"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    n = random.randint(n_min, n_max)
    project_dir = get_project_dir()
    distractor_dir = project_dir / "assets" / "distractors"
    usdc_files = list(distractor_dir.glob("*.usdc")) if distractor_dir.exists() else []
    
    cam = bpy.data.objects.get('Camera')
    cam_loc = cam.location.copy() if cam else Vector((0, 1.5, 1.35))
    
    primitives = ['cube', 'sphere', 'cylinder', 'cone', 'torus']
    
    for i in range(n):
        use_usdc = usdc_files and random.random() < 0.7
        
        if use_usdc:
            usdc_path = str(random.choice(usdc_files))
            scene = bpy.context.scene
            orig_frame_start = scene.frame_start
            orig_frame_end = scene.frame_end
            orig_fps = scene.render.fps
            existing = set(bpy.data.objects.keys())
            orig_cwd = os.getcwd()
            try:
                os.chdir(str(distractor_dir))
                bpy.ops.wm.usd_import(filepath=usdc_path)
            except:
                use_usdc = False
            finally:
                os.chdir(orig_cwd)
                scene.frame_start = orig_frame_start
                scene.frame_end = orig_frame_end
                scene.render.fps = orig_fps
            
            if use_usdc:
                new_objs = [o for o in bpy.data.objects if o.name not in existing and o.type == 'MESH']
                if not new_objs:
                    use_usdc = False
                else:
                    bpy.ops.object.select_all(action='DESELECT')
                    for o in new_objs:
                        o.select_set(True)
                    bpy.context.view_layer.objects.active = new_objs[0]
                    if len(new_objs) > 1:
                        bpy.ops.object.join()
                    obj = bpy.context.active_object
                    obj.name = f"DR_Distractor_{i}"
        
        if not use_usdc:
            prim = random.choice(primitives)
            if prim == 'cube':
                bpy.ops.mesh.primitive_cube_add()
            elif prim == 'sphere':
                bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
            elif prim == 'cylinder':
                bpy.ops.mesh.primitive_cylinder_add(vertices=16)
            elif prim == 'cone':
                bpy.ops.mesh.primitive_cone_add(vertices=16)
            elif prim == 'torus':
                bpy.ops.mesh.primitive_torus_add()
            obj = bpy.context.active_object
            obj.name = f"DR_Distractor_{i}"
        
        # Random position
        obj.location = (rand_range(-1.2, 1.2), rand_range(-0.5, 2.0), rand_range(0.0, 1.8))
        
        # Random rotation
        obj.rotation_euler = (rand_range(0, 2*math.pi), rand_range(0, 2*math.pi), rand_range(0, 2*math.pi))
        
        # Scale: USDC ~1.0 (real size), primitives small
        if use_usdc:
            s = rand_range(0.7, 1.5)
            obj.scale = (s, s, s)
        else:
            dist_to_cam = (obj.location - cam_loc).length
            max_scale = max(0.02, min(0.15, dist_to_cam * 0.12))
            s = rand_range(0.02, max_scale)
            obj.scale = (s * rand_range(0.7, 1.3), s * rand_range(0.7, 1.3), s * rand_range(0.7, 1.3))
        
        # Smooth shading on distractors
        for poly in obj.data.polygons:
            poly.use_smooth = True
        
        # Materials: USDC keeps original, primitives get random
        if not use_usdc:
            if texture_files and random.random() < 0.6:
                tex = random.choice(texture_files)
                mat = create_random_material(f"distractor_{i}", texture_path=str(tex))
            else:
                mat = create_random_material(f"distractor_{i}")
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        
        # Animate: move between two positions over timeline
        scene = bpy.context.scene
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        
        start_loc = obj.location.copy()
        end_loc = Vector((
            start_loc.x + rand_range(-0.5, 0.5),
            start_loc.y + rand_range(-0.5, 0.5),
            start_loc.z + rand_range(-0.3, 0.3)))
        
        obj.location = start_loc
        obj.keyframe_insert(data_path="location", frame=frame_start)
        obj.location = end_loc
        obj.keyframe_insert(data_path="location", frame=frame_end)
        
        obj.rotation_euler = Euler((rand_range(0, 2*math.pi), rand_range(0, 2*math.pi), rand_range(0, 2*math.pi)))
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_start)
        obj.rotation_euler = Euler((rand_range(0, 2*math.pi), rand_range(0, 2*math.pi), rand_range(0, 2*math.pi)))
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_end)
        
        # Linear interpolation
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            if action.is_action_layered:
                for layer in action.layers:
                    for strip in layer.strips:
                        if hasattr(strip, 'channelbags'):
                            for cb in strip.channelbags:
                                for fc in cb.fcurves:
                                    for kp in fc.keyframe_points:
                                        kp.interpolation = 'LINEAR'
            else:
                for fc in action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
        
        obj.select_set(False)


# ============================================================
# HDRI-ONLY BACKGROUND MODE
# ============================================================

def setup_hdri_only_background():
    """Hide floor/backdrop, use HDRI as background. Remove extra lights."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ('floor' in obj.name or 'backdrop' in obj.name):
            obj.hide_render = True
            obj.hide_viewport = True
    for obj in list(bpy.data.objects):
        if obj.name.startswith("DR_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.render.film_transparent = False
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs['Strength'].default_value = rand_range(0.8, 2.5)
                break


def restore_scene_geometry():
    """Restore floor/backdrop visibility."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ('floor' in obj.name or 'backdrop' in obj.name):
            obj.hide_render = False
            obj.hide_viewport = False
    bpy.context.scene.render.film_transparent = False


# ============================================================
# CORE: APPLY DR TO SCENE
# ============================================================

def apply_dr(seed, hdri_dir, texture_files, n_distractors_min=5, n_distractors_max=15):
    """Apply one full round of domain randomization to the current scene.
    Returns whether HDRI-only background was used."""
    
    set_seed(seed)
    apply_mesh_smoothing()
    
    # HDRI lighting
    if hdri_dir:
        setup_hdri_lighting(hdri_dir)
    
    # 35% chance HDRI-only background
    use_hdri_bg = random.random() < 0.35
    if use_hdri_bg:
        setup_hdri_only_background()
    else:
        restore_scene_geometry()
        add_random_lights()
        randomize_table_material(texture_files)
        randomize_table_geometry()
    
    # Materials
    randomize_robot_material(texture_files)
    randomize_object_material(texture_files)
    
    # Camera
    randomize_camera()
    
    # Distractors
    add_flying_distractors(n_distractors_min, n_distractors_max, texture_files)
    
    return use_hdri_bg


# ============================================================
# RENDER SETUP
# ============================================================

def setup_render_settings(engine, samples, resolution):
    """Configure render engine and resolution."""
    scene = bpy.context.scene
    scene.render.engine = engine
    
    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons.get('cycles')
        if prefs:
            try:
                prefs.preferences.compute_device_type = 'METAL'
                prefs.preferences.get_devices()
                for d in prefs.preferences.devices:
                    d.use = True
            except:
                pass
    elif engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = samples
    
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    
    # Color management randomization
    scene.view_settings.exposure = rand_range(-0.3, 0.3)
    try:
        scene.view_settings.view_transform = random.choice(['Filmic', 'Standard'])
    except:
        pass


def render_frame(output_path, frame):
    """Render a single frame to file."""
    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


# ============================================================
# MODE: GRID
# ============================================================

def run_grid(args, hdri_dir, texture_files):
    """Render NxN grid of DR variations at random frames."""
    n_total = args.grid_size * args.grid_size
    
    cam = bpy.data.objects.get('Camera')
    orig_cam_loc = cam.location.copy() if cam else None
    orig_cam_rot = cam.rotation_euler.copy() if cam else None
    orig_cam_lens = cam.data.lens if cam else None
    
    print(f"Rendering {n_total} DR variations ({args.grid_size}x{args.grid_size} grid)...")
    
    for i in range(n_total):
        seed = args.seed * 10000 + i
        
        # Reset camera before each variation
        if cam and orig_cam_loc:
            cam.location = orig_cam_loc.copy()
            cam.rotation_euler = orig_cam_rot.copy()
            cam.data.lens = orig_cam_lens
            cam.data.dof.use_dof = False
        
        # Apply DR
        apply_dr(seed, hdri_dir, texture_files, args.n_distractors_min, args.n_distractors_max)
        setup_render_settings(args.engine, args.samples, args.resolution)
        
        # Pick random frame
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        if args.frame >= 0:
            render_f = args.frame
        else:
            render_f = random.randint(frame_start, frame_end)
        
        # Render
        out_path = os.path.join(args.output_dir, f"cell_{i:03d}.png")
        render_frame(out_path, render_f)
        print(f"  [{i+1}/{n_total}] cell_{i:03d}.png (seed={seed}, frame={render_f})")
    
    print(f"All {n_total} cells rendered.")


# ============================================================
# MODE: ANIMATION
# ============================================================

def run_animation(args, hdri_dir, texture_files):
    """Render full animation sequences with DR."""
    cam = bpy.data.objects.get('Camera')
    orig_cam_loc = cam.location.copy() if cam else None
    orig_cam_rot = cam.rotation_euler.copy() if cam else None
    orig_cam_lens = cam.data.lens if cam else None
    
    for var_idx in range(args.n_variations):
        seed = args.seed * 1000 + var_idx
        
        if cam and orig_cam_loc:
            cam.location = orig_cam_loc.copy()
            cam.rotation_euler = orig_cam_rot.copy()
            cam.data.lens = orig_cam_lens
            cam.data.dof.use_dof = False
        
        apply_dr(seed, hdri_dir, texture_files, args.n_distractors_min, args.n_distractors_max)
        setup_render_settings(args.engine, args.samples, args.resolution)
        
        var_dir = os.path.join(args.output_dir, f"variation_{var_idx:04d}")
        os.makedirs(var_dir, exist_ok=True)
        
        scene = bpy.context.scene
        scene.render.filepath = os.path.join(var_dir, "frame_")
        scene.frame_step = args.frame_step
        
        print(f"  Variation {var_idx} (seed={seed}) -> {var_dir}")
        bpy.ops.render.render(animation=True)
        print(f"  ✓ Done")


# ============================================================
# MODE: SINGLE
# ============================================================

def run_single(args, hdri_dir, texture_files):
    """Render a single DR frame."""
    seed = args.seed
    apply_dr(seed, hdri_dir, texture_files, args.n_distractors_min, args.n_distractors_max)
    setup_render_settings(args.engine, args.samples, args.resolution)
    
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    render_f = args.frame if args.frame >= 0 else random.randint(frame_start, frame_end)
    
    out_path = os.path.join(args.output_dir, "render.png")
    render_frame(out_path, render_f)
    print(f"Rendered frame {render_f} -> {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    
    # Resolve asset directories
    project_dir = get_project_dir()
    assets_dir = project_dir / "assets"
    hdri_dir = args.hdri_dir or str(assets_dir / "hdri")
    texture_dir = args.texture_dir or str(assets_dir / "surface_textures")
    
    # Collect texture files
    texture_files = []
    if os.path.isdir(texture_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            texture_files.extend(Path(texture_dir).rglob(ext))
    print(f"Found {len(texture_files)} texture files")
    
    if not os.path.isdir(hdri_dir):
        print(f"WARNING: HDRI directory not found: {hdri_dir}")
        hdri_dir = None
    else:
        print(f"Found {len(list(Path(hdri_dir).glob('*.hdr')))} HDRI files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dispatch by mode
    if args.mode == 'grid':
        run_grid(args, hdri_dir, texture_files)
    elif args.mode == 'animation':
        run_animation(args, hdri_dir, texture_files)
    elif args.mode == 'single':
        run_single(args, hdri_dir, texture_files)
    
    print("DONE")


if __name__ == "__main__":
    main()
