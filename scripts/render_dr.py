"""
Domain Randomization Renderer for Sim2Real Jaco Robot Picking.

This script opens a .blend file containing animated robot motion,
applies aggressive visual domain randomization following best practices from:
- MolmoBot (2026): diversity > photorealism, randomize camera
- Tremblay et al. (2018): flying distractors
- RoboTwin 2.0 (2025): background textures, lighting, clutter
- OpenAI ADR (2019): aggressive randomization ranges

Randomization axes:
1. HDRI environment lighting (from env_map_hdri dataset)
2. Robot material randomization (color, metallic, roughness)
3. Table/floor/backdrop texture randomization
4. Camera pose perturbation (position + look-at jitter)
5. Flying distractors (random geometric shapes with random materials)
6. Additional area/point lights with random color/intensity
7. Object material randomization
8. Flat shading on robot (no smoothing — preserves sim geometry)
9. Post-processing (color management jitter)

Usage:
    blender --background <file.blend> --python render_dr.py -- \
        --output_dir /path/to/output \
        --seed 0 \
        --n_variations 10 \
        [--hdri_dir /path/to/hdris] \
        [--texture_dir /path/to/textures] \
        [--resolution 512]
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
    # Get args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Domain Randomization Renderer")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_variations", type=int, default=10,
                        help="Number of DR variations to render per blend file")
    parser.add_argument("--hdri_dir", type=str, default=None)
    parser.add_argument("--texture_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64,
                        help="Render samples (Cycles)")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--render_video", action="store_true", default=False,
                        help="Render as video (mp4) instead of image sequence")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Render every N frames")
    parser.add_argument("--n_distractors_min", type=int, default=5)
    parser.add_argument("--n_distractors_max", type=int, default=15)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    
    return parser.parse_args(argv)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rand_color():
    """Random RGB color."""
    return (random.random(), random.random(), random.random(), 1.0)


def rand_range(lo, hi):
    return random.uniform(lo, hi)


def get_scene_bounds():
    """Get approximate bounding box of the scene for distractor placement."""
    all_coords = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'robot' in obj.name:
            # Use object location as proxy
            all_coords.append(obj.location.copy())
    if not all_coords:
        return Vector((-1, -1, 0)), Vector((1, 1, 1.5))
    
    xs = [c.x for c in all_coords]
    ys = [c.y for c in all_coords]
    zs = [c.z for c in all_coords]
    return (Vector((min(xs) - 0.5, min(ys) - 0.5, 0)),
            Vector((max(xs) + 0.5, max(ys) + 0.5, max(zs) + 0.5)))


# ============================================================
# MESH SMOOTHING
# ============================================================

def apply_mesh_smoothing():
    """Explicitly set FLAT shading on robot — keep the blocky sim look."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if 'robot' in obj.name:
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
    
    # Clear existing nodes
    nodes.clear()
    
    # Create principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if texture_path and os.path.exists(texture_path):
        # Use texture image
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 0)
        tex_node.image = bpy.data.images.load(texture_path)
        
        # Add texture coordinate and mapping for random UV transform
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
        # Random solid color
        if base_color:
            bsdf.inputs['Base Color'].default_value = base_color
        else:
            bsdf.inputs['Base Color'].default_value = rand_color()
    
    # Randomize PBR properties — but keep object VISIBLE
    # Per DR reference tip 4: "Don't randomize away the cues the task actually depends on"
    # The robot/objects must remain opaque and non-mirror to be seen by a vision policy.
    bsdf.inputs['Metallic'].default_value = rand_range(0.0, 0.7)
    bsdf.inputs['Roughness'].default_value = rand_range(0.3, 1.0)  # avoid pure mirror
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = rand_range(0.0, 0.8)
    # Explicitly ensure opacity — no glass/transmission
    bsdf.inputs['Alpha'].default_value = 1.0
    if 'Transmission Weight' in bsdf.inputs:
        bsdf.inputs['Transmission Weight'].default_value = 0.0
    
    return mat


def randomize_robot_material(texture_files):
    """Randomize robot material PER LINK — each robot part gets its own random material.
    Heavily favors textures for aggressive DR (per MolmoBot: diversity > photorealism)."""
    
    robot_objects = [obj for obj in bpy.data.objects if 'robot' in obj.name and obj.type == 'MESH']
    
    for obj in robot_objects:
        # Each link gets an independent random material
        # 70% chance texture, 30% chance solid color
        if texture_files and random.random() < 0.7:
            tex = random.choice(texture_files)
            mat = create_random_material(f"robot_{obj.name}", texture_path=str(tex))
        else:
            mat = create_random_material(f"robot_{obj.name}")
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def randomize_table_material(texture_files):
    """Randomize table, floor, and backdrop — heavily favor textures."""
    table_objects = [obj for obj in bpy.data.objects 
                     if obj.type == 'MESH' and ('table' in obj.name or 'leg' in obj.name)]
    floor_objects = [obj for obj in bpy.data.objects 
                     if obj.type == 'MESH' and 'floor' in obj.name]
    backdrop_objects = [obj for obj in bpy.data.objects 
                        if obj.type == 'MESH' and 'backdrop' in obj.name]
    
    # Table — 80% textured
    if texture_files and random.random() < 0.8:
        tex = random.choice(texture_files)
        mat = create_random_material("table_dr", texture_path=str(tex))
    else:
        mat = create_random_material("table_dr")
    for obj in table_objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    
    # Floor — 80% textured
    if texture_files and random.random() < 0.8:
        tex = random.choice(texture_files)
        mat = create_random_material("floor_dr", texture_path=str(tex))
    else:
        mat = create_random_material("floor_dr")
    for obj in floor_objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    
    # Backdrop — 80% textured
    if texture_files and random.random() < 0.8:
        tex = random.choice(texture_files)
        mat = create_random_material("backdrop_dr", texture_path=str(tex))
    else:
        mat = create_random_material("backdrop_dr")
    for obj in backdrop_objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def randomize_object_material(texture_files):
    """Randomize the hope_object (target object) material — 80% textured."""
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
        print(f"WARNING: No HDRI files found in {hdri_dir}")
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
    
    # Environment texture
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.location = (-600, 0)
    env_tex.image = bpy.data.images.load(hdri_path)
    
    # Mapping for random rotation
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-800, 0)
    mapping.inputs['Rotation'].default_value = (
        rand_range(-0.1, 0.1),
        rand_range(-0.1, 0.1),
        rand_range(0, 2 * math.pi)  # Full Z rotation
    )
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-1000, 0)
    
    # Background node with random strength
    background = nodes.new('ShaderNodeBackground')
    background.location = (-200, 0)
    background.inputs['Strength'].default_value = rand_range(0.3, 3.0)
    
    # Output
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (0, 0)
    
    # Link
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    return True


def add_random_lights():
    """Add random additional lights to the scene."""
    # Remove any existing DR lights
    for obj in list(bpy.data.objects):
        if obj.name.startswith("DR_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    n_lights = random.randint(1, 4)
    
    for i in range(n_lights):
        light_type = random.choice(['POINT', 'AREA', 'SPOT'])
        
        light_data = bpy.data.lights.new(name=f"DR_Light_{i}", type=light_type)
        light_obj = bpy.data.objects.new(name=f"DR_Light_{i}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        
        # Random position around the scene
        light_obj.location = (
            rand_range(-2.0, 2.0),
            rand_range(-2.0, 2.0),
            rand_range(0.5, 3.0)
        )
        
        # Random color temperature (warm to cool)
        r = rand_range(0.7, 1.0)
        g = rand_range(0.7, 1.0)
        b = rand_range(0.7, 1.0)
        light_data.color = (r, g, b)
        
        # Random energy
        light_data.energy = rand_range(10, 500)
        
        if light_type == 'AREA':
            light_data.size = rand_range(0.5, 3.0)
        elif light_type == 'SPOT':
            light_data.spot_size = rand_range(0.3, 1.5)
            light_data.spot_blend = rand_range(0.0, 1.0)
            # Point roughly toward scene center
            light_obj.rotation_euler = Euler((
                rand_range(0.5, 1.5),
                rand_range(-0.5, 0.5),
                rand_range(0, 2 * math.pi)
            ))


# ============================================================
# CAMERA RANDOMIZATION
# ============================================================

def randomize_camera():
    """Randomize camera pose - critical for sim2real transfer (MolmoBot)."""
    cam = bpy.data.objects.get('Camera')
    if not cam:
        return
    
    # Get original camera location as center of randomization
    orig_loc = cam.location.copy()
    
    # Perturb position (moderate - we still want to see the robot)
    cam.location.x = orig_loc.x + rand_range(-0.3, 0.3)
    cam.location.y = orig_loc.y + rand_range(-0.3, 0.3)
    cam.location.z = orig_loc.z + rand_range(-0.2, 0.2)
    
    # Perturb rotation slightly
    orig_rot = cam.rotation_euler.copy()
    cam.rotation_euler.x = orig_rot.x + rand_range(-0.08, 0.08)
    cam.rotation_euler.y = orig_rot.y + rand_range(-0.08, 0.08)
    cam.rotation_euler.z = orig_rot.z + rand_range(-0.08, 0.08)
    
    # Randomize focal length
    cam.data.lens = rand_range(25, 60)
    
    # Slight DOF blur sometimes
    if random.random() > 0.7:
        cam.data.dof.use_dof = True
        cam.data.dof.aperture_fstop = rand_range(1.4, 8.0)
        cam.data.dof.focus_distance = rand_range(1.0, 3.0)
    else:
        cam.data.dof.use_dof = False


# ============================================================
# FLYING DISTRACTORS (Tremblay et al. 2018)
# ============================================================

def add_flying_distractors(n_min, n_max, texture_files):
    """Add random geometric shapes floating in the scene as distractors."""
    # Remove existing distractors
    for obj in list(bpy.data.objects):
        if obj.name.startswith("DR_Distractor"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    n = random.randint(n_min, n_max)
    
    primitives = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'monkey']
    
    for i in range(n):
        prim = random.choice(primitives)
        
        # Create primitive
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
        elif prim == 'monkey':
            bpy.ops.mesh.primitive_monkey_add()
        
        obj = bpy.context.active_object
        obj.name = f"DR_Distractor_{i}"
        
        # Random position (in and around the workspace)
        obj.location = (
            rand_range(-1.5, 1.5),
            rand_range(-1.0, 2.0),
            rand_range(0.0, 2.0)
        )
        
        # Random rotation
        obj.rotation_euler = (
            rand_range(0, 2 * math.pi),
            rand_range(0, 2 * math.pi),
            rand_range(0, 2 * math.pi)
        )
        
        # Random scale (small to medium)
        s = rand_range(0.02, 0.25)
        obj.scale = (s * rand_range(0.5, 2.0),
                     s * rand_range(0.5, 2.0),
                     s * rand_range(0.5, 2.0))
        
        # Smooth shading
        for poly in obj.data.polygons:
            poly.use_smooth = True
        
        # Random material
        if texture_files and random.random() > 0.5:
            tex = random.choice(texture_files)
            mat = create_random_material(f"distractor_{i}", texture_path=str(tex))
        else:
            mat = create_random_material(f"distractor_{i}")
        
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        
        obj.select_set(False)


# ============================================================
# RENDER SETTINGS
# ============================================================

def setup_render(args):
    """Configure render settings."""
    scene = bpy.context.scene
    
    # Engine
    scene.render.engine = args.engine
    
    if args.engine == 'CYCLES':
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        
        # GPU if available
        if args.use_gpu:
            prefs = bpy.context.preferences.addons.get('cycles')
            if prefs:
                prefs.preferences.compute_device_type = 'METAL'  # macOS
                bpy.context.preferences.addons['cycles'].preferences.get_devices()
                scene.cycles.device = 'GPU'
    
    elif args.engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = args.samples
    
    # Resolution
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    
    # Color management randomization (subtle)
    scene.view_settings.exposure = rand_range(-0.5, 0.5)
    # Randomly choose view transform
    try:
        if random.random() > 0.5:
            scene.view_settings.view_transform = 'Filmic'
        else:
            scene.view_settings.view_transform = 'Standard'
    except:
        pass  # Some Blender versions have different options


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_dr_variation(variation_idx, args, hdri_dir, texture_files):
    """Apply one variation of domain randomization and render."""
    
    seed = args.seed * 1000 + variation_idx
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"  DR Variation {variation_idx} (seed={seed})")
    print(f"{'='*60}")
    
    # 1. Mesh smoothing (only once, on first variation)
    if variation_idx == 0:
        print("  [1/7] Applying mesh smoothing...")
        apply_mesh_smoothing()
    
    # 2. HDRI environment
    print("  [2/7] Randomizing HDRI lighting...")
    if hdri_dir:
        setup_hdri_lighting(hdri_dir)
    
    # 3. Additional lights
    print("  [3/7] Adding random lights...")
    add_random_lights()
    
    # 4. Material randomization
    print("  [4/7] Randomizing materials...")
    randomize_robot_material(texture_files)
    randomize_table_material(texture_files)
    randomize_object_material(texture_files)
    
    # 5. Camera randomization
    print("  [5/7] Randomizing camera...")
    randomize_camera()
    
    # 6. Flying distractors
    print("  [6/7] Adding flying distractors...")
    add_flying_distractors(args.n_distractors_min, args.n_distractors_max, texture_files)
    
    # 7. Render settings
    print("  [7/7] Setting up render...")
    setup_render(args)
    
    # Set output path
    scene = bpy.context.scene
    var_dir = os.path.join(args.output_dir, f"variation_{variation_idx:04d}")
    os.makedirs(var_dir, exist_ok=True)
    
    if args.render_video:
        scene.render.filepath = os.path.join(var_dir, "render.mp4")
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    else:
        scene.render.filepath = os.path.join(var_dir, "frame_")
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
    
    scene.frame_step = args.frame_step
    
    # Render animation
    print(f"  Rendering to: {var_dir}")
    bpy.ops.render.render(animation=True)
    print(f"  ✓ Variation {variation_idx} complete!")
    
    return var_dir


def main():
    args = parse_args()
    
    # Resolve asset directories
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    assets_dir = project_dir / "assets"
    
    hdri_dir = args.hdri_dir or str(assets_dir / "hdri")
    texture_dir = args.texture_dir or str(assets_dir / "surface_textures")
    
    # Collect texture files
    texture_files = []
    if os.path.isdir(texture_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            texture_files.extend(Path(texture_dir).rglob(ext))
    print(f"Found {len(texture_files)} texture files")
    
    # Check HDRI dir
    if not os.path.isdir(hdri_dir):
        print(f"WARNING: HDRI directory not found: {hdri_dir}")
        hdri_dir = None
    else:
        n_hdri = len(list(Path(hdri_dir).glob("*.hdr")))
        print(f"Found {n_hdri} HDRI files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store original scene state for resetting between variations
    # (Camera position, materials are modified in-place so we save originals)
    cam = bpy.data.objects.get('Camera')
    orig_cam_loc = cam.location.copy() if cam else None
    orig_cam_rot = cam.rotation_euler.copy() if cam else None
    orig_cam_lens = cam.data.lens if cam else None
    
    # Run DR variations
    for var_idx in range(args.n_variations):
        # Reset camera to original before each variation
        if cam and orig_cam_loc:
            cam.location = orig_cam_loc.copy()
            cam.rotation_euler = orig_cam_rot.copy()
            cam.data.lens = orig_cam_lens
            cam.data.dof.use_dof = False
        
        run_dr_variation(var_idx, args, hdri_dir, texture_files)
    
    print(f"\n{'='*60}")
    print(f"  ALL DONE: {args.n_variations} variations rendered")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
