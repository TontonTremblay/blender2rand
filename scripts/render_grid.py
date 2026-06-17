"""
Render a 10x10 grid of DR variations from a single blend file.
Each cell is a different DR randomization of the same frame.

Usage:
    blender --background <file.blend> --python render_grid.py -- \
        --output_dir /path/to/output \
        --seed 0 \
        --resolution 256 \
        --frame 45
"""

import bpy
import sys
import os
import random
import math
import argparse
from pathlib import Path
from mathutils import Vector, Euler
import numpy as np

# ============================================================
# Import DR functions from render_dr.py
# ============================================================

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Read and exec the render_dr module (minus main)
_render_dr_code = open(script_dir / "render_dr.py").read()
_render_dr_code = _render_dr_code.split('if __name__')[0]
exec(_render_dr_code)


# ============================================================
# GRID RENDERING
# ============================================================

def parse_grid_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Render DR Grid")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution per cell")
    parser.add_argument("--frame", type=int, default=45,
                        help="Which frame to render")
    parser.add_argument("--grid_size", type=int, default=10,
                        help="Grid dimension (NxN)")
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE")
    parser.add_argument("--samples", type=int, default=32)
    
    return parser.parse_args(argv)


def render_single_frame(output_path, frame, resolution, engine, samples):
    """Render a single frame to a file."""
    scene = bpy.context.scene
    scene.render.engine = engine
    
    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        scene.cycles.device = 'GPU'
        # Use Metal on macOS
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
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.frame_set(frame)
    
    # Color management
    scene.view_settings.exposure = rand_range(-0.3, 0.3)
    try:
        scene.view_settings.view_transform = random.choice(['Filmic', 'Standard'])
    except:
        pass
    
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_grid_args()
    
    # Resolve asset directories
    project_dir = script_dir.parent
    assets_dir = project_dir / "assets"
    hdri_dir = str(assets_dir / "hdri")
    texture_dir = str(assets_dir / "textures")
    
    # Collect texture files
    texture_files = []
    if os.path.isdir(texture_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            texture_files.extend(Path(texture_dir).rglob(ext))
    print(f"Found {len(texture_files)} texture files")
    
    n_hdri = len(list(Path(hdri_dir).glob("*.hdr"))) if os.path.isdir(hdri_dir) else 0
    print(f"Found {n_hdri} HDRI files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save original camera state
    cam = bpy.data.objects.get('Camera')
    orig_cam_loc = cam.location.copy() if cam else None
    orig_cam_rot = cam.rotation_euler.copy() if cam else None
    orig_cam_lens = cam.data.lens if cam else None
    
    n_total = args.grid_size * args.grid_size
    print(f"\nRendering {n_total} DR variations ({args.grid_size}x{args.grid_size} grid)...")
    
    for i in range(n_total):
        seed = args.seed * 10000 + i
        set_seed(seed)
        
        # Reset camera
        if cam and orig_cam_loc:
            cam.location = orig_cam_loc.copy()
            cam.rotation_euler = orig_cam_rot.copy()
            cam.data.lens = orig_cam_lens
            cam.data.dof.use_dof = False
        
        # Apply DR
        apply_mesh_smoothing()
        if os.path.isdir(hdri_dir):
            setup_hdri_lighting(hdri_dir)
        add_random_lights()
        randomize_robot_material(texture_files)
        randomize_table_material(texture_files)
        randomize_object_material(texture_files)
        randomize_camera()
        add_flying_distractors(5, 15, texture_files)
        
        # Render
        out_path = os.path.join(args.output_dir, f"cell_{i:03d}.png")
        render_single_frame(out_path, args.frame, args.resolution, args.engine, args.samples)
        print(f"  [{i+1}/{n_total}] Rendered cell_{i:03d}.png (seed={seed})")
    
    print(f"\nAll {n_total} cells rendered. Stitching grid...")
    
    # Stitch into grid using numpy/PIL-free approach via Blender's compositor
    # Actually let's just use Python + basic image loading
    stitch_grid(args.output_dir, args.grid_size, args.resolution)


def stitch_grid(output_dir, grid_size, cell_res):
    """Stitch individual cell PNGs into a single grid image."""
    try:
        # Use Blender's image API to compose the grid
        grid_w = grid_size * cell_res
        grid_h = grid_size * cell_res
        
        # Create a new image for the grid
        grid_img = bpy.data.images.new("DR_Grid", width=grid_w, height=grid_h, alpha=False)
        grid_pixels = [0.0] * (grid_w * grid_h * 4)
        
        for idx in range(grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            
            cell_path = os.path.join(output_dir, f"cell_{idx:03d}.png")
            if not os.path.exists(cell_path):
                continue
            
            cell_img = bpy.data.images.load(cell_path)
            cell_pixels = list(cell_img.pixels)
            
            # Copy cell pixels into grid
            # Blender images are bottom-up
            for y in range(cell_res):
                for x in range(cell_res):
                    # Source pixel
                    src_idx = (y * cell_res + x) * 4
                    # Destination pixel (flip row order: row 0 = top)
                    dest_row = (grid_size - 1 - row) * cell_res + y
                    dest_col = col * cell_res + x
                    dest_idx = (dest_row * grid_w + dest_col) * 4
                    
                    grid_pixels[dest_idx] = cell_pixels[src_idx]
                    grid_pixels[dest_idx + 1] = cell_pixels[src_idx + 1]
                    grid_pixels[dest_idx + 2] = cell_pixels[src_idx + 2]
                    grid_pixels[dest_idx + 3] = 1.0
            
            bpy.data.images.remove(cell_img)
        
        grid_img.pixels = grid_pixels
        grid_path = os.path.join(output_dir, "grid.png")
        grid_img.filepath_raw = grid_path
        grid_img.file_format = 'PNG'
        grid_img.save()
        bpy.data.images.remove(grid_img)
        
        print(f"Grid saved: {grid_path}")
        
    except Exception as e:
        print(f"Grid stitching failed: {e}")
        print("Individual cells are still available in the output directory.")


if __name__ == "__main__":
    main()
