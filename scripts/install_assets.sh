#!/bin/bash
# Install/download all assets needed for domain randomization rendering.
# Run this once before using the render pipeline.
#
# Assets downloaded:
#   - HDRI environment maps (50) from TontonTremblay/env_map_hdri
#   - Surface textures (32) from polyhaven.org (CC0)
#   - Distractor objects (5 USDC + 15 textures) from TontonTremblay/meta_assets_2k

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="${PROJECT_DIR}/assets"

echo "=============================================="
echo "  Installing assets for blender2dr"
echo "=============================================="

# -----------------------------------------------
# 1. HDRI environment maps
# -----------------------------------------------
echo ""
echo "[1/3] Downloading HDRI environment maps..."
python3 << 'EOF'
import os, random
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree

random.seed(42)
hdri_dir = os.path.join(os.environ.get("ASSETS_DIR", "assets"), "hdri")
os.makedirs(hdri_dir, exist_ok=True)

existing = len(list(Path(hdri_dir).glob("*.hdr")))
if existing >= 50:
    print(f"  Already have {existing} HDRIs, skipping.")
else:
    files = [f for f in list_repo_tree("TontonTremblay/env_map_hdri", repo_type="dataset")
             if hasattr(f, 'rfilename') and f.rfilename.endswith('.hdr')]
    selected = random.sample(files, min(50, len(files)))
    for f in selected:
        dest = os.path.join(hdri_dir, f.rfilename)
        if os.path.exists(dest):
            continue
        hf_hub_download("TontonTremblay/env_map_hdri", f.rfilename, 
                       repo_type="dataset", local_dir=hdri_dir)
    print(f"  Done: {len(list(Path(hdri_dir).glob('*.hdr')))} HDRIs")
EOF

# -----------------------------------------------
# 2. Surface textures from polyhaven (CC0)
# -----------------------------------------------
echo ""
echo "[2/3] Downloading surface textures from polyhaven..."
python3 << 'EOF'
import urllib.request, os

out_dir = os.path.join(os.environ.get("ASSETS_DIR", "assets"), "surface_textures")
os.makedirs(out_dir, exist_ok=True)

names = [
    "wood_cabinet_worn_long", "concrete_floor_02", "metal_plate",
    "painted_plaster_wall", "stone_wall", "wood_floor_deck",
    "gravel_concrete", "rusty_metal_02", "plywood", "asphalt_04",
    "marble_01", "bark_willow", "corrugated_iron",
    "aerial_rocks_04", "brown_mud_leaves_01", "castle_brick_07",
    "concrete_wall_008", "dark_brick_wall", "green_metal_rust",
    "herringbone_parquet", "laminate_floor_02", "leather_white",
    "old_planks_02", "pavement_04", "red_bricks_04",
    "rubber_tiles", "slate_floor", "snow_02",
    "weathered_planks", "wood_table_001", "yellow_brick", "concrete_layers"
]

count = 0
for name in names:
    dest = os.path.join(out_dir, f"{name}.jpg")
    if os.path.exists(dest):
        count += 1
        continue
    for suffix in ["_diff_1k.jpg", "_diffuse_1k.jpg"]:
        url = f"https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/{name}/{name}{suffix}"
        try:
            urllib.request.urlretrieve(url, dest)
            count += 1
            break
        except:
            continue

print(f"  Done: {count} surface textures")
EOF

# -----------------------------------------------
# 3. Distractor objects from meta_assets_2k
# -----------------------------------------------
echo ""
echo "[3/3] Downloading distractor objects + textures..."
export ASSETS_DIR="${ASSETS_DIR}"
python3 << 'EOF'
import os
from huggingface_hub import hf_hub_download

assets_dir = os.environ.get("ASSETS_DIR", "assets")
distractor_dir = os.path.join(assets_dir, "distractors")
os.makedirs(os.path.join(distractor_dir, "textures"), exist_ok=True)

# 5 USDC objects
objects = [
    "DTC_1_0_Marker_B07Z5P84J2_PurplishPink_3d-asset.usdc",
    "DTC_1_0_Cup_B0CR45H24G_Blue_3d-asset.usdc",
    "DTC_1_0_Knife_B00421ATJK_Purple_3d-asset.usdc",
    "DTC_1_0_Bowl_B07WG43L2D_RedOrange_TU_3d-asset.usdc",
    "DTC_1_0_Shampoo_B09RX3PRL1_GrayWhiteCap_3d-asset.usdc",
]

for obj in objects:
    dest = os.path.join(distractor_dir, obj)
    if not os.path.exists(dest):
        print(f"  Downloading {obj}...")
        hf_hub_download("TontonTremblay/meta_assets_2k", obj, repo_type="dataset", local_dir=distractor_dir)

# Their textures (3 per object)
textures = [
    "textures/DTC_1_0_Bowl_B07WG43_10.png",
    "textures/DTC_1_0_Bowl_B07WG43_11.png",
    "textures/DTC_1_0_Bowl_B07WG43_9.png",
    "textures/DTC_1_0_Cup_B0CR45H2_0.png",
    "textures/DTC_1_0_Cup_B0CR45H2_1.png",
    "textures/DTC_1_0_Cup_B0CR45H2_2.png",
    "textures/DTC_1_0_Knife_B00421_0.png",
    "textures/DTC_1_0_Knife_B00421_1.png",
    "textures/DTC_1_0_Knife_B00421_2.png",
    "textures/DTC_1_0_Marker_B07Z5_30.png",
    "textures/DTC_1_0_Marker_B07Z5_31.png",
    "textures/DTC_1_0_Marker_B07Z5_32.png",
    "textures/DTC_1_0_Shampoo_B09R_6.png",
    "textures/DTC_1_0_Shampoo_B09R_7.png",
    "textures/DTC_1_0_Shampoo_B09R_8.png",
]

for tex in textures:
    dest = os.path.join(distractor_dir, tex)
    if not os.path.exists(dest):
        print(f"  Downloading {tex}...")
        hf_hub_download("TontonTremblay/meta_assets_2k", tex, repo_type="dataset", local_dir=distractor_dir)

print(f"  Done: {len(objects)} objects + {len(textures)} textures")
EOF

echo ""
echo "=============================================="
echo "  Asset installation complete!"
echo "  Location: ${ASSETS_DIR}"
echo "=============================================="
