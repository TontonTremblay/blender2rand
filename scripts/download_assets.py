"""
Download assets from HuggingFace datasets for domain randomization.
- HDRI environment maps from TontonTremblay/env_map_hdri
- Textures from TontonTremblay/synthetic-scene-content (cco_textures.zip)
- Distractor meshes from TontonTremblay/meta_assets_2k
"""

import os
import sys
import random
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree

ASSETS_DIR = Path(__file__).parent.parent / "assets"
HDRI_DIR = ASSETS_DIR / "hdri"
TEXTURES_DIR = ASSETS_DIR / "textures"
DISTRACTORS_DIR = ASSETS_DIR / "distractors"

# How many of each to download (for initial setup; increase for production)
N_HDRI = 50
N_TEXTURES = 100  # texture PNGs from meta_assets_2k/textures
N_DISTRACTORS = 30  # USD files


def download_hdris(n=N_HDRI):
    """Download a random subset of HDRI environment maps."""
    HDRI_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Listing HDRI files from TontonTremblay/env_map_hdri...")
    files = [f for f in list_repo_tree("TontonTremblay/env_map_hdri", repo_type="dataset")
             if hasattr(f, 'rfilename') and f.rfilename.endswith('.hdr')]
    
    print(f"Found {len(files)} HDR files, downloading {n}...")
    selected = random.sample(files, min(n, len(files)))
    
    for f in selected:
        dest = HDRI_DIR / f.rfilename
        if dest.exists():
            continue
        print(f"  Downloading {f.rfilename}...")
        hf_hub_download(
            "TontonTremblay/env_map_hdri",
            f.rfilename,
            repo_type="dataset",
            local_dir=str(HDRI_DIR),
        )
    print(f"HDRI download complete: {len(list(HDRI_DIR.glob('*.hdr')))} files")


def download_textures(n=N_TEXTURES):
    """Download texture PNGs from meta_assets_2k/textures."""
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Listing texture files from TontonTremblay/meta_assets_2k/textures...")
    files = [f for f in list_repo_tree("TontonTremblay/meta_assets_2k", repo_type="dataset",
                                        path_in_repo="textures")
             if hasattr(f, 'path') and f.path.endswith('.png')]
    
    print(f"Found {len(files)} texture PNGs, downloading {n}...")
    selected = random.sample(files, min(n, len(files)))
    
    for f in selected:
        fname = os.path.basename(f.path)
        dest = TEXTURES_DIR / fname
        if dest.exists():
            continue
        print(f"  Downloading {fname}...")
        hf_hub_download(
            "TontonTremblay/meta_assets_2k",
            f.path,
            repo_type="dataset",
            local_dir=str(TEXTURES_DIR),
        )
    print(f"Texture download complete: {len(list(TEXTURES_DIR.glob('**/*.png')))} files")


def download_distractors(n=N_DISTRACTORS):
    """Download USD distractor meshes."""
    DISTRACTORS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Listing distractor files from TontonTremblay/meta_assets_2k...")
    files = [f for f in list_repo_tree("TontonTremblay/meta_assets_2k", repo_type="dataset")
             if hasattr(f, 'rfilename') and f.rfilename.endswith('.usdc')]
    
    print(f"Found {len(files)} USDC files, downloading {n}...")
    selected = random.sample(files, min(n, len(files)))
    
    for f in selected:
        dest = DISTRACTORS_DIR / f.rfilename
        if dest.exists():
            continue
        print(f"  Downloading {f.rfilename}...")
        hf_hub_download(
            "TontonTremblay/meta_assets_2k",
            f.rfilename,
            repo_type="dataset",
            local_dir=str(DISTRACTORS_DIR),
        )
    print(f"Distractor download complete: {len(list(DISTRACTORS_DIR.glob('**/*.usdc')))} files")


if __name__ == "__main__":
    random.seed(42)
    download_hdris()
    download_textures()
    download_distractors()
    print("\n=== All assets downloaded ===")
