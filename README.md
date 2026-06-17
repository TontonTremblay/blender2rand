# Blender Domain Randomization for Sim2Real

Aggressive visual domain randomization pipeline for rendering sim2real robot manipulation training data. Uses pre-animated Blender scenes from physics simulation and applies DR following best practices from MolmoBot (2026), Tremblay et al. (2018), and RoboTwin 2.0 (2025).

## Overview

This pipeline takes `.blend` files containing animated Jaco robot picking motions (generated in a physics simulator) and renders them with aggressive visual domain randomization for sim-to-real transfer. **The motion is never modified** — only the visual appearance changes.

## DR Axes

Following the reference guide (`domain_randomization_reference.md`):

1. **HDRI Environment Lighting** — Random environment maps from [env_map_hdri](https://huggingface.co/datasets/TontonTremblay/env_map_hdri) with random rotation and intensity
2. **Robot Material** — Per-variation: uniform color, metallic, textured, per-part random, or dark robot
3. **Table/Floor/Backdrop Textures** — Random textures from [meta_assets_2k](https://huggingface.co/datasets/TontonTremblay/meta_assets_2k) or procedural colors
4. **Camera Perturbation** — Position, rotation, and focal length jitter
5. **Flying Distractors** — 3-10 random geometric shapes with random materials (à la Tremblay 2018)
6. **Additional Lights** — Random point/area/spot lights with varied color and intensity
7. **Object Material** — Target object texture/color randomization
8. **Mesh Smoothing** — Subdivision + smooth shading on robot (reduces sim blockiness)
9. **Post-Processing** — Exposure and color management variation

## Asset Sources

| Source | Content | URL |
|--------|---------|-----|
| env_map_hdri | ~430 HDR environment maps | https://huggingface.co/datasets/TontonTremblay/env_map_hdri |
| meta_assets_2k | Distractor meshes (USDC) + textures | https://huggingface.co/datasets/TontonTremblay/meta_assets_2k |
| synthetic-scene-content | CCO textures + dome HDRIs | https://huggingface.co/datasets/TontonTremblay/synthetic-scene-content |

## Setup

```bash
# 1. Install dependencies (uses system Blender 5.1+)
pip install huggingface_hub

# 2. Download assets from HuggingFace
python scripts/download_assets.py

# 3. Quick test (renders 3 variations of 1 blend file)
./scripts/test_single_frame.sh
```

## Usage

### Single file with DR
```bash
blender --background <file.blend> --python scripts/render_dr.py -- \
    --output_dir output/my_render \
    --seed 0 \
    --n_variations 10 \
    --resolution 512 \
    --engine CYCLES \
    --samples 64 \
    --frame_step 3
```

### Batch render all scenes
```bash
./scripts/batch_render.sh [n_variations] [resolution] [engine] [samples] [frame_step]

# Examples:
./scripts/batch_render.sh 10 512 CYCLES 64 3    # Production quality
./scripts/batch_render.sh 3 256 BLENDER_EEVEE 16 10  # Fast preview
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | required | Output directory |
| `--seed` | 0 | Random seed base |
| `--n_variations` | 10 | Number of DR variations per scene |
| `--resolution` | 512 | Render resolution (square) |
| `--engine` | CYCLES | CYCLES or BLENDER_EEVEE |
| `--samples` | 64 | Render samples |
| `--frame_step` | 1 | Render every N frames |
| `--n_distractors_min` | 3 | Min flying distractors |
| `--n_distractors_max` | 10 | Max flying distractors |
| `--render_video` | false | Output MP4 instead of PNG sequence |

## Input Data

Blend files from `/Users/jtremblay/Downloads/sim2real_jaco_examples/`:
- 5 objects × 2 seeds = 10 scenes
- Objects: mayonnaise_bottle, pineapple_can, ranch_bottle, tomato_sauce_can, yogurt_cup
- Each contains ~90-141 frames of animated robot pick motion
- Scene structure: Jaco robot (35 mesh parts) + table + target object + backdrop

## Project Structure

```
blender2dr/
├── README.md
├── domain_randomization_reference.md   # Literature reference guide
├── scripts/
│   ├── download_assets.py              # Download HF assets
│   ├── render_dr.py                    # Main DR rendering script
│   ├── batch_render.sh                 # Batch processing
│   └── test_single_frame.sh            # Quick test
├── assets/                             # Downloaded assets (git-ignored)
│   ├── hdri/                           # Environment maps
│   ├── textures/                       # Texture images
│   └── distractors/                    # USD distractor meshes
└── output/                             # Rendered output (git-ignored)
```

## Design Principles

From the DR reference (see `domain_randomization_reference.md`):

1. **Diversity > Photorealism** — More variations of textures/lighting/viewpoints beats photorealistic rendering
2. **Randomize the camera** — Camera pose randomization is critical for zero-shot transfer
3. **Flying distractors** — Force the policy to learn task-relevant features, not background
4. **Aggressive material randomization** — Non-realistic textures/colors are fine and beneficial
5. **Preserve motion** — The physics-simulated trajectories are correct; only randomize visuals
6. **Structured placement** — Distractors respect approximate scene bounds (not purely random)
