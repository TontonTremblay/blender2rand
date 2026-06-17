#!/bin/bash
# Quick test: render a single frame from one blend file with DR
# Usage: ./test_single_frame.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BLEND_FILE="/Users/jtremblay/Downloads/sim2real_jaco_examples/mayonnaise_bottle/mayonnaise_bottle_seed0.blend"
OUTPUT_DIR="${PROJECT_DIR}/output/test_single_frame"

echo "Testing DR on single frame..."
echo "  Blend: ${BLEND_FILE}"
echo "  Output: ${OUTPUT_DIR}"

blender --background "${BLEND_FILE}" --python "${SCRIPT_DIR}/render_dr.py" -- \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --n_variations 3 \
    --resolution 256 \
    --engine BLENDER_EEVEE \
    --samples 32 \
    --frame_step 90

echo "Test complete! Check: ${OUTPUT_DIR}"
