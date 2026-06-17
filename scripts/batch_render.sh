#!/bin/bash
# Batch Domain Randomization Rendering for Sim2Real Jaco
#
# Usage: ./batch_render.sh [n_variations] [resolution] [engine]
#
# Examples:
#   ./batch_render.sh 10 512 CYCLES        # 10 variations, 512x512, Cycles
#   ./batch_render.sh 5 256 BLENDER_EEVEE_NEXT  # 5 variations, fast preview

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BLEND_DIR="/Users/jtremblay/Downloads/sim2real_jaco_examples"
OUTPUT_DIR="${PROJECT_DIR}/output"

N_VARIATIONS=${1:-10}
RESOLUTION=${2:-512}
ENGINE=${3:-CYCLES}
SAMPLES=${4:-64}
FRAME_STEP=${5:-3}

echo "=============================================="
echo "  Blender Domain Randomization Batch Render"
echo "=============================================="
echo "  Blend files:   ${BLEND_DIR}"
echo "  Output:        ${OUTPUT_DIR}"
echo "  Variations:    ${N_VARIATIONS}"
echo "  Resolution:    ${RESOLUTION}x${RESOLUTION}"
echo "  Engine:        ${ENGINE}"
echo "  Samples:       ${SAMPLES}"
echo "  Frame step:    ${FRAME_STEP}"
echo "=============================================="

# Find all blend files
BLEND_FILES=$(find "${BLEND_DIR}" -name "*.blend" | sort)
N_FILES=$(echo "$BLEND_FILES" | wc -l | tr -d ' ')
echo "Found ${N_FILES} blend files to process"
echo ""

COUNTER=0
for BLEND_FILE in $BLEND_FILES; do
    COUNTER=$((COUNTER + 1))
    BASENAME=$(basename "$BLEND_FILE" .blend)
    OBJECT_DIR=$(basename "$(dirname "$BLEND_FILE")")
    
    OUT="${OUTPUT_DIR}/${OBJECT_DIR}/${BASENAME}"
    
    echo "[${COUNTER}/${N_FILES}] Processing: ${OBJECT_DIR}/${BASENAME}"
    echo "  Output: ${OUT}"
    
    blender --background "${BLEND_FILE}" --python "${SCRIPT_DIR}/render_dr.py" -- \
        --output_dir "${OUT}" \
        --seed ${COUNTER} \
        --n_variations ${N_VARIATIONS} \
        --resolution ${RESOLUTION} \
        --engine ${ENGINE} \
        --samples ${SAMPLES} \
        --frame_step ${FRAME_STEP} \
        --use_gpu
    
    echo "  ✓ Done: ${BASENAME}"
    echo ""
done

echo "=============================================="
echo "  Batch rendering complete!"
echo "  Total: ${N_FILES} files × ${N_VARIATIONS} variations"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="
