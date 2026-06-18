#!/bin/bash
# Render a 9x9 video grid: each cell is a different DR variation rendered as video.
# Uses different blend files (scenes) across cells.
#
# Usage: ./render_video_grid.sh

set -e

PROJECT_DIR="/Users/jtremblay/code/blender2dr"
BLEND_DIR="/Users/jtremblay/Downloads/sim2real_jaco_examples"
OUTPUT_DIR="${PROJECT_DIR}/output/video_grid"
GRID_SIZE=9
RESOLUTION=128  # per cell (9x9 = 1152x1152 final)
FRAME_STEP=3
SAMPLES=8

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Get all blend files
BLEND_FILES=($(find "${BLEND_DIR}" -name "*.blend" | sort))
N_FILES=${#BLEND_FILES[@]}

echo "=============================================="
echo "  Rendering ${GRID_SIZE}x${GRID_SIZE} video grid"
echo "  ${N_FILES} blend files, ${RESOLUTION}px per cell"
echo "=============================================="

N_TOTAL=$((GRID_SIZE * GRID_SIZE))

for i in $(seq 0 $((N_TOTAL - 1))); do
    # Pick blend file (cycle through them)
    BLEND_IDX=$((i % N_FILES))
    BLEND_FILE="${BLEND_FILES[$BLEND_IDX]}"
    SEED=$((100 + i))
    CELL_DIR="${OUTPUT_DIR}/cell_$(printf '%03d' $i)"
    
    echo "[$(($i+1))/${N_TOTAL}] seed=${SEED} file=$(basename ${BLEND_FILE})"
    
    cd "${PROJECT_DIR}/assets/distractors"
    blender --background "${BLEND_FILE}" --python "${PROJECT_DIR}/scripts/render_dr.py" -- \
        --mode animation \
        --output_dir "${CELL_DIR}" \
        --seed ${SEED} \
        --n_variations 1 \
        --resolution ${RESOLUTION} \
        --engine CYCLES \
        --samples ${SAMPLES} \
        --frame_step ${FRAME_STEP} \
        --handheld \
        2>/dev/null
    
    # Convert frame sequence to mp4
    ffmpeg -y -framerate 10 -i "${CELL_DIR}/variation_0000/frame_%04d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 23 \
        "${OUTPUT_DIR}/cell_$(printf '%03d' $i).mp4" 2>/dev/null
done

echo ""
echo "All ${N_TOTAL} cells rendered. Stitching grid video..."

# Build ffmpeg filter for 9x9 grid
# First create a file list
INPUTS=""
FILTER=""
for i in $(seq 0 $((N_TOTAL - 1))); do
    INPUTS="${INPUTS} -i ${OUTPUT_DIR}/cell_$(printf '%03d' $i).mp4"
done

# Build xstack layout for 9x9
LAYOUT=""
for row in $(seq 0 $((GRID_SIZE - 1))); do
    for col in $(seq 0 $((GRID_SIZE - 1))); do
        idx=$((row * GRID_SIZE + col))
        if [ -n "$LAYOUT" ]; then
            LAYOUT="${LAYOUT}|"
        fi
        LAYOUT="${LAYOUT}${col}*${RESOLUTION}_${row}*${RESOLUTION}"
    done
done

# Use xstack filter
ffmpeg -y ${INPUTS} \
    -filter_complex "xstack=inputs=${N_TOTAL}:layout=${LAYOUT}" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 \
    "${OUTPUT_DIR}/grid_9x9.mp4" 2>/dev/null

echo "=============================================="
echo "  Done! Output: ${OUTPUT_DIR}/grid_9x9.mp4"
echo "=============================================="
