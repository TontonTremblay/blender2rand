#!/bin/bash
# Render a 3x3 video grid: each cell is a different DR variation rendered as video.
# Properly handles non-sequential frame numbering from Blender.
#
# Usage: ./scripts/render_video_grid_3x3.sh

set -e

PROJECT_DIR="/Users/jtremblay/code/blender2dr"
BLEND_FILE="/Users/jtremblay/Downloads/scene.blend"
OUTPUT_DIR="${PROJECT_DIR}/output/video_grid_3x3"
GRID_SIZE=3
RESOLUTION=256
FRAME_STEP=2
SAMPLES=64
ENGINE=BLENDER_EEVEE

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

N_TOTAL=$((GRID_SIZE * GRID_SIZE))

echo "=============================================="
echo "  Rendering ${GRID_SIZE}x${GRID_SIZE} video grid"
echo "  Blend: $(basename ${BLEND_FILE})"
echo "  Resolution: ${RESOLUTION}px per cell"
echo "  Engine: ${ENGINE}, Samples: ${SAMPLES}"
echo "=============================================="

for i in $(seq 0 $((N_TOTAL - 1))); do
    SEED=$((200 + i))
    CELL_DIR="${OUTPUT_DIR}/cell_$(printf '%03d' $i)"
    CELL_MP4="${OUTPUT_DIR}/cell_$(printf '%03d' $i).mp4"
    
    echo ""
    echo "[$(($i+1))/${N_TOTAL}] Rendering cell_$(printf '%03d' $i) (seed=${SEED})..."
    
    cd "${PROJECT_DIR}/assets/distractors"
    blender --background "${BLEND_FILE}" --python "${PROJECT_DIR}/scripts/render_dr.py" -- \
        --mode animation \
        --output_dir "${CELL_DIR}" \
        --seed ${SEED} \
        --n_variations 1 \
        --resolution ${RESOLUTION} \
        --engine ${ENGINE} \
        --samples ${SAMPLES} \
        --frame_step ${FRAME_STEP} \
        --handheld \
        2>/dev/null
    
    # Convert frame sequence to mp4 using glob pattern (handles non-sequential frames)
    FRAME_DIR="${CELL_DIR}/variation_0000"
    
    if [ -d "${FRAME_DIR}" ]; then
        # Get sorted list of frames and use concat demuxer for reliable ordering
        FILELIST="${FRAME_DIR}/filelist.txt"
        rm -f "${FILELIST}"
        for f in $(ls "${FRAME_DIR}"/frame_*.png 2>/dev/null | sort); do
            echo "file '${f}'" >> "${FILELIST}"
            echo "duration 0.1" >> "${FILELIST}"
        done
        
        if [ -f "${FILELIST}" ]; then
            ffmpeg -y -f concat -safe 0 -i "${FILELIST}" \
                -c:v libx264 -pix_fmt yuv420p -crf 23 -vf "fps=10" \
                "${CELL_MP4}" 2>/dev/null
            echo "  ✓ Created $(basename ${CELL_MP4})"
        else
            echo "  ✗ No frames found in ${FRAME_DIR}"
        fi
    else
        echo "  ✗ No variation_0000 directory found"
    fi
done

echo ""
echo "All ${N_TOTAL} cells rendered. Stitching 3x3 grid video..."

# Verify all mp4s exist
MISSING=0
for i in $(seq 0 $((N_TOTAL - 1))); do
    if [ ! -f "${OUTPUT_DIR}/cell_$(printf '%03d' $i).mp4" ]; then
        echo "ERROR: Missing cell_$(printf '%03d' $i).mp4"
        MISSING=$((MISSING + 1))
    fi
done

if [ ${MISSING} -gt 0 ]; then
    echo "Cannot create grid: ${MISSING} cells missing."
    exit 1
fi

# Build ffmpeg inputs
INPUTS=""
for i in $(seq 0 $((N_TOTAL - 1))); do
    INPUTS="${INPUTS} -i ${OUTPUT_DIR}/cell_$(printf '%03d' $i).mp4"
done

# Build xstack layout for 3x3
# Format: x0_y0|x1_y1|...
LAYOUT=""
for row in $(seq 0 $((GRID_SIZE - 1))); do
    for col in $(seq 0 $((GRID_SIZE - 1))); do
        if [ -n "$LAYOUT" ]; then
            LAYOUT="${LAYOUT}|"
        fi
        LAYOUT="${LAYOUT}${col}*${RESOLUTION}_${row}*${RESOLUTION}"
    done
done

# Find shortest video duration to avoid sync issues
# Use xstack with shortest=1
ffmpeg -y ${INPUTS} \
    -filter_complex "xstack=inputs=${N_TOTAL}:layout=${LAYOUT}:shortest=1" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "${OUTPUT_DIR}/grid_3x3.mp4" 2>/dev/null

echo ""
echo "=============================================="
echo "  ✓ Done! Output: ${OUTPUT_DIR}/grid_3x3.mp4"
echo "=============================================="
