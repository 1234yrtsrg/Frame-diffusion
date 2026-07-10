#!/usr/bin/env bash
set -e

python inference/infer_blendshapes_linear.py \
  --keyframes_json "${KEYFRAMES_JSON:-data/blendshapes.json}" \
  --frames_per_segment "${FRAMES_PER_SEGMENT:-12}" \
  --output_dir "${OUTPUT_DIR:-outputs/blendshapes_linear}" \
  "$@"
