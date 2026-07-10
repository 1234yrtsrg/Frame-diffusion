#!/usr/bin/env bash
set -e

python inference/infer_blendshapes_condition.py \
  --model keyframe_dataset_60fps \
  --checkpoint "${KEYFRAME_DATASET_60FPS_CHECKPOINT:-save/keyframe_dataset_60fps/checkpoint_step_50000.pth}" \
  --keyframes_json "${KEYFRAMES_JSON:-data/blendshapes.json}" \
  --condition "${CONDITION:-3}" \
  --device "${DEVICE:-auto}" \
  --output_dir "${OUTPUT_DIR:-outputs/blendshapes_keyframe_dataset_60fps}" \
  "$@"
