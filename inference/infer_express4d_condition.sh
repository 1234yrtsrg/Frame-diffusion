#!/usr/bin/env bash
set -e

python inference/infer_blendshapes_condition.py \
  --model express4d_condition \
  --checkpoint "${EXPRESS4D_CONDITION_CHECKPOINT:-save/express4d_condition/checkpoint_step_50000.pth}" \
  --keyframes_json "${KEYFRAMES_JSON:-data/blendshapes.json}" \
  --condition "${CONDITION:-3}" \
  --device "${DEVICE:-auto}" \
  --output_dir "${OUTPUT_DIR:-outputs/blendshapes_express4d_condition}" \
  "$@"
