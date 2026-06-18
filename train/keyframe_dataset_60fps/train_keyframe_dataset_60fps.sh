#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

python train/keyframe_dataset_60fps/train_keyframe_dataset_60fps.py \
  --config CSDI/config/keyframe_dataset_60fps.yaml \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-50000}" \
  --save_interval_steps "${SAVE_INTERVAL_STEPS:-10000}" \
  --dataset_root "${DATASET_ROOT:-dataset/keyframe_dataset_60fps}" \
  --data_dirs "${DATA_DIRS:-dfew,express4d}" \
  --data_parallel \
  "$@"
