#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

python train/express4d_condition/train_express4d_condition.py \
  --config CSDI/config/express4d_condition.yaml \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-2048}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-50000}" \
  --save_interval_steps "${SAVE_INTERVAL_STEPS:-10000}" \
  --data_parallel \
  "$@"
