#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=WARN

python train/express4d_duration/train_express4d.py \
  --config CSDI/config/express4d.yaml \
  --device cuda:0 \
  --max_train_steps 50000 \
  --save_interval_steps 10000 \
  --data_parallel
