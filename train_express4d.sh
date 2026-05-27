#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

python train_express4d.py \
  --config config/express4d.yaml \
  --device cuda:0 \
  --max_train_steps 100000
