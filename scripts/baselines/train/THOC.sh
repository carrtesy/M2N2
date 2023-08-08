#!/bin/bash
# THOC
echo "GPU $1"
for SEED in 2021 2022 2023 2024 2025
do
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SWaT window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=WADI window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-1-4 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=MSL +dataset_id=P-15 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMAP +dataset_id=T-3 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=CreditCard window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_20 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_55 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC
done
