#!/bin/bash
# THOC
echo "gpu $1"
for SEED in 2021 2022 2023 2024 2025
do
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SWaT save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=q97
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-1-4 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=q99
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=q97
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=CreditCard save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_55 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
done