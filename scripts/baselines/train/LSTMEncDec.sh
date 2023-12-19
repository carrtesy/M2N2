#!/bin/bash
# LSTMEncDec
CUDA_VISIBLE_DEVICES=$1;
echo "GPU $1"
for SEED in 2021 2022 2023 2024 2025
do
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SWaT window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=WADI window_size=10 stride=10 eval_stride=10 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=MSL +dataset_id=P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMAP +dataset_id=T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=CreditCard window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_20 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_55 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None
done



