#!/bin/bash

for SEED in 2021 2022 2023 2024 2025
do
#  # SWaT
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#
  # WADI
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=WADI window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=WADI window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128
#
#  # SMD (machine-1-4)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#
#  # SMD (machine-2-1)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#
#  # MSL (P-15)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=MSL +dataset_id=P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=MSL +dataset_id=P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16
#
#  # SMAP (T-3)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMAP +dataset_id=T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=SMAP +dataset_id=T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#
#  # CreditCard
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=CreditCard window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=CreditCard window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#
#  # Yahoo (A1-R20)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_20 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=None model=MLP model.latent_dim=2 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_20 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 &&
#
#  # Yahoo (A1-R55)
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_55 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=None model=MLP model.latent_dim=2 &&
#  CUDA_VISIBLE_DEVICES=$1 python train.py SEED=$SEED dataset=yahoo +dataset_id=real_55 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2

done