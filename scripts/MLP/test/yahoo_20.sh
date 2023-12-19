#!/bin/bash
echo "gpu $1"
for SEED in 2021 2022 2023 2024 2025
do
  # offline
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=2 \
   infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=2 \
   infer_options=["offline"] thresholding=off_f1_best

  # offline + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 \
   gamma=0.9 \
   infer_options=["offline_detrend"] thresholding=q88.0
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 \
   gamma=0.9 \
   infer_options=["offline_detrend"] thresholding=q90.0

  # online
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=2 \
   ttlr=0.005 \
   infer_options=["online"] thresholding=q81.0
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=2 \
   ttlr=0.005 \
   infer_options=["online"] thresholding=q100

  # online + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=2 \
   normalization=Detrend \
   ttlr=0.005 \
   gamma=0.9 \
   infer_options=["online"] thresholding=q87.0
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=yahoo +dataset_id=real_20 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=2 \
   normalization=Detrend \
   ttlr=0.005 \
   gamma=0.9 \
   infer_options=["online"] thresholding=q90.0

done