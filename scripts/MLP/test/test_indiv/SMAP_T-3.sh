#!/bin/bash
echo "gpu $1"
for SEED in 2021 2022 2023 2024 2025
do
  # offline
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   infer_options=["offline"] thresholding=off_f1_best &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   infer_options=["offline"] thresholding=q100 &&

  # offline + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
   gamma=0.8 \
   infer_options=["offline_detrend"] thresholding=q95.0 &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
   gamma=0.8 \
   infer_options=["offline_detrend"] thresholding=q100 &&

  # online
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   ttlr=1 \
   infer_options=["online"] thresholding=off_f1_best &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   ttlr=1 \
   infer_options=["online"] thresholding=off_f1_best &&

  # online + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
   normalization=Detrend \
   ttlr=1 \
   gamma=0.8 \
   infer_options=["online"] thresholding=q97.0 &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMAP +dataset_id=T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
   normalization=Detrend \
   ttlr=1 \
   gamma=0.8 \
   infer_options=["online"] thresholding=q98.0

done