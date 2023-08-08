#!/bin/bash
echo "gpu $1"
for SEED in 2021 2022 2023 2024 2025
do
  # offline
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   infer_options=["offline"] thresholding=off_f1_best &
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   infer_options=["offline"] thresholding=off_f1_best &&

  # offline + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
   gamma=0.6 \
   infer_options=["offline_detrend"] thresholding=q99.0 &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
   gamma=0.6 \
   infer_options=["offline_detrend"] thresholding=off_f1_best &&

  # online
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   ttlr=0.1 \
   infer_options=["online"] thresholding=q93.0 &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
   ttlr=0.1 \
   infer_options=["online"] thresholding=q100 &&

  # online + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
   normalization=Detrend \
   ttlr=0.1 \
   gamma=0.6 \
   infer_options=["online"] thresholding=q99.9 &&
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=MSL +dataset_id=P-15 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
   normalization=Detrend \
   ttlr=0.1 \
   gamma=0.6 \
   infer_options=["online"] thresholding=q99.9
done