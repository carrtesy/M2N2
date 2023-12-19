#!/bin/bash
echo "gpu $1"
for SEED in 2021 2022 2023 2024 2025
do
  # offline
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
   infer_options=["offline"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
   infer_options=["offline"] thresholding=q99.6

  # offline + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
   gamma=0.99 \
   infer_options=["offline_detrend"] thresholding=off_f1_best
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
   gamma=0.99 \
   infer_options=["offline_detrend"] thresholding=q99.7

  # online
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
   ttlr=1e-03 \
   infer_options=["online"] thresholding=q92.0
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
   ttlr=1e-03 \
   infer_options=["online"] thresholding=q99.5

  # online + detrend
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
   normalization=Detrend \
   ttlr=1e-03 \
   gamma=0.99 \
   infer_options=["online"] thresholding=q99.9
  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
   normalization=Detrend \
   ttlr=1e-03 \
   gamma=0.99 \
   infer_options=["online"] thresholding=q99.8
done