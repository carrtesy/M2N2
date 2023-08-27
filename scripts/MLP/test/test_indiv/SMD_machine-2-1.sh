#!/bin/bash
echo "gpu $1"
for thr in q98.1 q98.2 q98.3 q98.4 q98.5 q98.6 q98.7 q98.8 q98.9
do
  for SEED in 2021 2022 2023 2024 2025
  do
    # online + detrend
    CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
     normalization=Detrend \
     ttlr=0.05 \
     gamma=0.99 \
     infer_options=["online"] thresholding=$thr
  done
done
#  # offline
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
#   infer_options=["offline"] thresholding=off_f1_best &&
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
#   infer_options=["offline"] thresholding=q99.0 &&
#
#  # offline + detrend
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
#   gamma=0.99 \
#   infer_options=["offline_detrend"] thresholding=q98.0 &&
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
#   gamma=0.99 \
#   infer_options=["offline_detrend"] thresholding=q99.0 &&
#
#  # online
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
#   ttlr=0.05 \
#   infer_options=["online"] thresholding=off_f1_best &&
#  CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
#   ttlr=0.05 \
#   infer_options=["online"] thresholding=q99.0 &&

#    # online + detrend
#    CUDA_VISIBLE_DEVICES=$1 python test.py SEED=$SEED dataset=SMD +dataset_id=machine-2-1 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
#     normalization=Detrend \
#     ttlr=0.05 \
#     gamma=0.99 \
#     infer_options=["online"] thresholding=q9976
