export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=SMAP_D-13 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
 normalization=Detrend \
 ttlr=0.05 \
 gamma=0.9 \
 infer_options=["online_label_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01