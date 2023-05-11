export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=yahoo_55 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 \
 gamma=0.1 \
 infer_options=["offline_detrend_all"] +qStart=0.80 +qEnd=1.00 +qStep=0.01