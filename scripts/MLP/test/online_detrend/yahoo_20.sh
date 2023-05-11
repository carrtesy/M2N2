export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=yahoo_20 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=2 \
 normalization=Detrend \
 ttlr=0.005 \
 gamma=0.9 \
 infer_options=["online_all"] +qStart=0.80 +qEnd=1.00 +qStep=0.01