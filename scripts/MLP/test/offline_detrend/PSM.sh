export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=PSM window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.999 \
 infer_options=["offline_detrend_all"] +qStart=0.80 +qEnd=1.00 +qStep=0.01