export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
python test.py dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline_all"] +qStart=0.99 +qEnd=1.00 +qStep=0.001 &&
python test.py dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline_all"] +qStart=0.999 +qEnd=1.00 +qStep=0.0001