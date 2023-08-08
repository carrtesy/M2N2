CUDA_VISIBLE_DEVICES=0 python cc.py dataset=MSL_P-15 normalization=None model=MLP model.latent_dim=128 infer_options=["offline"]
#CUDA_VISIBLE_DEVICES=0 python cc.py dataset=MSL_P-15 normalization=None model=USAD model.latent_dim=128 infer_options=["offline"] &&
#CUDA_VISIBLE_DEVICES=0 python cc.py dataset=MSL_P-15 normalization=None model=THOC model.hidden_dim=128 infer_options=["offline"] &&
#CUDA_VISIBLE_DEVICES=0 python cc.py dataset=MSL_P-15 normalization=None model=AnomalyTransformer infer_options=["offline"] &&
#CUDA_VISIBLE_DEVICES=0 python cc.py dataset=MSL_P-15 normalization=None model=LSTMEncDec model.latent_dim=128  infer_options=["offline"]