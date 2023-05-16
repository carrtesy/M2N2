# offline
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
infer_options=["offline"] thresholding=q99.980 &&

# offline + detrend
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.99999 \
 infer_options=["offline_detrend"] thresholding=q99.98 &&
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.99999 \
 infer_options=["offline_detrend"] thresholding=q99.97 &&

# online
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=0.005 \
 infer_options=["online"] thresholding=q99.6 &&
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=0.005 \
 infer_options=["online"] thresholding=q99.5 &&

# online + detrend
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=0.005 \
 gamma=0.99999 \
 infer_options=["online"] thresholding=q99.6 &&
python test.py dataset=SWaT save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=0.005 \
 gamma=0.99999 \
 infer_options=["online"] thresholding=q99.3