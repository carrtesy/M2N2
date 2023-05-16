# offline
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline"] thresholding=q99.6 &&

# offline + detrend
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.99 \
 infer_options=["offline_detrend"] thresholding=off_f1_best &&
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.99 \
 infer_options=["offline_detrend"] thresholding=q99.7 &&

# online
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=1e-03 \
 infer_options=["online"] thresholding=q92.0 &&
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=1e-03 \
 infer_options=["online"] thresholding=q99.5 &&

# online + detrend
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=1e-03 \
 gamma=0.99 \
 infer_options=["online"] thresholding=q99.9 &&
python test.py dataset=WADI save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=1e-03 \
 gamma=0.99 \
 infer_options=["online"] thresholding=q99.8