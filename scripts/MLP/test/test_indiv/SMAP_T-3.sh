# offline
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
 infer_options=["offline"] thresholding=q100 &&

# offline + detrend
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
 gamma=0.8 \
 infer_options=["offline_detrend"] thresholding=q95.0 &&
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 \
 gamma=0.8 \
 infer_options=["offline_detrend"] thresholding=q100 &&

# online
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
 ttlr=1 \
 infer_options=["online"] thresholding=off_f1_best &&
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 \
 ttlr=1 \
 infer_options=["online"] thresholding=off_f1_best &&

# online + detrend
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
 normalization=Detrend \
 ttlr=1 \
 gamma=0.8 \
 infer_options=["online"] thresholding=q97.0 &&
python test.py dataset=SMAP_T-3 save_outputs=True window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 model=MLP model.latent_dim=16 \
 normalization=Detrend \
 ttlr=1 \
 gamma=0.8 \
 infer_options=["online"] thresholding=q98.0