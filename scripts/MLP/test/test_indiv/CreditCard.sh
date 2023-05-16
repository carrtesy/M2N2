# offline
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 infer_options=["offline"] thresholding=off_f1_best &&

# offline + detrend
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.999 \
 infer_options=["offline_detrend"] thresholding=q99.93 &&
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 \
 gamma=0.999 \
 infer_options=["offline_detrend"] thresholding=q99.93 &&

# online
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=0.01 \
 infer_options=["online"] thresholding=q99.930 &&
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 \
 ttlr=0.01 \
 infer_options=["online"] thresholding=q99.930 &&

# online + detrend
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=0.01 \
 gamma=0.999 \
 infer_options=["online"] thresholding=q99.93 &&
python test.py dataset=CreditCard save_outputs=True window_size=12 stride=12 eval_stride=12 model=MLP model.latent_dim=128 \
 normalization=Detrend \
 ttlr=0.01 \
 gamma=0.999 \
 infer_options=["online"] thresholding=q99.93