# USAD
export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=SWaT save_roc_curve=True window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 scaler=minmax model=USAD normalization=None model.latent_dim=40 infer_options=["offline"] thresholding=off_f1_best  &&
python test.py dataset=WADI window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 scaler=minmax model=USAD normalization=None  model.latent_dim=40 infer_options=["offline"] thresholding=off_f1_best  &&

python test.py dataset=SMD_machine-1-4 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=38 infer_options=["offline"] thresholding=off_f1_best  &&
python test.py dataset=SMD_machine-2-1 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=38 infer_options=["offline"] thresholding=off_f1_best

python test.py dataset=MSL_P-15 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=33 infer_options=["offline"] thresholding=off_f1_best  &&
python test.py dataset=SMAP_T-3 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=33 infer_options=["offline"] thresholding=off_f1_best  &&

python test.py dataset=CreditCard save_roc_curve=True window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 scaler=minmax model=USAD normalization=None model.latent_dim=40 infer_options=["offline"] thresholding=off_f1_best  &&

python test.py dataset=yahoo_20 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=40 infer_options=["offline"] thresholding=off_f1_best
python test.py dataset=yahoo_55 save_roc_curve=True window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 scaler=minmax model=USAD normalization=None model.latent_dim=40 infer_options=["offline"] thresholding=off_f1_best
