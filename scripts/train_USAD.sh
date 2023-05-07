# USAD
export CUDA_VISIBLE_DEVICES=$1
python train.py dataset=SWaT model.latent_dim=40 window_size=12 stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=WADI model.latent_dim=40 window_size=10 stride=10 eval_stride=10 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=PSM model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=MSL_P-15 model.latent_dim=33 window_size=5 stride=5 eval_stride=5 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=yahoo_20 model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=2 eval_batch_size=2 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=CreditCard model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
python train.py dataset=SMD_machine-1-4 model.latent_dim=38 window_size=5 stride=5 eval_stride=5 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None