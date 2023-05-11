# USAD
export CUDA_VISIBLE_DEVICES=$1;
#python train.py dataset=SWaT window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 scaler=minmax model=USAD normalization=None model.latent_dim=40 &&
#python train.py dataset=WADI window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None  model.latent_dim=40 &&
#python train.py dataset=PSM window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=40 &&
#python train.py dataset=CreditCard window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=40 &&
#python train.py dataset=MSL_P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=33 &&
#python train.py dataset=SMD_machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=38 &&
#python train.py dataset=yahoo_20 window_size=5 stride=1 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=40

python train.py dataset=SMAP_D-13 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=33 &&
python train.py dataset=SMAP_T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=33 &&
python train.py dataset=yahoo_55 window_size=5 stride=1 eval_stride=5 batch_size=8 eval_batch_size=8 epochs=30 scaler=minmax model=USAD scaler=minmax normalization=None model.latent_dim=40
