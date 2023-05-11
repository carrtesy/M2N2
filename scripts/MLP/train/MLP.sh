CUDA_VISIBLE_DEVICES=$1;
#python train.py dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#python train.py dataset=SWaT window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#python train.py dataset=WADI window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#python train.py dataset=WADI window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#python train.py dataset=PSM window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#python train.py dataset=PSM window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#python train.py dataset=CreditCard window_size=12 stride=12 eval_stride=12 normalization=None model=MLP model.latent_dim=128 &&
#python train.py dataset=CreditCard window_size=12 stride=12 eval_stride=12 normalization=Detrend model=MLP model.latent_dim=128 &&
#python train.py dataset=MSL_P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=MSL_P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#python train.py dataset=SMD_machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=SMD_machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&

#python train.py dataset=SMD_machine-3-9 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=SMD_machine-3-9 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#

python train.py dataset=SMD_machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &
python train.py dataset=SMD_machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &



#python train.py dataset=SMAP_G-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=SMAP_G-1 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#
#
#python train.py dataset=SMAP_D-13 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=SMAP_D-13 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&
#
#
#python train.py dataset=SMAP_T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=None model=MLP model.latent_dim=16 &&
#python train.py dataset=SMAP_T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 log_freq=1 normalization=Detrend model=MLP model.latent_dim=16 &&


#python train.py dataset=yahoo_20 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=None model=MLP model.latent_dim=2 &&
#python train.py dataset=yahoo_20 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 &&


#python train.py dataset=yahoo_55 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=None model=MLP model.latent_dim=2 &&
#python train.py dataset=yahoo_55 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2 &&
#
#
#python train.py dataset=yahoo_60 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=None model=MLP model.latent_dim=2 &&
#python train.py dataset=yahoo_60 window_size=5 stride=1 eval_stride=5 batch_size=8 epochs=30 log_freq=1 normalization=Detrend model=MLP model.latent_dim=2
