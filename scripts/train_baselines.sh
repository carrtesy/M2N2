
# LSTMEncDec
dataset=SWaT batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
dataset=WADI batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
dataset=PSM batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
dataset=CreditCard batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None
dataset=MSL_P-15 batch_size=2 eval_batch_size=2 model=LSTMEncDec scaler=std normalization=None
dataset=yahoo_20 batch_size=2 eval_batch_size=2 model=LSTMEncDec scaler=std normalization=None
dataset=SMD_machine-1-4 log_freq=3 batch_size=2 eval_batch_size=2 model=LSTMEncDec scaler=std normalization=None

# USAD
dataset=SWaT model.latent_dim=40 window_size=12 stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
dataset=WADI model.latent_dim=40 window_size=10 stride=10 eval_stride=10 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
dataset=PSM model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
dataset=MSL_P-15 model.latent_dim=33 window_size=5 stride=5 eval_stride=5 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
dataset=yahoo_20 model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=2 eval_batch_size=2 epochs=30 model=USAD scaler=minmax normalization=None
dataset=CreditCard model.latent_dim=40 window_size=12 stride=12 eval_stride=12 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None
dataset=SMD_machine-1-4 model.latent_dim=38 window_size=5 stride=5 eval_stride=5 batch_size=64 eval_batch_size=64 epochs=30 model=USAD scaler=minmax normalization=None

# THOC
dataset=SWaT window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 model=THOC scaler=std normalization=None
dataset=WADI window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 model=THOC scaler=std normalization=None
dataset=PSM window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 model=THOC scaler=std normalization=None
dataset=MSL_P-15 window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 model=THOC scaler=std normalization=None
dataset=yahoo_20 window_size=100 stride=1 eval_stride=1 batch_size=2 eval_batch_size=2 model=THOC scaler=std normalization=None
dataset=CreditCard window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 model=THOC scaler=std normalization=None
dataset=SMD_machine-1-4 window_size=100 stride=1 eval_stride=1 batch_size=2 eval_batch_size=2 model=THOC scaler=std normalization=None

# AnomalyTransformer
dataset=SWaT epochs=3 batch_size=256 eval_batch_size=256 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=0.5 scaler=std normalization=None
dataset=WADI epochs=3 batch_size=256 eval_batch_size=256 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=0.5 scaler=std normalization=None
dataset=PSM epochs=3 batch_size=256 eval_batch_size=256 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=1.0 scaler=std normalization=None
dataset=CreditCard epochs=3 batch_size=256 eval_batch_size=256 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=0.5 scaler=std normalization=None
dataset=MSL_P-15 epochs=3 batch_size=2 eval_batch_size=2 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=1.0 scaler=std normalization=None
dataset=yahoo_20 epochs=3 batch_size=2 eval_batch_size=2 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=0.5 scaler=std normalization=None
dataset=SMD_machine-1-4 epochs=10 batch_size=2 eval_batch_size=2 window_size=100 stride=100 model=AnomalyTransformer model.anomaly_ratio=0.5 scaler=std normalization=None