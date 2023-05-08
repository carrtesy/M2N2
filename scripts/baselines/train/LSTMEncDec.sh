# LSTMEncDec
export CUDA_VISIBLE_DEVICES=$1;
python train.py dataset=SWaT window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=WADI window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=PSM window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=CreditCard window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=MSL_P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=SMD_machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None &&
python train.py dataset=yahoo_20 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None