# LSTMEncDec
export CUDA_VISIBLE_DEVICES=$1;
python test.py save_roc_curve=True dataset=SWaT window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best  &&
python test.py save_roc_curve=True dataset=WADI window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best  &&

python test.py save_roc_curve=True dataset=SMD_machine-1-4 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best &&
python test.py save_roc_curve=True dataset=SMD_machine-2-1 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best &&

python test.py save_roc_curve=True dataset=MSL_P-15 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best  &&
python test.py save_roc_curve=True dataset=SMAP_T-3 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best  &&

python test.py save_roc_curve=True dataset=CreditCard window_size=12 stride=12 eval_stride=12 batch_size=256 eval_batch_size=256 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best  &&

python test.py save_roc_curve=True dataset=yahoo_20 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best &&
python test.py save_roc_curve=True dataset=yahoo_55 window_size=5 stride=5 eval_stride=5 batch_size=8 eval_batch_size=8 log_freq=1 model=LSTMEncDec scaler=std normalization=None infer_options=["offline"] thresholding=off_f1_best
