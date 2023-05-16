# THOC
export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=SWaT save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=WADI save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=SMD_machine-1-4 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best  &&
python test.py dataset=SMD_machine-2-1 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best

python test.py dataset=MSL_P-15 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=SMAP_T-3 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=CreditCard save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=yahoo_20 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
python test.py dataset=yahoo_55 save_roc_curve=True window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline"] thresholding=off_f1_best
