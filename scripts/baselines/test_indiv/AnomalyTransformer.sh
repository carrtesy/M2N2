# AnomalyTransformer
export CUDA_VISIBLE_DEVICES=$1;
python test.py dataset=SWaT save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=WADI save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=SMD_machine-1-4 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=SMD_machine-2-1 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline"] thresholding=off_f1_best

python test.py dataset=MSL_P-15 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline"] thresholding=off_f1_best &&
python test.py dataset=SMAP_T-3 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=CreditCard save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline"] thresholding=off_f1_best &&

python test.py dataset=yahoo_20 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline"] thresholding=off_f1_best
python test.py dataset=yahoo_55 save_roc_curve=True window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline"] thresholding=off_f1_best
