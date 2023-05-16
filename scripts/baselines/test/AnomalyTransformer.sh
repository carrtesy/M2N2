# AnomalyTransformer
export CUDA_VISIBLE_DEVICES=$1;
#python test.py dataset=SWaT window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=WADI window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=PSM window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=CreditCard window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=MSL_P-15 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=SMD_machine-1-4 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=yahoo_20 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01



python test.py dataset=SMD_machine-2-1 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01

#python test.py dataset=SMAP_D-13 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=SMAP_T-3 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=yahoo_55 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5  infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01
