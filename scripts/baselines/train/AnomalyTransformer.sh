# AnomalyTransformer
export CUDA_VISIBLE_DEVICES=$1;
#python train.py dataset=SWaT window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 &&
#python train.py dataset=WADI window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 &&
#python train.py dataset=PSM window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0 &&
#python train.py dataset=CreditCard window_size=100 stride=100 eval_stride=100 batch_size=256 eval_batch_size=256 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 &&
#python train.py dataset=MSL_P-15 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0 &&
#python train.py dataset=SMD_machine-1-4 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5
#python train.py dataset=yahoo_20 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5 &&

python train.py dataset=SMD_machine-2-1 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5

#python train.py dataset=SMAP_D-13 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0 &&
#python train.py dataset=SMAP_T-3 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=1.0 &&
#python train.py dataset=yahoo_55 window_size=100 stride=100 eval_stride=100 batch_size=1 eval_batch_size=1 normalization=None model=AnomalyTransformer model.anomaly_ratio=0.5

