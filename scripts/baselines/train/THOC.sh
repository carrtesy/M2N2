# THOC
export CUDA_VISIBLE_DEVICES=$1;
python train.py dataset=SWaT window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
python train.py dataset=WADI window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
python train.py dataset=PSM window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
python train.py dataset=CreditCard window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC &&
python train.py dataset=MSL_P-15 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
python train.py dataset=SMD_machine-1-4 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC &&
python train.py dataset=yahoo_20 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC
