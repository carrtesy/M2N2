# THOC
export CUDA_VISIBLE_DEVICES=$1;
#python test.py dataset=SWaT window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=WADI window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=PSM window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=CreditCard window_size=100 stride=1 eval_stride=1 batch_size=64 eval_batch_size=64 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=MSL_P-15 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
#python test.py dataset=SMD_machine-1-4 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01  &&
#python test.py dataset=yahoo_20 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01

python test.py dataset=SMAP_D-13 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
python test.py dataset=SMAP_T-3 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01 &&
python test.py dataset=yahoo_55 window_size=100 stride=1 eval_stride=1 batch_size=1 eval_batch_size=1 normalization=None model=THOC infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01
