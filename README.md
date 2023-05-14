# TSAD-on-the-run


## Options

set threshold to offline best f1 score
``` 
python test.py (...) infer_options=["offline"] thresholding=off_f1_best
```

run all thresholds in range(a, b, c)

``` 
python test.py (...) infer_options=["offline_all"] +qStart=0.90 +qEnd=1.00 +qStep=0.01
```

save run information
```
plot_anomaly_scores: False
plot_recon_status: False
save_result: True # save result in pandas dataframe
load_anoscs: True # load previously calcuated anomaly score (if not exist, start from scratch.)
save_outputs: False # gt (X.pt) and reconstructed (Xhat.pt)
save_roc_curve: False # whether to save fpr, tpr, thrs from sklearn.roc_curve
```
