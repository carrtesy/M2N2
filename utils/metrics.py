import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import os

def get_summary_stats(gt, pred):
    '''
    return acc, prec, recall, f1, (tn, fp, fn, tp)
    '''

    # filter out -1's (paddings)
    mask = (gt != -1)
    _gt = np.copy(gt)[mask]
    _pred = np.copy(pred)[mask]

    # Without Adjust
    acc = accuracy_score(_gt, _pred)
    p = precision_score(_gt, _pred, zero_division=1)
    r = recall_score(_gt, _pred, zero_division=1)
    f1 = f1_score(_gt, _pred, zero_division=1)
    tn, fp, fn, tp = confusion_matrix(_gt, _pred).ravel()

    # Point Adjust
    _pred_PA = PA(_gt, _pred)
    acc_PA = accuracy_score(_gt, _pred_PA)
    p_PA = precision_score(_gt, _pred_PA, zero_division=1)
    r_PA = recall_score(_gt, _pred_PA, zero_division=1)
    f1_PA = f1_score(_gt, _pred_PA, zero_division=1)
    tn_PA, fp_PA, fn_PA, tp_PA = confusion_matrix(_gt, _pred_PA).ravel()

    result = {
        # non-adjusted metrics
        f"Accuracy": acc,
        f"Precision": p,
        f"Recall": r,
        f"F1": f1,
        f"tn": tn,
        f"fp": fp,
        f"fn": fn,
        f"tp": tp,

        # adjusted metrics
        f"Accuracy_PA": acc_PA,
        f"Precision_PA": p_PA,
        f"Recall_PA": r_PA,
        f"F1_PA": f1_PA,
        f"tn_PA": tn_PA,
        f"fp_PA": fp_PA,
        f"fn_PA": fn_PA,
        f"tp_PA": tp_PA,
    }

    del _gt, _pred, _pred_PA
    return result


def calculate_roc_auc(gt, anomaly_scores, path, save_roc_curve=False, drop_intermediate=True):
    # filter out pads
    mask = (gt != -1)
    _gt = gt[mask]
    _anomaly_scores = anomaly_scores[mask]

    # get roc curve
    fpr, tpr, thr = roc_curve(_gt, _anomaly_scores, drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)

    if save_roc_curve:
        with open(os.path.join(path, "fpr.npy"), 'wb') as f:
            np.save(f, fpr)
        with open(os.path.join(path, "tpr.npy"), 'wb') as f:
            np.save(f, tpr)
        with open(os.path.join(path, "thr.npy"), 'wb') as f:
            np.save(f, thr)
    del _gt, _anomaly_scores, fpr, tpr, thr

    return roc_auc

def calculate_pr_auc(gt, anomaly_scores, path, save_pr_curve=False):
    # filter out pads
    mask = (gt != -1)
    _gt = gt[mask]
    _anomaly_scores = anomaly_scores[mask]

    # get roc curve
    prec, rec, thr = precision_recall_curve(_gt, _anomaly_scores)
    pr_auc = auc(rec, prec)

    if save_pr_curve:
        with open(os.path.join(path, "prec.npy"), 'wb') as f:
            np.save(f, prec)
        with open(os.path.join(path, "rec.npy"), 'wb') as f:
            np.save(f, rec)
        with open(os.path.join(path, "thr_prauc.npy"), 'wb') as f:
            np.save(f, thr)
    del _gt, _anomaly_scores, prec, rec, thr

    return pr_auc




'''
Point-Adjust
https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py
'''
def PA(y, y_pred):
    anomaly_state = False
    y_pred_pa = copy.deepcopy(y_pred)
    for i in range(len(y)):
        if y[i] == 1 and y_pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
            for j in range(i, len(y)):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
        elif y[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred_pa[i] = 1

    return y_pred_pa