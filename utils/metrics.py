import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def get_summary_stats(gt, pred):
    '''
    return acc, prec, recall, f1, (tn, fp, fn, tp)
    '''

    # filter out -1's (paddings)
    mask = (gt != -1)
    gt = np.copy(gt)[mask]
    pred = np.copy(pred)[mask]

    # Without Adjust
    acc = accuracy_score(gt, pred)
    p = precision_score(gt, pred, zero_division=1)
    r = recall_score(gt, pred, zero_division=1)
    f1 = f1_score(gt, pred, zero_division=1)
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()

    # Point Adjust
    pred_PA = PA(gt, pred)
    acc_PA = accuracy_score(gt, pred_PA)
    p_PA = precision_score(gt, pred_PA, zero_division=1)
    r_PA = recall_score(gt, pred_PA, zero_division=1)
    f1_PA = f1_score(gt, pred_PA, zero_division=1)
    tn_PA, fp_PA, fn_PA, tp_PA = confusion_matrix(gt, pred_PA).ravel()

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
    return result



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