
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix


def get_summary_stats(gt, pred, desc=""):
    '''
    return acc, prec, recall, f1, (tn, fp, fn, tp)
    '''
    acc = accuracy_score(gt, pred)
    p = precision_score(gt, pred, zero_division=1)
    r = recall_score(gt, pred, zero_division=1)
    f1 = f1_score(gt, pred, zero_division=1)
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()

    result = {
        f"Accuracy{desc}": acc,
        f"Precision{desc}": p,
        f"Recall{desc}": r,
        f"F1{desc}": f1,
        f"tn{desc}": tn,
        f"fp{desc}": fp,
        f"fn{desc}": fn,
        f"tp{desc}": tp
    }
    return result