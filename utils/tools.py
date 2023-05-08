import torch
import numpy as np
import random
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import roc_curve


def SEED_everything(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def read_xlsx_and_convert_to_csv(path):
    excelFile = pd.read_excel(path, skiprows=[0])
    filename = path[:-5]
    excelFile.to_csv(f"{filename}.csv", index=None, header=True)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def plot_interval(ax, interval, facecolor="red", alpha=0.5):
    s, e = None, None
    n = len(interval)
    for i in range(n):
        if interval[i] == 1 and s is None:
            s = i
        elif interval[i] == 0 and s is not None:
            e = i - 1
            if (e - s) > 0:
                ax.axvspan(s, e, facecolor=facecolor, alpha=alpha)
            else:
                ax.axvspan(s-0.5, s+0.5, facecolor=facecolor, alpha=alpha)
            s, e = None, None

    if s is not None:
        if (n - 1 - s) > 0:
            ax.axvspan(s, n-1, facecolor=facecolor, alpha=alpha)
        else:
            ax.axvspan(s - 0.5, s + 0.5, facecolor=facecolor, alpha=alpha)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def get_best_static_threshold(gt, anomaly_scores, th_ubd=None, th_lbd=None):
    '''
    Find the threshold that maximizes f1-score
    '''

    # filter out pads
    mask = (gt != -1)
    gt = np.copy(gt)[mask]
    anomaly_scores = np.copy(anomaly_scores)[mask]

    P, N = (gt == 1).sum(), (gt == 0).sum()
    fpr, tpr, thresholds = roc_curve(gt, anomaly_scores, drop_intermediate=False)

    # filter with availablity
    if th_ubd:
        tidx = (thresholds <= th_ubd)
        fpr, tpr, thresholds = fpr[tidx], tpr[tidx], thresholds[tidx]

    if th_lbd:
        tidx = (thresholds >= th_ubd)
        fpr, tpr, thresholds = fpr[tidx], tpr[tidx], thresholds[tidx]


    fp = np.array(fpr * N, dtype=int)
    tn = np.array(N - fp, dtype=int)
    tp = np.array(tpr * P, dtype=int)
    fn = np.array(P - tp, dtype=int)

    # precision, recall score from confusion matrix
    eps = 1e-12
    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)

    # f1 score
    den = precision + recall
    den[den==0.0] = 1
    f1 = 2 * (precision * recall) / den
    idx = np.argmax(f1)
    best_threshold = thresholds[idx]
    return best_threshold
