import torch
import numpy as np
import random
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import roc_curve
import matplotlib as mat
from matplotlib import font_manager, rc


def setPlotStyle():    
    # mat.rcParams['font.family'] = "serif"
    # mat.rcParams['font.serif'] = ["Arial"]
    # mat.rcParams['font.family'] = "sans-serif"
    # mat.rcParams['font.sans-serif'] = "Times new roman"
    # mat.rcParams["axes.titlesize"] = "x-large"
    mat.rcParams['font.size'] = 15
    mat.rcParams['legend.fontsize'] = 15
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.color'] = 'r'
    mat.rcParams['axes.grid'] = 1     
    mat.rcParams['axes.xmargin'] = 0.1     
    mat.rcParams['axes.ymargin'] = 0.1     
    mat.rcParams["mathtext.fontset"] = "dejavuserif" #"cm", "stix", etc.
    mat.rcParams['figure.dpi'] = 500
    mat.rcParams['savefig.dpi'] = 500


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
    _gt = gt[mask]
    _anomaly_scores = anomaly_scores[mask]

    P, N = (_gt == 1).sum(), (_gt == 0).sum()
    fpr, tpr, thresholds = roc_curve(_gt, _anomaly_scores, drop_intermediate=False)

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
    # if den is zero, tp will also be zero, and hence result is zero for both precision/recall.
    den_p = tp+fp
    den_p[den_p==0] = 1
    precision = tp / den_p
    den_r = tp+fn
    den_r[den_r==0] = 1
    recall = tp / den_r

    # f1 score
    den = precision + recall
    den[den==0.0] = 1
    f1 = 2 * (precision * recall) / den
    idx = np.argmax(f1)
    best_threshold = thresholds[idx]
    return best_threshold
