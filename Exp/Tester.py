import copy

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle
from utils.metrics import get_summary_stats
from sklearn.metrics import roc_curve

matplotlib.rcParams['agg.path.chunksize'] = 10000


class Tester:
    '''
    Test-time logics,
    including offline evaluation and online adaptation.
    '''
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader


    def calculate_recon_errors(self, dataloader):
        pass


    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)


    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)


    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)


    @torch.no_grad()
    def get_best_static_threshold(self, gt, anomaly_scores):
        '''
        Find the threshold that maximizes f1-score
        '''
        P, N = (gt == 1).sum(), (gt == 0).sum()
        fpr, tpr, thresholds = roc_curve(gt, anomaly_scores)

        fp = np.array(fpr * N, dtype=int)
        tn = np.array(N - fp, dtype=int)
        tp = np.array(tpr * P, dtype=int)
        fn = np.array(P - tp, dtype=int)

        eps = 1e-6
        precision = tp / np.maximum(tp + fp, eps)
        recall = tp / np.maximum(tp + fn, eps)
        f1 = 2 * (precision * recall) / np.maximum(precision + recall, eps)
        idx = np.argmax(f1)
        best_threshold = thresholds[idx]
        self.logger.info(f"Best threshold found at: {best_threshold}, "
                         f"with fpr: {fpr[idx]}, tpr: {tpr[idx]}\n"
                         f"tn: {tn[idx]} fn: {fn[idx]}\n"
                         f"fp: {fp[idx]} tp: {tp[idx]}")
        return best_threshold
