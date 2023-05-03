import wandb

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.AnomalyTransformer import AnomalyTransformer
from pyod.models.deep_svdd import DeepSVDD
#from models.DAGMM import DAGMM
#from models.THOC import THOC

# utils
from utils.metrics import PA
#from utils.optim import adjust_learning_rate
#from utils.custom_loss import my_kl_loss
from utils.metrics import PA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from utils.tools import plot_interval, get_best_static_threshold
from utils.tools import to_var # for DAGMM.
from utils.loss import my_kl_loss
from utils.optim import adjust_learning_rate

# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import datetime

from tqdm import tqdm
import pickle

import pandas as pd
from utils.metrics import get_summary_stats
import matplotlib.pyplot as plt


class AnomalyTransformer_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader):
        super(AnomalyTransformer_Tester, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.model = AnomalyTransformer(
            win_size=self.args.window_size,
            enc_in=self.args.num_channels,
            c_out=self.args.num_channels,
            e_layers=self.args.model.e_layers,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.load_trained_model()
        self.prepare_stats()


    def prepare_stats(self):
        '''
        prepare 95, 99, 100 threshold of train errors.
        prepare test errors, gt, and best threshold with test errors to check optimal bound of offline approaches.
        '''
        # train
        train_error_pt_path = os.path.join(self.args.output_path, "train_errors.pt")
        if self.args.load_previous_error and os.path.isfile(train_error_pt_path):
            self.logger.info("train_errors.pt file exists, loading...")
            with open(train_error_pt_path, 'rb') as f:
                train_errors = torch.load(f)
                train_errors.to(self.args.device)
            self.logger.info(f"{train_errors.shape}")
        else:
            self.logger.info("train_errors.pt file does not exist, calculating...")
            train_errors = self.calculate_anomaly_scores(self.train_loader)
            train_errors = torch.Tensor(train_errors)
            self.logger.info("saving train_errors.pt...")
            with open(train_error_pt_path, 'wb') as f:
                torch.save(train_errors, f)
        torch.cuda.empty_cache()

        # test
        test_error_pt_path = os.path.join(self.args.output_path, "test_errors.pt")
        if self.args.load_previous_error and os.path.isfile(test_error_pt_path):
            self.logger.info("test_errors.pt file exists, loading...")
            with open(test_error_pt_path, 'rb') as f:
                test_errors = torch.load(f)
                test_errors.to(self.args.device)
            self.logger.info(f"{test_errors.shape}")
        else:
            self.logger.info("test_errors.pt file does not exist, calculating...")
            test_errors = self.calculate_anomaly_scores(self.test_loader)
            test_errors = torch.Tensor(test_errors)
            self.logger.info("saving test_errors.pt...")
            with open(test_error_pt_path, 'wb') as f:
                torch.save(test_errors, f)
            self.logger.info(f"{test_errors.shape}")
        torch.cuda.empty_cache()

        # test errors (T=B*L, C) and ground truth
        self.train_errors = train_errors.reshape(-1).detach().cpu().numpy()
        self.test_errors = test_errors.reshape(-1).detach().cpu().numpy()
        self.gt = self.test_loader.dataset.y

        # thresholds
        ## quantile-based
        self.th_q95 = np.quantile(self.train_errors, 0.95)
        self.th_q99 = np.quantile(self.train_errors, 0.99)
        self.th_q100 = np.quantile(self.train_errors, 1.00)

        ## with test data
        self.th_best_static = get_best_static_threshold(gt=self.gt, anomaly_scores=self.test_errors)


    def calculate_anomaly_scores(self, dataloader):
        temperature = self.args.model.temperature
        criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        for i, (input_data, labels) in enumerate(dataloader):
            input = input_data.float().to(self.args.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        return test_energy

    def infer(self, mode, cols):
        result_df = pd.DataFrame(columns=cols)
        gt = self.test_loader.dataset.y

        if mode == "offline":
            pred = self.offline(self.args.thresholding)
            result = get_summary_stats(gt, pred)
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            self.logger.info(f"{mode}-{self.args.thresholding} \n {result_df.to_string()}")
            return result_df

        elif mode == "offline_all":
            result_df = self.offline_all(
                cols=cols, qStart=self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )
            self.logger.info(f"{mode} \n {result_df.to_string()}")
            return result_df


    def offline(self, th="q95.1"):
        if th[0] == "q":
            th = float(th[1:]) / 100
            tau = np.quantile(self.train_errors, th)
        elif th == "otsu":
            tau = self.th_otsu
        elif th == "pot":
            tau = self.th_pot
        elif th == "tbest":
            tau = self.th_best_static

        # plot results w/o TTA
        plt.figure(figsize=(20, 6))
        plt.plot(self.test_errors, color="blue", label="anomaly score w/o online learning")
        plt.axhline(self.th_q95, color="C1", label="Q95 threshold")
        plt.axhline(self.th_q99, color="C2", label="Q99 threshold")
        plt.axhline(self.th_q100, color="C3", label="Q100 threshold")
        plt.axhline(self.th_best_static, color="C4", label="threshold w/ test data")
        plt.axhline(self.th_otsu, color="C5", label="otsu threshold")
        plt.axhline(self.th_pot, color="C6", label="pot+otsu threshold")
        plot_interval(plt, self.gt)
        plt.legend()
        plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_offline.png"))
        wandb.log({f"{self.args.exp_id}_offline": wandb.Image(plt)})

        return (self.test_errors > tau)


    def offline_all(self, cols, qStart=0.90, qEnd=1.00, qStep=0.01):
        result_df = pd.DataFrame(columns=cols)

        # according to quantiles.
        for q in np.arange(qStart, qEnd + qStep, qStep):
            q = min(q, qEnd)
            th = np.quantile(self.train_errors, q)
            result = get_summary_stats(self.gt, self.test_errors > th)
            result_df = pd.concat([result_df, pd.DataFrame([result], index=[f"Q{q*100:.3f}"], columns=result_df.columns)])
            result_df.at[f"Q{q*100:.3f}", "tau"] = th

        # threshold with test data
        best_result = get_summary_stats(self.gt, self.test_errors > self.th_best_static)
        result_df = pd.concat([result_df, pd.DataFrame([best_result], index=[f"Qbest"], columns=result_df.columns)])
        result_df.at[f"Qbest", "tau"] = self.th_best_static
        result_df.to_csv(os.path.join(self.args.result_path, f"{self.args.exp_id}_offline_{qStart}_{qEnd}_{qStep}.csv"))

        # plot results w/o TTA
        plt.figure(figsize=(20, 6))
        plt.plot(self.test_errors, color="blue", label="anomaly score w/o online learning")
        plt.axhline(self.th_q95, color="C1", label="Q95 threshold")
        plt.axhline(self.th_q99, color="C2", label="Q99 threshold")
        plt.axhline(self.th_q100, color="C3", label="Q100 threshold")
        plt.axhline(self.th_best_static, color="C4", label="threshold w/ test data")
        plt.legend()
        plot_interval(plt, self.gt)
        plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_woTTA.png"))
        wandb.log({"woTTA": wandb.Image(plt)})
        return result_df
