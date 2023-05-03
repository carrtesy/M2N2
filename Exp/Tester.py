import copy

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle
from utils.metrics import get_summary_stats
from sklearn.metrics import roc_curve
import os
from utils.tools import plot_interval, get_best_static_threshold

import wandb
import pandas as pd

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


    def calculate_anomaly_scores(self, dataloader):
        raise NotImplementedError()


    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)


    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)


    def load_trained_model(self):
        self.load(os.path.join(self.args.checkpoint_path, f"best.pth"))


    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)



    def prepare_stats(self):
        '''
        prepare anomaly scores of train data / test data.
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
            train_errors = torch.Tensor(self.calculate_anomaly_scores(self.train_loader))  # (B, L, C) => (B, L)
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
            test_errors = torch.Tensor(self.calculate_anomaly_scores(self.test_loader))  # (B, L, C) => (B, L)
            self.logger.info("saving test_errors.pt...")
            with open(test_error_pt_path, 'wb') as f:
                torch.save(test_errors, f)
        torch.cuda.empty_cache()

        # test errors (T=B*L, ) and ground truth
        self.train_errors = train_errors.detach().cpu().numpy()
        self.test_errors = test_errors.detach().cpu().numpy()
        self.gt = self.test_loader.dataset.y

        # thresholds
        ## quantile-based
        self.th_q95 = np.quantile(self.train_errors, 0.95)
        self.th_q99 = np.quantile(self.train_errors, 0.99)
        self.th_q100 = np.quantile(self.train_errors, 1.00)

        ## with test data
        self.th_best_static = get_best_static_threshold(gt=self.gt, anomaly_scores=self.test_errors)


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
                cols=cols, qStart= self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )
            self.logger.info(f"{mode} \n {result_df.to_string()}")
            return result_df

        elif mode == "offline_wSR_all":
            result_df = self.offline_with_SlowRevIN_all(
                cols=cols, qStart=self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )
            self.logger.info(f"{mode} \n {result_df.to_string()}")
            return result_df

        elif mode == "online":
            th = self.args.thresholding
            if th[0] == "q":
                th = float(th[1:]) / 100
                tau = np.quantile(self.train_errors, th)
            elif th == "otsu":
                tau = self.th_otsu
            elif th == "pot":
                tau = self.th_pot
            elif th == "tbest":
                tau = self.th_best_static

            pred = self.online(self.test_loader, tau, normalization=self.args.normalization)
            result = get_summary_stats(gt, pred)
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            result_df.to_csv(os.path.join(self.args.result_path, f"{self.args.exp_id}_online_{th}.csv"))
            self.logger.info(f"{mode} \n {result_df.to_string()}")

            return result_df

        elif mode == "online_all":
            result_df = self.online_all(
                cols=cols, qStart= self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )
            self.logger.info(f"{mode} \n {result_df.to_string()}")
            return result_df

        elif mode == "online_label_all":
            result_df = self.online_label_all(
                cols=cols, qStart= self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )
            self.logger.info(f"{mode} \n {result_df.to_string()}")
            return result_df


    def online(self, *args):
        raise NotImplementedError()


    def online_all(self, *args):
        raise NotImplementedError()


    def online_label_all(self, *args):
        raise NotImplementedError()


    def online_label(self, *args):
        raise NotImplementedError


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