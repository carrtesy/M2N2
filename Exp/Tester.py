import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle
from utils.metrics import get_summary_stats
import os
from utils.tools import plot_interval, get_best_static_threshold

import wandb
import pandas as pd


from sklearn.metrics import roc_curve, roc_auc_score
from utils.metrics import calculate_roc_auc

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
        '''
        # train
        train_anoscs_pt_path = os.path.join(self.args.output_path, "train_anoscs.pt")
        if self.args.load_anoscs and os.path.isfile(train_anoscs_pt_path):
            self.logger.info("train_anoscs.pt file exists, loading...")
            with open(train_anoscs_pt_path, 'rb') as f:
                train_anoscs = torch.load(f)
                train_anoscs.to(self.args.device)
            self.logger.info(f"{train_anoscs.shape}")
        else:
            self.logger.info("train_anoscs.pt file does not exist, calculating...")
            train_anoscs = torch.Tensor(self.calculate_anomaly_scores(self.train_loader))  # (B, L, C) => (B, L)
            self.logger.info("saving train_anoscs.pt...")
            with open(train_anoscs_pt_path, 'wb') as f:
                torch.save(train_anoscs, f)
        torch.cuda.empty_cache()

        # test
        test_anosc_pt_path = os.path.join(self.args.output_path, "test_anoscs.pt")
        if self.args.load_anoscs and os.path.isfile(test_anosc_pt_path):
            self.logger.info("test_anoscs.pt file exists, loading...")
            with open(test_anosc_pt_path, 'rb') as f:
                test_anoscs = torch.load(f)
                test_anoscs.to(self.args.device)
            self.logger.info(f"{test_anoscs.shape}")
        else:
            self.logger.info("test_anoscs.pt file does not exist, calculating...")
            test_anoscs = torch.Tensor(self.calculate_anomaly_scores(self.test_loader))  # (B, L, C) => (B, L)
            self.logger.info("saving test_anoscs.pt...")
            with open(test_anosc_pt_path, 'wb') as f:
                torch.save(test_anoscs, f)
        torch.cuda.empty_cache()

        # train_anoscs, test anoscs (T=B*L, ) and ground truth
        train_mask = (self.train_loader.dataset.y != -1)
        self.train_anoscs = train_anoscs.detach().cpu().numpy()[train_mask] # does not include -1's
        self.test_anoscs = test_anoscs.detach().cpu().numpy() # may include -1's, filtered when calculating final results.
        self.gt = self.test_loader.dataset.y

        # thresholds for visualization
        self.th_q95 = np.quantile(self.train_anoscs, 0.95)
        self.th_q99 = np.quantile(self.train_anoscs, 0.99)
        self.th_q100 = np.quantile(self.train_anoscs, 1.00)
        self.th_off_f1_best = get_best_static_threshold(gt=self.gt, anomaly_scores=self.test_anoscs)


    def infer(self, mode, cols):
        result_df = pd.DataFrame(columns=cols)
        gt = self.test_loader.dataset.y

        # for single inference: select specific threshold tau
        th = self.args.thresholding
        if th[0] == "q":
            th = float(th[1:]) / 100
            tau = np.quantile(self.train_anoscs, th)
        elif th == "off_f1_best":
            tau = self.th_off_f1_best
        else:
            raise ValueError(f"Thresholding mode {self.args.thresholding} is not supported.")

        # get result
        if mode == "offline":
            anoscs, pred = self.offline(tau)
            result = get_summary_stats(gt, pred)
            roc_auc = calculate_roc_auc(gt, anoscs,
                                        path=self.args.output_path,
                                        save_roc_curve=self.args.save_roc_curve,
                                        drop_intermediate=False
                                        )
            result["ROC_AUC"] = roc_auc
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            result_df.at[mode, "tau"] = tau

        elif mode == "offline_all":
            result_df = self.offline_all(
                cols=cols, qStart=self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )

        elif mode == "offline_detrend":
            anoscs, pred = self.offline_detrend(tau)
            result = get_summary_stats(gt, pred)
            roc_auc = calculate_roc_auc(gt, anoscs,
                                        path=self.args.output_path,
                                        save_roc_curve=self.args.save_roc_curve,
                                        drop_intermediate=False
                                        )

            result["ROC_AUC"] = roc_auc
            wandb.log(result)
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            result_df.at[mode, "tau"] = tau

        elif mode == "offline_detrend_all":
            result_df = self.offline_detrend_all(
                cols=cols, qStart=self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )

        elif mode == "online":
            anoscs, pred = self.online(self.test_loader, tau, normalization=self.args.normalization)
            result = get_summary_stats(gt, pred)
            roc_auc = calculate_roc_auc(gt, anoscs,
                                        path=self.args.output_path,
                                        save_roc_curve=self.args.save_roc_curve,
                                        drop_intermediate=False,
                                        )
            result["ROC_AUC"] = roc_auc
            wandb.log(result)
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            result_df.at[mode, "tau"] = tau

        elif mode == "online_all":
            result_df = self.online_all(
                cols=cols, qStart= self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )

        elif mode == "online_label":
            anoscs, pred = self.online_label(self.test_loader, tau, normalization=self.args.normalization)
            result = get_summary_stats(gt, pred)
            roc_auc = calculate_roc_auc(gt, anoscs,
                                        path=self.args.output_path,
                                        save_roc_curve=self.args.save_roc_curve,
                                        drop_intermediate=False)
            result["ROC_AUC"] = roc_auc
            wandb.log(result)
            result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            result_df.at[mode, "tau"] = tau

        elif mode == "online_label_all":
            result_df = self.online_label_all(
                cols=cols, qStart= self.args.qStart, qEnd=self.args.qEnd, qStep=self.args.qStep
            )

        if self.args.save_result:
            filename = f"{self.args.exp_id}_{mode}_{th}.csv" if (not hasattr(self.args, "qStart")) \
                else f"{self.args.exp_id}_{mode}_{self.args.qStart}_{self.args.qEnd}_{self.args.qStep}.csv"
            path = os.path.join(self.args.result_path, filename)
            self.logger.info(f"Saving result to {path}")
            result_df.to_csv(path)

        self.logger.info(f"{mode} \n {result_df.to_string()}")
        return result_df


    def offline(self, tau):
        # plot results
        plt.figure(figsize=(20, 6))
        plt.plot(self.test_anoscs, color="blue", label="anomaly score w/o online learning")
        plt.axhline(self.th_q95, color="C1", label="Q95 threshold")
        plt.axhline(self.th_q99, color="C2", label="Q99 threshold")
        plt.axhline(self.th_q100, color="C3", label="Q100 threshold")
        plt.axhline(self.th_off_f1_best, color="C4", label="threshold w/ test data")

        plot_interval(plt, self.gt)
        plt.legend()
        plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_offline.png"))
        wandb.log({f"{self.args.exp_id}_offline": wandb.Image(plt)})

        pred = (self.test_anoscs >= tau)
        return self.test_anoscs, pred


    def offline_all(self, cols, qStart=0.90, qEnd=1.00, qStep=0.01):
        result_df = pd.DataFrame(columns=cols)

        # according to quantiles.
        for q in np.arange(qStart, qEnd + 1e-07, qStep):
            th = np.quantile(self.train_anoscs, min(q, qEnd))
            result = get_summary_stats(self.gt, self.test_anoscs >= th)
            result_df = pd.concat([result_df, pd.DataFrame([result], index=[f"Q{q*100:.3f}"], columns=result_df.columns)])
            result_df.at[f"Q{q*100:.3f}", "tau"] = th

        # off_f1_best
        best_result = get_summary_stats(self.gt, self.test_anoscs >= self.th_off_f1_best)
        roc_auc = calculate_roc_auc(self.gt, self.test_anoscs, path=self.args.output_path,
                                    save_roc_curve=self.args.save_roc_curve)
        best_result["ROC_AUC"] = roc_auc

        result_df = pd.concat(
            [result_df, pd.DataFrame([best_result], index=[f"Q_off_f1_best"], columns=result_df.columns)])
        result_df.at[f"Q_off_f1_best", "tau"] = self.th_off_f1_best

        # plot results w/o TTA
        plt.figure(figsize=(20, 6))
        plt.plot(self.test_anoscs, color="blue", label="anomaly score w/o online learning")
        plt.axhline(self.th_q95, color="C1", label="Q95 threshold")
        plt.axhline(self.th_q99, color="C2", label="Q99 threshold")
        plt.axhline(self.th_q100, color="C3", label="Q100 threshold")

        plt.axhline(self.th_off_f1_best, color="C4", label="threshold w/ test data")
        plt.legend()
        plot_interval(plt, self.gt)
        plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_woTTA.png"))
        wandb.log({"woTTA": wandb.Image(plt)})
        return result_df


    def offline_detrend(self, *args):
        raise NotImplementedError()


    def offline_detrend_all(self, *args):
        raise NotImplementedError()


    def online(self, *args):
        raise NotImplementedError()


    def online_all(self, *args):
        raise NotImplementedError()


    def online_label(self, *args):
        raise NotImplementedError()


    def online_label_all(self, *args):
        raise NotImplementedError()

