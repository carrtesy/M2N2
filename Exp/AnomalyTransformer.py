import wandb

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.AnomalyTransformer import AnomalyTransformer

# utils
from utils.tools import plot_interval, get_best_static_threshold
from utils.loss import my_kl_loss
from utils.optim import adjust_learning_rate

# others
import torch
import torch.nn as nn
import numpy as np
import time
import os

import pandas as pd
from utils.metrics import get_summary_stats
import matplotlib.pyplot as plt


# code referrence:
# https://github.com/thuml/Anomaly-Transformer/blob/72a71e5f0847bd14ba0253de899f7b0d5ba6ee97/solver.py#L130

class AnomalyTransformer_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(AnomalyTransformer_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader)
        self.model = AnomalyTransformer(
            win_size=self.args.window_size,
            enc_in=self.args.num_channels,
            c_out=self.args.num_channels,
            e_layers=self.args.model.e_layers,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()


    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)
        self.model.train()
        time_now = time.time()
        train_steps = len(self.train_loader)

        for epoch in range(self.args.epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.args.device)
                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()))
                                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.window_size)).detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)),series[u].detach()))
                                   + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.args.model.k * series_loss).item())
                loss1 = rec_loss - self.args.model.k * series_loss
                loss2 = rec_loss + self.args.model.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    self.logger.info(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            self.logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(loss1_list)
            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} ")
            adjust_learning_rate(self.optimizer, epoch + 1, self.args.lr)
            self.logger.info(f'Updating learning rate to {self.optimizer.param_groups[0]["lr"]}')

            self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))


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
