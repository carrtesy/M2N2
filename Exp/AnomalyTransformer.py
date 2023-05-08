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
        self.criterion = nn.MSELoss()

        self.load_trained_model()
        self.prepare_stats()



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