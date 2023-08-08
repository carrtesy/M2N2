import os
import pandas as pd
import wandb
from tqdm import tqdm

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.LSTMEncDec import LSTMEncDec

# others
import torch
import torch.nn.functional as F
import numpy as np



class LSTMEncDec_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(LSTMEncDec_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader)

        self.model = LSTMEncDec(
            input_dim=self.args.num_channels,
            window_size=self.args.window_size,
            latent_dim=self.args.model.latent_dim,
            num_layers=self.args.model.num_layers,
            dropout=self.args.model.dropout,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)


    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)

        train_iterator = tqdm(
            range(1, self.args.epochs + 1),
            total=self.args.epochs,
            desc="training epochs",
            leave=True
        )

        best_train_stats = None

        for epoch in train_iterator:
            # train
            train_stats = self.train_epoch()
            self.logger.info(f"epoch {epoch} | train_stats: {train_stats}")
            self.checkpoint(os.path.join(self.args.checkpoint_path, f"epoch{epoch}.pth"))

            if best_train_stats is None or train_stats < best_train_stats:
                self.logger.info(f"Saving best results @epoch{epoch}")
                self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))
                best_train_stats = train_stats
        return


    def train_epoch(self):
        self.model.train()
        log_freq = len(self.train_loader) // self.args.log_freq
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data)
            if (i+1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary


    def _process_batch(self, batch_data):
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat = self.model(X)
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()

        out = {
            "loss": loss.item(),
            "summary": loss.item(),
        }
        return out


class LSTMEncDec_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader, load=False):
        super(LSTMEncDec_Tester, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        self.model = LSTMEncDec(
            input_dim=self.args.num_channels,
            window_size=self.args.window_size,
            latent_dim=self.args.model.latent_dim,
            num_layers=self.args.model.num_layers,
            dropout=self.args.model.dropout,
        ).to(self.args.device)

        if load:
            self.load_trained_model()
            self.prepare_stats()


    def _process_batch(self, batch_data):
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat = self.model(X)
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()

        out = {
            "loss": loss.item(),
            "summary": loss.item(),
        }
        return out


    def calculate_anomaly_scores(self, dataloader):
        # get train statistics.
        train_recon_errors = self.calculate_recon_errors(dataloader=self.train_loader) # (B, L, C)
        B, L, C = train_recon_errors.shape
        train_recon_errors = train_recon_errors.reshape(B*L, C)
        train_error_mu, train_error_cov = np.mean(train_recon_errors, axis=0), np.cov(train_recon_errors.T)

        # test statistics
        test_recon_errors = self.calculate_recon_errors(dataloader=dataloader)
        B, L, C = test_recon_errors.shape
        test_recon_errors = test_recon_errors.reshape(B*L, C)

        # anomaly scores
        r = test_recon_errors - train_error_mu # e-mu, (T, C)
        ic = np.linalg.pinv(train_error_cov) if C > 1 else np.array([[train_error_cov]]) # inverse of covariance matrix, (C, C)
        anomaly_scores = np.einsum("TC,CC,TC->T", r, ic, r)
        return anomaly_scores


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader):
        '''
        :param dataloader (self.train or self.test)
        :return:  returns (B, L, C) recon loss tensor
        '''
        self.model.eval()
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(iterator):
            X = batch_data[0].to(self.args.device)
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors
