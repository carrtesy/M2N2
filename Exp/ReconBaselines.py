import os
import pandas as pd
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.MLP import MLP

# others
import torch
import torch.nn.functional as F
import numpy as np

from utils.metrics import get_summary_stats


class MLP_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(MLP_Trainer, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader
        )

        self.model = MLP(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.model.latent_dim,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)


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


    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        # recon
        Xhat = self.model(X.reshape(B, L*C)).reshape(B, L, C)

        # optimize
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()

        out = {
            "recon_loss": loss.item(),
            "summary": loss.item(),
        }
        return out


class MLP_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader):
        super(MLP_Tester, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader,
            test_loader=test_loader
        )

        self.model = MLP(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.model.latent_dim,
        ).to(self.args.device)


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader):
        '''
        :return:  returns (B, L, C) recon loss tensor
        '''
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat = self.model(X.reshape(B, L * C)).reshape(B, L, C)
            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_errors.append(recon_error)
        recon_errors = torch.cat(recon_errors, axis=0)
        return recon_errors


    def infer(self):
        # offline

        ## train
        train_error_pt_path = os.path.join(self.args.output_path, "train_errors.pt")
        if os.path.isfile(train_error_pt_path):
            self.logger.info("train_errors.pt file exists, loading...")
            with open(train_error_pt_path, 'rb') as f:
                train_errors = torch.load(f)
                train_errors.to(self.args.device)
            self.logger.info(f"{train_errors.shape}")
        else:
            self.logger.info("train_errors.pt file does not exist, calculating...")
            train_errors = self.calculate_recon_errors(self.train_loader).mean(dim=2)  # (B, L, C) => (B, L)
            self.logger.info("saving train_errors.pt...")
            with open(train_error_pt_path, 'wb') as f:
                torch.save(train_errors, f)

        ## test
        test_error_pt_path = os.path.join(self.args.output_path, "test_errors.pt")
        if os.path.isfile(test_error_pt_path):
            self.logger.info("test_errors.pt file exists, loading...")
            with open(test_error_pt_path, 'rb') as f:
                test_errors = torch.load(f)
                test_errors.to(self.args.device)
            self.logger.info(f"{test_errors.shape}")
        else:
            self.logger.info("test_errors.pt file does not exist, calculating...")
            test_errors = self.calculate_recon_errors(self.test_loader).mean(dim=2)  # (B, L, C) => (B, L)
            self.logger.info("saving test_errors.pt...")
            with open(test_error_pt_path, 'wb') as f:
                torch.save(test_errors, f)

        th_q95 = torch.quantile(train_errors, 0.95).item()
        th_q99 = torch.quantile(train_errors, 0.99).item()
        th_q100 = torch.max(train_errors).item()

        gt = self.test_loader.dataset.y
        test_errors = test_errors.reshape(-1).detach().cpu().numpy()

        th_best_static = self.get_best_static_threshold(gt=gt, anomaly_scores=test_errors)

        q95_result = get_summary_stats(gt, test_errors > th_q95)
        q99_result = get_summary_stats(gt, test_errors > th_q99)
        q100_result = get_summary_stats(gt, test_errors > th_q100)
        best_static_result = get_summary_stats(gt, test_errors > th_best_static)

        result_df = pd.DataFrame(columns=q95_result.keys())
        result_df = result_df.append(pd.DataFrame([q95_result], index=["Q95"], columns=result_df.columns))
        result_df = result_df.append(pd.DataFrame([q99_result], index=["Q99"], columns=result_df.columns))
        result_df = result_df.append(pd.DataFrame([q100_result], index=["Q100"], columns=result_df.columns))
        result_df = result_df.append(pd.DataFrame([best_static_result], index=["BEST"], columns=result_df.columns))

        # online
        online_pred = self.online_inference(self.test_loader, th_q95)
        online_result = get_summary_stats(gt, online_pred)
        result_df = result_df.append(pd.DataFrame([online_result], index=["Online"], columns=result_df.columns))

        # log result
        wt = wandb.Table(dataframe=result_df)
        wandb.log({"result_table": wt})
        self.logger.info("\n" + result_df)


    def online_inference(self, dataloader, init_thr):
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        thr = torch.tensor(init_thr, requires_grad=True)

        self.optimizer = torch.optim.SGD(
            params=[thr] + [p for p in self.model.parameters()],
            lr=self.args.lr
        )

        bce = torch.nn.BCELoss()

        anoscs, preds = [], []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            y = batch_data[1].to(self.args.device)

            # infer
            B, L, C = X.shape
            Xhat = self.model(X.reshape(B, L * C)).reshape(B, L, C)

            e = F.mse_loss(Xhat, X, reduction='none')
            anosc = e.mean(dim=2)
            anoscs.append(anosc)

            pred = (anosc > thr)
            preds.append(pred)

            # update
            self.optimizer.zero_grad()
            yhat = torch.sigmoid(anosc - thr)
            cls_loss = bce(yhat, y.float())
            mask = ((y == 0).unsqueeze(2))
            recon_loss = (e * mask).mean()
            loss = cls_loss + recon_loss
            loss.backward()
            self.optimizer.step()

        preds = torch.cat(preds).reshape(-1).detach().cpu().numpy()

        return preds