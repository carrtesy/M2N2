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
from utils.tools import plot_interval
from utils.metrics import calculate_roc_auc, calculate_pr_auc


class MLP_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(MLP_Trainer, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader
        )

        self.model = MLP(
            seq_len=args.window_size,
            num_channels=args.num_channels,
            latent_space_size=args.model.latent_dim,
            gamma=args.gamma,
            normalization=args.normalization,
        ).to(self.args.device)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=args.lr, weight_decay=self.args.L2_reg) # L2 Reg is set to zero by default, but can be set as needed.
        self.logger.info(f"\n{self.model}")


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
        Xhat = self.model(X)

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
    def __init__(self, args, logger, train_loader, test_loader, load=False):
        super(MLP_Tester, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader,
            test_loader=test_loader
        )

        self.model = MLP(
            seq_len=args.window_size,
            num_channels=args.num_channels,
            latent_space_size=args.model.latent_dim,
            gamma=args.gamma,
            normalization=args.normalization,
        ).to(self.args.device)

        if load:
            self.load_trained_model()
            self.prepare_stats()


    @torch.no_grad()
    def calculate_anomaly_scores(self, dataloader):
        recon_errors = self.calculate_recon_errors(dataloader) # B, L, C
        #anomaly_scores = recon_errors.mean(dim=2).reshape(-1).detach().cpu() # B, L -> (T=B*L, )
        anomaly_scores = recon_errors.mean(axis=2).reshape(-1)
        return anomaly_scores


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
        Xs, Xhats = [], []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat = self.model(X)

            if self.args.save_outputs:
                Xs.append(X)
                Xhats.append(Xhat)

            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_error = recon_error.detach().cpu().numpy()
            recon_errors.append(recon_error)
            torch.cuda.empty_cache()

        # save recon outputs
        if self.args.save_outputs:
            self.logger.info(f"saving outputs: {self.args.output_path}")
            Xs = torch.cat(Xs, axis=0)
            Xhats = torch.cat(Xhats, axis=0)
            X_path = os.path.join(self.args.output_path, "Xs.pt")
            Xhat_path = os.path.join(self.args.output_path, "Xhats.pt")
            with open(X_path, 'wb') as f:
                torch.save(Xs, f)
            with open(Xhat_path, 'wb') as f:
                torch.save(Xhats, f)

        #recon_errors = torch.cat(recon_errors, axis=0)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors


    def offline_detrend(self, dataloader, init_thr, normalization="Detrend"):
        self.load_trained_model()  # reset

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        tau = init_thr

        preds = []
        anomaly_scores = []
        Xs, Xhats = [], []
        mus = []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape

            # Update of test-time statistics.
            if normalization == "Detrend":
                self.model.normalizer._update_statistics(X)

            # inference
            Xhat = self.model(X)
            E = (Xhat - X) ** 2
            A = E.mean(dim=2)
            ytilde = (A >= tau).float()
            pred = ytilde

            if self.args.save_outputs:
                Xs.append(X)
                Xhats.append(Xhat)
                mus.append(self.model.normalizer.mean.clone())

            # log model outputs
            preds.append(pred.clone().detach())
            anomaly_scores.append(A.clone().detach())

        # outputs
        preds = torch.cat(preds).reshape(-1).detach().cpu().numpy().astype(int)
        anomaly_scores = torch.cat(anomaly_scores).reshape(-1).detach().cpu().numpy()

        # save recon outputs
        if self.args.save_outputs:
            self.logger.info(f"saving outputs: {self.args.output_path}")
            Xs = torch.cat(Xs, axis=0)
            Xhats = torch.cat(Xhats, axis=0)
            mus = torch.cat(mus, axis=0)
            X_path = os.path.join(self.args.output_path, "Xs.pt")
            Xhat_path = os.path.join(self.args.output_path, "Xhats_off.pt")
            mu_path = os.path.join(self.args.output_path, "mus_off.pt")

            with open(X_path, 'wb') as f:
                torch.save(Xs, f)
            with open(Xhat_path, 'wb') as f:
                torch.save(Xhats, f)
            with open(mu_path, 'wb') as f:
                torch.save(mus, f)


        return anomaly_scores, preds


    def offline_detrend_all(self, cols, qStart=0.90, qEnd=1.00, qStep=0.01):
        result_df = pd.DataFrame(columns=cols)
        for q in np.arange(qStart, qEnd + 1e-07, qStep):
            th = np.quantile(self.train_anoscs, min(q, qEnd))
            anoscs, pred = self.offline_detrend(self.test_loader, init_thr=th, normalization=self.args.normalization)
            result = get_summary_stats(self.gt, pred)

            self.logger.info(result)
            result_df = pd.concat(
                [result_df, pd.DataFrame([result], index=[f"Q{q * 100:.3f}"], columns=result_df.columns)])
            result_df.at[f"Q{q * 100:.3f}", "tau"] = th

        # off_f1_best
        anoscs, pred = self.offline_detrend(self.test_loader, init_thr=self.th_off_f1_best, normalization=self.args.normalization)
        best_result = get_summary_stats(self.gt, pred)
        roc_auc = calculate_roc_auc(self.gt, anoscs, path=self.args.output_path,
                                    save_roc_curve=self.args.save_roc_curve)
        best_result["ROC_AUC"] = roc_auc
        pr_auc = calculate_pr_auc(self.gt, anoscs,
                                  path=self.args.output_path,
                                  save_pr_curve=self.args.save_pr_curve,
                                  )
        best_result["PR_AUC"] = pr_auc

        result_df = pd.concat([result_df, pd.DataFrame([best_result], index=[f"Q_off_f1_best"], columns=result_df.columns)])
        result_df.at[f"Q_off_f1_best", "tau"] = self.th_off_f1_best

        return result_df


    def online(self, dataloader, init_thr, normalization="None"):
        self.load_trained_model() # reset

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        tau = init_thr
        TT_optimizer = torch.optim.SGD([p for p in self.model.parameters()], lr=self.args.ttlr)

        Xs, Xhats = [], []
        preds = []
        As, thrs = [], []

        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape

            # Update of test-time statistics.
            if normalization == "Detrend":
                self.model.normalizer._update_statistics(X)

            # inference
            Xhat = self.model(X)
            E = (Xhat-X)**2
            A = E.mean(dim=2)
            ytilde = (A >= tau).float()
            pred = ytilde

            # log model outputs
            Xs.append(X)
            Xhats.append(Xhat.clone().detach())
            As.append(A.clone().detach())
            preds.append(pred.clone().detach())
            thrs.append(tau)

            # learn new-normals
            TT_optimizer.zero_grad()
            mask = (ytilde == 0)
            recon_loss = (A * mask).mean()
            recon_loss.backward()
            TT_optimizer.step()


        # outputs
        Xs = torch.cat(Xs, axis=0).detach().cpu()
        Xhats = torch.cat(Xhats, axis=0).detach().cpu()
        anoscs = torch.cat(As, axis=0).reshape(-1).detach().cpu().numpy()
        thrs = np.repeat(np.array(thrs), self.args.window_size * self.args.eval_batch_size)
        preds = torch.cat(preds).reshape(-1).detach().cpu().numpy().astype(int)

        # save recon outputs
        if self.args.save_outputs:
            self.logger.info(f"saving outputs: {self.args.output_path}")
            X_path = os.path.join(self.args.output_path, "Xs.pt")
            Xhat_path = os.path.join(self.args.output_path, "Xhats_on.pt")
            tta_anosc_path = os.path.join(self.args.output_path, "tta_anosc.pt")
            with open(X_path, 'wb') as f:
                torch.save(Xs, f)
            with open(Xhat_path, 'wb') as f:
                torch.save(Xhats, f)
            with open(tta_anosc_path, 'wb') as f:
                torch.save(anoscs, f)

        # save plots
        ## anomaly scores
        if self.args.plot_anomaly_scores:
            self.logger.info("Plotting anomaly scores...")
            plt.figure(figsize=(20, 6), dpi=500)
            plt.title(f"online_{self.args.dataset}_tau_{tau}")
            plt.plot(self.test_anoscs, color="blue", label="anomaly score w/o online learning")
            plt.plot(anoscs, color="green", label="anomaly score w/ online learning")
            plt.plot(thrs, color="black", label="threshold")

            plot_interval(plt, self.gt)
            plot_interval(plt, preds, facecolor="gray", alpha=1)
            plt.legend()
            plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_wTTA_tau_{tau}.png"))
            wandb.log({f"wTTA_tau_{tau}": wandb.Image(plt)})

        ## reconstruction status
        if self.args.plot_recon_status:
            self.logger.info("Plotting reconstruction status...")

            for c in range(self.args.num_channels):
                B, L, C = Xhats.shape
                title = f"{self.args.exp_id}_wTTA_recon_ch{c}"

                plt.figure(figsize=(20, 6), dpi=500)
                plt.title(f"{title}")
                plt.plot(Xs[:, c], color="black", label="gt")
                plt.plot(Xhats[:, :, c].reshape(-1), linewidth=0.5, color="green", label="reconed w/ online learning")
                plt.legend()
                plot_interval(plt, self.gt)
                plt.savefig(os.path.join(self.args.plot_path, f"{title}.png"))
                wandb.log({f"{title}": wandb.Image(plt)})

        return anoscs, preds


    def online_all(self, cols, qStart=0.90, qEnd=1.00, qStep=0.01):
        result_df = pd.DataFrame(columns=cols)
        for q in np.arange(qStart, qEnd + qStep, qStep):
            q = min(q, qEnd)
            th = np.quantile(self.train_anoscs, q)
            anoscs, pred = self.online(self.test_loader, init_thr=th, normalization=self.args.normalization)
            result = get_summary_stats(self.gt, pred)
            roc_auc = calculate_roc_auc(self.gt, anoscs, path=self.args.output_path, save_roc_curve=self.args.save_roc_curve)
            result["ROC_AUC"] = roc_auc
            pr_auc = calculate_pr_auc(self.gt, anoscs,
                                      path=self.args.output_path,
                                      save_pr_curve=self.args.save_pr_curve,
                                      )
            result["PR_AUC"] = pr_auc

            self.logger.info(result)
            result_df = pd.concat(
                [result_df, pd.DataFrame([result], index=[f"Q{q * 100:.3f}"], columns=result_df.columns)])
            result_df.at[f"Q{q * 100:.3f}", "tau"] = th

        # off_f1_best
        anoscs, pred = self.online(self.test_loader, init_thr=self.th_off_f1_best,
                                    normalization=self.args.normalization)
        best_result = get_summary_stats(self.gt, pred)

        roc_auc = calculate_roc_auc(self.gt, anoscs, path=self.args.output_path, save_roc_curve=self.args.save_roc_curve)
        best_result["ROC_AUC"] = roc_auc
        pr_auc = calculate_pr_auc(self.gt, anoscs,
                                  path=self.args.output_path,
                                  save_pr_curve=self.args.save_pr_curve,
                                  )
        best_result["PR_AUC"] = pr_auc

        result_df = pd.concat(
            [result_df, pd.DataFrame([best_result], index=[f"Q_off_f1_best"], columns=result_df.columns)])
        result_df.at[f"Q_off_f1_best", "tau"] = self.th_off_f1_best

        return result_df


    def online_label(self, dataloader, init_thr, normalization="None"):
        self.load_trained_model() # reset

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        tau = init_thr
        TT_optimizer = torch.optim.SGD([p for p in self.model.parameters()], lr=self.args.ttlr)

        Xs, Xhats = [], []
        preds = []
        As, thrs = [], []

        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            y = batch_data[1].to(self.args.device)
            B, L, C = X.shape

            # Update of test-time statistics.
            if normalization == "Detrend":
                self.model.normalizer._update_statistics(X)

            # inference
            Xhat = self.model(X)
            E = (Xhat-X)**2
            A = E.mean(dim=2)
            ytilde = (A >= tau).float()
            pred = ytilde

            # log model outputs
            Xs.append(X)
            Xhats.append(Xhat.clone().detach())
            As.append(A.clone().detach())
            preds.append(pred.clone().detach())
            thrs.append(tau)

            # learn new-normals
            TT_optimizer.zero_grad()
            mask = (y == 0)
            recon_loss = (A * mask).mean()
            recon_loss.backward()
            TT_optimizer.step()


        # outputs
        Xs = torch.cat(Xs, axis=0).detach().cpu().numpy()
        Xhats = torch.cat(Xhats, axis=0).detach().cpu().numpy()
        anoscs = torch.cat(As, axis=0).reshape(-1).detach().cpu().numpy()
        thrs = np.repeat(np.array(thrs), self.args.window_size * self.args.eval_batch_size)
        preds = torch.cat(preds).reshape(-1).detach().cpu().numpy().astype(int)

        # save plots
        ## anomaly scores
        if self.args.plot_anomaly_scores:
            self.logger.info("Plotting anomaly scores...")
            plt.figure(figsize=(20, 6), dpi=500)
            plt.title(f"online_{self.args.dataset}_tau_{tau}")
            plt.plot(self.test_anoscs, color="blue", label="anomaly score w/o online learning")
            plt.plot(anoscs, color="green", label="anomaly score w/ online learning")
            plt.plot(thrs, color="black", label="threshold")

            plot_interval(plt, self.gt)
            plot_interval(plt, preds, facecolor="gray", alpha=1)
            plt.legend()
            plt.savefig(os.path.join(self.args.plot_path, f"{self.args.exp_id}_wTTA_label_tau_{tau}.png"))
            wandb.log({f"wTTA_tau_{tau}": wandb.Image(plt)})

        ## reconstruction status
        if self.args.plot_recon_status:
            self.logger.info("Plotting reconstruction status...")

            for c in range(self.args.num_channels):
                B, L, C = Xhats.shape
                title = f"{self.args.exp_id}_wTTA_label_recon_ch{c}"

                plt.figure(figsize=(20, 6), dpi=500)
                plt.title(f"{title}")
                plt.plot(Xs[:, c], color="black", label="gt")
                plt.plot(Xhats[:, :, c].reshape(-1), linewidth=0.5, color="green", label="reconed w/ online learning")
                plt.legend()
                plot_interval(plt, self.gt)
                plt.savefig(os.path.join(self.args.plot_path, f"{title}.png"))
                wandb.log({f"{title}": wandb.Image(plt)})

        return anoscs, preds


    def online_label_all(self, cols, qStart=0.90, qEnd=1.00, qStep=0.01):
        result_df = pd.DataFrame(columns=cols)
        for q in np.arange(qStart, qEnd + 1e-07, qStep):
            th = np.quantile(self.train_anoscs, min(q, qEnd))
            anoscs, pred = self.online_label(self.test_loader, init_thr=th, normalization=self.args.normalization)
            result = get_summary_stats(self.gt, pred)
            roc_auc = calculate_roc_auc(self.gt, anoscs, path=self.args.output_path, save_roc_curve=self.args.save_roc_curve)
            result["ROC_AUC"] = roc_auc
            pr_auc = calculate_pr_auc(self.gt, anoscs,
                                      path=self.args.output_path,
                                      save_pr_curve=self.args.save_pr_curve,
                                      )
            result["PR_AUC"] = pr_auc

            self.logger.info(result)

            result_df = pd.concat(
                [result_df, pd.DataFrame([result], index=[f"Q{q * 100:.3f}"], columns=result_df.columns)])
            result_df.at[f"Q{q * 100:.3f}", "tau"] = th


        # off_f1_best
        anoscs, pred = self.online_label(self.test_loader, init_thr=self.th_off_f1_best,
                           normalization=self.args.normalization)
        best_result = get_summary_stats(self.gt, pred)

        roc_auc = calculate_roc_auc(self.gt, anoscs, path=self.args.output_path, save_roc_curve=self.args.save_roc_curve)
        best_result["ROC_AUC"] = roc_auc
        pr_auc = calculate_pr_auc(self.gt, anoscs,
                                  path=self.args.output_path,
                                  save_pr_curve=self.args.save_pr_curve,
                                  )
        best_result["PR_AUC"] = pr_auc

        range_metrics = get_range_vus_roc(score=anoscs, labels=self.gt, slidingWindow=self.args.range_window_size)
        result.update(range_metrics)

        result_df = pd.concat(
            [result_df, pd.DataFrame([best_result], index=[f"Q_off_f1_best"], columns=result_df.columns)])
        result_df.at[f"Q_off_f1_best", "tau"] = self.th_off_f1_best

        return result_df