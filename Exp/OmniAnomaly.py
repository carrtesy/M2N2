import os
import wandb
from tqdm import tqdm

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.OmniAnomaly import OmniAnomaly

# others
import torch
import torch.nn.functional as F


class OmniAnomaly_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(OmniAnomaly_Trainer, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader
        )

        self.model = OmniAnomaly(
            in_dim=self.args.num_channels,
            hidden_dim=self.args.model.hidden_dim,
            z_dim=self.args.model.z_dim,
            dense_dim=self.args.model.dense_dim,
            out_dim=self.args.num_channels,
            K=self.args.model.K,
        ).to(self.args.device)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-04)


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


    @staticmethod
    def gaussian_prior_KLD(mu, logvar):
        return -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())


    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat, mu, logvar = self.model(X)
        self.optimizer.zero_grad()
        recon_loss = F.mse_loss(Xhat, X)
        KLD_loss = self.gaussian_prior_KLD(mu, logvar)
        loss = recon_loss + (1/L)*KLD_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        out = {
            "recon_loss": recon_loss.item(),
            "KLD_loss": KLD_loss.item(),
            "total_loss": loss.item(),
            "summary": loss.item(),
        }
        return out



class OmniAnomaly_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader):
        super(OmniAnomaly_Tester, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader,
            test_loader=test_loader,
        )

        self.model = OmniAnomaly(
            in_dim=self.args.num_channels,
            hidden_dim=self.args.model.hidden_dim,
            z_dim=self.args.model.z_dim,
            dense_dim=self.args.model.dense_dim,
            out_dim=self.args.num_channels,
            K=self.args.model.K,
        ).to(self.args.device)

        self.load_trained_model()
        self.prepare_stats()


    @torch.no_grad()
    def calculate_anomaly_scores(self, dataloader) -> torch.Tensor:
        recon_errors = self.calculate_recon_errors(dataloader) # B, L, C
        anomaly_scores = recon_errors.mean(dim=2).reshape(-1).detach().cpu() # B, L -> (T=B*L, )
        return anomaly_scores


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader) -> torch.Tensor:
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat, _, _ = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_errors.append(recon_error)
        recon_errors = torch.cat(recon_errors, dim=0)
        return recon_errors