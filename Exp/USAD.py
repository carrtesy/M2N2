import os
import wandb
from tqdm import tqdm

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.USAD import USAD

# others
import torch
import torch.nn.functional as F


class USAD_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(USAD_Trainer, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader
        )

        self.model = USAD(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.model.latent_dim,
        ).to(self.args.device)

        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0


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
            train_log = self._process_batch(batch_data, self.epoch+1)
            if (i+1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        self.epoch += 1
        return train_summary


    def _process_batch(self, batch_data, epoch) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        # AE1
        z = self.model.encoder(X.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2p = self.model.decoder2(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)

        self.optimizer1.zero_grad()
        loss_AE1 = (1 / epoch) * F.mse_loss(X, Wt1p) + (1 - (1 / epoch)) * F.mse_loss(X, Wt2dp)
        loss_AE1.backward()
        self.optimizer1.step()

        # AE2
        z = self.model.encoder(X.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2p = self.model.decoder2(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)

        self.optimizer2.zero_grad()
        loss_AE2 = (1 / epoch) * F.mse_loss(X, Wt2p) - (1 - (1 / epoch)) * F.mse_loss(X, Wt2dp)
        loss_AE2.backward()
        self.optimizer2.step()

        out = {
            "loss_AE1": loss_AE1.item(),
            "loss_AE2": loss_AE2.item(),
            "summary": loss_AE1.item() + loss_AE2.item()
        }
        return out


class USAD_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader):
        super(USAD_Tester, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader,
            test_loader=test_loader,
        )

        self.model = USAD(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.model.latent_dim,
        ).to(self.args.device)

        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0

        self.load_trained_model()
        self.prepare_stats()


    @torch.no_grad()
    def calculate_anomaly_scores(self, dataloader):
        recon_errors = self.calculate_recon_errors(dataloader)  # B, L, C
        anomaly_scores = recon_errors.mean(dim=2).reshape(-1).detach().cpu()  # B, L -> (T=B*L, )
        return anomaly_scores


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader):
        '''
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
            recon_error = self.recon_error_criterion(X, self.args.model.alpha, self.args.model.beta)
            recon_errors.append(recon_error)
        recon_errors = torch.cat(recon_errors, dim=0)
        return recon_errors


    def recon_error_criterion(self, Wt, alpha=0.5, beta=0.5):
        '''
        :param Wt: model input
        :param alpha: low detection sensitivity
        :param beta: high detection sensitivity
        :return: recon error (B, L, C)
        '''
        B, L, C = Wt.shape
        z = self.model.encoder(Wt.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)
        return alpha * F.mse_loss(Wt, Wt1p, reduction='none') + beta * F.mse_loss(Wt, Wt2dp, reduction='none')