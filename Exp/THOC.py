import os
import wandb
from tqdm import tqdm

# Trainer
from Exp.Trainer import Trainer
from Exp.Tester import Tester

# models
from models.THOC import THOC

# others
import torch


class THOC_Trainer(Trainer):
    def __init__(self, args, logger, train_loader):
        super(THOC_Trainer, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader
        )

        self.model = THOC(
            C=self.args.num_channels,
            W=self.args.window_size,
            n_hidden=self.args.model.hidden_dim,
            device=self.args.device
        ).to(self.args.device)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr,
                                           weight_decay=self.args.model.L2_reg)

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
            if (i + 1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary


    def _process_batch(self, batch_data) -> dict:
        out = dict()
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        anomaly_score, loss_dict = self.model(X)
        for k in loss_dict:
            out.update({k: loss_dict[k].item()})

        self.optimizer.zero_grad()
        loss = loss_dict["L_THOC"] + self.args.model.LAMBDA_orth * loss_dict["L_orth"] + self.args.model.LAMBDA_TSS * loss_dict["L_TSS"]
        loss.backward()
        self.optimizer.step()

        out.update({
            "summary": loss.item(),
        })
        return out



class THOC_Tester(Tester):
    def __init__(self, args, logger, train_loader, test_loader, load=False):
        super(THOC_Tester, self).__init__(
            args=args,
            logger=logger,
            train_loader=train_loader,
            test_loader=test_loader,
        )

        self.model = THOC(
            C=self.args.num_channels,
            W=self.args.window_size,
            n_hidden=self.args.model.hidden_dim,
            device=self.args.device
        ).to(self.args.device)

        if load:
            self.load_trained_model()
            self.prepare_stats()


    @torch.no_grad()
    def calculate_anomaly_scores(self, dataloader):
        eval_iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating anomaly scores",
            leave=True
        )

        anomaly_scores = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            anomaly_score, loss_dict = self.model(X)
            anomaly_scores.append(anomaly_score)

        anomaly_scores = torch.cat(anomaly_scores, dim=0)
        init_pred = anomaly_scores[0].repeat(self.args.window_size - 1)
        anomaly_scores = torch.cat((init_pred, anomaly_scores)).cpu().numpy()
        return anomaly_scores


    def online(self, dataloader, init_thr, normalization="None"):
        self.load_trained_model()  # reset

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        tau = init_thr
        TT_optimizer = torch.optim.SGD([p for p in self.model.parameters()], lr=self.args.ttlr)

        anoscs = []
        preds = []

        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape

            # Update of test-time statistics.
            if normalization == "Detrend":
                self.model.normalizer._update_statistics(X)

            # inference
            anomaly_score, loss_dict = self.model(X)
            ytilde = (anomaly_score >= tau).float()
            pred = ytilde
            mask = (ytilde == 0)

            # log model outputs
            anoscs.append(anomaly_score)
            preds.append(pred.clone().detach())

            # learn new-normals
            TT_optimizer.zero_grad()
            mask = (ytilde == 0)
            loss = mask*(loss_dict["L_THOC"] + self.args.model.LAMBDA_orth * loss_dict[
                "L_orth"] + self.args.model.LAMBDA_TSS * loss_dict["L_TSS"])
            loss.backward()
            TT_optimizer.step()

        init_pred = anoscs[0].repeat(self.args.window_size - 1)
        anoscs = torch.cat((init_pred, anoscs)).cpu().numpy()

        return anoscs, preds