import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from abc import ABC


class BaseModule(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.per_batch_metrics = nn.ModuleDict({})
        self.per_epoch_metrics = nn.ModuleDict({})

    def _log_metrics(self, *, prefix, y, y_pred):

        """
        Log all metrics
        Parameters
        ----------
        prefix: str
            One of train, val, test
        y: tensor
        y_pred: tensor
        Returns
        -------
        """
        for key, metric in self.per_batch_metrics.items():
            if key[: len(prefix)] == prefix:
                self.log(key, metric(y_pred, y), on_step=True, on_epoch=True)
        for key, metric in self.per_epoch_metrics.items():
            if key[: len(prefix)] == prefix:
                self.log(key, metric(y_pred, y), on_step=False, on_epoch=True)


class ClassificationModule(BaseModule):
    def __init__(self, model, *, lr=1e-3):
        """
        Multiclass classification trainer

        Parameters
        ----------
        model : nn.Module
        lr : float
        """
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.criterion = nn.CrossEntropyLoss()
        self.per_epoch_metrics = nn.ModuleDict(
            dict(
                train_accuracy=Accuracy(),
                val_accuracy=Accuracy(),
                test_accuracy=Accuracy(),
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def _step(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        y_pred_class = torch.argmax(y_pred, dim=-1)
        return y, y_pred_class, loss

    def training_step(self, batch, batch_index):
        x, target_class = batch
        target_class, y_pred, loss = self._step(x, target_class)
        self.log("train_loss", float(loss), on_step=True, on_epoch=True)
        self._log_metrics(prefix="train", y=target_class, y_pred=y_pred)
        return loss

    def validation_step(self, val_batch, batch_index):
        x, target_class = val_batch
        target_class, y_pred, loss = self._step(x, target_class)
        self.log("val_loss", float(loss), on_step=True, on_epoch=True)
        self._log_metrics(prefix="val", y=target_class, y_pred=y_pred)
        return loss

    def test_step(self, test_batch, batch_index):
        x, target_class = test_batch
        target_class, y_pred, loss = self._step(x, target_class)
        self.log("test_loss", float(loss), on_step=False, on_epoch=True)
        self._log_metrics(prefix="test", y=target_class, y_pred=y_pred)
        return loss


class VAEModule(BaseModule):
    from xca.ml.torch.vae import VAE

    def __init__(self, vae: VAE, *, lr=1e-3, kl_weight=1.0):
        super().__init__()
        self.model = vae
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.learning_rate = lr
        self.reconstruction_loss = nn.MSELoss()
        self.kl_weight = kl_weight

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def kl_loss(mu, log_var):
        """Computes the KL-divergence loss with a batchwise mean as a reduction"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean(
            dim=0
        )

    def _step(self, batch):
        # Takes the first tensor off a list/tuple batch. I.e. removes target and other auxilary info.
        if isinstance(batch, torch.Tensor):
            x = batch
        else:
            x, *_ = batch
        recon, mu, log_var = self.model(x)
        kl_loss = self.kl_loss(mu, log_var)
        recon_loss = self.reconstruction_loss(x, recon)
        loss = recon_loss + self.kl_weight * kl_loss
        return recon, loss, recon_loss, kl_loss

    def training_step(self, batch, batch_index):
        recon, total_loss, recon_loss, kl_loss = self._step(batch)
        # Not using torch metrics and only logging losses
        for loss, key in zip(
            (total_loss, recon_loss, kl_loss),
            ("loss", "reconstruction_loss", "kl_divergence"),
        ):
            self.log(f"train_{key}", float(loss), on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_index):
        recon, total_loss, recon_loss, kl_loss = self._step(val_batch)
        for loss, key in zip(
            (total_loss, recon_loss, kl_loss),
            ("loss", "reconstruction_loss", "kl_divergence"),
        ):
            self.log(f"val_{key}", float(loss), on_step=True, on_epoch=True)
        return total_loss

    def test_step(self, test_batch, batch_index):
        recon, total_loss, recon_loss, kl_loss = self._step(test_batch)
        for loss, key in zip(
            (total_loss, recon_loss, kl_loss),
            ("loss", "reconstruction_loss", "kl_divergence"),
        ):
            self.log(f"test_{key}", float(loss), on_step=True, on_epoch=True)
        return total_loss

    def predict_step(self, batch, batch_idx):
        recon, total_loss, recon_loss, kl_loss = self._step(batch)
        return dict(
            predicted_reconstruction=recon,
            total_loss=total_loss,
            reconstruction_loss=recon_loss,
            kl_divergence=kl_loss,
        )


class JointVAEClassifierModule(BaseModule):
    """This should have 2 models that are independent until the loss sum.
    The optimizers will need specific parameter access.
    Useful for case where 2 models are being trained on the same data access, and data access is costly.
    """

    def __init__(self):
        raise NotImplementedError


class RegressionTrainer(BaseModule):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


def dynamic_training(pl_module, max_epochs, gpus=None, **kwargs):
    """
    Dynamic training utility function. See documentation for
    xca.data_synthesis.dynamic for explanation of kwargs

    Parameters
    ----------
    pl_module : Module
        Pytorch Lightning module for training
    max_epochs : int
        Maximum number of epochs to continue training. Must be greater than 0.
    gpus : Optional[List]
    kwargs :
        Keyword arguments to be passed into DynamicDataModule

    Returns
    -------

    """
    from pytorch_lightning.loggers import WandbLogger
    from xca.data_synthesis.dynamic import DynamicDataModule

    if gpus is None:
        gpus = [0] if torch.cuda.is_available() else None
    wandb_logger = WandbLogger(project="XCA")
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs, logger=wandb_logger)
    pl_data_module = DynamicDataModule(**kwargs)
    trainer.fit(pl_module, pl_data_module)
    return trainer.logged_metrics
