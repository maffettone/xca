import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from abc import ABC


class Trainer(pl.LightningModule, ABC):
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


class ClassificationTrainer(Trainer):
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


class RegressionTrainer(Trainer):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
