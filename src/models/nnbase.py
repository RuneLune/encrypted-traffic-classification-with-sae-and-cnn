from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from pathlib import Path
from typing import TYPE_CHECKING, Any, Type, Optional

from torch import argmax
from torch.nn import (
    CrossEntropyLoss,
)
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)

from dataset import CustomDataset

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.modules import Module

    pass


class NNBase(LightningModule):
    def __init__(self, batch_size: int = 2048, n_jobs: int = 52) -> None:
        r"""Base class for all neural networks.

        Args:
            batch_size: The batch size to use for training, validation, and testing.
            n_jobs: The number of jobs to use for loading the data.
        """
        super(NNBase, self).__init__()

        self.save_hyperparameters()
        self._batch_size: int = batch_size
        self._n_jobs: int = n_jobs
        self._data_path = Path("./data/splited")
        self._layers: Optional[Type[Module]] = None

        self._loss = CrossEntropyLoss()

        return None

    def _pre_forward(self, x: Tensor) -> Tensor:
        return x

    def _post_forward(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor):
        r"""Forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        x = self._pre_forward(x)
        x = self._layers(x)
        return self._post_forward(x)

    def configure_optimizers(self) -> Any:
        r"""Configure the optimizer to use for training.

        Returns:
            The optimizer to use for training.
        """
        optimizer = Adam(self.parameters())
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        r"""Returns the training dataloader.

        Returns:
            The training dataloader.
        """
        return DataLoader(
            CustomDataset(self._data_path / "train"),
            batch_size=self._batch_size | self.hparams.batch_size,
            num_workers=self._n_jobs,
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        r"""Training step.

        Args:
            batch: The batch of data to train on.
            batch_idx: The index of the batch.

        Returns:
            The loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def val_dataloader(self) -> EVAL_DATALOADERS:
        r"""Returns the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            CustomDataset(self._data_path / "val"),
            batch_size=self._batch_size | self.hparams.batch_size,
            num_workers=self._n_jobs,
        )

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        r"""Validation step.

        Args:
            batch: The batch of data to validate on.
            batch_idx: The index of the batch.

        Returns:
            The loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        r"""Returns the testing dataloader.

        Returns:
            The testing dataloader.
        """
        return DataLoader(
            CustomDataset(self._data_path / "test"),
            batch_size=self._batch_size | self.hparams.batch_size,
            num_workers=self._n_jobs,
        )

    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        r"""Testing step.

        Args:
            batch: The batch of data to test on.
            batch_idx: The index of the batch.

        Returns:
            The ground truth and the predictions.
        """
        x, y = batch
        y_hat = self(x)
        return y, argmax(y_hat, dim=1)

    pass


if __name__ == "__main__":
    pass
