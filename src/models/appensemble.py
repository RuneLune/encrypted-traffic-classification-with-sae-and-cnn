from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from typing import List, Type, TYPE_CHECKING
from pathlib import Path

from torch.nn import Linear, Sequential

from .ensemblebase import EnsembleBase
from . import AppSAE, AppCNN

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule


class AppEnsemble(EnsembleBase):
    def __init__(
        self,
        sae_hparam,
        cnn_hparam,
        sae_param,
        cnn_param,
        batch_size: int = 2048,
        n_jobs: int = 52,
        device_id: int = 5,
    ) -> None:
        r"""Ensemble model for the application identification task.

        Args:
            batch_size: The batch size to use for training, validation, and testing.
            n_jobs: The number of jobs to use for loading the data.
        """
        super(AppEnsemble, self).__init__(batch_size, n_jobs)

        self._data_path = Path("./data/splited/app")
        self._device_id = device_id

        self._models: List[Type[LightningModule]] = [
            AppSAE(**sae_hparam).cuda(self._device_id),
            AppCNN(**cnn_hparam).cuda(self._device_id),
        ]
        self._models[0].load_state_dict(sae_param)
        self._models[1].load_state_dict(cnn_param)

        self._layers = Sequential(Linear(17 * 2, 17))

        return None

    def _pre_forward(self, x):
        output = super(AppEnsemble, self)._pre_forward(x).cuda(self._device_id)
        return output.reshape(-1, 17 * len(self._models)).cuda(self._device_id)

    pass
