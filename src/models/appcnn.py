from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from pathlib import Path
from typing import TYPE_CHECKING

from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    Dropout,
    Conv1d,
    BatchNorm1d,
    MaxPool1d,
    Flatten,
)

from nnbase import NNBase

if TYPE_CHECKING:
    from torch import Tensor

    pass


class AppCNN(NNBase):
    def __init__(self, batch_size: int = 2048, n_jobs: int = 52) -> None:
        r"""Model used to train the CNN for the application identification.

        Args:
            batch_size: The batch size to use for training, validation, and testing.
            n_jobs: The number of jobs to use for loading the data.
        """
        super(AppCNN, self).__init__(batch_size, n_jobs)

        self._data_path = Path("./data/splited/app")

        self._layers = Sequential(
            Sequential(Conv1d(1, 200, 4, 3), BatchNorm1d(200), ReLU(), Dropout(0.05)),
            Sequential(Conv1d(200, 200, 5, 1), BatchNorm1d(200), ReLU(), Dropout(0.05)),
            Sequential(MaxPool1d(2, 2), Dropout(0.05)),
            Flatten(),
            Sequential(Linear(200 * 247, 200), ReLU(), Dropout(0.05)),
            Sequential(Linear(200, 100), ReLU(), Dropout(0.05)),
            Sequential(Linear(100, 50), ReLU(), Dropout(0.05)),
            Linear(50, 17),
        )

        return None

    def forward(self, x: Tensor):
        r"""Forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return super(AppCNN, self).forward(x.reshape(x.shape[0], 1, x.shape[1]))

    pass


if __name__ == "__main__":
    from torch import rand

    input = rand(4, 1500)
    net = AppCNN(1, 1)
    output = net(input)
    print(output)

    pass
