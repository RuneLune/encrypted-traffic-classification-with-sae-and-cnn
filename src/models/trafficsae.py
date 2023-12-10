from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from pathlib import Path

from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    Dropout,
)

from .nnbase import NNBase


class TrafficSAE(NNBase):
    def __init__(self, batch_size: int = 4096, n_jobs: int = 52) -> None:
        r"""Model used to train the autoencoder for the traffic characterization.

        Args:
            batch_size: The batch size to use for training, validation, and testing.
            n_jobs: The number of jobs to use for loading the data.
        """
        super(TrafficSAE, self).__init__(batch_size, n_jobs)

        self._data_path = Path("./data/splited/traffic")

        self._layers = Sequential(
            Sequential(Linear(1500, 400), ReLU(), Dropout(0.05)),
            Sequential(Linear(400, 300), ReLU(), Dropout(0.05)),
            Sequential(Linear(300, 200), ReLU(), Dropout(0.05)),
            Sequential(Linear(200, 100), ReLU(), Dropout(0.05)),
            Sequential(Linear(100, 50), ReLU(), Dropout(0.05)),
            Linear(50, 12),
        )

        return None

    pass


if __name__ == "__main__":
    from torch import rand

    input = rand(4, 1500)
    net = TrafficSAE()
    output = net(input)
    print(output)

    pass
