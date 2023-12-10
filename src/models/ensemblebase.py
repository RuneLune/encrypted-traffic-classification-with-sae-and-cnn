import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from overrides import overrides

from torch import cat

from .nnbase import NNBase


class EnsembleBase(NNBase):
    @overrides
    def __init__(self, batch_size: int = 4096, n_jobs: int = 52) -> None:
        r"""Base class for the ensemble models.

        Args:
            batch_size: The batch size to use for training, validation, and testing.
            n_jobs: The number of jobs to use for loading the data.
        """
        super(EnsembleBase, self).__init__(batch_size, n_jobs)
        self._models = None

        return None

    @overrides
    def _pre_forward(self, x):
        output = []
        for model in self._models:
            output.append(model(x))
            pass
        return cat(output, dim=1)

    pass
