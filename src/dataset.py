from __future__ import annotations

from pathlib import Path

from numpy import array
from pandas import read_parquet, concat

from torch import tensor
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir: Path | str) -> None:
        r"""Custom Dataset for torch.utils.data.Dataloader.

        Example::

            CustomDataset(Path("/path/to/directory/containing/parquet/files"))

        Args:
            data_dir: Path to the directory containing the parquet files. Can be a ``Path`` or ``str``.
        """
        dataframe_list = []

        self.data_dir = Path(str(data_dir))

        # Add all DataFrames loaded from parquet files in the directory to the list
        for dataframe_path in sorted(data_dir.glob("*.dataframe.gzip.parquet")):
            dataframe_list.append(read_parquet(dataframe_path))
            continue

        # Concatenate all DataFrames in the list and shuffle the rows
        self.df = (
            concat(dataframe_list, ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )
        return None

    def __len__(self):
        r"""Returns the number of rows in the dataset.

        Returns:
            The number of rows in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        r"""Returns the row at the given index.

        Args:
            idx: The index of the row to return. Can be an ``int`` or ``slice``.

        Returns:
            The row at the given index.
        """
        return tensor(array(self.df.iloc[idx, 0])), int(self.df.iloc[idx, 1])

    pass


def main() -> None:
    dataset = CustomDataset(Path("./data/splited/app/train"))
    print(dataset.df["app_id"].value_counts())
    dataset = CustomDataset(Path("./data/splited/app/val"))
    print(dataset.df["app_id"].value_counts())
    dataset = CustomDataset(Path("./data/splited/app/test"))
    print(dataset.df["app_id"].value_counts())
    dataset = CustomDataset(Path("./data/splited/traffic/train"))
    print(dataset.df["traffic_id"].value_counts())
    dataset = CustomDataset(Path("./data/splited/traffic/val"))
    print(dataset.df["traffic_id"].value_counts())
    dataset = CustomDataset(Path("./data/splited/traffic/test"))
    print(dataset.df["traffic_id"].value_counts())
    return None


if __name__ == "__main__":
    main()
    pass
