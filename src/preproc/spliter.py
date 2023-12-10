from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from typing import TYPE_CHECKING

from pandas import read_parquet

if TYPE_CHECKING:
    from pathlib import Path


class Spliter:
    def __init__(self, src_file_path: Path, dst_dir_path: Path) -> None:
        self.src_file_path = src_file_path
        self.dst_dir_path = dst_dir_path
        pass

    def run(self) -> None:
        print(f"Spliting {self.src_file_path.name}...")
        train_df = read_parquet(self.src_file_path)
        test_df = train_df.sample(frac=0.2, random_state=25)
        train_df.drop(test_df.index, inplace=True)
        val_df = train_df.sample(frac=0.2, random_state=25)
        train_df.drop(val_df.index, inplace=True)
        train_df.to_parquet(
            self.dst_dir_path / "train" / self.src_file_path.name,
            compression="gzip",
            index=False,
        )
        val_df.to_parquet(
            self.dst_dir_path / "val" / self.src_file_path.name,
            compression="gzip",
            index=False,
        )
        test_df.to_parquet(
            self.dst_dir_path / "test" / self.src_file_path.name,
            compression="gzip",
            index=False,
        )
        del train_df, val_df, test_df
        return None

    pass
