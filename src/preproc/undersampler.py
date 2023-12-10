import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from abc import ABC, abstractmethod

from pandas import read_parquet, concat
from overrides import overrides

from utils.id import (
    AppName,
    TrafficName,
    get_app_regex,
    get_traffic_regex,
)


class Undersampler(ABC):
    @abstractmethod
    def run(self) -> None:
        pass

    pass


class AppUndersampler(Undersampler):
    def __init__(self, app_id, min_app, src_dir_path, dst_dir_path):
        self.app_id = app_id
        self.min_app = min_app
        self.src_dir_path = src_dir_path
        self.dst_dir_path = dst_dir_path
        return None

    @overrides
    def run(self):
        print(f"Undersampling {AppName[self.app_id]}...")
        df_list = []
        parquet_list = [
            file_path
            for file_path in sorted(
                list(self.src_dir_path.glob("*.dataframe.gzip.parquet"))
            )
            if get_app_regex(self.app_id).match(file_path.name)
        ]
        for parquet_path in parquet_list:
            df_list.append(read_parquet(parquet_path).drop(["traffic_id"], axis=1))
            continue
        df = concat(df_list, ignore_index=True).sample(
            n=self.min_app, random_state=25, replace=False
        )
        dst_file_path = (
            self.dst_dir_path / f"{AppName[self.app_id]}.dataframe.gzip.parquet"
        )
        df.to_parquet(dst_file_path, compression="gzip", index=False)
        print(f"Undersampled {AppName[self.app_id]}: {len(df)}")
        del df

        return None


class TrafficUndersampler(Undersampler):
    def __init__(self, traffic_id, min_traffic, src_dir_path, dst_dir_path):
        self.traffic_id = traffic_id
        self.min_traffic = min_traffic
        self.src_dir_path = src_dir_path
        self.dst_dir_path = dst_dir_path
        return None

    @overrides
    def run(self):
        print(f"Undersampling {TrafficName[self.traffic_id]}...")
        df_list = []
        parquet_list = [
            file_path
            for file_path in sorted(
                list(self.src_dir_path.glob("*.dataframe.gzip.parquet"))
            )
            if get_traffic_regex(self.traffic_id).match(file_path.name)
        ]
        for parquet_path in parquet_list:
            df_list.append(read_parquet(parquet_path).drop(["app_id"], axis=1))
            continue
        df = concat(df_list, ignore_index=True).sample(
            n=self.min_traffic, random_state=25, replace=False
        )
        traffic_name = TrafficName[self.traffic_id].replace(":", "").replace(" ", "_")
        dst_file_path = self.dst_dir_path / f"{traffic_name}.dataframe.gzip.parquet"
        df.to_parquet(dst_file_path, compression="gzip", index=False)
        print(f"Undersampled {TrafficName[self.traffic_id]}: {len(df)}")
        del df
