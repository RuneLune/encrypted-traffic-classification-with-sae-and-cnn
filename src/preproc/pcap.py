from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

import gc
from pathlib import Path
import json

from pandas import DataFrame
from numpy import uint8
from scapy.utils import rdpcap

from preproc.packet import PacketPreprocessor as PckPreproc
from utils.id import get_app_id, get_traffic_id


class PcapPreprocessor:
    def __init__(
        self,
        src_file_path: Path,
        dst_dir_path: Path = Path(".") / "data" / "preprocessed",
        packets_per_file: int = 65536,
        packet_len: int = 1500,
    ) -> None:
        self.__file_name = src_file_path.name.split(".")[0]
        self.__dst_dir_path = dst_dir_path
        if (self.__dst_dir_path / f"{self.__file_name}.json").exists():
            print(f"Data {self.__file_name} already processed, skipping...")
            return None
        self.__app_id = get_app_id(self.__file_name)
        self.__traffic_id = get_traffic_id(self.__file_name)
        print(f"Reading {self.__file_name}...")
        self.__packet_list = rdpcap(str(src_file_path))
        self.__packets_per_file = packets_per_file
        self.__packet_preprocessor = PckPreproc(packet_len)
        self.__process()
        return None

    def __process(self) -> None:
        print(f"Processing {self.__file_name}...")
        file_counter = 0
        data_list = []
        for packet in self.__packet_list:
            res = self.__packet_preprocessor.process(packet)
            if res is not None:
                data_list.append(
                    [
                        res,
                        self.__app_id,
                        self.__traffic_id,
                    ]
                )
                pass
            if len(data_list) >= self.__packets_per_file:
                self.__save(data_list, file_counter)
                file_counter += 1
                data_list.clear()
                pass
            continue
        last_rows = len(data_list)
        if len(data_list) > 0:
            self.__save(data_list, file_counter)
            data_list.clear()
            pass
        del self.__packet_list
        gc.collect()

        total_rows = self.__packets_per_file * file_counter + last_rows
        with open(self.__dst_dir_path / f"{self.__file_name}.json", "w") as f:
            json.dump(
                {
                    "files": file_counter + 1,
                    "file_rows": self.__packets_per_file
                    if file_counter > 0
                    else last_rows,
                    "last_file_rows": last_rows,
                    "total_rows": total_rows,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
            pass
        print(f" - Saved {self.__file_name}.json")
        print(f"Processing {self.__file_name} complete. (total {total_rows} rows)")
        return None

    def __save(self, data_list: list, file_counter: int) -> None:
        dst_file_name = f"{self.__file_name}_{file_counter:02d}.dataframe.gzip.parquet"
        dst_file_path = self.__dst_dir_path / dst_file_name
        df = DataFrame(data_list, columns=["packet", "app_id", "traffic_id"])
        df = df.astype({"packet": object, "app_id": uint8, "traffic_id": uint8})
        df.to_parquet(str(dst_file_path), compression="gzip", index=False)
        print(f" - Saved {dst_file_name} ({df.shape[0]} rows)")
        return None

    pass


def main() -> None:
    return None


if __name__ == "__main__":
    main()
    pass
