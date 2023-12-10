from pathlib import Path
from typing import Final, List

from joblib import Parallel, delayed
from numpy.random import seed

from utils.id import (
    AppName,
    TrafficName,
)
from preproc import Spliter

app_types: Final[int] = len(AppName)
traffic_types: Final[int] = len(TrafficName)


def main(
    src_dir=str(Path(".") / "data" / "undersampled"),
    dst_dir=str(Path(".") / "data" / "splited"),
    jobs=-1,
) -> None:
    seed(25)
    src_dir_path = Path(src_dir)
    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    (dst_dir_path / "app").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "traffic").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "app" / "train").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "app" / "val").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "app" / "test").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "traffic" / "train").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "traffic" / "val").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "traffic" / "test").mkdir(parents=True, exist_ok=True)

    spliter_list: List[Spliter] = []

    for app_parquet_path in sorted(
        list((src_dir_path / "app").glob("*.dataframe.gzip.parquet"))
    ):
        spliter_list.append(
            Spliter(
                app_parquet_path,
                dst_dir_path / "app",
            )
        )
        continue

    for traffic_parquet_path in sorted(
        list((src_dir_path / "traffic").glob("*.dataframe.gzip.parquet"))
    ):
        spliter_list.append(
            Spliter(
                traffic_parquet_path,
                dst_dir_path / "traffic",
            )
        )
        continue

    if jobs == 1:
        for spliter in spliter_list:
            spliter.run()
            continue
        pass
    else:
        Parallel(n_jobs=jobs, verbose=10)(
            delayed(spliter.run)() for spliter in spliter_list
        )
        pass

    return None


if __name__ == "__main__":
    main()
    pass
