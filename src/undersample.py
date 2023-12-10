from json import load, dump
from pathlib import Path
from typing import Final, List

from numpy.random import seed
from joblib import Parallel, delayed

from utils.id import (
    AppName,
    TrafficName,
    get_app_id,
    get_traffic_id,
)
from preproc import AppUndersampler, TrafficUndersampler


app_types: Final[int] = len(AppName)
traffic_types: Final[int] = len(TrafficName)


def main(
    src_dir=str(Path(".") / "data" / "preprocessed"),
    dst_dir=str(Path(".") / "data" / "undersampled"),
    jobs=-1,
) -> None:
    seed(25)
    src_dir_path = Path(src_dir)
    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "app").mkdir(parents=True, exist_ok=True)
    (dst_dir_path / "traffic").mkdir(parents=True, exist_ok=True)
    app_count = [0] * (app_types)
    traffic_count = [0] * (traffic_types)
    for src_json_path in sorted(list(src_dir_path.glob("*.json"))):
        with open(src_json_path, "r") as f:
            data = load(f)
            pass
        file_name = src_json_path.name.split(".")[0]
        app_id = get_app_id(file_name)
        traffic_id = get_traffic_id(file_name)
        app_count[app_id] += data.get("total_rows")
        traffic_count[traffic_id] += data.get("total_rows")
        continue

    with open(dst_dir_path / "app_before.json", "w") as f:
        dump(app_count, f)
        pass
    with open(dst_dir_path / "traffic_before.json", "w") as f:
        dump(traffic_count, f)
        pass
    min_app = min(app_count)
    min_traffic = min(traffic_count)

    print(f"Minimum app: {min_app}")
    print(f"Minimum traffic: {min_traffic}")

    undersampler_list: List[AppUndersampler | TrafficUndersampler] = []

    for app_id in range(app_types):
        undersampler_list.append(
            AppUndersampler(app_id, min_app, src_dir_path, dst_dir_path / "app")
        )
        continue
    for traffic_id in range(traffic_types):
        undersampler_list.append(
            TrafficUndersampler(
                traffic_id, min_traffic, src_dir_path, dst_dir_path / "traffic"
            )
        )
        continue

    if jobs == 1:
        for undersampler in undersampler_list:
            undersampler.run()
            continue
        pass
    else:
        Parallel(n_jobs=jobs, verbose=10)(
            delayed(undersampler.run)() for undersampler in undersampler_list
        )
        pass

    return None


if __name__ == "__main__":
    main()
    pass
