from pathlib import Path

from joblib import Parallel, delayed

from preproc import PcapPreprocessor


def main(
    src_dir=str(Path(".") / "data" / "raw"),
    dst_dir=str(Path(".") / "data" / "preprocessed"),
    jobs=-1,
):
    src_dir_path = Path(src_dir)
    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    if jobs == 1:
        for src_pcap_path in sorted(src_dir_path.iterdir()):
            PcapPreprocessor(src_pcap_path)
    else:
        Parallel(n_jobs=jobs, verbose=10)(
            delayed(PcapPreprocessor)(src_pcap_path)
            for src_pcap_path in sorted(src_dir_path.iterdir())
        )


if __name__ == "__main__":
    main()
    pass
