from pathlib import Path

from torch import set_float32_matmul_precision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models import TrafficSAE, TrafficCNN, TrafficEnsemble


def main() -> None:
    set_float32_matmul_precision("medium")

    appsae = TrafficSAE.load_from_checkpoint(
        "checkpoints/trafficsae/TrafficSAE-epoch=000-val_loss=0.0000.ckpt"
    )
    appcnn = TrafficCNN.load_from_checkpoint(
        "checkpoints/trafficcnn/TrafficCNN-epoch=000-val_loss=0.0000.ckpt"
    )
    model = TrafficEnsemble(
        appsae.hparams,
        appcnn.hparams,
        appsae.state_dict(),
        appcnn.state_dict(),
        n_jobs=4,
        device_id=6,
    )

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=128))

    checkpoint_path = Path("./checkpoints/trafficensemble/")
    checkpoint_path.mkdir(exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_path,
            filename="TrafficEnsemble-{epoch:03d}-{val_loss:.4f}",
            save_top_k=3,
        )
    )

    logger_path = Path("tb_logs")
    logger_path.mkdir(exist_ok=True)

    logger = TensorBoardLogger(logger_path, name="TrafficEnsemble", default_hp_metric=False)

    trainer = Trainer(
        accelerator="gpu",
        devices=[6],
        callbacks=callbacks,
        min_epochs=256,
        max_epochs=2048,
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    return None


if __name__ == "__main__":
    main()
    pass
