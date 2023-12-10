from pathlib import Path

from torch import set_float32_matmul_precision
from torch.cuda import device_count
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models import AppSAE


def main() -> None:
    set_float32_matmul_precision("medium")

    model = AppSAE(n_jobs=4)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=128))

    checkpoint_path = Path("./checkpoints/appsae/")
    checkpoint_path.mkdir(exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_path,
            filename="AppSAE-{epoch:03d}-{val_loss:.4f}",
            save_top_k=3,
        )
    )

    logger_path = Path("tb_logs")
    logger_path.mkdir(exist_ok=True)

    logger = TensorBoardLogger(logger_path, name="AppSAE")

    trainer = Trainer(
        accelerator="gpu",
        devices=device_count(),
        # devices=[2],
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
