from matplotlib import pyplot as plt
from pandas import DataFrame
from pytorch_lightning import Trainer
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch import set_float32_matmul_precision

from models import AppSAE, AppCNN, AppEnsemble
from utils.id import AppName


def main() -> None:
    set_float32_matmul_precision("medium")

    appsae = AppSAE.load_from_checkpoint(
        "checkpoints/appsae/AppSAE-epoch=000-val_loss=0.0000.ckpt"
    )
    appcnn = AppCNN.load_from_checkpoint(
        "checkpoints/appcnn/AppCNN-epoch=000-val_loss=0.0000.ckpt"
    )
    model = AppEnsemble(
        appsae.hparams,
        appcnn.hparams,
        appsae.state_dict(),
        appcnn.state_dict(),
        n_jobs=4,
        device_id=5,
    ).load_from_checkpoint(
        "checkpoints/appensemble/AppEnsemble-epoch=099-val_loss=0.2409.ckpt"
    )
    model.eval()

    trainer = Trainer(accelerator="gpu", devices=[5], logger=False)
    predictions = trainer.predict(model)

    y = []
    y_hat = []

    for batch_pred in predictions:
        y.extend(batch_pred[0])
        y_hat.extend(batch_pred[1])
        continue

    cm = confusion_matrix(y, y_hat)
    df = DataFrame(cm / 1055 * 100, index=AppName, columns=AppName)
    plt.figure(figsize=(16, 9))
    plot = heatmap(df, annot=True, cmap="Purples", fmt="02.02f")
    plot.figure.savefig("appensemble_confusion_matrix.png")

    return None


if __name__ == "__main__":
    main()
    pass
