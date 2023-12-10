from matplotlib import pyplot as plt
from pandas import DataFrame
from pytorch_lightning import Trainer
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch import set_float32_matmul_precision

from models import TrafficSAE, TrafficCNN, TrafficEnsemble
from utils.id import TrafficName


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
    ).load_from_checkpoint(
        "checkpoints/trafficensemble/TrafficEnsemble-epoch=000-val_loss=0.0000.ckpt"
    )
    model.eval()

    trainer = Trainer(accelerator="gpu", devices=[6], logger=False)
    predictions = trainer.predict(model)

    y = []
    y_hat = []

    for batch_pred in predictions:
        y.extend(batch_pred[0])
        y_hat.extend(batch_pred[1])
        continue

    cm = confusion_matrix(y, y_hat)
    df = DataFrame(cm / 3198 * 100, index=TrafficName, columns=TrafficName)
    plt.figure(figsize=(16, 9))
    plot = heatmap(df, annot=True, cmap="Purples", fmt="02.02f")
    plot.figure.savefig("trafficensemble_confusion_matrix.png")

    return None


if __name__ == "__main__":
    main()
    pass
