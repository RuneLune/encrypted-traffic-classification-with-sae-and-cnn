from matplotlib import pyplot as plt
from pandas import DataFrame
from pytorch_lightning import Trainer
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch import set_float32_matmul_precision

from models import TrafficCNN
from utils.id import TrafficName


def main() -> None:
    set_float32_matmul_precision("medium")

    model = TrafficCNN.load_from_checkpoint(
        "checkpoints/trafficcnn/TrafficCNN-epoch=000-val_loss=0.0000.ckpt"
    )
    model.eval()

    trainer = Trainer(accelerator="gpu", devices=[3])
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
    plot = heatmap(df, annot=True, cmap="Reds", fmt="02.02f")
    plot.figure.savefig("trafficcnn_confusion_matrix.png")

    return None


if __name__ == "__main__":
    main()
    pass
