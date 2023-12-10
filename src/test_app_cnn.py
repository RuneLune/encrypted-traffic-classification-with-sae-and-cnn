from matplotlib import pyplot as plt
from pandas import DataFrame
from pytorch_lightning import Trainer
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import set_float32_matmul_precision

from models import AppCNN
from utils.id import AppName


def main() -> None:
    set_float32_matmul_precision("medium")

    model = AppCNN.load_from_checkpoint(
        "checkpoints/appcnn/AppCNN-epoch=000-val_loss=0.0000.ckpt"
    )
    model.eval()

    trainer = Trainer(accelerator="gpu", devices=[1])
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
    plot = heatmap(df, annot=True, cmap="Reds", fmt="02.02f")
    plot.figure.savefig("appcnn_confusion_matrix.png")

    precision, recall, f1, _ = precision_recall_fscore_support(y, y_hat)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    return None


if __name__ == "__main__":
    main()
    pass
